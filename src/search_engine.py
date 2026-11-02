import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from transformers import AutoModel, AutoTokenizer

# Optional: use NLTK WordNet for simple query expansion.
# If NLTK data is not present, the expansion will gracefully fallback to the original query.
try:
    import nltk

    nltk.data.find("corpora/wordnet")
    from nltk.corpus import wordnet

    _WORDNET_AVAILABLE = True
except Exception:  # pragma: no cover
    _WORDNET_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Perform mean pooling on token embeddings.

    Args:
        model_output: Tensor of shape (batch_size, seq_len, hidden_dim).
        attention_mask: Tensor of shape (batch_size, seq_len) with 1 for real tokens.

    Returns:
        Tensor of shape (batch_size, hidden_dim) representing sentence embeddings.
    """
    token_embeddings = model_output  # (batch_size, seq_len, hidden_dim)
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class Document(BaseModel):
    """Model representing a document to be indexed."""

    id: str = Field(..., description="Unique identifier for the document.")
    text: str = Field(..., description="Raw textual content of the document.")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Optional auxiliary information."
    )


class SearchRequest(BaseModel):
    """Model representing a search request."""

    query: str = Field(..., description="User query string.")
    top_k: int = Field(5, ge=1, description="Number of top results to return.")
    expand: bool = Field(
        True,
        description="Whether to perform query expansion before searching.",
    )


class SearchResult(BaseModel):
    """Model representing a single search result."""

    id: str
    text: str
    metadata: Dict[str, Any]
    score: float


class SemanticSearchEngine:
    """
    Core engine handling embedding generation, FAISS indexing,
    and optional query expansion.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialise the engine.

        Args:
            model_name: HuggingFace model identifier for embedding generation.
            index_path: Optional path to a persisted FAISS index and document store.
            device: Torch device; defaults to CUDA if available.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading tokenizer and model %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self._lock = threading.Lock()
        self.doc_store: List[Dict[str, Any]] = []  # In‑memory storage of documents

        # Initialise FAISS index (inner product for cosine similarity after normalisation)
        self.dimension = self.model.config.hidden_size
        self.index = faiss.IndexFlatIP(self.dimension)

        if index_path and os.path.isdir(index_path):
            self._load(index_path)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of texts into L2‑normalised embeddings.

        Args:
            texts: List of raw strings.

        Returns:
            Numpy array of shape (len(texts), dimension) with dtype float32.
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded)
            embeddings = _mean_pooling(model_output.last_hidden_state, encoded["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().astype(np.float32)

    def add_documents(self, documents: List[Document]) -> None:
        """
        Index a batch of documents.

        Args:
            documents: List of Document instances.
        """
        if not documents:
            return

        texts = [doc.text for doc in documents]
        embeddings = self._embed_texts(texts)

        with self._lock:
            start_id = self.index.ntotal
            self.index.add(embeddings)
            for i, doc in enumerate(documents):
                self.doc_store.append(
                    {
                        "id": doc.id,
                        "text": doc.text,
                        "metadata": doc.metadata or {},
                        "faiss_id": start_id + i,
                    }
                )
        logger.info("Added %d documents to the index (total=%d).", len(documents), self.index.ntotal)

    def _expand_query(self, query: str, max_synonyms: int = 5) -> List[str]:
        """
        Generate simple synonym‑based expansions using WordNet.

        Args:
            query: Original query string.
            max_synonyms: Maximum number of synonyms per token.

        Returns:
            List containing the original query and its expansions.
        """
        if not _WORDNET_AVAILABLE:
            logger.debug("WordNet not available – skipping query expansion.")
            return [query]

        tokens = query.split()
        expansions = set([query])

        for i, token in enumerate(tokens):
            synonyms = set()
            for syn in wordnet.synsets(token):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace("_", " ")
                    if synonym.lower() != token.lower():
                        synonyms.add(synonym)
                if len(synonyms) >= max_synonyms:
                    break
            for synonym in list(synonyms)[:max_synonyms]:
                new_tokens = tokens.copy()
                new_tokens[i] = synonym
                expansions.add(" ".join(new_tokens))

        logger.debug("Query expansion generated %d variants.", len(expansions))
        return list(expansions)

    def _search_embeddings(
        self, query_embeddings: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a FAISS search.

        Args:
            query_embeddings: (n_queries, dim) array.
            top_k: Number of neighbours to retrieve.

        Returns:
            distances and indices arrays of shape (n_queries, top_k).
        """
        if self.index.ntotal == 0:
            raise ValueError("The index is empty – add documents before searching.")
        distances, indices = self.index.search(query_embeddings, top_k)
        return distances, indices

    def query(
        self,
        query: str,
        top_k: int = 5,
        expand: bool = True,
    ) -> List[SearchResult]:
        """
        Perform a semantic search.

        Args:
            query: User supplied query string.
            top_k: Number of results to return.
            expand: Whether to apply query expansion.

        Returns:
            List of SearchResult objects sorted by descending similarity.
        """
        queries = self._expand_query(query) if expand else [query]
        query_embeddings = self._embed_texts(queries)

        # Average embeddings of all query variants to obtain a single representation.
        combined_embedding = np.mean(query_embeddings, axis=0, keepdims=True)
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding, axis=1, keepdims=True)

        distances, indices = self._search_embeddings(combined_embedding, top_k)

        results: List[SearchResult] = []
        with self._lock:
            for score, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.doc_store):
                    continue
                doc = self.doc_store[idx]
                results.append(
                    SearchResult(
                        id=doc["id"],
                        text=doc["text"],
                        metadata=doc["metadata"],
                        score=float(score),
                    )
                )
        # FAISS with inner product returns higher scores for more similar vectors.
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def save(self, directory: str) -> None:
        """
        Persist the FAISS index and document store to ``directory``.

        Args:
            directory: Target folder path.
        """
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, "faiss.index")
        meta_path = os.path.join(directory, "documents.json")
        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_store, f, ensure_ascii=False, indent=2)
        logger.info("Search engine state saved to %s.", directory)

    def _load(self, directory: str) -> None:
        """
        Load a previously saved index and document store.

        Args:
            directory: Folder containing ``faiss.index`` and ``documents.json``.
        """
        index_path = os.path.join(directory, "faiss.index")
        meta_path = os.path.join(directory, "documents.json")
        if not os.path.isfile(index_path) or not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Index files not found in {directory}")

        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.doc_store = json.load(f)
        logger.info(
            "Loaded FAISS index with %d vectors and %d documents from %s.",
            self.index.ntotal,
            len(self.doc_store),
            directory,
        )


# --------------------------------------------------------------------------- #
# FastAPI integration
# --------------------------------------------------------------------------- #

router = APIRouter()
engine = SemanticSearchEngine()


@router.post("/index", status_code=201, summary="Add documents to the index")
def index_documents(docs: List[Document]) -> Dict[str, Any]:
    """
    Endpoint to ingest a batch of documents.
    """
    try:
        engine.add_documents(docs)
        return {"status": "success", "added": len(docs)}
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to index documents")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/search",
    response_model=List[SearchResult],
    summary="Perform a semantic search",
)
def search(request: SearchRequest) -> List[SearchResult]:
    """
    Endpoint to execute a semantic search with optional query expansion.
    """
    try:
        return engine.query(
            query=request.query,
            top_k=request.top_k,
            expand=request.expand,
        )
    except ValueError as ve:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:  # pragma: no cover
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(exc))


def create_app() -> FastAPI:
    """
    Factory function to create a FastAPI application with the search router.
    """
    app = FastAPI(title="Adaptive Semantic Search Engine")
    app.include_router(router)
    return app


__all__ = [
    "SemanticSearchEngine",
    "Document",
    "SearchRequest",
    "SearchResult",
    "create_app",
]