import os
import logging
from typing import List, Dict, Any

import numpy as np
import faiss
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from transformers import AutoTokenizer, AutoModel
import uvicorn

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("adaptive_semantic_search")

# --------------------------------------------------------------------------- #
# Pydantic models for request/response validation
# --------------------------------------------------------------------------- #
class Document(BaseModel):
    """Schema representing a document to be indexed."""

    id: str = Field(..., description="Unique identifier of the document")
    text: str = Field(..., description="Raw text content of the document")

    @validator("id")
    def id_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Document id must be a non‑empty string")
        return v

    @validator("text")
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Document text must be a non‑empty string")
        return v


class SearchResult(BaseModel):
    """Schema representing a single search result."""

    id: str = Field(..., description="Identifier of the matched document")
    score: float = Field(..., description="Similarity score (higher is more similar)")


# --------------------------------------------------------------------------- #
# Core search engine implementation
# --------------------------------------------------------------------------- #
class AdaptiveSemanticSearchEngine:
    """
    Adaptive Semantic Search Engine powered by a Transformer encoder and FAISS.

    The engine maintains an in‑memory FAISS index (inner‑product) and a mapping
    from vector IDs to original document metadata.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        """
        Initialise the encoder model and FAISS index.

        Args:
            model_name: HuggingFace model identifier.
            device: Torch device string (e.g., "cpu", "cuda"). If ``None``, auto‑detect.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading model %s on %s", model_name, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Determine embedding dimension from the model's hidden size
        hidden_size = self.model.config.hidden_size
        self.dimension = hidden_size
        logger.info("Embedding dimension: %d", self.dimension)

        # FAISS index for inner product (cosine similarity after normalisation)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_to_doc: Dict[int, Document] = {}
        self._next_vector_id = 0

    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling on token embeddings.

        Args:
            model_output: Last hidden state from the transformer (batch, seq_len, hidden).
            attention_mask: Mask indicating non‑padding tokens (batch, seq_len).

        Returns:
            Tensor of shape (batch, hidden) representing pooled sentence embeddings.
        """
        token_embeddings = model_output  # (batch, seq_len, hidden)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of texts into L2‑normalised embeddings.

        Args:
            texts: List of raw strings.

        Returns:
            Numpy array of shape (len(texts), dimension) with dtype ``float32``.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self._mean_pooling(model_output.last_hidden_state, attention_mask)

        # L2‑normalise for cosine similarity via inner product
        embeddings = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return embeddings.cpu().numpy().astype(np.float32)

    def add_documents(self, docs: List[Document]) -> int:
        """
        Index a batch of documents.

        Args:
            docs: List of :class:`Document` objects.

        Returns:
            Number of documents successfully added.
        """
        if not docs:
            return 0

        texts = [doc.text for doc in docs]
        embeddings = self.embed(texts)

        # FAISS expects a contiguous float32 array
        self.index.add(embeddings)

        for doc, _ in zip(docs, embeddings):
            self.id_to_doc[self._next_vector_id] = doc
            self._next_vector_id += 1

        logger.info("Added %d documents to the index", len(docs))
        return len(docs)

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """
        Perform a semantic search for the given query.

        Args:
            query: Query string.
            k: Number of top results to return.

        Returns:
            List of :class:`SearchResult` ordered by decreasing similarity.
        """
        if not query.strip():
            raise ValueError("Query string must be non‑empty")

        query_emb = self.embed([query])  # shape (1, dim)
        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_emb, k)  # inner product scores

        results: List[SearchResult] = []
        for score, idx in zip(distances[0], indices[0]):
            doc = self.id_to_doc.get(int(idx))
            if doc is None:
                continue
            results.append(SearchResult(id=doc.id, score=float(score)))
        return results


# --------------------------------------------------------------------------- #
# FastAPI application setup
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Adaptive Semantic Search Engine",
    description="A lightweight semantic search service based on Transformers and FAISS.",
    version="0.1.0",
)

# Enable CORS for all origins (adjust in production as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise the search engine singleton
engine = AdaptiveSemanticSearchEngine(
    model_name=os.getenv("ASSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
)


@app.get("/healthz", tags=["Health"])
def health_check() -> Dict[str, Any]:
    """Simple health‑check endpoint."""
    return {"status": "ok", "indexed_documents": engine.index.ntotal}


@app.post("/index", tags=["Index"], response_model=Dict[str, int])
def index_documents(docs: List[Document]) -> Dict[str, int]:
    """
    Index a batch of documents.

    Returns the number of documents added.
    """
    try:
        added = engine.add_documents(docs)
        return {"added": added}
    except Exception as exc:
        logger.exception("Failed to index documents")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/search", tags=["Search"], response_model=List[SearchResult])
def search(
    query: str = Query(..., description="Query string"),
    k: int = Query(5, ge=1, le=100, description="Number of results to return"),
) -> List[SearchResult]:
    """
    Perform a semantic search over the indexed corpus.
    """
    try:
        results = engine.search(query, k)
        return results
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(exc))


# --------------------------------------------------------------------------- #
# Entry point for running with ``python -m src.main``
# --------------------------------------------------------------------------- #
def _run() -> None:
    """
    Run the FastAPI application with Uvicorn.
    """
    uvicorn.run(
        "src.main:app",
        host=os.getenv("ASSE_HOST", "0.0.0.0"),
        port=int(os.getenv("ASSE_PORT", "8000")),
        log_level="info",
        reload=os.getenv("ASSE_RELOAD", "false").lower() == "true",
    )


if __name__ == "__main__":
    _run()
