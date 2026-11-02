"""tests/test_main.py
Unit tests for the Adaptive Semantic Search Engine project.

These tests cover:
- Deterministic embedding generation.
- FAISS index construction and search.
- FastAPI endpoint behavior.

The tests use pytest and FastAPI's TestClient.
"""

import json
from typing import List

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Import the objects under test from the project's main module.
# Adjust the import path if the project's entry point has a different name.
from main import app, get_embedding, build_index, search  # type: ignore


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def dummy_documents() -> List[str]:
    """A small collection of documents used for indexing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence and machine learning are evolving rapidly.",
        "FastAPI provides high performance APIs with Python.",
        "FAISS enables efficient similarity search at scale.",
        "Transformers have revolutionized natural language processing.",
    ]


@pytest.fixture
def dummy_embedding(monkeypatch):
    """
    Monkey‑patch ``get_embedding`` to return a deterministic, low‑dimensional vector.
    This avoids loading large transformer models during testing.
    """

    def _dummy_embedding(text: str) -> np.ndarray:
        """
        Create a simple deterministic embedding:
        - length of the text
        - sum of Unicode code points
        - count of characters with an odd code point
        The result is a float32 vector of shape (3,).
        """
        length = len(text)
        code_sum = sum(ord(ch) for ch in text)
        odd_count = sum(1 for ch in text if ord(ch) % 2)
        return np.array([length, code_sum, odd_count], dtype=np.float32)

    monkeypatch.setattr("main.get_embedding", _dummy_embedding)
    return _dummy_embedding


def test_get_embedding_returns_expected_shape(dummy_embedding):
    """Verify that the (patched) embedding function returns a vector of length 3."""
    text = "sample query"
    embedding = get_embedding(text)  # patched version
    assert isinstance(embedding, np.ndarray), "Embedding should be a NumPy array"
    assert embedding.shape == (3,), f"Expected shape (3,), got {embedding.shape}"
    # Ensure deterministic values
    expected = dummy_embedding(text)
    np.testing.assert_array_equal(embedding, expected)


def test_faiss_index_construction_and_search(dummy_documents, dummy_embedding):
    """
    Build a FAISS index from dummy documents, perform a search,
    and verify that the most similar document is returned.
    """
    # Build the index using the (patched) deterministic embeddings.
    index = build_index(dummy_documents)

    # Perform a search with a query that is identical to the first document.
    query = dummy_documents[0]
    results = search(query, index, k=3)

    # ``results`` should be a list of dictionaries containing at least ``doc`` and ``score``.
    assert isinstance(results, list), "Search results should be a list"
    assert len(results) > 0, "Search should return at least one result"

    # The top result must be the identical document.
    top_result = results[0]
    assert "doc" in top_result and "score" in top_result, "Result items must contain 'doc' and 'score'"
    assert top_result["doc"] == dummy_documents[0], "The most similar document should be returned first"

    # Verify that scores are non‑negative and decreasing.
    scores = [r["score"] for r in results]
    assert all(s >= 0 for s in scores), "All scores should be non‑negative"
    assert scores == sorted(scores, reverse=True), "Scores should be sorted in descending order"


def test_search_endpoint_returns_correct_structure(client, dummy_documents, dummy_embedding):
    """
    Test the ``/search`` FastAPI endpoint.
    The endpoint is expected to accept JSON ``{\"query\": \"...\"}`` and return
    a JSON payload with a ``results`` list.
    """
    # Populate the in‑memory index used by the API.
    # The project may expose a helper to set the documents; if not, we rely on
    # the module-level state being initialised on import.
    # Here we explicitly rebuild the index to guarantee deterministic behaviour.
    index = build_index(dummy_documents)

    # Attach the index to the FastAPI app's state if the project follows that pattern.
    # This is a defensive step; if the app uses a different mechanism the test will
    # still pass because the endpoint will internally rebuild the index.
    if hasattr(app, "state"):
        app.state.index = index  # type: ignore[attr-defined]

    # Prepare a request payload.
    payload = {"query": dummy_documents[2]}  # use the third document as the query
    response = client.post("/search", json=payload)

    # Basic HTTP checks.
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    data = response.json()
    assert isinstance(data, dict), "Response JSON should be a dictionary"
    assert "results" in data, "Response must contain a 'results' key"
    assert isinstance(data["results"], list), "'results' should be a list"
    assert len(data["results"]) > 0, "Results list should not be empty"

    # Verify the top result matches the query document.
    top_result = data["results"][0]
    assert "doc" in top_result and "score" in top_result, "Each result must contain 'doc' and 'score'"
    assert top_result["doc"] == dummy_documents[2], "The most relevant document should be returned first"

    # Ensure scores are numeric.
    for item in data["results"]:
        assert isinstance(item["score"], (int, float)), "Score must be a numeric type"


def test_search_endpoint_handles_missing_query(client):
    """
    The endpoint should return a 422 Unprocessable Entity error when the ``query``
    field is missing from the request payload.
    """
    response = client.post("/search", json={})
    assert response.status_code == 422, f"Expected 422 for missing query, got {response.status_code}"
    error_detail = response.json()
    assert "detail" in error_detail, "Error response should contain a 'detail' field"


def test_search_endpoint_handles_empty_query(client):
    """
    An empty query string should be processed without error and return an empty
    results list (or whatever the implementation defines). The test asserts that
    the response is well‑formed.
    """
    response = client.post("/search", json={"query": ""})
    assert response.status_code == 200, f"Expected 200 OK for empty query, got {response.status_code}"
    data = response.json()
    assert "results" in data, "Response must contain a 'results' key"
    # The exact behaviour (empty list vs. all documents) depends on implementation;
    # we only assert that the field exists and is a list.
    assert isinstance(data["results"], list), "'results' should be a list"
