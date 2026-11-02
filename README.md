# Adaptive Semantic Search Engine

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Service](#running-the-service)
- [Indexing Documents](#indexing-documents)
- [Performing a Search](#performing-a-search)
- [Testing](#testing)
- [Performance Tips](#performance-tips)
- [Contributing](#contributing)
- [License](#license)

## Overview
The **Adaptive Semantic Search Engine** is a production‑ready, end‑to‑end semantic search solution built on state‑of‑the‑art transformer models. It encodes raw text into dense vectors, stores them in a FAISS index, and serves similarity search via a FastAPI HTTP API. The system automatically adapts to new data by re‑encoding and updating the index without downtime.

## Features
- **Transformer‑based embeddings** – uses Hugging Face `sentence-transformers` models (e.g., `all-MiniLM-L6-v2`).
- **Scalable similarity search** – FAISS IVF‑PQ index for fast, approximate nearest‑neighbor lookup.
- **RESTful API** – FastAPI with automatic OpenAPI documentation.
- **Hot‑reloading of the index** – new documents can be added on the fly.
- **Docker ready** – official `Dockerfile` provided.
- **Comprehensive test suite** – `pytest` with coverage > 90 %.
- **Type‑hints & linting** – fully typed code, passes `flake8` and `mypy`.

## Architecture
```
┌─────────────────────┐   encode   ┌─────────────────────┐
│   FastAPI (Uvicorn) │──────────►│  Transformer Model  │
└─────────┬───────────┘            └─────────┬───────────┘
          │                                 │
          │   query vectors                 │   document vectors
          ▼                                 ▼
   ┌───────────────┐                ┌───────────────┐
   │   FAISS Index │◄──────────────►│  Vector Store │
   └───────────────┘                └───────────────┘
```

## Prerequisites
- Python ≥ 3.9
- `pip` (latest)
- (Optional) Docker ≥ 20.10 for containerized deployment
- A GPU is **recommended** for encoding large corpora but not required; CPU fallback is supported.

## Installation
```bash
# Clone the repository
git clone https://github.com/your-org/adaptive-semantic-search-engine.git
cd adaptive-semantic-search-engine

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install pre‑commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

### Docker (alternative)
```bash
docker build -t adaptive-semantic-search .
docker run -p 8000:8000 adaptive-semantic-search
```

## Configuration
All configurable options live in `config.yaml`. A minimal example:

```yaml
model_name: "sentence-transformers/all-MiniLM-L6-v2"
faiss:
  index_path: "./faiss_index"
  metric: "inner_product"
  nlist: 100
  nprobe: 10
api:
  host: "0.0.0.0"
  port: 8000
  reload: true   # only for development
logging:
  level: "INFO"
```

You can override any setting via environment variables using the `pydantic`‑compatible `APP_` prefix, e.g.:

```bash
export APP_API_PORT=8080
```

## Running the Service
```bash
# Load the config and start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The OpenAPI docs are available at `http://localhost:8000/docs`.

## Indexing Documents
The service ships a CLI helper for bulk indexing.

```bash
# Index a JSONL file where each line is {"id": "...", "text": "..."}
python -m scripts.index --input data/documents.jsonl --batch-size 64
```

The command:
- Loads the transformer model once.
- Encodes texts in batches.
- Adds vectors to the FAISS index (creates it if missing).
- Persists the index to the path defined in `config.yaml`.

### Incremental Updates
To add new documents without rebuilding the whole index:

```bash
python -m scripts.index --input data/new_docs.jsonl --incremental
```

## Performing a Search
Send a POST request to `/search` with a JSON payload:

```json
{
  "query": "deep learning optimization techniques",
  "top_k": 5,
  "filter": {
    "metadata.field": "value"
  }
}
```

Example using `curl`:

```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query":"machine translation", "top_k":10}'
```

Response format:

```json
{
  "results": [
    {"id": "doc_42", "score": 0.987, "metadata": {...}},
    {"id": "doc_17", "score": 0.965, "metadata": {...}}
  ],
  "query_vector": [0.12, -0.03, ...]   // optional, for debugging
}
```

## Testing
The test suite lives under `tests/`. Run it with:

```bash
pytest -vv --cov=app
```

Key test categories:
- **Unit tests** for model loading, encoding, and FAISS wrapper.
- **Integration tests** for the FastAPI endpoints (using `httpx.AsyncClient`).
- **Performance tests** that verify indexing time scales linearly with dataset size.

All tests are CI‑validated on Linux, macOS, and Windows.

## Performance Tips
- **GPU encoding**: Set `CUDA_VISIBLE_DEVICES` to the desired GPU ID. The encoder automatically uses `torch.cuda`.
- **FAISS tuning**: Increase `nlist` for larger corpora; adjust `nprobe` at query time for a trade‑off between speed and recall.
- **Batch size**: Larger batches improve GPU utilization but require more RAM. Typical values: 64–256.

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feat/awesome-feature`).
3. Write tests for your changes.
4. Ensure all linting passes (`pre-commit run --all-files`).
5. Open a Pull Request with a clear description and reference any related issues.

See `CONTRIBUTING.md` for detailed guidelines.

## License
This project is licensed under the **MIT License** – see the `LICENSE` file for details.