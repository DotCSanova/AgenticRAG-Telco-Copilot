# Agentic RAG Engineering Copilot for Telco

Technical copilot based on Agentic-RAG for telecom engineers working with standards docs (e.g. O-RAN).

## Surfaces

| Surface | Role | Image / entrypoint |
|---|---|---|
| **Serving (chat)** | `/chat`, `/reset-memory`, `/eval` stub | `Dockerfile.serving` → `main_chat` |
| **Ingest** | Parse PDF → chunk → embed → Qdrant (delete+upsert by stem); Cloud Run `POST /` push worker | `Dockerfile.ingest` → `main_ingest` (CLI: `scripts/ingest_local.py`). [docs/ingest.md](docs/ingest.md) |
| **Dev UI** | ADK web UI (same serving image) | compose profile `dev` |

Chat never loads Docling/torch. Ingest never loads ADK.

## Local setup

```bash
# Default DX: chat + tests (no Docling/torch)
uv sync

# Host-side indexing / GPU torch (optional; compose ingest is preferred)
uv sync --group ingest
```

Copy `.env` with `COHERE_API_KEY` and Qdrant/session settings as needed.

## Chat (Docker)

```bash
docker compose up -d postgres qdrant agent-api
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d "{\"message\":\"ping\",\"user_id\":\"t\"}"
```

ADK Dev UI:

```bash
docker compose --profile dev up -d agent-dev-ui
# http://localhost:8080
```

## Ingest

How it works: [docs/ingest.md](docs/ingest.md).  
GCP deploy: [docs/gcp_ingest_deployment.md](docs/gcp_ingest_deployment.md).

### Ingest a PDF (local)

Preferred (CPU image; override default worker CMD):

```bash
docker compose up -d qdrant
docker compose --profile ingest run --rm agent-ingest \
  python scripts/ingest_local.py /data/your-doc.pdf
```

Host script (needs `--group ingest`; uses CUDA torch on Windows/Linux when available):

```bash
uv run --group ingest python scripts/ingest_local.py path/to/doc.pdf
```

Re-running the same path replaces prior chunks for that PDF stem (`delete_by_doc_id` + upsert). Parse-only: add `--no-index`.

## Dependency groups

| Group | Installs |
|---|---|
| *(default)* `serving` + `dev` | FastAPI, ADK, LiteLLM, sessions, pytest, … |
| `ingest` | Docling, torch, pymupdf, sentence-transformers |
| `notebooks` | ipykernel |

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [agent-api-cookiecutter](https://github.com/neural-maze/agent-api-cookiecutter) project template.
