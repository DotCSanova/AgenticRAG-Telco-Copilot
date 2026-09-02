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

# Host-side Docling (optional; local indexing uses compose ingest)
# CPU wheels: --extra cpu. NVIDIA GPU: --extra cu124.
uv sync --group ingest --extra cpu
```

Copy `.env` with `COHERE_API_KEY` and Qdrant/session settings as needed.

## Chat (Docker)

```bash
docker compose up -d db qdrant agent-api
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
Local commands: [docs/gcp_ingest_local.md](docs/gcp_ingest_local.md).  
GCP deploy: [docs/gcp_ingest_deployment.md](docs/gcp_ingest_deployment.md).

### Ingest a PDF (local)

Compose Qdrant only (same store as `agent-api`). Put the file in `data/`:

```bash
docker compose up -d qdrant
docker compose --profile ingest run --rm agent-ingest \
  python scripts/ingest_local.py /data/your-doc.pdf
```

Production ingest is Cloud Run ([docs/gcp_ingest_deployment.md](docs/gcp_ingest_deployment.md)), not this service.

Host `uv run … ingest_local.py` follows `.env` `QDRANT_URL` (can be Qdrant Cloud). Use it for `--no-index` dumps, not to feed local chat.

Re-running the same path replaces prior chunks for that PDF stem (`delete_by_doc_id` + upsert).

## Dependency groups

| Group | Installs |
|---|---|
| *(default)* `serving` + `dev` | FastAPI, ADK, LiteLLM, sessions, pytest, … |
| `ingest` | Docling, pymupdf, sentence-transformers. Add `--extra cpu` or `--extra cu124` for torch. |
| `notebooks` | ipykernel |

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [agent-api-cookiecutter](https://github.com/neural-maze/agent-api-cookiecutter) project template.
