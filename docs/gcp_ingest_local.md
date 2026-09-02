# Local ingest

Index a PDF into the **Compose Qdrant** so local chat can retrieve it. Same `run_ingest` core as Cloud Run.

```text
PDF under data/ → agent-ingest (ingest_local.py) → parse → chunk → embed → qdrant container
```

Local testing is **Docker only**. Production ingest is GCS → Pub/Sub → Cloud Run ([gcp_ingest_deployment.md](./gcp_ingest_deployment.md)). This path does not use GCS, Pub/Sub, or Secret Manager.

| | |
|---|---|
| Files | PDF under `data/` → `/data/...` in the container. `doc_id` = filename stem. Same stem again **deletes** then upserts. |
| Qdrant | Always `http://qdrant:6333` (empty API key). Compose **ignores** `QDRANT_URL` in `.env`. |
| Secrets | Compose interpolates `COHERE_API_KEY` from `.env`. `USE_SECRET_MANAGER` stays false. |
| Chunker | `CHUNKER` from `.env` (default `section`). |

Commands are **Windows PowerShell**, from the **repository root**.

---



## Prerequisites

1. Docker Desktop.
2. `.env` with `COHERE_API_KEY` (Compose reads it for substitution only).
3. A PDF in `data/` (start with a short file).

NOTE: Compose ingest matches Cloud Run: CPU torch (`uv sync --extra cpu` in `Dockerfile.ingest`). No NVIDIA libs in the image. Host CLI with a GPU: `--extra cu124`.

Chat (`agent-api`) is optional until you want to ask about the doc. Postgres is not required to ingest.

---



## Ingest (Compose)

```powershell
docker compose up -d qdrant

docker compose --profile ingest run --rm agent-ingest `
    python scripts/ingest_local.py /data/your-doc.pdf
```

If chat is already up (`docker compose up -d db qdrant agent-api`), skip the first line: it is the same `qdrant` service.

Expect a log line with `doc_id`, `indexed=True`, `chunk_count`. Dashboard: [http://localhost:6333/dashboard](http://localhost:6333/dashboard) → collection `tech_docs`.

Parse only (no Qdrant write):

```powershell
docker compose --profile ingest run --rm agent-ingest `
    python scripts/ingest_local.py /data/your-doc.pdf --no-index
```

`data/` is mounted **read-only**. `--canonical-out` / `--json-out` cannot write under `/data`. For dumps, use the optional host CLI below.

Compose bind-mounts `./scripts` at `/app/scripts` so `ingest_local.py` is not baked into the Cloud Run image.

---



## Talk with your docs

Same Compose Qdrant as ingest (`tech_docs`). Postgres (`db`) is required for sessions; ingest did not start it.

**FastAPI** (`POST /chat`):

```powershell
docker compose up -d db qdrant agent-api
```

- Chat: `POST http://localhost:8000/chat`
- OpenAPI: [http://localhost:8000/docs](http://localhost:8000/docs)

**ADK Dev UI** (graphical ADK web UI):

```powershell
docker compose --profile dev up -d agent-dev-ui
```

Open [http://localhost:8080](http://localhost:8080). Compose starts `db` and `qdrant` if they are not already up.

---



## Optional: host CLI (parse dumps / GPU)

Does **not** share Qdrant with Docker chat unless `.env` has `QDRANT_URL=http://localhost:6333`. Prefer Compose to index.

```powershell
uv sync --group ingest --extra cpu
uv run --group ingest --extra cpu python scripts/ingest_local.py data\your-doc.pdf --no-index `
    --canonical-out data\out\your-doc.canonical.json `
    --canonical-md-out data\out\your-doc.canonical.md `
    --json-out data\out\your-doc.chunks.json `
    --md-out data\out\your-doc.chunks.md
```

Host NVIDIA GPU: replace `--extra cpu` with `--extra cu124`.

---



## Smoke check

1. CLI log: `doc_id=<stem> indexed=True chunk_count=<n>`.
2. Qdrant dashboard → `tech_docs`.
3. Chat: `POST http://localhost:8000/chat` about that stem.

Re-run the same path to replace vectors for that stem.

Do not run `main_ingest` or `gcloud` for this loop. Cloud ingest: [gcp_ingest_deployment.md](./gcp_ingest_deployment.md).
