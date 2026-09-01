# Ingest

Index a technical PDF into Qdrant so chat can retrieve it.

```text
PDF → parse (Docling) → chunk → embed → Qdrant upsert
```

`doc_id` is the filename stem (`WG1.pdf` → `WG1`). Ingesting the same stem again **deletes** existing vectors for that id, then upserts.

Local CLI and Cloud Run call the same `run_ingest` path. Domain code sees a **local file**; GCS and Pub/Sub stay in infrastructure.

## Local

Command runbook: [gcp_ingest_local.md](./gcp_ingest_local.md).

```bash
uv sync --group ingest
uv run --group ingest python scripts/ingest_local.py path/to/doc.pdf
```

Docker (CPU image):

```bash
docker compose up -d qdrant
docker compose --profile ingest run --rm agent-ingest \
  python scripts/ingest_local.py /data/doc.pdf
```

Parse without indexing: `--no-index`. Optional dumps: `--canonical-out`, `--canonical-md-out`, `--json-out`, `--md-out`.

## Cloud Run

GCS `OBJECT_FINALIZE` → Pub/Sub **push** → `POST /` on `Dockerfile.ingest` (`main_ingest`).

Deploy: [gcp_ingest_deployment.md](./gcp_ingest_deployment.md).

The app does not use a Pub/Sub client. It only handles the push HTTP envelope. Secrets are loaded in-process when `USE_SECRET_MANAGER=true` (not Cloud Run `--set-secrets`).

### `POST /`

GCS fields come from **message attributes**:

| Attribute | Role |
|---|---|
| `bucketId` | Bucket |
| `objectId` | Object name (prefixes allowed, e.g. `docs/WG1.pdf`) |
| `objectGeneration` | Logged only |
| `eventType` | If set and not `OBJECT_FINALIZE` → **200** ignored (ack, no ingest) |

| Object | Result |
|---|---|
| `.pdf` | Download → `run_ingest(..., index=True)` |
| `.doc` / `.docx` | **400** |
| Other | **400** |

| HTTP | Meaning |
|---|---|
| **200** | Indexed, or non-finalize ignored |
| **400** | Bad envelope or unsupported type (Pub/Sub does not retry) |
| **404** | Object missing in GCS |
| **5xx** | Failure; Pub/Sub retries, then dead-letter topic |

Indexed body:

```json
{
  "status": "ok",
  "doc_id": "WG1",
  "chunk_count": 42,
  "deleted": "0",
  "upserted": "42"
}
```

Ignored body (non-finalize):

```json
{
  "status": "ignored",
  "eventType": "OBJECT_DELETE"
}
```

No HTTP health route. Cloud Run probes TCP on `$PORT` (image listens on **8080**). OpenAPI UI: `/docs`.

**Ack:** Pub/Sub push deadline is **600 seconds**. If ingest is still running, Pub/Sub redelivers. Use `--concurrency=1` and `--memory=8Gi`. Cloud Run `--timeout=3600` does not raise that 600 s ceiling.

## Configuration

| Variable | Role |
|---|---|
| `.env` / pydantic `Settings` | Local default |
| `USE_SECRET_MANAGER=true` | Load `cohere-api-key`, `qdrant-url`, `qdrant-api-key` from Secret Manager |
| `GOOGLE_CLOUD_PROJECT` | Project id for Secret Manager |
| `CHUNKER` | `section` or `semantic` (local default `semantic`; deploy sets `section`) |
| `INGEST_PROFILE` | `local` or `cloud` (deploy sets `cloud`) |
| `QDRANT_COLLECTION` | Default `tech_docs` |

## Code

| | |
|---|---|
| HTTP worker | `src/RAG_Agent/infrastructure/api/main_ingest.py` |
| Envelope / file type | `src/RAG_Agent/infrastructure/api/ingest_events.py` |
| `run_ingest` | `src/RAG_Agent/infrastructure/composition/ingest.py` |
| GCS download | `src/RAG_Agent/infrastructure/storage/gcs.py` |
| Secrets | `src/RAG_Agent/infrastructure/secrets/gcp_secrets.py` |
| CLI | `scripts/ingest_local.py` |
