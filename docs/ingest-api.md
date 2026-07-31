# Ingest process and HTTP API

Canonical reference for the **ingest surface**: how a technical PDF becomes searchable chunks in Qdrant, and how the Cloud Run worker exposes that process over HTTP.

Design / GCP decisions: [gcp-ingest-pubsub.md](./gcp-ingest-pubsub.md).  
Serving/chat API is separate (`main_chat`); this document covers ingest only.

---

## Purpose

Index telecom standards PDFs (e.g. O-RAN) into the vector store so the chat agent can retrieve them.

| Concern | Behavior |
|---|---|
| Identity | `doc_id` = PDF filename stem (e.g. `WG1.pdf` → `WG1`) |
| Re-ingest | Delete existing vectors for that stem, then upsert (idempotent) |
| Shared core | Local CLI and Cloud Run both call `run_ingest` → `IngestDocumentService` |

---

## Process pipeline

```text
PDF on disk
  → parse (Docling / native pipeline → CanonicalDocument)
  → chunk (section or semantic)
  → embed (Cohere dense ± BM25 sparse)
  → Qdrant upsert
```

Hexagonal boundary: domain/application know only a **local path**. GCS and Pub/Sub stay in infrastructure drivers.

---

## Entrypoints

### Local CLI

```bash
uv run --group ingest python scripts/ingest_local.py path/to/doc.pdf
docker compose --profile ingest run --rm agent-ingest \
  python scripts/ingest_local.py /data/doc.pdf
```

- Builds the ingest service for that process, then `run_ingest(path)`.
- `--no-index` parses only (no chunk/embed/upsert).

### Cloud Run worker (`main_ingest`)

```text
GCS OBJECT_FINALIZE → Pub/Sub push → POST / → download → run_ingest → 2xx/4xx/5xx
```

- Image: `Dockerfile.ingest` (uvicorn `RAG_Agent.infrastructure.api.main_ingest:app`).
- On startup (lifespan): optional Secret Manager hydrate, then **one** `build_ingest_service()` kept warm and passed into `run_ingest(..., service=…)`.
- No Pub/Sub client in code; the app only receives the push HTTP envelope.
- Recommended deploy: `concurrency=1`, memory ≈ 8Gi (see GCP design doc).

---

## HTTP API

Base app: `RAG_Agent.infrastructure.api.main_ingest:app`.

OpenAPI UI: `/docs` when the service is running.

### `POST /`

Pub/Sub **push** endpoint. Body is the standard push envelope; GCS fields are read from **message attributes** (not custom app fields).

**Relevant attributes**

| Attribute | Role |
|---|---|
| `bucketId` | GCS bucket |
| `objectId` | Object name (may include prefixes, e.g. `docs/WG1.pdf`) |
| `objectGeneration` | Object version — logged only |
| `eventType` | If present and not `OBJECT_FINALIZE`, response **200** ignored (ack/drop) |

**Example body (abridged)**

```json
{
  "message": {
    "attributes": {
      "eventType": "OBJECT_FINALIZE",
      "bucketId": "my-project-input-docs",
      "objectId": "docs/WG1.pdf",
      "objectGeneration": "1710000000000000"
    }
  }
}
```

**Object types**

| Extension | Result |
|---|---|
| `.pdf` | Download to temp (basename preserved) → `run_ingest` |
| `.doc` / `.docx` | **400** — not supported yet |
| Other | **400** — unsupported |

**Status codes**

| Code | Meaning |
|---|---|
| **200** | Indexed (ack) or non-finalize ignored |
| **400** | Bad envelope, Word, or unsupported type |
| **404** | GCS object not found |
| **5xx** | Transient/unexpected failure (Pub/Sub retries → DLQ) |

**Success body (example)**

```json
{
  "status": "ok",
  "doc_id": "WG1",
  "chunk_count": 42,
  "deleted": "0",
  "upserted": "42"
}
```

There is no separate health HTTP route; Cloud Run can use a TCP probe on `$PORT`.

---

## Configuration (ingest worker)

| Mechanism | When |
|---|---|
| `.env` / env vars via pydantic `Settings` | Local and default |
| `USE_SECRET_MANAGER=true` | Cloud Run: load `cohere-api-key`, `qdrant-url`, `qdrant-api-key` from Secret Manager at startup |
| `GOOGLE_CLOUD_PROJECT` | Required for Secret Manager paths |
| `QDRANT_COLLECTION` | Default `tech_docs` |

Chat/serving secrets are out of scope for this API.

---

## Related code

| Piece | Location |
|---|---|
| HTTP worker | `src/RAG_Agent/infrastructure/api/main_ingest.py` |
| Envelope parse / type gate | `…/api/ingest_events.py` |
| Shared entry | `…/composition/ingest.py` → `run_ingest` |
| Use case | `…/application/ingest_documents_service/ingest_document.py` |
| GCS download | `…/infrastructure/storage/gcs.py` |
| Secrets | `…/infrastructure/secrets/gcp_secrets.py` |
| Local CLI | `scripts/ingest_local.py` |
