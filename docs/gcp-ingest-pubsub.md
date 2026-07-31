# GCP ingest — Pub/Sub + GCS + Cloud Run Service

Next phase after [serving-ingest-split.md](./serving-ingest-split.md).  
Design north: [split-ingest-serving.md](./split-ingest-serving.md) §7.1.  
**API / process reference (canonical for operators and OpenAPI):** [ingest-api.md](./ingest-api.md).

**Branch:** `feature/gcp-ingest-pubsub`  
**Goal:** event-driven indexing in GCP without Docling in the chat service. Same application core as local CLI: `run_ingest(path)` → `IngestDocumentService` (delete-by-stem + upsert).

**Runtime shape (crowned):**

```text
Upload / ops ──► GCS (docs)
                   │ OBJECT_FINALIZE notification
                   ▼
                Pub/Sub topic
                   │ push subscription (OIDC → Cloud Run invoker)
                   ▼
                Cloud Run Ingest Service
                   POST /  (Pub/Sub push envelope)
                       │
                       ├─ parse attributes: bucketId, objectId, objectGeneration
                       ├─ validate extension (.pdf | .doc | .docx)
                       ├─ Word → 400 (not supported yet)
                       ├─ PDF → download GCS → temp → run_ingest → cleanup
                       └─ Qdrant (Cloud)
```

**Not in this phase:** Cloud Run **Job** on the hot path (Jobs reserved for scraper / backfills). Multi-agent / web search. Fan-Out Docling. Canonical Fan-In (§7.4). Terraform / CI-CD deploy. Word→PDF converter. SQL ingest registry. Local Pub/Sub emulator.

**Manual GCP setup:** commands live in a dedicated runbook markdown (created in M3/M4), not in application code. CI/CD comes later.

---

## 1. Design decisions (closed — do not reopen)

### 1.1 Architecture & surfaces

| Topic | Decision |
|---|---|
| Trigger | GCS → Pub/Sub → **Cloud Run Service** (HTTP push), not Job |
| Shared core | `run_ingest(path, *, index=True) -> IngestResult` in `composition/ingest.py` |
| CLI | `scripts/ingest_local.py` becomes a thin wrapper over `run_ingest` |
| Re-ingest / idempotency | `delete_by_doc_id` + upsert; `doc_id` = file stem |
| Chat API | No sync `POST /ingest` on serving; `/eval` stub **kept** for future |
| Images | `Dockerfile.serving` + `Dockerfile.ingest` (ingest image = worker HTTP + CLI override) |
| Hexagonal | GCS + Pub/Sub envelope parsing stay in `infrastructure/`; domain/application free of GCP SDKs |
| SQL doc registry | **No** in this phase (Qdrant + structured logs only) |
| Local compose HTTP | **No** ingest HTTP service in compose; CLI + unit tests only |

### 1.2 HTTP contract (Ingest Service)

| Topic | Decision |
|---|---|
| Entrypoint | `main_ingest.py` (minimal FastAPI app) |
| Push path | **`POST /`** — same pattern as the reference Telco ingest service |
| Health endpoint | **None** — Cloud Run TCP/startup on `$PORT` is enough for this single-purpose worker |
| Handler style | Sync `POST /` (blocking download + ingest; `concurrency=1`) |
| Pub/Sub client in app | **None.** App only receives push HTTP. No Publisher/Subscriber, no topic/subscription IDs in code |
| Envelope | Standard Pub/Sub push JSON; GCS fields from **message attributes** |
| Attribute fields | `bucketId`, `objectId`, `objectGeneration` (optional: `eventType`) |
| Auth to Cloud Run | **IAM only**: push subscription uses a service account with `roles/run.invoker`. Service not `allUsers`. App does **not** validate OIDC JWT in v1 |
| Lifespan | Build `IngestDocumentService` once at startup (warm), same idea as chat |

**Envelope shape (conceptual):**

```json
{
  "message": {
    "attributes": {
      "eventType": "OBJECT_FINALIZE",
      "bucketId": "<bucket>",
      "objectId": "path/to/doc.pdf",
      "objectGeneration": "<generation>"
    },
    "data": "<base64 optional JSON_API_V1 metadata>",
    "messageId": "...",
    "publishTime": "..."
  },
  "subscription": "projects/.../subscriptions/..."
}
```

Wiring example (names illustrative; exact names in the runbook):

```text
Topic:        pdf-ingestion-topic
Notification: GCS OBJECT_FINALIZE → topic
Subscription: pdf-processor-sub (push → https://<ingest-service>.run.app/)
```

### 1.3 File types & validation

| Topic | Decision |
|---|---|
| Accepted for future product | `.pdf`, `.doc`, `.docx` |
| Processed in this phase | **`.pdf` only** → download + `run_ingest` |
| Word (`.doc` / `.docx`) | **HTTP 400** — `not supported yet` (converter later). Avoids poison retries on unsupported parse |
| Other extensions | **HTTP 400** |
| GCS notification filter | **No** suffix-only PDF filter on the bucket notification (Word may land later). Worker validates |
| `objectGeneration` | **Log only.** No skip-if-already-indexed (needs registry = M5) |

### 1.4 Ack model, timeouts, concurrency

| Topic | Decision |
|---|---|
| Ack model | **Synchronous:** download + ingest complete before `200`. No “202 + background work” in this phase |
| Success | HTTP **2xx** → Pub/Sub ack |
| Client / bad input | **400** (bad envelope, unsupported type, Word) → no useful retry |
| Missing object | **404** (object gone from GCS) |
| Transient failure | **5xx** → Pub/Sub retry → eventually DLQ |
| Idempotency under redelivery | Stem delete+upsert (already implemented) |
| Pub/Sub **ack deadline** | **600 s (max).** Hard limit for push; Pub/Sub may redeliver after ~10 min if still processing |
| Cloud Run **request timeout** | High (document **3600 s** in runbook; adjustable). Does not extend Pub/Sub’s 10 min ack ceiling |
| Measure in prod | Time real O-RAN PDFs. If P95 approaches/exceeds ~8–9 min, revisit ack pattern (async or Job) — **out of this phase unless forced** |
| Concurrency | **`1` per instance** (must set explicitly; Cloud Run default is ~80 with ≥1 vCPU — unsafe with Docling) |
| Memory (starting point) | **8 Gi** (tune after measuring) |
| Parallelism | Horizontal: more instances, not more concurrent requests per instance |

**Why sync + 10 min matters:**

```text
t=0      Pub/Sub POST / → ingest starts
t≤10min  Prefer finish and return 200
t>10min  Pub/Sub may redeliver while first request still runs
         → second instance may process same object (idempotent index, double cost / race)
```

### 1.5 GCS download & dependencies

| Topic | Decision |
|---|---|
| Storage abstraction | Infra **helper** (no domain `BlobStore` port until a second backend exists) |
| Auth | ADC / Cloud Run runtime SA (no keys in image) |
| Temp files | Download to temp path; **always** cleanup in `finally` |
| `pyproject` group `ingest` | Add **`fastapi`**, **`google-cloud-storage`**, **`google-cloud-secret-manager`** (GCP secrets) |
| `google-cloud-pubsub` | **Not** added — push receiver does not need the client library |
| ADK / LiteLLM | Stay out of ingest image |

### 1.6 Ops / infra provisioning

| Topic | Decision |
|---|---|
| IaC | **No Terraform** in this phase |
| CI/CD deploy | **Later** |
| GCP wiring | Dedicated **manual runbook** markdown with `gcloud` + PowerShell placeholders (M3/M4) |
| DLQ | Dead-letter topic/subscription in runbook; `maxDeliveryAttempts=5`; idempotent ingest |
| Chat Cloud Run slim | Ops checklist in runbook; **not** a blocker for ingest M0–M2 code (§8.5) |
| Secrets | `USE_SECRET_MANAGER=true` → `get_secret` on **ingest only**; local `.env` / Settings otherwise; collection = `QDRANT_COLLECTION` / `tech_docs` (§8.6) |
| Scale flags | `gcloud run deploy`: cpu=2, memory=8Gi, concurrency=1, max-instances=3, min=0, timeout=3600 |

---

## 2. HTTP status taxonomy (worker)

| Status | When |
|---|---|
| **200** | PDF ingested successfully (ack) |
| **400** | Invalid envelope; missing attributes; unsupported extension; Word not supported yet |
| **404** | GCS object not found |
| **5xx** | Transient / unexpected (Qdrant, Cohere, Docling crash, etc.) → retry |

Structured logs (minimum): `bucket`, `object`, `objectGeneration`, `doc_id`, `chunk_count`, `deleted`, `upserted`, outcome/error.

---

## 3. Target code layout (implementation)

```text
src/RAG_Agent/
  infrastructure/
    api/main_ingest.py           # POST / push handler (sync); OpenAPI description ← docs/ingest-api.md
    storage/…                    # GCS download helper (exact module name at impl time)
    composition/ingest.py        # build_ingest_service + run_ingest
scripts/ingest_local.py          # thin CLI → run_ingest
Dockerfile.ingest                # CMD: uvicorn main_ingest (CLI via override)
docs/gcp-ingest-pubsub.md        # this design
docs/gcp-ingest-runbook.md       # manual gcloud (M3/M4; create if missing)
tests/…                          # envelope parse, extension gate, GCS fake, handler
```

Keep `/eval` on `main_chat`. Do not reintroduce monolith `main.py`.

---

## 4. Milestones

### M0 — Shared `run_ingest(path)` entry

- [x] `run_ingest(path: Path, *, index: bool = True) -> IngestResult` in `composition/ingest.py`
- [x] `scripts/ingest_local.py` thin CLI over `run_ingest`
- [x] Unit tests unchanged in spirit (idempotent ingest still green)

**Done when:** local `uv run` / compose ingest still work; one function is the only indexing entry.

---

### M1 — GCS download helper

- [x] Infra helper: download `bucket` + `object` to a temp file
- [x] Cleanup temp in `finally`
- [x] Auth via ADC (no keys in image)
- [x] `google-cloud-storage` in group `ingest`
- [x] Tests with fake storage client / recorded bytes

**Done when:** given bucket+object, worker can materialize a local path for `run_ingest`.

---

### M2 — Pub/Sub push handler (Cloud Run Ingest Service)

- [x] `main_ingest.py`: lifespan warm + sync `POST /`
- [x] Parse attributes → validate extension → Word 400 → PDF download → `run_ingest` → status codes above
- [x] Structured logs as in §2
- [x] Idempotent under redelivery (stem delete+upsert)
- [x] `fastapi` in group `ingest`
- [x] `Dockerfile.ingest` CMD → uvicorn `main_ingest` (CLI remains available via override)
- [x] Unit tests for envelope / extension gate / handler (no real GCP)
- [x] `USE_SECRET_MANAGER` → Secret Manager hydrate on ingest lifespan only

**Done when:** a test push envelope (or real upload once wired) can drive indexing without touching chat.

---

### M3 — GCP wiring (manual `gcloud` runbook)

- [ ] Create `docs/gcp-ingest-runbook.md` with copy-pasteable `gcloud` steps
- [ ] GCS bucket + Pub/Sub topic + push subscription → Cloud Run URL `/`
- [ ] GCS notification `OBJECT_FINALIZE` (no PDF-only notification filter)
- [ ] Cloud Run Ingest: `Dockerfile.ingest`, `--concurrency=1`, `--memory=8Gi`, high `--timeout`, ack deadline **600** on subscription
- [ ] IAM: ingest SA `storage.objects.get`; Pub/Sub SA `run.invoker`; secrets/env for Cohere + Qdrant
- [ ] DLQ topic/subscription + retry notes
- [ ] Document chat Service slim deploy if not already done
- [ ] Document measure plan: ingest duration vs 10 min Pub/Sub ack limit

**Done when:** another engineer can provision by following the runbook (no Terraform/CI-CD required).

---

### M4 — Ops & verification

- [ ] Runbook: manual upload, redrive DLQ, force re-ingest (re-upload or publish synthetic message)
- [ ] Optional snippet: publish synthetic Pub/Sub message for a known object
- [ ] Confirm serving image has no Docling/torch; ingest Service does not load ADK
- [ ] Update README + [split-ingest-serving.md](./split-ingest-serving.md) §7.5 log on merge
- [ ] Mark checkboxes in this doc when done

**Done when:** another engineer can upload a PDF and debug a failed ingest without reading application code.

---

### M5 — Optional stretch (not planned unless needed)

- [ ] SQL/registry: `doc_id`, `gcs_uri`, `generation`, `status`, `indexed_at` (enables skip-by-generation)
- [ ] Ops-only enqueue HTTP (202 → Pub/Sub) — **not** on chat Service
- [ ] MinIO + local Pub/Sub emulator for compose parity
- [ ] Word → PDF conversion then ingest
- [ ] Async ack pattern if measured durations break the 10 min ceiling

---

## 5. Explicitly out of scope

| Item | Why later |
|---|---|
| Cloud Run **Job** for hot-path ingest | Crowned as Service + push; Jobs for scraper/backfill |
| Scraper | Writes GCS only; separate surface |
| Fan-Out / canonical store (§7.4) | After single-worker path is stable |
| Multi-agent / web search | Product track |
| `SessionStore` adapter | Orthogonal ([architecture-improvements.md](./architecture-improvements.md) §2) |
| Terraform / CI-CD | Explicitly deferred; manual runbook first |
| Remove `/eval` | Kept empty for future eval work |

---

## 6. Suggested implementation order

```text
M0 run_ingest  →  M1 GCS download  →  M2 main_ingest handler
                                      →  M3 gcloud runbook
                                      →  M4 verify + README / §7.5 log
                         (later) M5 registry / Word converter / async if needed
```

Suggested git commits on `feature/gcp-ingest-pubsub`: one per milestone M0–M4 (docs can be M3+M4 together).

---

## 7. Definition of done (phase)

- [ ] PDF uploaded to GCS is indexed in Cloud Qdrant without laptop `ingest_local`
- [ ] Chat Cloud Run Service role unchanged (slim; no ingest pipeline)
- [ ] Ingest path is Cloud Run **Service** + Pub/Sub push (not Job)
- [ ] Same `run_ingest` used locally and in GCP
- [ ] Redelivery does not duplicate chunks for the same stem
- [ ] Word objects get 400 (not supported yet), not endless retries as 5xx
- [ ] Manual `gcloud` runbook exists; no Terraform required for v1
- [ ] Ack/timeout/concurrency limits documented; measure note included

---

## 8. Ops knobs (closed) + where each is defined

| # | Point | Decision | Where it is defined |
|---|---|---|---|
| R1 | Cloud Run request timeout | **3600 s** | `gcloud run deploy … --timeout=3600` (runbook) |
| R2 | CPU / instances / concurrency | **2 vCPU**, **max-instances=3**, **min-instances=0**, **concurrency=1**, **memory=8Gi** | Same `gcloud run deploy` flags (see §8.1). Not in Python |
| R3 | Resource names | Bucket/object at **runtime** from Pub/Sub attributes; infra names = PowerShell placeholders in runbook (see §8.2) | Attributes in app; topic/bucket creation in runbook |
| R4 | DLQ max delivery attempts | **5** | `gcloud pubsub subscriptions create/update … --max-delivery-attempts=5` (runbook) |
| R5 | Non-`OBJECT_FINALIZE` | **200** ack/drop | **Application code** in `main_ingest` when reading `attributes.eventType` (see §8.3) |
| R6 | Missing GCS object | **404** | Application code (GCS download error mapping) |
| R7 | Object path prefixes | `doc_id` = basename stem | Application: `Path(objectId).stem` (see §8.4) |
| R8 | Type filter | Extension only | Application code |
| R9 | Ephemeral disk | Watch `/tmp`; revisit if errors | Ops note in runbook |
| R10 | Port | `$PORT` (8080) | Dockerfile / uvicorn / Cloud Run |
| R11 | Chat deploy same wave | See §8.5 — **not required to land ingest code**; runbook documents both services | Ops choice |
| R12 | Secrets | Secret Manager in GCP (see §8.6); local keeps `.env` / Settings | Code helper + IAM + runbook secret IDs |

### 8.1 R2 — CPU / concurrency: yes, `gcloud` (not code)

Cloud Run revision settings. Example (runbook will use env placeholders):

```powershell
gcloud run deploy $env:INGEST_SERVICE_NAME `
  --image=$env:INGEST_IMAGE `
  --region=$env:REGION `
  --cpu=2 `
  --memory=8Gi `
  --concurrency=1 `
  --max-instances=3 `
  --min-instances=0 `
  --timeout=3600 `
  --no-allow-unauthenticated `
  --service-account=$env:INGEST_SA_EMAIL
```

If you omit `--concurrency`, default is ~**80** → unsafe with Docling.

### 8.2 R3 — Bucket name vs project env vs gcloud placeholders

**Two different things:**

| Concern | Source | Example |
|---|---|---|
| Which bucket/object to download **for this event** | Pub/Sub message attributes | `bucketId`, `objectId` |
| GCP project / collection / wiring names | Env / Secret Manager / runbook vars | `GOOGLE_CLOUD_PROJECT`, collection name |

**In application (per request):**

```python
attributes = pubsub_message.get("attributes", {})
bucket_name = attributes.get("bucketId")
file_name = attributes.get("objectId")
# objectGeneration → logs only
```

The app does **not** hardcode the docs bucket. Whatever bucket notified Pub/Sub is what gets downloaded (IAM must allow that bucket).

**Process-level config (env):**

```text
GOOGLE_CLOUD_PROJECT   # project id (Secret Manager paths, ADC project)
QDRANT_COLLECTION      # existing Settings field (default today: tech_docs)
```

> Naming note: reference services sometimes use `COLLECTION_NAME` / `tech_documents`. This repo already uses pydantic `qdrant_collection` ← env `QDRANT_COLLECTION` default **`tech_docs`**. Prefer keeping that name unless we deliberately migrate the Qdrant collection. Do **not** introduce a second parallel `COLLECTION_NAME` without mapping.

**Runbook placeholders (PowerShell), for topic/bucket creation — not read from the push handler for download:**

```powershell
$env:PROJECT_ID   = "your-gcp-project"
$env:REGION       = "europe-west1"
$env:BUCKET_NAME  = "$env:PROJECT_ID-input-docs"
$env:TOPIC_NAME   = "doc-ingestion-topic"
$env:SUB_NAME     = "doc-processor-sub"
$env:DLQ_TOPIC    = "doc-ingestion-dlq"
$env:INGEST_SERVICE_NAME = "rag-ingest"
```

Flow remains: upload to `$BUCKET_NAME` → notification → topic `$TOPIC_NAME` → push sub → Cloud Run `POST /` → attributes carry the real `bucketId`/`objectId`.

### 8.3 R5 — Where non-FINALIZE is defined

In **`main_ingest`**, after parsing attributes:

```text
if eventType is present AND eventType != OBJECT_FINALIZE:
    log + return 200   # ack and drop (no retry storm)
```

Not a `gcloud` flag. Optional hardening: create the GCS notification with `--event-types=OBJECT_FINALIZE` only (runbook) so other events never publish; the code check is still defense in depth.

### 8.4 R7 — Stem example

| `objectId` (GCS object name) | Basename | `doc_id` (`Path(objectId).stem`) |
|---|---|---|
| `WG1.pdf` | `WG1.pdf` | `WG1` |
| `docs/O-RAN/WG1.pdf` | `WG1.pdf` | `WG1` |
| `inbox/2026/WG11.TR.ZTA.pdf` | `WG11.TR.ZTA.pdf` | `WG11.TR.ZTA` |

Re-upload to the same object name → same `doc_id` → delete+upsert replaces vectors. Different folder + same filename → **same** `doc_id` (stem collision). Ops convention: avoid two different PDFs that share the same basename if they must be distinct docs.

### 8.5 R11 — “Chat deploy in the same wave” (context)

This phase adds the **ingest** Cloud Run Service. Chat is a **separate** service (`Dockerfile.serving` / `main_chat`).

| Situation | What you do |
|---|---|
| Chat slim already deployed and talking to Cloud Qdrant | **Nothing required** for ingest PR to work. Upload PDF → ingest indexes → chat already retrieves |
| Prod still runs an old fat monolith image | You should deploy **serving** slim separately (same “wave” of ops work, not the same git milestone as M0 code). Runbook documents both deploys |
| Chat not on Cloud Run yet | Ingest can still be developed/tested; end-to-end “ask the agent” needs chat deployed later |

**Conclusion for this PR:** implement ingest worker + runbook. Chat deploy is an **ops checklist item**, not a code dependency of M0–M2. No need to block ingest on chat redeploy if serving is already split in prod.

### 8.6 R12 — Secrets (Secret Manager vs env)

Today the app reads config via **pydantic Settings** (`.env` locally: `COHERE_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, …).

Your reference pattern fetches secrets at process start:

```python
# Conceptual — Secret Manager
QDRANT_URL = get_secret("qdrant-url")
QDRANT_API_KEY = get_secret("qdrant-api-key")
COHERE_API_KEY = get_secret("cohere-api-key")
```

**Two valid GCP approaches:**

| Approach | How | Pros | Cons |
|---|---|---|---|
| **A. Cloud Run mounts secrets as env** | `gcloud run deploy --set-secrets=COHERE_API_KEY=cohere-api-key:latest,…` | Zero SM code; Settings works unchanged; local `.env` unchanged | Secrets appear as env inside the container |
| **B. App calls Secret Manager API** | `get_secret` + `google-cloud-secret-manager`; SA needs `secretmanager.secretAccessor` | Matches your other service; explicit warm-start fetch | Extra dep; must not break local DX; dual path env vs SM |

**Decision for this phase (aligned with your ask): prefer B in GCP**, with a clean split:

1. Small infra helper `get_secret(secret_id)` (ADC + `projects/{GOOGLE_CLOUD_PROJECT}/secrets/.../versions/latest`).
2. **Local / tests:** keep Settings + `.env` (no SM calls).
3. **Cloud Run ingest (and later serving):** on startup, if `GOOGLE_CLOUD_PROJECT` is set (or an explicit flag like `USE_SECRET_MANAGER=true`), resolve `qdrant-url`, `qdrant-api-key`, `cohere-api-key` and feed the existing Settings fields / composition — **do not** scatter `os.environ` reads across domain.
4. Add `google-cloud-secret-manager` to the groups that need it (`ingest` now; `serving` when chat uses the same helper).
5. Strip surrounding quotes in secret payloads (as in your snippet).
6. Warm-start: resolve secrets once at lifespan / module init of composition, not per Pub/Sub message.

**Secrets sub-decisions (closed):**

| # | Decision |
|---|---|
| S1 | Gate Secret Manager with **`USE_SECRET_MANAGER=true`** (not merely “project env is set”) |
| S2 | Keep **`QDRANT_COLLECTION`** / default **`tech_docs`** (no `COLLECTION_NAME` / `tech_documents` rename) |
| S3 | **`get_secret` only on ingest** in this PR; serving stays Settings/`.env` (or Cloud Run env) until a later PR |

Local DX unchanged: without the flag, composition uses pydantic Settings + `.env` as today.

---

## 9. Decision log

| Date | Decision |
|---|---|
| 2026-07-30 | Phase doc created; open questions listed |
| 2026-07-31 | Closed design session: `POST /` + GCS attributes; sync ack; concurrency=1 / 8Gi; fastapi+gcs storage; generation log-only; Word→400; manual gcloud runbook (no TF/CI); measure vs 10 min ack limit |
| 2026-07-31 | Remaining ops knobs listed in §8 with suggested defaults |
| 2026-07-31 | Closed R1–R11 details: gcloud flags for scale; attributes for bucket/object; runbook PowerShell placeholders; eventType drop in app; stem examples; chat deploy optional ops; Secret Manager helper for GCP + Settings/.env local |
| 2026-07-31 | Closed S1–S3: `USE_SECRET_MANAGER=true`; `QDRANT_COLLECTION`/`tech_docs`; SM helper ingest-only this PR. **Design ready for implementation.** |
