# Local ingest

Laptop commands to parse a PDF and optionally index it in Qdrant. Same application core as Cloud Run: `scripts/ingest_local.py` → `run_ingest`.

```text
PDF on disk → ingest_local.py → parse (Docling) → chunk → embed → Qdrant
```

This path does **not** use GCS, Pub/Sub, Secret Manager, or a service account. Cloud Run deploy: [gcp_ingest_deployment.md](./gcp_ingest_deployment.md). HTTP worker contract: [ingest.md](./ingest.md).

| | |
|---|---|
| Files | Any local PDF path. `doc_id` = filename stem. Same stem again **deletes** then upserts. |
| Secrets | `.env` (`COHERE_API_KEY`, `QDRANT_*`). Keep `USE_SECRET_MANAGER=false`. |
| Qdrant | Compose on `localhost:6333` (no API key), or Qdrant Cloud URL + key in `.env`. |
| Chunker | Local default `semantic`. Cloud Run sets `section`. |
| Profile | `INGEST_PROFILE=local` (larger Docling batches than `cloud`). |

Two ways to run the CLI:

| | Host (`uv`) | Compose (`agent-ingest`) |
|---|---|---|
| When | Day-to-day on the laptop (CUDA torch on Windows/Linux when available). | CPU image, no host Docling/torch install. |
| PDF path | Host path, e.g. `data\doc.pdf`. | File must live under `data/` → `/data/doc.pdf` in the container. |

Commands are **Windows PowerShell**, from the **repository root**.

---



## Prerequisites

1. [uv](https://docs.astral.sh/uv/) and Python as in the repo `README`.
2. Docker Desktop (for local Qdrant; also for the compose ingest image).
3. Cohere API key — [https://dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys).
4. A PDF on disk (start with a short file; a 400+ page TS is slow on a laptop).

Chat and Postgres are **not** required to ingest.

---



## Setup (one-time)

```powershell
Copy-Item .env.example .env
# Edit .env: COHERE_API_KEY. Leave QDRANT_URL=http://localhost:6333 for compose Qdrant.
# USE_SECRET_MANAGER=false
```

Load `.env` into the current PowerShell process (optional; `pydantic` Settings also reads `.env` from the repo root):

```powershell
Get-Content .env | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith('#')) {
        $name, $value = $line -split '=', 2
        $value = $value.Trim().Trim('"').Trim("'")
        [Environment]::SetEnvironmentVariable($name.Trim(), $value, 'Process')
    }
}

$k = $env:COHERE_API_KEY
"len=$($k.Length)  " + $(if ($k -and $k -eq $k.Trim()) { "OK" } else { "SUSPECT — re-check .env" })
```

Install the ingest extra **once** if you use the host CLI (skip if you only use compose):

```powershell
uv sync --group ingest
```

---



## Phase 0 — Qdrant

```powershell
docker compose up -d qdrant
```

Dashboard: [http://localhost:6333/dashboard](http://localhost:6333/dashboard). Collection name defaults to `tech_docs`.

To index into **Qdrant Cloud** instead, set `QDRANT_URL` and `QDRANT_API_KEY` in `.env` and skip this compose service.

---



## Phase 1 — Host CLI (typical)

Index a PDF (replace the path):

```powershell
uv run --group ingest python scripts/ingest_local.py data\your-doc.pdf
```

Expect a log line with `doc_id`, `indexed=True`, `chunk_count`, `deleted`, `upserted`.

Parse only (no embed / no Qdrant write):

```powershell
uv run --group ingest python scripts/ingest_local.py data\your-doc.pdf --no-index
```

Dump canonical JSON/Markdown and chunk JSON/Markdown (parse + chunk; indexing still on unless you add `--no-index`):

```powershell
uv run --group ingest python scripts/ingest_local.py data\your-doc.pdf --no-index `
    --canonical-out data\out\your-doc.canonical.json `
    --canonical-md-out data\out\your-doc.canonical.md `
    --json-out data\out\your-doc.chunks.json `
    --md-out data\out\your-doc.chunks.md
```

Match Cloud Run chunking while still on the laptop:

```powershell
$env:CHUNKER = "section"
uv run --group ingest python scripts/ingest_local.py data\your-doc.pdf
```

(`INGEST_PROFILE` stays `local` unless you set it; that only changes Docling batch sizes, not the chunker.)

---



## Phase 2 — Compose CLI (CPU image)

PDF must be under `data/` (compose mounts `./data` → `/data` read-only).

```powershell
docker compose up -d qdrant

docker compose --profile ingest run --rm agent-ingest `
    python scripts/ingest_local.py /data/your-doc.pdf
```

Same flags as the host CLI (`--no-index`, `--canonical-out`, …). Output paths must also be under `/data/...` if you want files on the host (`data/` is the mount; it is **read-only**, so dumps to `/data/out` will fail). Prefer the host CLI for `--canonical-out` / `--json-out`.

---



## Smoke check

1. CLI log: `doc_id=<stem> indexed=True chunk_count=<n>`.
2. Qdrant dashboard → collection `tech_docs` → points for that stem.
3. Optional chat (needs serving stack): [README ingest](../README.md#ingest) then `/chat` asking about that document.

Re-run the same path to replace vectors for that stem.

---



## Configuration (local)

| Variable | Local value |
|---|---|
| `USE_SECRET_MANAGER` | `false` (`.env` / Settings) |
| `QDRANT_URL` | `http://localhost:6333` with compose Qdrant |
| `QDRANT_COLLECTION` | `tech_docs` |
| `CHUNKER` | `semantic` (default) or `section` |
| `INGEST_PROFILE` | `local` (default) |

Do not run `main_ingest` or `gcloud` for this loop. Do not set `USE_SECRET_MANAGER=true` on the laptop.
