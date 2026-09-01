# GCP ingest deployment

Manual `gcloud` steps to run the **current** ingest worker on Google Cloud: upload a PDF to GCS, Pub/Sub pushes to Cloud Run, the service downloads the object and indexes it in Qdrant.

```text
PDF → GCS (OBJECT_FINALIZE) → Pub/Sub → Cloud Run (POST /) → run_ingest → Qdrant
```

Same application core as local: `scripts/ingest_local.py` / `run_ingest`. Laptop commands: [gcp_ingest_local.md](./gcp_ingest_local.md). Chat and Postgres are **not** deployed here.

| | |
|---|---|
| Files | `.pdf` only. `.doc` / `.docx` and other types → HTTP **400** (no retry). |
| Identity | `doc_id` = filename stem. Re-upload replaces that stem in Qdrant. |
| Time limit | Pub/Sub **ack deadline is 600 s**. Cloud Run `--timeout=3600` does not extend it. Use a PDF that finishes in well under 10 minutes. |
| Scale | `concurrency=1` (required with Docling). `--max-instances=3`. |

Worker HTTP details: [ingest.md](./ingest.md).

Commands are **Windows PowerShell**, from the **repository root**.

---



## Resource names (single source of truth)

Use these names everywhere (create, deploy, subscribe, cleanup):


| Resource               | Value                                            |
| ---------------------- | ------------------------------------------------ |
| Cloud Run service      | `rag-ingest`                                     |
| Artifact Registry repo | `cloud-run-source`                               |
| Image                  | `rag-ingest`                                     |
| Pub/Sub topic          | `doc-ingestion-topic`                            |
| Pub/Sub DLQ topic      | `doc-ingestion-dlq`                              |
| Push subscription      | `doc-ingestion-sub`                              |
| Runtime + push SA      | `rag-ingest-sa`                                  |
| Secrets                | `cohere-api-key`, `qdrant-url`, `qdrant-api-key` |


One service account is used both as Cloud Run runtime identity and as Pub/Sub push auth (simpler). Invoker is granted **on the Cloud Run service**, not project-wide.

---



## Prerequisites

1. Google Cloud project with billing enabled.
2. [gcloud CLI](https://cloud.google.com/sdk/docs/install) (new PowerShell window after install).
3. Docker Desktop (optional for local image builds; Cloud Build does not require it).
4. Cohere API key — [https://dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys).
5. Qdrant Cloud cluster — [https://cloud.qdrant.io](https://cloud.qdrant.io) (local compose Qdrant is for laptop only).

Chat/Postgres is **out of scope** for this ingest deploy.

---



## Setup (one-time on the laptop)

```powershell
Copy-Item .env.example .env
# Edit .env: COHERE_API_KEY, QDRANT_*, GOOGLE_CLOUD_PROJECT, REGION
```

Load `.env` into the current PowerShell process (sets `GOOGLE_CLOUD_PROJECT`, `REGION`, etc.):

```powershell
Get-Content .env | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith('#')) {
        $name, $value = $line -split '=', 2
        $value = $value.Trim().Trim('"').Trim("'")
        [Environment]::SetEnvironmentVariable($name.Trim(), $value, 'Process')
    }
}
```

Sanity-check Cohere key (typical length 40; adjust if your key format differs):

```powershell
$k = $env:COHERE_API_KEY
"len=$($k.Length)  " + $(if ($k -and $k -eq $k.Trim()) { "OK" } else { "SUSPECT — re-check .env" })
```

---



## Phase 0 — Project

Uses `GOOGLE_CLOUD_PROJECT` and `REGION` already loaded from `.env` (no second project variable).

```powershell
gcloud auth login

if (-not $env:GOOGLE_CLOUD_PROJECT -or -not $env:REGION) {
    Write-Error "Set GOOGLE_CLOUD_PROJECT and REGION in .env, then re-run the Setup loader."
    return
}

gcloud config set project $env:GOOGLE_CLOUD_PROJECT

$env:BUCKET_NAME = "$env:GOOGLE_CLOUD_PROJECT-input-docs"
$env:TOPIC_NAME = "doc-ingestion-topic"
$env:DLQ_TOPIC = "doc-ingestion-dlq"
$env:SUB_NAME = "doc-ingestion-sub"
$env:INGEST_SERVICE_NAME = "rag-ingest"
$env:SA_NAME = "rag-ingest-sa"
$env:SA_EMAIL = "$env:SA_NAME@$env:GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com"
$env:AR_REPO = "cloud-run-source"
$env:IMAGE_URL = "$env:REGION-docker.pkg.dev/$env:GOOGLE_CLOUD_PROJECT/$env:AR_REPO/rag-ingest:latest"
```

---



## Phase 1 — APIs, bucket, Pub/Sub, secrets



### 1.1 Enable APIs

```powershell
gcloud services enable `
    artifactregistry.googleapis.com `
    cloudbuild.googleapis.com `
    run.googleapis.com `
    storage.googleapis.com `
    pubsub.googleapis.com `
    secretmanager.googleapis.com `
    iamcredentials.googleapis.com
```



### 1.2 GCS service agent → Pub/Sub publisher

New projects often need the GCS service agent before bucket notifications work:

```powershell
if (-not $env:GOOGLE_CLOUD_PROJECT) { Write-Error "GOOGLE_CLOUD_PROJECT is not set."; return }

$env:PN = $(gcloud projects describe $env:GOOGLE_CLOUD_PROJECT --format="value(projectNumber)")
gcloud storage service-agent | Out-Null
gcloud projects add-iam-policy-binding $env:GOOGLE_CLOUD_PROJECT `
    --member="serviceAccount:service-$env:PN@gs-project-accounts.iam.gserviceaccount.com" `
    --role="roles/pubsub.publisher"
```



### 1.3 Bucket

```powershell
gcloud storage buckets create "gs://$env:BUCKET_NAME" --location=$env:REGION
```



### 1.4 Topics (main + DLQ)

```powershell
gcloud pubsub topics create $env:TOPIC_NAME
gcloud pubsub topics create $env:DLQ_TOPIC
```

Optional: pull subscription on the DLQ to inspect failed messages:

```powershell
gcloud pubsub subscriptions create "$env:DLQ_TOPIC-pull" --topic=$env:DLQ_TOPIC
```



### 1.5 GCS → Pub/Sub notification

Create **once**. `notifications create` is **not** idempotent: each run adds another config. Two configs on the same topic → two Pub/Sub messages (and two ingest runs) per PDF upload. Existing objects are not re-notified; only new/overwritten objects fire `OBJECT_FINALIZE`.

List first; create only if none exist:

```powershell
gcloud storage buckets notifications list "gs://$env:BUCKET_NAME"
```

If the list is empty:

```powershell
gcloud storage buckets notifications create "gs://$env:BUCKET_NAME" `
    --topic=$env:TOPIC_NAME `
    --event-types=OBJECT_FINALIZE `
    --payload-format=json
```

If duplicates already exist, keep one and delete the extra (API resource name, not `gs://bucket/notificationConfigs/ID`):

```powershell
gcloud storage buckets notifications delete `
    projects/_/buckets/$env:BUCKET_NAME/notificationConfigs/2
```

No PDF-only suffix filter on the bucket: the worker accepts `.pdf` and returns **400** for Word and other types (Pub/Sub does not retry 4xx).

### 1.6 Secrets

Secret IDs must match `gcp_secrets.py`: `cohere-api-key`, `qdrant-url`, `qdrant-api-key`.

PowerShell pipes can add encoding noise; write temp files instead:

```powershell
function New-SecretFromValue([string]$SecretId, [string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) {
        Write-Error "Empty value for secret $SecretId."
        return
    }
    $tmp = New-TemporaryFile
    [System.IO.File]::WriteAllText($tmp.FullName, $Value.Trim(), [System.Text.UTF8Encoding]::new($false))
    $filePath = $tmp.FullName
    gcloud secrets create $SecretId --replication-policy="automatic" --data-file $filePath
    
    Remove-Item $tmp.FullName -Force
}

New-SecretFromValue "qdrant-url" $env:QDRANT_URL
New-SecretFromValue "qdrant-api-key" $env:QDRANT_API_KEY
New-SecretFromValue "cohere-api-key" $env:COHERE_API_KEY
```

If a secret already exists, add a version instead:

```powershell
# gcloud secrets versions add qdrant-url --data-file=...
```

---



## Phase 2 — Service account & IAM

```powershell
gcloud iam service-accounts create $env:SA_NAME `
    --display-name="RAG-Agent ingest (runtime + Pub/Sub push)"

# Read objects from any bucket in the project (or bind only this bucket if you want a tighter grant)
gcloud projects add-iam-policy-binding $env:GOOGLE_CLOUD_PROJECT `
    --member="serviceAccount:$env:SA_EMAIL" `
    --role="roles/storage.objectViewer"

gcloud projects add-iam-policy-binding $env:GOOGLE_CLOUD_PROJECT `
    --member="serviceAccount:$env:SA_EMAIL" `
    --role="roles/secretmanager.secretAccessor"
```

`roles/aiplatform.user` is **not** required for this ingest worker (Cohere + Docling only).

Invoker on the Cloud Run service is granted in Phase 3 after deploy.

---



## Phase 3 — Build, deploy, connect Pub/Sub



### 3.1 Artifact Registry + Cloud Build

From the **repository root** (where `Dockerfile.ingest` and `cloudbuild.ingest.yaml` live):

```powershell
gcloud artifacts repositories create $env:AR_REPO `
    --repository-format=docker `
    --location=$env:REGION `
    --description="Docker images for RAG-Agent Cloud Run" `
    2>$null  # ignore if repo already exists

gcloud builds submit `
    --config=cloudbuild.ingest.yaml `
    --substitutions="_IMAGE_URL=$env:IMAGE_URL" `
    .
```

`cloudbuild.ingest.yaml` builds with `-f Dockerfile.ingest` (required: `--tag` alone only finds a file named `Dockerfile`).

### 3.2 Deploy Cloud Run

```powershell
gcloud run deploy $env:INGEST_SERVICE_NAME `
    --image=$env:IMAGE_URL `
    --region=$env:REGION `
    --service-account=$env:SA_EMAIL `
    --cpu=2 `
    --memory=8Gi `
    --concurrency=1 `
    --max-instances=3 `
    --min-instances=0 `
    --timeout=3600 `
    --execution-environment=gen2 `
    --no-allow-unauthenticated `
    --set-env-vars="USE_SECRET_MANAGER=true,GOOGLE_CLOUD_PROJECT=$env:GOOGLE_CLOUD_PROJECT,CHUNKER=section,INGEST_PROFILE=cloud"
```

Notes:

- `concurrency=1` — required with Docling (Cloud Run default is ~80).
- `timeout=3600` — Cloud Run request timeout. Pub/Sub still acks at **600 s**; the worker must finish before that or the message is redelivered.
- `USE_SECRET_MANAGER=true` — required for `main_ingest` to load Secret Manager secrets.
- `INGEST_PROFILE=cloud` — smaller Docling batches than the local default (fits `--memory=8Gi`).
- Raise `--cpu` (e.g. 4–8) if you measure CPU-bound Docling; start with 2.



### 3.3 Pub/Sub can mint tokens for the push SA

```powershell
$env:PROJECT_NUMBER = $(gcloud projects describe $env:GOOGLE_CLOUD_PROJECT --format="value(projectNumber)")

gcloud iam service-accounts add-iam-policy-binding $env:SA_EMAIL `
    --member="serviceAccount:service-$env:PROJECT_NUMBER@gcp-sa-pubsub.iam.gserviceaccount.com" `
    --role="roles/iam.serviceAccountTokenCreator"
```

(Prefer binding on the SA, not only project-wide.)

### 3.4 Allow push SA to invoke Cloud Run

```powershell
gcloud run services add-iam-policy-binding $env:INGEST_SERVICE_NAME `
    --region=$env:REGION `
    --member="serviceAccount:$env:SA_EMAIL" `
    --role="roles/run.invoker"
```



### 3.5 Push subscription → `POST /`

```powershell
$env:SERVICE_URL = $(gcloud run services describe $env:INGEST_SERVICE_NAME `
    --region=$env:REGION `
    --format="value(status.url)")

# Pub/Sub service agent needs publisher on DLQ topic
gcloud pubsub topics add-iam-policy-binding $env:DLQ_TOPIC `
    --member="serviceAccount:service-$env:PROJECT_NUMBER@gcp-sa-pubsub.iam.gserviceaccount.com" `
    --role="roles/pubsub.publisher"

gcloud pubsub subscriptions create $env:SUB_NAME `
    --topic=$env:TOPIC_NAME `
    --push-endpoint="$env:SERVICE_URL/" `
    --push-auth-service-account=$env:SA_EMAIL `
    --ack-deadline=600 `
    --dead-letter-topic="projects/$env:GOOGLE_CLOUD_PROJECT/topics/$env:DLQ_TOPIC" `
    --max-delivery-attempts=5
```

---



## Smoke test

```powershell
gcloud storage cp path\to\sample.pdf "gs://$env:BUCKET_NAME/sample.pdf"

gcloud run services logs read $env:INGEST_SERVICE_NAME --region=$env:REGION --limit=50
```

Expect logs with `bucket`, `object`, `doc_id`, `chunk_count`. Then query Qdrant / chat for that stem.

---



## Cleanup

```powershell
# Empty bucket first (required if it still has objects)
gcloud storage rm -r "gs://$env:BUCKET_NAME/**" 2>$null

gcloud run services delete $env:INGEST_SERVICE_NAME --region=$env:REGION --quiet
gcloud pubsub subscriptions delete $env:SUB_NAME --quiet
gcloud pubsub subscriptions delete "$env:DLQ_TOPIC-pull" --quiet 2>$null
gcloud pubsub topics delete $env:TOPIC_NAME --quiet
gcloud pubsub topics delete $env:DLQ_TOPIC --quiet
gcloud storage buckets delete "gs://$env:BUCKET_NAME" --quiet
gcloud artifacts repositories delete $env:AR_REPO --location=$env:REGION --quiet
gcloud iam service-accounts delete $env:SA_EMAIL --quiet
gcloud secrets delete qdrant-url --quiet
gcloud secrets delete qdrant-api-key --quiet
gcloud secrets delete cohere-api-key --quiet
```

