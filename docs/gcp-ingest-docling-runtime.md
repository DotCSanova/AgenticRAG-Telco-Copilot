# Runtime GCP de Docling (Jobs) — diseño futuro

**Fecha:** 2026-08-31  
**Estado (2026-09-01):** **no implementado.** No forma parte de `feature/gcp-ingest-pubsub`. El camino GCP **actual** es Service + Pub/Sub push: [gcp-ingest-pubsub.md](./gcp-ingest-pubsub.md) y [gcp_ingest_deployment.md](./gcp_ingest_deployment.md).  
**Alcance:** cómo **se plantearía** ejecutar Docling en GCP para un corpus grande (~154 PDFs + deltas) o PDFs que no caben en el ack de 600 s. Paralelismo entre documentos, shards **dentro** de un PDF, Jobs vs Service.  
**No cubre:** parser, mapper, JSON canónico, orden de refiners — [ingest-canonical-review.md](./ingest-canonical-review.md). No sustituye el runbook del Service.

Relacionado:

| Doc | Rol |
|---|---|
| [ingest-canonical-review.md](./ingest-canonical-review.md) | Qué hace Docling/normalizer (código) |
| [gcp-ingest-pubsub.md](./gcp-ingest-pubsub.md) | Contrato **ya implementado**: Service + Pub/Sub push (`main_ingest`) |
| [gcp_ingest_deployment.md](./gcp_ingest_deployment.md) | Runbook `gcloud` del Service |
| [split-ingest-serving.md](./split-ingest-serving.md) §7.1 / §7.4 | Norte chat⊥ingest; fan-out de **páginas** = no MVP |

El core de aplicación **no cambia**: `run_ingest(path)` → `IngestDocumentService`. Cambia el **disparo** y el **límite de tiempo**.

---

## 1. Carga real

| Fase | Volumen | Cadencia |
|---|---|---|
| Carga inicial | ~**154** PDFs, tamaños mezclados (17 págs. … 468 págs.) | Una vez |
| Delta | Lote **más pequeño** de docs nuevos/actualizados | Cada X meses |

Eso es **batch**, no un API con latencia de usuario. Chat sigue siendo Cloud Run **Service**. Ingest no tiene que responder en 10 minutos.

---

## 2. Veredicto (cuando se implemente)

**Para el lote ~154 y PDFs que superan ~10 min: Cloud Run Job.**  
Una tarea = un PDF. Shards de páginas **en serie** dentro de la tarea. Paralelismo **entre PDFs** (`parallelism` 2–4).

**Hasta entonces:** el Service + Pub/Sub push **sí** es el ingest GCP (PDFs que acaban dentro del ack). El mismo `run_ingest` / imagen / idempotencia. Un TS largo en el `POST /` de un push reintenta a los ~10 min (ack máximo **600 s**); eso no se arregla subiendo `--timeout` de Cloud Run. Esos docs: CLI local ahora, Job más adelante.

Delta trimestral (futuro): el **mismo Job**, otro prefijo GCS. No hace falta un Service 24/7 para el lote.

---

## 3. Por qué no Service + ack síncrono

Lo desplegado / documentado hoy:

```text
GCS OBJECT_FINALIZE → Pub/Sub push → Cloud Run Service POST /
  concurrency=1, 8Gi, timeout HTTP 3600 s
```

El techo real es Pub/Sub push: **ack máximo 600 s**. El `--timeout=3600` de Cloud Run **no** alarga ese ack. A los ~10 min Pub/Sub reenvía; otro contenedor puede parsear el mismo PDF (índice idempotente, coste doble).

| Documento | ¿Cabe en 10 min? (CPU, TableFormer, embed) |
|---|---|
| `O-RAN.SuFG.CE-v01.00` (17 págs., 0.3 MB) | Sí, típico |
| `O-RAN.WG1.TS.Use-Cases-Detailed-…-v19.00` (468 págs., 12.8 MB, ~10 shards) | No de forma fiable |

[gcp-ingest-pubsub.md](./gcp-ingest-pubsub.md) §1.4 ya decía: si el P95 se acerca a 8–9 min, salir del ack síncrono. Con este corpus ese “si” es el caso de producción.

Un Job: timeout de **horas** por tarea, sin ack. Falla la tarea del PDF 47; las otras 153 no se rehacen.

---

## 4. Diseño de la fase Docling

Misma imagen `Dockerfile.ingest`, mismos pesos horneados (evitar 429 HuggingFace al arrancar).

```text
GCS prefix (154 PDFs)
        │
        ▼
Cloud Run Job   parallelism=3
   task i  ──► download objeto i
           ──► preprocess PDF entero
           ──► Docling page_range SECUENCIAL (50 págs.)
           ──► merge canónico
           ──► opcional: gs://…/canonical/{stem}.json
           ──► chunk → embed (Cohere) → Qdrant (delete stem + upsert)
```

### 4.1 Unidad de trabajo

| Eje | Decisión |
|---|---|
| 1 tarea Job | **1 PDF** (`CLOUD_RUN_TASK_INDEX` → objeto `i` del listado del prefijo) |
| Shards de páginas | **Secuenciales** en esa tarea (`page_range`, techos en [ingest-canonical-review.md](./ingest-canonical-review.md) §13) |
| Paralelismo | Entre documentos: `--parallelism=3` (alineado con 8 Gi y el `max-instances=3` del Service) |
| Fan-out de páginas a otras tareas | **No** (merge, bordes de tabla, N cargas de modelo por PDF). Ver §7.4 de split-ingest |

### 4.2 Cómo se lanza el lote

1. Subir PDFs a un prefijo (`gs://…/corpus/oran/` o `gs://…/inbox/2026-09/`).
2. El Job lista el prefijo; la tarea `i` procesa el objeto `i`.
3. Disparo: `gcloud run jobs execute` (ops) o Cloud Scheduler (delta periódico).

No hace falta Eventarc por fichero para 154 + un lote cada X meses.

### 4.3 Delta cada X meses

Mismo Job, otro prefijo, `--tasks=N` del lote nuevo. `min-instances=0`: se paga solo mientras corre.

### 4.4 Idempotencia y reintentos

- `doc_id` = stem; delete + upsert (ya implementado).
- Reejecutar el Job o una tarea fallida es seguro.
- Timeout por tarea: **1–4 h** (no 600 s). Un PDF que peta a las 2 h no tumba el lote.

### 4.5 Canónico en GCS (recomendado)

Tras merge, escribir `gs://…/canonical/{stem}.json` (y stats: páginas, bloques, `docling_version`). Si Cohere/Qdrant fallan, reindex **sin** volver a Docling. Desacopla parse caro de index reintento. El codec del canónico es el de domain ([ingest-canonical-review.md](./ingest-canonical-review.md) §6), no un dump `asdict` del CLI.

#### 4.5.1 Checkpoint por rango (futuro)

Si un Job **reintenta un PDF** a mitad (timeout a las 2 h, OOM en el rango 7), hoy se re-parsean también los rangos 1–6.

Evolución: después de cada `normalize` de un `page_range`, subir p. ej. `gs://…/shards/{stem}/p051-100.json` (`shard_3.json` equivalente: índice + rango + stem). Al arrancar la tarea:

1. Listar shards ya escritos para ese stem.
2. Si hay JSON para `(lo, hi)` y el `docling_version` (y el hash/generation del PDF) coinciden → **no** llamar a `convert`; cargar el canónico parcial.
3. Procesar solo los rangos que falten.
4. `merge_canonical_shards` al final igual que ahora.

Invalidar el prefijo `shards/{stem}/` si cambia el PDF, la versión de Docling o el pipeline de refiners. No sustituye `canonical/{stem}.json` (eso es el documento **ya mergeado**, listo para indexar).

### 4.6 Modelos y arranque

`--tasks=154 --parallelism=3` arranca muchos contenedores. Pesos de Docling **en la imagen**. Si cada tarea descarga de HF, el backfill es inviable. Con pesos horneados el coste es cargar torch en RAM (~1 min por tarea): aceptable en un lote.

### 4.7 Qué limita el `parallelism`

El mínimo de: cuota 8 Gi × N, y **rate limit de Cohere**. 3 es el punto de partida; no 154.

### 4.8 GPU

Opcional después de medir CPU. Un L4 por tarea acelera layout/TableFormer; **no** cambia Jobs vs Service ni el “1 tarea = 1 PDF”.

---

## 5. Papel del Service (opcional, más adelante)

`main_ingest` + push **no parsea** Docling en el request. Como mucho:

1. Recibe `OBJECT_FINALIZE`.
2. Lanza **un Job de 1 tarea** para ese objeto.
3. Devuelve **200** en segundos (dentro del ack de 600 s).

Útil si ops quiere “soltar un PDF en el bucket”. No es necesario para la carga de 154 ni para el delta trimestral (prefijo + `jobs execute`).

No: 202 + trabajo en background dentro del mismo proceso del Service. No: Docling en el `POST /`.

---

## 6. Qué no hacer

| Idea | Por qué no |
|---|---|
| Una sola tarea que recorre los 154 PDFs | Un crash a mitad; wall-clock de días; no usáis paralelismo |
| Varios `convert` en el **mismo** contenedor | OOM en 8 Gi |
| Fan-out de **páginas** a Cloud Tasks | Complejidad de merge; 10 cargas de modelo por PDF largo; no es el cuello de 154 ficheros |
| `parallelism=20` | Cuota RAM + 429 Cohere |
| Service + ack síncrono para el TS de 468 págs. | Redelivery a los 10 min |
| GPU como requisito del MVP | Acelera; no desbloquea el diseño |

---

## 7. Relación con lo ya coronado (Service)

| Qué | Estado |
|---|---|
| `run_ingest` / `IngestDocumentService` / imagen ingest | Se reutiliza |
| Hexagonal, chat slim | Se mantiene |
| Service + push | **Camino GCP actual** ([gcp-ingest-pubsub.md](./gcp-ingest-pubsub.md)); este doc es el recambio para lote / ack |
| Runbook del Service | Válido ahora; un Job futuro puede reutilizar la misma imagen |
| Scraper futuro | Sigue siendo Job que **escribe GCS**; el Job de ingest **lee** el prefijo |

Orden práctico **después** de cerrar el worker Service (no bloquea esta rama):

1. Parser/normalizer en local ya cerrado ([ingest-canonical-review.md](./ingest-canonical-review.md) §19).
2. Medir un PDF corto en el **Service** (ack 600 s). El TS de 468 págs. en un Job de 1 tarea cuando exista.
3. Backfill 154 con `--tasks` + `--parallelism=3`.
4. Deltas = mismo Job. Dispatcher Service solo si hace falta UX de bucket.

---

## 8. Checklist de Job (cuando se implemente)

- [ ] Cloud Run Job `rag-ingest-job`, misma imagen que `Dockerfile.ingest`, entry CLI/`run_ingest` (no uvicorn).
- [ ] SA con lectura GCS + escritura `canonical/` + Qdrant/Cohere (secrets como el Service).
- [ ] Listado de prefijo + `CLOUD_RUN_TASK_INDEX` / `CLOUD_RUN_TASK_COUNT`.
- [ ] `--parallelism=3`, `--task-timeout` ≥ 1 h, `--max-retries` por tarea.
- [ ] Modelos Docling en imagen.
- [ ] Escritura opcional `canonical/{stem}.json`.
- [ ] Logs estructurados: `object`, `doc_id`, `pages`, `shards`, `wall_clock_s`, `docling_version`.
- [ ] Delta: prefijo distinto, mismo Job.

No Terraform en esta nota (igual que el runbook actual). Provisioning cuando se implemente, en runbook propio o extensión de [gcp_ingest_deployment.md](./gcp_ingest_deployment.md).
