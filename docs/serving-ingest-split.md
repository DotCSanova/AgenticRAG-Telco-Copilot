# Serving/ingest split — chat slim ⊥ local ingest

Implementation of the **serving/ingest split** (local, no GCP) defined in [split-ingest-serving.md](./split-ingest-serving.md) (§7 decisions, §9 checklist).

**Goal:** split the monolith into two images/processes, without GCP (no Pub/Sub/GCS), keeping hexagonal boundaries.

**Out of scope:** GCS, Pub/Sub, SQL document registry, scraper, Fan-Out, `SessionStore` adapter, richer agent domain.

---

## Decisiones ya cerradas (no reabrir)

| Tema | Decisión |
|---|---|
| Orden | serving/ingest split primero; GCP después |
| Imágenes | A2 — `Dockerfile.serving` + `Dockerfile.ingest` |
| Disparo ingest | Script local `run_ingest(path)` (mismo núcleo que el futuro worker) |
| API chat | Sin `POST /ingest` |
| Dev UI | Misma imagen serving, otro CMD |
| Re-ingest | `delete_by_doc_id` + upsert; `doc_id` = stem del PDF |
| Registro SQL | Not in this phase |
| Hexagonal | Intact: entrypoints + composition + packaging only |

Already on main (no work needed here except regressions): `search_documents` tool async + `asyncio.to_thread`.

---

## Criterio de done

- [x] API chat sin `/ingest`; entrypoint `main_chat`.
- [x] Dos Dockerfiles; serving **sin** Docling/torch.
- [x] Dev UI sobre imagen serving (`profile: dev`).
- [x] `scripts/ingest_local.py` indexa contra Qdrant con delete+upsert por stem.
- [x] Composition serving no importa `infrastructure.ingestion` / Docling.
- [x] Tests verdes + guardrail de imports serving.
- [x] README/ops actualizado (cómo chat / cómo ingest local).

**Status:** merged to `main` (2026-07-30) from `feature/pr1-solo-a-split` (full commit history preserved).

### Packaging notes landed with this split

- uv `dependency-groups`: `serving`, `ingest`, `dev`, `notebooks`; `default-groups = ["serving", "dev"]`.
- Host ingest: `--group ingest --extra cu124` (CUDA) or `--extra cpu`. `Dockerfile.ingest` uses `--extra cpu` (no NVIDIA wheels).
- Docling OpenCV: `opencv-python-headless` + `override-dependencies` excluding `opencv-python` (no X11/GL apt in ingest image).
- Compose: keep service name `agent-api`; add `agent-ingest` (`profiles: [ingest]`).

---

## Cambios por capa (checklist ejecutable)

### 1. Domain — puerto VectorStore

**Archivos**

- [`src/RAG_Agent/domain/ports/vector_store.py`](../src/RAG_Agent/domain/ports/vector_store.py)

**Cambios**

- Añadir `delete_by_doc_id(self, doc_id: str) -> int` (points borrados o equivalente).

### 2. Infrastructure — Qdrant

**Archivos**

- [`src/RAG_Agent/infrastructure/indexing/qdrant_vector_store.py`](../src/RAG_Agent/infrastructure/indexing/qdrant_vector_store.py)
- `tests/test_qdrant_vector_store.py` (ampliar)

**Cambios**

- Implementar delete por filtro payload `doc_id`.
- Tests (cliente fake o `:memory:`).

### 3. Application — ingest idempotente

**Archivos**

- [`src/RAG_Agent/application/ingest_documents_service/ingest_document.py`](../src/RAG_Agent/application/ingest_documents_service/ingest_document.py)
- Tests nuevos o existentes de ingest

**Cambios**

- En `execute(..., index=True)`:
  1. parse → canonical  
  2. `doc_id = canonical.metadata.source_path.stem` (alineado con chunkers)  
  3. `vector_store.delete_by_doc_id(doc_id)`  
  4. chunk → embed → upsert  
- `index=False` sin cambios de semántica (solo parse).

### 4. Composition split

**Archivos**

- Sustituir / partir [`src/RAG_Agent/infrastructure/composition.py`](../src/RAG_Agent/infrastructure/composition.py)
  - Opción preferida: `infrastructure/composition/serving.py` + `ingest.py` (+ `__init__` si hace falta)
  - Alternativa aceptable: un solo módulo con secciones claras y **imports lazy** para que `import serving` no cargue Docling

**Cambios**

- `build_serving_*`: embedder query, vector store, reranker, search service, sessions, ADK + tool.
- `build_ingest_*`: `NativePdfPipeline`, chunker, embedder doc, vector store.
- Regla dura: serving **no** importa `NativePdfPipeline` / `docling` / semantic chunker torch.

### 5. API chat (sin ingest)

**Archivos**

- Nuevo: `src/RAG_Agent/infrastructure/api/main_chat.py`
- Eliminar (preferido): monolito [`main.py`](../src/RAG_Agent/infrastructure/api/main.py) con chat+ingest
- [`models.py`](../src/RAG_Agent/infrastructure/api/models.py): quitar `IngestDocumentsRequest` / `IngestDocumentsResponse` de la API de chat (o dejar de usarlos; sin shims eternos)
- Tests: `tests/test_api_chat.py`; retirar/mover `tests/test_api_ingest.py` hacia script o borrar si solo cubría HTTP ingest

**Cambios**

- Lifespan solo: search + chat + reset.
- Rutas: `/`, `/chat`, `/reset-memory` (sin `/ingest`; `/eval` stub — quitar si sigue vacío y no aporta).
- CMD prod / compose → `main_chat`.

### 6. Script ingest local

**Archivos**

- Nuevo: `scripts/ingest_local.py` (o `python -m` equivalente bajo infrastructure)
- Migrar callers de `scripts/index_pdf_append.py` / `scripts/reindex_tr_section.py` al nuevo root (actualizar o reemplazar; sin aliases muertos)

**Cambios**

- CLI: `path` al PDF; `--index/--no-index` (default index true).
- Wire `build_ingest_*` → `IngestDocumentService.execute`.
- Logs: `doc_id`, `chunk_count`, `indexed`.

### 7. Packaging + Docker

**Archivos**

- [`pyproject.toml`](../pyproject.toml) — extras o dependency-groups `serving` / `ingest`
- Nuevo: `Dockerfile.serving`, `Dockerfile.ingest`
- Retirar o dejar de usar el [`Dockerfile`](../Dockerfile) monolito como default (actualizar referencias)

**Reparto orientativo de deps**

| Grupo | Incluye | Excluye |
|---|---|---|
| **serving** | FastAPI, ADK, LiteLLM, sqlalchemy/asyncpg, cohere, qdrant-client, fastembed (si hybrid query) | docling, torch, torchvision, sentence-transformers, pymupdf (salvo que algo de serving lo exija — no debería) |
| **ingest** | docling, torch, pymupdf, sentence-transformers, cohere, qdrant-client, fastembed | google-adk (no necesario en worker) |

**Nota BM25:** con `qdrant_enable_sparse=True`, **serving** necesita sparse en query → fastembed/BM25 en imagen serving; pre-download en build de serving si aplica. No es Docling.

### 8. Compose

**Archivo:** [`docker-compose.yaml`](../docker-compose.yaml)

**Cambios**

| Service | Imagen | Comando |
|---|---|---|
| `agent-chat` | `Dockerfile.serving` | `main_chat` |
| `agent-dev-ui` | misma serving | `dev_ui` (`profiles: [dev]`) |
| `qdrant` / `postgres` | como hoy | — |

- Documentar ingest: `uv run` / `docker compose run` con imagen ingest + `ingest_local.py`.
- No montar HTTP ingest en el servicio chat.

### 9. Tests y CI

**Cambios**

- Unit: `delete_by_doc_id`, ingest idempotente (fake store).
- Guardrail: test que `main_chat` (o composition serving) no importe `docling` / `NativePdfPipeline`.
- Ajustar tests que apuntaban a `main:app` monolito.
- CI (si aplica): build `Dockerfile.serving` (+ ingest si el runner aguanta tiempo/tamaño).

### 10. Docs

**Archivos**

- [`README.md`](../README.md) — arranque chat vs ingest local
- [split-ingest-serving.md](./split-ingest-serving.md) — entrada en log §7.5 cuando se mergee
- Este archivo — marcar done al cerrar la PR

---

## Orden de commits sugerido (dentro de la PR)

1. `delete_by_doc_id` (puerto + Qdrant + tests)  
2. Ingest idempotente  
3. Composition serving/ingest  
4. `main_chat` + borrar monolito + tests API  
5. `ingest_local.py` + scripts  
6. `pyproject` extras + Dockerfiles + compose  
7. Guardrail imports + README  

Así cada commit es revisable y reversible.

---

## Árbol objetivo tras serving/ingest split

```text
src/RAG_Agent/
  domain/ports/vector_store.py          # + delete_by_doc_id
  application/ingest_documents_service/ # delete luego upsert
  infrastructure/
    composition/serving.py
    composition/ingest.py
    api/main_chat.py                    # sin /ingest
    api/models.py                       # sin modelos ingest (o sin uso)
    agent/...                           # solo serving
    ingestion/...                       # solo ingest
    indexing/...
scripts/ingest_local.py
Dockerfile.serving
Dockerfile.ingest
docker-compose.yaml                     # agent-chat (+ dev-ui)
docs/serving-ingest-split.md            # this doc
```

---

## Verificación manual rápida

```bash
# Chat
docker compose up -d postgres qdrant agent-chat
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d "{\"message\":\"ping\",\"user_id\":\"t\"}"

# Ingest (host o contenedor ingest)
uv run python scripts/ingest_local.py path/al.pdf
# Re-ejecutar el mismo path → no duplicar chunks (delete+upsert)

# Guardrail mental
# La imagen serving no debe listar torch/docling en pip freeze del contenedor
```

---

## Después de merge (PR2+, no aquí)

1. Handler Pub/Sub → download GCS → `run_ingest(path)`.  
2. Registro SQL opcional.  
3. Canonical en GCS / Fan-Out solo si hace falta ([§7.4](./split-ingest-serving.md)).
