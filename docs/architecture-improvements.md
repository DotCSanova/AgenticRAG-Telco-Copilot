# Mejoras de arquitectura pendientes

Notas de deuda estructural detectadas en revisión (2026). El hexagonal actual es sólido; estos puntos son el siguiente salto de madurez, no refactors cosméticos.

---

## 1. Partir el monolito de arranque (ingest/index vs serving/chat) — DONE (serving/ingest split)

**Problema.** Un solo proceso carga Docling/torch + embedders + agente en el `lifespan` de FastAPI. En local funciona; en Cloud Run ya ha costado OOM y rate-limits (p. ej. descarga BM25/HuggingFace al arrancar).

**Objetivo.** En producción, separar al menos dos surfaces:

| Surface | Responsabilidad | Dependencias pesadas |
|---|---|---|
| **Ingest / index** | PDF → CanonicalDocument → chunk → embed → upsert | Docling, torch, sentence-transformers, embedders |
| **Serving / chat** | `/chat`, `/reset-memory`, retrieval ligero | Agente (ADK), embedder de query, Qdrant client, reranker |

**Hecho (2026-07-30).** serving/ingest split en `main`: `main_chat` + `Dockerfile.serving` / `Dockerfile.ingest`, composition `serving`⊥`ingest`, `ingest_local.py`, uv groups. Detalle: [serving-ingest-split.md](./serving-ingest-split.md), diseño [split-ingest-serving.md](./split-ingest-serving.md).

**Siguiente.** GCS + Pub/Sub + Cloud Run Job reusing `build_ingest_service` / `ingest_local` core (no HTTP ingest).

---

## 2. Adapter explícito `SessionStore` ↔ ADK

**Problema.** El puerto `SessionStore` existe en domain, pero `build_session_service()` devuelve el `BaseSessionService` de ADK y confía en tipado estructural. Funciona; el contrato no es honesto.

**Objetivo.** Un adapter en infrastructure que:

- implemente `SessionStore` de forma explícita;
- encapsule `InMemorySessionService` / `DatabaseSessionService` detrás del puerto;
- deje a `ChatService` / `ResetMemoryService` tipados solo contra el puerto de dominio.

---

## 3. Enriquecer el dominio de agente (cuando crezca)

**Problema.** El dominio de agente es mínimo a propósito: un agente + una tool (`search_documents`), spec + prompt en domain, lógica de turno en el adapter ADK. Está bien para el tamaño actual.

**Objetivo.** Si el producto crece (multi-tool, guardrails, citation post-processing, multi-agent), mover más modelo a domain y menos lógica al adapter:

- contratos de tools / resultados tipados;
- post-procesado de citaciones (`doc_id`, `section_path`) fuera de ADK;
- guardrails / políticas de respuesta en application o domain;
- mantener ADK (u otro runtime) como detalle de infrastructure detrás de `AgentRuntime`.

No adelantar abstracciones: crecer el dominio **cuando** aparezca la segunda tool o el segundo agente, no antes.
