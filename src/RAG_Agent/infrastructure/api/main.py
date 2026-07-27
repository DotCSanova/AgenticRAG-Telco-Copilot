from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from RAG_Agent.application.ingest_documents_service.ingest_document import IngestDocumentService
from RAG_Agent.config import settings
from RAG_Agent.infrastructure.api.models import (
    ChatRequest,
    EvalRequest,
    IngestDocumentsRequest,
    IngestDocumentsResponse,
    ResetMemoryRequest,
)
from RAG_Agent.infrastructure.indexing.bm25_embedder import BM25Embedder
from RAG_Agent.infrastructure.indexing.cohere_embedder import CohereEmbedder
from RAG_Agent.infrastructure.indexing.hybrid_embedder import HybridEmbedder
from RAG_Agent.infrastructure.indexing.qdrant_vector_store import QdrantVectorStore
from RAG_Agent.infrastructure.indexing.section_chunker import SectionChunker
from RAG_Agent.infrastructure.indexing.semantic_chunker import SemanticChunker
from RAG_Agent.infrastructure.ingestion.cascading_profile_resolver import CascadingProfileResolver
from RAG_Agent.infrastructure.ingestion.exceptions import PDFParsingException, PDFValidationError
from RAG_Agent.infrastructure.ingestion.native_pdf_pipeline import NativePdfPipeline

logger = logging.getLogger(__name__)


def _build_embedder():
    dense = CohereEmbedder()
    if settings.qdrant_enable_sparse:
        return HybridEmbedder(dense=dense, sparse=BM25Embedder())
    return dense


def _build_chunker():
    name = settings.chunker.lower().strip()
    if name == "section":
        return SectionChunker()
    if name == "semantic":
        return SemanticChunker(
            model_name=settings.semantic_chunk_model,
            threshold=settings.semantic_chunk_threshold,
            min_tokens=settings.semantic_chunk_min_tokens,
            max_tokens=settings.semantic_chunk_max_tokens,
        )
    raise ValueError(f"Unknown chunker={settings.chunker!r} (expected section|semantic)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Composition root: parse + chunk + hybrid embed + Qdrant.
    parser = NativePdfPipeline(CascadingProfileResolver())
    chunker = _build_chunker()
    app.state.ingest_service = IngestDocumentService(
        parser=parser,
        chunker=chunker,
        embedder=_build_embedder(),
        vector_store=QdrantVectorStore(),
    )
    logger.info(
        "IngestDocumentService ready (chunker=%s, sparse=%s)",
        settings.chunker,
        settings.qdrant_enable_sparse,
    )
    yield


app = FastAPI(
    title="RAG-Agent API",
    description="API del agente RAG sobre documentos técnicos.",
    docs_url="/docs",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {"message": "Welcome to RAG-Agent API. Visit /docs for documentation"}


@app.post("/chat")
async def chat(request: ChatRequest):
    return {"message": "Chat request received"}


@app.post("/eval")
async def eval_endpoint(request: EvalRequest):
    return {"message": "Eval request received"}


@app.post("/ingest", response_model=IngestDocumentsResponse)
async def ingest_documents(request: IngestDocumentsRequest, http_request: Request):
    """Parsea un PDF. Con ``index=true`` también chunk/embed/upsert."""
    path = Path(request.path).expanduser()
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"PDF not found: {path}")

    service: IngestDocumentService = http_request.app.state.ingest_service

    try:
        result = await asyncio.to_thread(service.execute, path, index=request.index)
    except PDFValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PDFParsingException as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=501,
            detail=f"Indexing stub not implemented yet: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Ingest failed for %s", path)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc

    document = result.canonical
    extra = dict(document.metadata.extra)
    extra.update(result.extra)
    return IngestDocumentsResponse(
        title=document.metadata.title,
        profile_id=document.metadata.profile_id,
        source_path=str(document.metadata.source_path),
        parser=document.metadata.parser,
        num_pages=int(extra.get("num_pages", len(document.pages))),
        num_blocks=int(extra.get("num_blocks", len(document.blocks))),
        num_sections=int(extra.get("num_sections", len(document.sections))),
        indexed=result.indexed,
        chunk_count=result.chunk_count,
        extra=extra,
    )


@app.post("/reset-memory")
async def reset_memory(request: ResetMemoryRequest):
    return {"message": "Reset memory request received"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
