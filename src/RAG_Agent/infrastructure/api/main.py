from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from RAG_Agent.application.ingest_documents_service.ingest_document import IngestDocumentService
from RAG_Agent.config import settings
from RAG_Agent.infrastructure.agent.agent import build_root_agent
from RAG_Agent.infrastructure.agent.tools.search_documents import make_search_documents_tool
from RAG_Agent.infrastructure.api.models import (
    ChatRequest,
    ChatResponse,
    EvalRequest,
    IngestDocumentsRequest,
    IngestDocumentsResponse,
    ResetMemoryRequest,
    ResetMemoryResponse,
)
from RAG_Agent.infrastructure.composition import (
    build_chunker,
    build_embedder,
    build_search_service,
    build_vector_store,
)
from RAG_Agent.infrastructure.ingestion.cascading_profile_resolver import CascadingProfileResolver
from RAG_Agent.infrastructure.ingestion.exceptions import PDFParsingException, PDFValidationError
from RAG_Agent.infrastructure.ingestion.native_pdf_pipeline import NativePdfPipeline

logger = logging.getLogger(__name__)


def _final_text_from_event(event) -> str | None:
    if not event.is_final_response():
        return None
    content = event.content
    if content is None or not content.parts:
        return None
    texts = [part.text for part in content.parts if getattr(part, "text", None)]
    if not texts:
        return None
    return "\n".join(texts)


async def _ensure_session(
    session_service: InMemorySessionService,
    *,
    app_name: str,
    user_id: str,
    session_id: str | None,
):
    if session_id:
        existing = await session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if existing is not None:
            return existing
    return await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id or str(uuid.uuid4()),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Composition root: ingest + search + ADK runner.
    parser = NativePdfPipeline(CascadingProfileResolver())
    chunker = build_chunker()
    embedder = build_embedder()
    vector_store = build_vector_store()

    app.state.ingest_service = IngestDocumentService(
        parser=parser,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    search_service = build_search_service(embedder=embedder, vector_store=vector_store)
    search_tool = make_search_documents_tool(search_service)
    root_agent = build_root_agent(tools=[search_tool])
    session_service = InMemorySessionService()
    app_name = settings.agent_app_name
    runner = Runner(
        agent=root_agent,
        app_name=app_name,
        session_service=session_service,
    )

    app.state.search_service = search_service
    app.state.session_service = session_service
    app.state.runner = runner
    app.state.agent_app_name = app_name

    logger.info(
        "Services ready (chunker=%s, sparse=%s, agent_model=%s)",
        settings.chunker,
        settings.qdrant_enable_sparse,
        settings.agent_model,
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


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request):
    runner: Runner = http_request.app.state.runner
    session_service: InMemorySessionService = http_request.app.state.session_service
    app_name: str = http_request.app.state.agent_app_name

    session = await _ensure_session(
        session_service,
        app_name=app_name,
        user_id=request.user_id,
        session_id=request.session_id,
    )

    user_message = types.Content(
        role="user",
        parts=[types.Part(text=request.message)],
    )
    final_text = ""
    try:
        async for event in runner.run_async(
            user_id=request.user_id,
            session_id=session.id,
            new_message=user_message,
        ):
            text = _final_text_from_event(event)
            if text is not None:
                final_text = text
    except Exception as exc:
        logger.exception("Chat failed for session=%s", session.id)
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    if not final_text:
        raise HTTPException(status_code=502, detail="Agent returned no final response")

    return ChatResponse(
        message=final_text,
        user_id=request.user_id,
        session_id=session.id,
    )


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


@app.post("/reset-memory", response_model=ResetMemoryResponse)
async def reset_memory(request: ResetMemoryRequest, http_request: Request):
    session_service: InMemorySessionService = http_request.app.state.session_service
    app_name: str = http_request.app.state.agent_app_name

    existing = await session_service.get_session(
        app_name=app_name,
        user_id=request.user_id,
        session_id=request.session_id,
    )
    if existing is not None:
        await session_service.delete_session(
            app_name=app_name,
            user_id=request.user_id,
            session_id=request.session_id,
        )

    await session_service.create_session(
        app_name=app_name,
        user_id=request.user_id,
        session_id=request.session_id,
    )
    return ResetMemoryResponse(
        user_id=request.user_id,
        session_id=request.session_id,
        reset=True,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
