"""Chat serving FastAPI app: /chat, /reset-memory, /eval stub (no ingest)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from RAG_Agent.application.chat_service.chat import ChatService
from RAG_Agent.application.reset_memory_service.reset_memory import ResetMemoryService
from RAG_Agent.config import settings
from RAG_Agent.domain.exceptions import AgentEmptyResponseError
from RAG_Agent.domain.tools.search_documents import make_search_documents_tool
from RAG_Agent.infrastructure.agent.adk_runtime import build_adk_runtime
from RAG_Agent.infrastructure.agent.sessions import build_session_service
from RAG_Agent.infrastructure.api.models import (
    ChatRequest,
    ChatResponse,
    EvalRequest,
    ResetMemoryRequest,
    ResetMemoryResponse,
)
from RAG_Agent.infrastructure.composition.serving import build_search_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    search_service = build_search_service()
    search_tool = make_search_documents_tool(search_service.execute)
    session_service = build_session_service()
    app_name = settings.agent_app_name
    runtime = build_adk_runtime(
        tools=[search_tool],
        session_service=session_service,
        app_name=app_name,
    )

    app.state.search_service = search_service
    app.state.session_service = session_service
    app.state.agent_app_name = app_name
    app.state.chat_service = ChatService(
        runtime=runtime,
        sessions=session_service,
        app_name=app_name,
    )
    app.state.reset_memory_service = ResetMemoryService(
        sessions=session_service,
        app_name=app_name,
    )

    logger.info(
        "Serving ready (sparse=%s, agent_model=%s)",
        settings.qdrant_enable_sparse,
        settings.agent_model,
    )
    yield


app = FastAPI(
    title="RAG-Agent API",
    description="Serving API for the technical-document RAG agent for Telco Copilot.",
    docs_url="/docs",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {"message": "Welcome to RAG-Agent API. Visit /docs for documentation"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request):
    service: ChatService = http_request.app.state.chat_service
    try:
        result = await service.execute(
            user_id=request.user_id,
            message=request.message,
            session_id=request.session_id,
        )
    except AgentEmptyResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat failed for user=%s", request.user_id)
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    return ChatResponse(
        message=result.message,
        user_id=result.user_id,
        session_id=result.session_id,
    )


@app.post("/eval")
async def eval_endpoint(request: EvalRequest):
    return {"message": "Eval request received"}


@app.post("/reset-memory", response_model=ResetMemoryResponse)
async def reset_memory(request: ResetMemoryRequest, http_request: Request):
    service: ResetMemoryService = http_request.app.state.reset_memory_service
    try:
        result = await service.execute(
            user_id=request.user_id,
            session_id=request.session_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Reset memory failed for session=%s", request.session_id)
        raise HTTPException(status_code=500, detail=f"Reset memory failed: {exc}") from exc

    return ResetMemoryResponse(
        user_id=result.user_id,
        session_id=result.session_id,
        reset=result.reset,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
