"""Cloud Run ingest worker: Pub/Sub push → GCS download → run_ingest (no ADK).

Canonical process and HTTP contract: ``docs/ingest-api.md`` (also used as this
app's OpenAPI description when the file is present in the image/repo).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from RAG_Agent.application.ingest_documents_service.ingest_document import IngestDocumentService
from RAG_Agent.config import settings
from RAG_Agent.infrastructure.api.ingest_events import (
    EnvelopeError,
    classify_object,
    parse_pubsub_gcs_envelope,
)
from RAG_Agent.infrastructure.composition.ingest import build_ingest_service, run_ingest
from RAG_Agent.infrastructure.secrets.gcp_secrets import apply_ingest_secrets_from_secret_manager
from RAG_Agent.infrastructure.storage.gcs import download_gcs_object

logger = logging.getLogger(__name__)

_FALLBACK_DESCRIPTION = (
    "Pub/Sub push worker: GCS object finalize → index into Qdrant. "
    "Canonical reference: docs/ingest-api.md in the repository."
)


def _api_description() -> str:
    """Load docs/ingest-api.md when packaged with the image or running from the repo."""
    doc = Path(__file__).resolve().parents[4] / "docs" / "ingest-api.md"
    if doc.is_file():
        return doc.read_text(encoding="utf-8")
    return _FALLBACK_DESCRIPTION


def _is_gcs_not_found(exc: BaseException) -> bool:
    return type(exc).__name__ == "NotFound" and type(exc).__module__.startswith("google.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    apply_ingest_secrets_from_secret_manager()
    app.state.ingest_service = build_ingest_service()
    logger.info(
        "Ingest worker ready (sparse=%s, secret_manager=%s)",
        settings.qdrant_enable_sparse,
        settings.use_secret_manager,
    )
    yield


app = FastAPI(
    title="RAG-Agent Ingest",
    description=_api_description(),
    docs_url="/docs",
    lifespan=lifespan,
)


@app.post("/")
def pubsub_push(request: Request, body: dict[str, Any]):
    try:
        notification = parse_pubsub_gcs_envelope(body)
    except EnvelopeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if notification.event_type and notification.event_type != "OBJECT_FINALIZE":
        logger.info(
            "Skipping non-finalize event bucket=%s object=%s eventType=%s",
            notification.bucket,
            notification.object_id,
            notification.event_type,
        )
        return JSONResponse(
            status_code=200,
            content={"status": "ignored", "eventType": notification.event_type},
        )

    kind = classify_object(notification.object_id)
    if kind == "word":
        raise HTTPException(
            status_code=400,
            detail="Word documents (.doc/.docx) are not supported yet",
        )
    if kind == "unsupported":
        raise HTTPException(
            status_code=400,
            detail=f"unsupported object type: {notification.object_id!r}",
        )

    service: IngestDocumentService = request.app.state.ingest_service
    try:
        with download_gcs_object(notification.bucket, notification.object_id) as path:
            result = run_ingest(path, index=True, service=service)
    except Exception as exc:
        if _is_gcs_not_found(exc):
            logger.warning(
                "GCS object not found bucket=%s object=%s generation=%s",
                notification.bucket,
                notification.object_id,
                notification.generation,
            )
            raise HTTPException(status_code=404, detail="GCS object not found") from exc
        logger.exception(
            "Ingest failed bucket=%s object=%s generation=%s",
            notification.bucket,
            notification.object_id,
            notification.generation,
        )
        raise HTTPException(status_code=500, detail=f"ingest failed: {exc}") from exc

    doc_id = result.extra.get("doc_id", "")
    logger.info(
        "Ingest ok bucket=%s object=%s generation=%s doc_id=%s "
        "chunk_count=%s deleted=%s upserted=%s",
        notification.bucket,
        notification.object_id,
        notification.generation,
        doc_id,
        result.chunk_count,
        result.extra.get("deleted", "0"),
        result.extra.get("upserted", "0"),
    )
    return {
        "status": "ok",
        "doc_id": doc_id,
        "chunk_count": result.chunk_count,
        "deleted": result.extra.get("deleted", "0"),
        "upserted": result.extra.get("upserted", "0"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
