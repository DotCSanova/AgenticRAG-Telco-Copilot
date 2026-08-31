from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    user_id: str = Field("default", description="User id used for session scoping")
    session_id: str | None = Field(
        None,
        description="Session id; if missing or unknown, a new session is created",
    )


class ChatResponse(BaseModel):
    message: str
    user_id: str
    session_id: str


class EvalRequest(BaseModel):
    """Reserved for a future evaluation endpoint."""


class ResetMemoryRequest(BaseModel):
    user_id: str = Field("default")
    session_id: str = Field(..., min_length=1)


class ResetMemoryResponse(BaseModel):
    user_id: str
    session_id: str
    reset: bool


class IngestOkResponse(BaseModel):
    """Pub/Sub push success body after indexing a PDF."""

    status: Literal["ok"] = "ok"
    doc_id: str
    chunk_count: int
    deleted: str = "0"
    upserted: str = "0"


class IngestIgnoredResponse(BaseModel):
    """Ack for non-OBJECT_FINALIZE notifications (no ingest)."""

    status: Literal["ignored"] = "ignored"
    eventType: str | None = None
