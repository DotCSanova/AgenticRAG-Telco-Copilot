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
