from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Mensaje del usuario")
    user_id: str = Field("default", description="Identificador de usuario para la sesión")
    session_id: str | None = Field(
        None,
        description="Id de sesión; si falta o no existe, se crea una nueva",
    )


class ChatResponse(BaseModel):
    message: str
    user_id: str
    session_id: str


class EvalRequest(BaseModel):
    pass


class ResetMemoryRequest(BaseModel):
    user_id: str = Field("default")
    session_id: str = Field(..., min_length=1)


class ResetMemoryResponse(BaseModel):
    user_id: str
    session_id: str
    reset: bool
