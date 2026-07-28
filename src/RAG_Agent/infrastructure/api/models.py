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


class IngestDocumentsRequest(BaseModel):
    """Ruta a un PDF accesible por el servidor (absoluta o relativa al CWD)."""

    path: str = Field(..., description="Ruta del PDF a ingerir")
    index: bool = Field(
        False,
        description="Si true: chunk + embed + upsert (requiere adapters implementados)",
    )


class IngestDocumentsResponse(BaseModel):
    title: str | None
    profile_id: str | None
    source_path: str
    parser: str | None
    num_pages: int
    num_blocks: int
    num_sections: int
    indexed: bool = False
    chunk_count: int = 0
    extra: dict[str, str] = Field(default_factory=dict)


class ResetMemoryRequest(BaseModel):
    user_id: str = Field("default")
    session_id: str = Field(..., min_length=1)


class ResetMemoryResponse(BaseModel):
    user_id: str
    session_id: str
    reset: bool
