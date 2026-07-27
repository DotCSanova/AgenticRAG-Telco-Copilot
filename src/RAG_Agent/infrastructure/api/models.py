from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    pass


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
    pass
