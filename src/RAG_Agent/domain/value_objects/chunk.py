from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """Unidad indexable para RAG (texto + citas al canónico)."""

    id: str
    doc_id: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section_id: str | None = None
    block_ids: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)
