from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Section:
    """Vista lógica: agrupa bloques por heading; page_start/end son derivados."""

    id: str
    title: str
    level: int
    order: int
    parent_id: str | None = None
    block_ids: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)
