from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Page:
    """Vista física: qué bloques aparecen en una página del PDF."""

    number: int
    block_ids: list[str] = field(default_factory=list)
    width: float | None = None
    height: float | None = None
