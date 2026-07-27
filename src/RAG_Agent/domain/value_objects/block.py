from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    FORMULA = "formula"
    NOTE = "note"


@dataclass(frozen=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float
    coord_origin: str = "BOTTOMLEFT"


@dataclass(frozen=True)
class ImageRef:
    """Referencia a una imagen sin embeber bytes."""

    uri: str | None = None
    alt: str | None = None
    caption: str | None = None


@dataclass(frozen=True)
class TableData:
    """Tabla como texto estructurado (mismo shape para PDF nativo y OCR)."""

    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    caption: str | None = None


@dataclass(frozen=True)
class Block:
    """Unidad de contenido canónica. Fuente de verdad del texto/tabla/imagen."""

    id: str
    type: BlockType
    order: int
    page: int | None = None
    text: str | None = None
    level: int | None = None
    table: TableData | None = None
    image: ImageRef | None = None
    bbox: BoundingBox | None = None
    source_ref: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
