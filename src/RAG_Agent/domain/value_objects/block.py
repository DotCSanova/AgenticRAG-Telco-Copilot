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
class LayoutSpan:
    """Rectángulo en una página PDF. Un block lógico puede tener varios (tabla partida)."""

    page: int | None = None
    bbox: BoundingBox | None = None
    source_ref: str | None = None


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
    layout_spans: tuple[LayoutSpan, ...] = ()

    def pdf_layout(self) -> tuple[LayoutSpan, ...]:
        """Spans para overlay PDF. Varios si el contenido cruza páginas."""
        if self.layout_spans:
            return self.layout_spans
        return (LayoutSpan(page=self.page, bbox=self.bbox, source_ref=self.source_ref),)

    def page_numbers(self) -> tuple[int, ...]:
        """Páginas en las que este block debe indexarse (chunk por página / overlay)."""
        numbers: list[int] = []
        seen: set[int] = set()

        def add(page: int | None) -> None:
            if page is not None and page not in seen:
                seen.add(page)
                numbers.append(page)

        if self.layout_spans:
            for span in self.layout_spans:
                add(span.page)
            return tuple(numbers)

        add(self.page)
        raw_end = self.metadata.get("page_end")
        if raw_end and self.page is not None:
            try:
                end = int(raw_end)
            except ValueError:
                return tuple(numbers)
            lo, hi = (self.page, end) if end >= self.page else (end, self.page)
            for page in range(lo, hi + 1):
                add(page)
        return tuple(numbers)
