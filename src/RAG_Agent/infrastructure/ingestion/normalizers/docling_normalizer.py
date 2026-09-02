from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from docling_core.types.doc import (
    CodeItem,
    DocItemLabel,
    DoclingDocument,
    FormulaItem,
    ListItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)
from docling_core.types.doc.document import (
    ContentLayer,
    FieldHeadingItem,
    FieldValueItem,
)

from RAG_Agent.domain.doc_processing_rules.document_processing import (
    DocumentProcessingRules,
    DocumentProfile,
)
from RAG_Agent.domain.value_objects._block_utils import index_block_ids_by_page
from RAG_Agent.domain.value_objects.block import (
    Block,
    BlockType,
    BoundingBox,
    ImageRef,
    LayoutSpan,
    TableData,
    coord_origin_name,
)
from RAG_Agent.domain.value_objects.block_pipeline import refine_block_sequence
from RAG_Agent.domain.value_objects.canonical_document import (
    CanonicalDocument,
    DocumentMetadata,
)
from RAG_Agent.domain.value_objects.page import Page
from RAG_Agent.domain.value_objects.section import Section

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_BODY_LAYERS = {ContentLayer.BODY}
_DROP_LABELS = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}


def _linked_caption_refs(doc: DoclingDocument) -> set[str]:
    linked: set[str] = set()
    for element, _depth in doc.iterate_items(included_content_layers=_BODY_LAYERS):
        refs = list(getattr(element, "captions", None) or ())
        refs.extend(getattr(element, "footnotes", None) or ())
        for ref in refs:
            cref = getattr(ref, "cref", None)
            if cref:
                linked.add(cref)
            resolved = _resolve_ref(doc, ref)
            self_ref = getattr(resolved, "self_ref", None) if resolved is not None else None
            if self_ref:
                linked.add(self_ref)
    return linked


def _resolve_ref(doc: DoclingDocument, ref: Any) -> Any | None:
    resolve = getattr(ref, "resolve", None)
    if not callable(resolve):
        return None
    try:
        return resolve(doc)
    except (AttributeError, RuntimeError, IndexError, TypeError, ValueError):
        return None


def _resolve_caption_text(doc: DoclingDocument, refs: Any) -> str | None:
    for ref in refs or ():
        resolved = _resolve_ref(doc, ref)
        text = (getattr(resolved, "text", None) or "").strip()
        if text:
            return text
    return None


def _bbox_from_raw(raw_bbox: Any) -> BoundingBox | None:
    if raw_bbox is None:
        return None
    return BoundingBox(
        x0=float(getattr(raw_bbox, "l", getattr(raw_bbox, "x0", 0.0))),
        y0=float(getattr(raw_bbox, "b", getattr(raw_bbox, "y0", 0.0))),
        x1=float(getattr(raw_bbox, "r", getattr(raw_bbox, "x1", 0.0))),
        y1=float(getattr(raw_bbox, "t", getattr(raw_bbox, "y1", 0.0))),
        coord_origin=coord_origin_name(getattr(raw_bbox, "coord_origin", "BOTTOMLEFT")),
    )


def _layout_from_element(
    element: Any,
) -> tuple[int | None, BoundingBox | None, tuple[LayoutSpan, ...]]:
    source_ref = getattr(element, "self_ref", None)
    spans: list[LayoutSpan] = []
    for prov in getattr(element, "prov", None) or ():
        page = getattr(prov, "page_no", None)
        spans.append(
            LayoutSpan(page=page, bbox=_bbox_from_raw(getattr(prov, "bbox", None)), source_ref=source_ref)
        )
    if not spans:
        return None, None, ()
    layout_spans = tuple(spans) if len(spans) > 1 else ()
    return spans[0].page, spans[0].bbox, layout_spans


def _grid_to_table_data(table: TableItem) -> TableData:
    data = getattr(table, "data", None)
    grid = getattr(data, "grid", None) if data is not None else None
    if not grid:
        return TableData()

    header_rows: list[list[str]] = []
    body_rows: list[list[str]] = []
    saw_column_header = False
    for row in grid:
        cells = list(row) if isinstance(row, list) else [row]
        texts = [str(getattr(cell, "text", cell) or "") for cell in cells]
        if any(getattr(cell, "column_header", False) for cell in cells):
            saw_column_header = True
            header_rows.append(texts)
        else:
            body_rows.append(texts)

    if not saw_column_header:
        return TableData(headers=[], rows=body_rows)

    width = max((len(row) for row in header_rows), default=0)
    headers = [""] * width
    for row in header_rows:
        for index, cell in enumerate(row):
            if not cell.strip():
                continue
            headers[index] = f"{headers[index]} {cell}".strip() if headers[index] else cell
    return TableData(headers=headers, rows=body_rows)


class DoclingNormalizer:
    """Mapea DoclingDocument → CanonicalDocument (blocks, pages, sections).

    La lógica específica de familia vive en ``DocumentProcessingRules``
    inyectadas vía ``DocumentProfile``.
    """

    def normalize(
        self,
        doc: DoclingDocument,
        *,
        source_path: Path,
        profile: DocumentProfile,
        parser_name: str = "docling",
    ) -> CanonicalDocument:
        """Map a Docling document to a canonical document via the shared block pipeline.

        Args:
            doc: Parser output (Docling tree).
            source_path: Original PDF path stored in metadata.
            profile: Family identity and processing rules.
            parser_name: Value stored in ``metadata.parser``.

        Returns:
            Canonical document with refined blocks, pages, sections, and title.
        """
        rules = profile.rules
        blocks = refine_block_sequence(self._extract_blocks(doc, rules), rules=rules)
        pages = self._build_pages(doc, blocks)
        sections = self.build_sections(blocks, rules=rules)
        title = self.resolve_title(
            sections, blocks, profile.identity.title_hint, rules=rules
        )

        extra = dict(profile.identity.metadata)
        extra["num_pages"] = str(len(pages))
        extra["num_blocks"] = str(len(blocks))
        extra["num_sections"] = str(len(sections))

        return CanonicalDocument(
            metadata=DocumentMetadata(
                source_path=source_path,
                title=title,
                profile_id=rules.profile_id,
                parser=parser_name,
                extra=extra,
            ),
            blocks={block.id: block for block in blocks},
            pages=pages,
            sections=sections,
        )

    def _extract_blocks(
        self,
        doc: DoclingDocument,
        rules: DocumentProcessingRules,
    ) -> list[Block]:
        linked_refs = _linked_caption_refs(doc)
        drop_index = "contents" in rules.removable_sections
        blocks: list[Block] = []
        order = 0

        for element, _depth in doc.iterate_items(included_content_layers=_BODY_LAYERS):
            label = getattr(element, "label", None)
            if label in _DROP_LABELS:
                continue
            if drop_index and label == DocItemLabel.DOCUMENT_INDEX:
                continue
            self_ref = getattr(element, "self_ref", None)
            if self_ref and self_ref in linked_refs:
                continue

            if isinstance(element, (SectionHeaderItem, TitleItem)):
                title = (element.text or "").strip()
                if not title:
                    continue

                extracted_level = 1
                if isinstance(element, SectionHeaderItem):
                    extracted_level = int(element.level or 1)
                level = rules.infer_heading_level(title, extracted_level=extracted_level)

                block = self._make_text_block(
                    element=element,
                    block_type=BlockType.HEADING,
                    order=order,
                    text=title,
                    level=level,
                )
                blocks.append(block)
                order += 1
                continue

            block = self._element_to_block(element, order=order, doc=doc, rules=rules)
            if block is None:
                continue
            blocks.append(block)
            order += 1

        return blocks

    def _element_to_block(
        self,
        element: Any,
        *,
        order: int,
        doc: DoclingDocument,
        rules: DocumentProcessingRules,
    ) -> Block | None:
        if isinstance(element, ListItem):
            text = (element.text or "").strip()
            if not text:
                return None
            return self._make_text_block(
                element=element,
                block_type=BlockType.LIST_ITEM,
                order=order,
                text=text,
            )

        if isinstance(element, CodeItem):
            text = (element.text or "").strip()
            if not text:
                return None
            return self._make_text_block(
                element=element,
                block_type=BlockType.CODE,
                order=order,
                text=text,
            )

        if isinstance(element, FormulaItem):
            text = (element.text or "").strip()
            if not text:
                return None
            return self._make_text_block(
                element=element,
                block_type=BlockType.FORMULA,
                order=order,
                text=text,
            )

        if isinstance(element, (TextItem, FieldHeadingItem, FieldValueItem)):
            text = (element.text or "").strip()
            if not text or rules.is_noise_paragraph(text):
                return None
            return self._make_text_block(
                element=element,
                block_type=BlockType.PARAGRAPH,
                order=order,
                text=text,
            )

        if isinstance(element, TableItem):
            page, bbox, spans = _layout_from_element(element)
            caption = _resolve_caption_text(doc, getattr(element, "captions", None))
            table = self._table_to_data(element)
            if caption and table.caption is None:
                table = TableData(headers=table.headers, rows=table.rows, caption=caption)
            return Block(
                id=f"block_{order}",
                type=BlockType.TABLE,
                order=order,
                page=page,
                table=table,
                bbox=bbox,
                source_ref=getattr(element, "self_ref", None),
                layout_spans=spans,
            )

        if isinstance(element, PictureItem):
            page, bbox, spans = _layout_from_element(element)
            caption = _resolve_caption_text(doc, getattr(element, "captions", None))
            return Block(
                id=f"block_{order}",
                type=BlockType.IMAGE,
                order=order,
                page=page,
                image=ImageRef(uri=None, alt=None, caption=caption),
                bbox=bbox,
                source_ref=getattr(element, "self_ref", None),
                layout_spans=spans,
            )

        return None

    def _make_text_block(
        self,
        *,
        element: Any,
        block_type: BlockType,
        order: int,
        text: str,
        level: int | None = None,
    ) -> Block:
        page, bbox, spans = _layout_from_element(element)
        return Block(
            id=f"block_{order}",
            type=block_type,
            order=order,
            page=page,
            text=text,
            level=level,
            bbox=bbox,
            source_ref=getattr(element, "self_ref", None),
            layout_spans=spans,
        )

    @staticmethod
    def _table_to_data(table: TableItem) -> TableData:
        try:
            return _grid_to_table_data(table)
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "Degraded TABLE %s: %s",
                getattr(table, "self_ref", None),
                exc,
            )
            return TableData()

    def _build_pages(self, doc: DoclingDocument, blocks: list[Block]) -> list[Page]:
        by_page = index_block_ids_by_page(blocks)

        doc_pages = getattr(doc, "pages", {}) or {}
        page_numbers = sorted(set(by_page) | {int(key) for key in doc_pages})

        pages: list[Page] = []
        for number in page_numbers:
            page_meta = doc_pages.get(number) or doc_pages.get(str(number))
            width = height = None
            if page_meta is not None:
                size = getattr(page_meta, "size", None)
                if size is not None:
                    width = float(getattr(size, "width", 0) or 0) or None
                    height = float(getattr(size, "height", 0) or 0) or None
            pages.append(
                Page(
                    number=number,
                    block_ids=by_page.get(number, []),
                    width=width,
                    height=height,
                )
            )
        return pages

    def build_sections(
        self,
        blocks: list[Block],
        *,
        rules: DocumentProcessingRules,
    ) -> list[Section]:
        sections: list[Section] = []
        stack: list[tuple[str, int]] = []
        current: dict[str, Any] | None = None

        def flush_current() -> None:
            nonlocal current
            if current is None:
                return
            block_ids: list[str] = list(current["block_ids"])
            page_by_id = {block.id: block.page for block in blocks}
            page_numbers = [
                page
                for block_id in block_ids
                if (page := page_by_id.get(block_id)) is not None
            ]
            page_start = min(page_numbers) if page_numbers else None
            page_end = max(page_numbers) if page_numbers else None
            sections.append(
                Section(
                    id=str(current["id"]),
                    title=str(current["title"]),
                    level=int(current["level"]),
                    order=int(current["order"]),
                    parent_id=current["parent_id"],
                    block_ids=block_ids,
                    page_start=page_start,
                    page_end=page_end,
                )
            )
            current = None

        section_order = 0
        orphan_title = rules.orphan_section_title
        for block in blocks:
            if block.type == BlockType.HEADING:
                flush_current()
                level = block.level or 1
                while stack and stack[-1][1] >= level:
                    stack.pop()
                parent_id = stack[-1][0] if stack else None
                section_id = self._section_id(block.text or "section", section_order)
                current = {
                    "id": section_id,
                    "title": block.text or "",
                    "level": level,
                    "order": section_order,
                    "parent_id": parent_id,
                    "block_ids": [block.id],
                }
                stack.append((section_id, level))
                section_order += 1
                continue

            if current is None:
                section_id = self._section_id(orphan_title, section_order)
                current = {
                    "id": section_id,
                    "title": orphan_title,
                    "level": 1,
                    "order": section_order,
                    "parent_id": None,
                    "block_ids": [block.id],
                }
                stack = [(section_id, 1)]
                section_order += 1
            else:
                current["block_ids"].append(block.id)

        flush_current()
        return sections

    @staticmethod
    def _section_id(title: str, order: int) -> str:
        slug = _SLUG_RE.sub("-", title.lower()).strip("-")
        slug = slug[:48] or "section"
        return f"sec_{order:04d}_{slug}"

    @staticmethod
    def resolve_title(
        sections: list[Section],
        blocks: list[Block],
        title_hint: str,
        *,
        rules: DocumentProcessingRules,
    ) -> str:
        """Resuelve el título del documento.

        Prioridad:
        1. Texto de portada (``rules.cover_page_number``): no boilerplate
        2. ``title_hint`` del perfil
        3. Primer heading/sección no genérico (según ``rules.is_generic_doc_title``)
        """
        cover = DoclingNormalizer._cover_title_from_blocks(blocks, rules=rules)
        if cover:
            return cover

        hint = title_hint.replace("-", " ").replace("_", " ").strip()
        if hint:
            return hint

        for section in sections:
            title = (section.title or "").strip()
            if title and not rules.is_generic_doc_title(title):
                return title

        for block in blocks:
            if block.type != BlockType.HEADING or not block.text:
                continue
            title = block.text.strip()
            if title and not rules.is_generic_doc_title(title):
                return title

        return hint or "Untitled"

    @staticmethod
    def _cover_title_from_blocks(
        blocks: list[Block],
        *,
        rules: DocumentProcessingRules,
    ) -> str | None:
        """Junta líneas de título en la página de portada; ignora boilerplate y prosa larga."""
        parts: list[str] = []
        ordered = sorted(blocks, key=lambda item: item.order)
        cover_page = rules.cover_page_number
        max_paragraph = rules.cover_title_max_paragraph_len
        max_joined = rules.cover_title_joined_max_len

        for block in ordered:
            if block.page is not None and block.page != cover_page:
                if parts:
                    break
                continue
            if block.page != cover_page:
                continue
            if block.type not in {BlockType.HEADING, BlockType.PARAGRAPH} or not block.text:
                continue

            text = block.text.strip()
            if not text or rules.is_title_boilerplate(text):
                continue
            if parts and len(text) > max_paragraph:
                break

            parts.append(text)
            joined_len = sum(len(part) for part in parts) + max(0, len(parts) - 1)
            if joined_len >= max_joined:
                break

        if not parts:
            return None
        return " ".join(parts)
