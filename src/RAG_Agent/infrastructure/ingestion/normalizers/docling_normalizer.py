from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from docling_core.types.doc import (
    CodeItem,
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
    TableData,
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
        blocks: list[Block] = []
        order = 0

        for element, _depth in doc.iterate_items():
            if getattr(element, "content_layer", None) == ContentLayer.FURNITURE:
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

        if isinstance(element, (CodeItem,)):
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
            page, bbox = self._prov_page_bbox(element)
            return Block(
                id=f"block_{order}",
                type=BlockType.TABLE,
                order=order,
                page=page,
                table=self._table_to_data(element, doc),
                bbox=bbox,
                source_ref=getattr(element, "self_ref", None),
            )

        if isinstance(element, PictureItem):
            page, bbox = self._prov_page_bbox(element)
            alt = None
            captions = getattr(element, "captions", None) or []
            if captions:
                # captions are often RefItems; best-effort text
                alt = str(captions[0]) if captions else None
            return Block(
                id=f"block_{order}",
                type=BlockType.IMAGE,
                order=order,
                page=page,
                image=ImageRef(uri=None, alt=alt),
                bbox=bbox,
                source_ref=getattr(element, "self_ref", None),
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
        page, bbox = self._prov_page_bbox(element)
        return Block(
            id=f"block_{order}",
            type=block_type,
            order=order,
            page=page,
            text=text,
            level=level,
            bbox=bbox,
            source_ref=getattr(element, "self_ref", None),
        )

    @staticmethod
    def _prov_page_bbox(element: Any) -> tuple[int | None, BoundingBox | None]:
        provs = getattr(element, "prov", None) or []
        if not provs:
            return None, None
        prov = provs[0]
        page = getattr(prov, "page_no", None)
        raw_bbox = getattr(prov, "bbox", None)
        bbox = None
        if raw_bbox is not None:
            bbox = BoundingBox(
                x0=float(getattr(raw_bbox, "l", getattr(raw_bbox, "x0", 0.0))),
                y0=float(getattr(raw_bbox, "b", getattr(raw_bbox, "y0", 0.0))),
                x1=float(getattr(raw_bbox, "r", getattr(raw_bbox, "x1", 0.0))),
                y1=float(getattr(raw_bbox, "t", getattr(raw_bbox, "y1", 0.0))),
                coord_origin=str(getattr(raw_bbox, "coord_origin", "BOTTOMLEFT")),
            )
        return page, bbox

    @staticmethod
    def _table_to_data(table: TableItem, doc: DoclingDocument) -> TableData:
        try:
            dataframe = table.export_to_dataframe(doc=doc)
            headers = [str(column) for column in dataframe.columns.tolist()]
            rows = [[str(cell) for cell in row] for row in dataframe.to_numpy().tolist()]
            return TableData(headers=headers, rows=rows)
        except Exception:
            logger.debug("export_to_dataframe failed for %s; using grid fallback", table.self_ref)

        data = getattr(table, "data", None)
        grid = getattr(data, "grid", None) if data is not None else None
        if not grid:
            return TableData()

        rows: list[list[str]] = []
        for row in grid:
            if isinstance(row, list):
                rows.append([str(getattr(cell, "text", cell) or "") for cell in row])
            else:
                # flat cell list variant in some exports
                rows.append([str(getattr(row, "text", row) or "")])

        headers = rows[0] if rows else []
        body = rows[1:] if len(rows) > 1 else []
        return TableData(headers=headers, rows=body)

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
