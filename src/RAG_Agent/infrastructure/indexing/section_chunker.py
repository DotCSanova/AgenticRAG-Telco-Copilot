from __future__ import annotations

from RAG_Agent.domain.ports.chunker import Chunker
from RAG_Agent.domain.value_objects.block import BlockType
from RAG_Agent.domain.value_objects.block_render import BlockTextFormat, render_blocks
from RAG_Agent.domain.value_objects.canonical_document import CanonicalDocument
from RAG_Agent.domain.value_objects.chunk import Chunk
from RAG_Agent.domain.value_objects.section import Section


class SectionChunker(Chunker):
    """Chunk por sección con texto propio.

    Las hojas siempre se emiten. Un padre solo-heading (estructural) se omite;
    un padre con preámbulo (tablas/párrafos antes del primer hijo) sí se indexa.
    """

    def chunk(self, document: CanonicalDocument) -> list[Chunk]:
        doc_id = _doc_id(document)
        sections = sorted(document.sections, key=lambda section: section.order)
        if not sections:
            return _chunks_from_orphan_blocks(document, doc_id)

        by_id = {section.id: section for section in sections}
        parent_ids_with_children = {
            section.parent_id for section in sections if section.parent_id is not None
        }

        chunks: list[Chunk] = []
        index = 0
        for section in sections:
            if _is_structural_parent(section, document, parent_ids_with_children):
                continue
            blocks = document.blocks_for_section(section.id)
            text, block_ids = render_blocks(blocks, fmt=BlockTextFormat.MARKDOWN)
            if not text:
                continue
            path = _section_path(section, by_id)
            chunks.append(
                Chunk(
                    id=f"{doc_id}:{section.id}",
                    doc_id=doc_id,
                    text=text,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    section_id=section.id,
                    block_ids=block_ids,
                    metadata={
                        "section_title": section.title,
                        "section_level": str(section.level),
                        "section_path": path,
                        "chunk_index": str(index),
                        "text_format": "markdown",
                    },
                )
            )
            index += 1
        return chunks


def _is_structural_parent(
    section: Section,
    document: CanonicalDocument,
    parent_ids_with_children: set[str | None],
) -> bool:
    """True si la sección tiene hijos y no aporta cuerpo más allá del heading."""
    if section.id not in parent_ids_with_children:
        return False
    body = [
        block
        for block in document.blocks_for_section(section.id)
        if block.type != BlockType.HEADING
    ]
    text, _ = render_blocks(body, fmt=BlockTextFormat.MARKDOWN)
    return not text


def _doc_id(document: CanonicalDocument) -> str:
    return document.metadata.source_path.stem


def _section_path(section: Section, by_id: dict[str, Section], *, sep: str = " > ") -> str:
    titles: list[str] = []
    current: Section | None = section
    seen: set[str] = set()
    while current is not None and current.id not in seen:
        seen.add(current.id)
        title = current.title.strip()
        if title:
            titles.append(title)
        parent_id = current.parent_id
        current = by_id.get(parent_id) if parent_id is not None else None
    titles.reverse()
    return sep.join(titles)


def _chunks_from_orphan_blocks(document: CanonicalDocument, doc_id: str) -> list[Chunk]:
    blocks = sorted(document.blocks.values(), key=lambda item: item.order)
    text, block_ids = render_blocks(blocks, fmt=BlockTextFormat.MARKDOWN)
    if not text:
        return []
    pages = [block.page for block in blocks if block.page is not None]
    return [
        Chunk(
            id=f"{doc_id}:orphan",
            doc_id=doc_id,
            text=text,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            block_ids=block_ids,
            metadata={"chunk_index": "0", "section_path": "", "text_format": "markdown"},
        )
    ]
