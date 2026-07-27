from __future__ import annotations

from RAG_Agent.domain.ports.chunker import Chunker
from RAG_Agent.domain.value_objects.canonical_document import CanonicalDocument
from RAG_Agent.domain.value_objects.chunk import Chunk
from RAG_Agent.domain.value_objects.section import Section
from RAG_Agent.infrastructure.indexing.block_text import render_blocks


class SectionChunker(Chunker):
    """Chunk por secciones hoja del árbol; ``section_path`` en metadata."""

    def chunk(self, document: CanonicalDocument) -> list[Chunk]:
        doc_id = _doc_id(document)
        sections = sorted(document.sections, key=lambda section: section.order)
        if not sections:
            return _chunks_from_orphan_blocks(document, doc_id)

        by_id = {section.id: section for section in sections}
        parent_ids_with_children = {
            section.parent_id for section in sections if section.parent_id is not None
        }
        leaves = [section for section in sections if section.id not in parent_ids_with_children]

        chunks: list[Chunk] = []
        for index, section in enumerate(leaves):
            blocks = document.blocks_for_section(section.id)
            text, block_ids = render_blocks(blocks)
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
                    },
                )
            )
        return chunks


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
    blocks = sorted(document.blocks.values(), key=lambda block: block.order)
    text, block_ids = render_blocks(blocks)
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
            metadata={"chunk_index": "0", "section_path": ""},
        )
    ]
