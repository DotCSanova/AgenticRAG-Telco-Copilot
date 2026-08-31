from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from RAG_Agent.domain.value_objects.block import Block
from RAG_Agent.domain.value_objects.page import Page
from RAG_Agent.domain.value_objects.section import Section


@dataclass(frozen=True)
class DocumentMetadata:
    source_path: Path
    title: str | None = None
    profile_id: str | None = None
    parser: str | None = None
    extra: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalDocument:
    """Documento canónico normalizado: blocks + pages + sections por referencias."""

    metadata: DocumentMetadata
    blocks: dict[str, Block] = field(default_factory=dict)
    pages: list[Page] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)

    def get_block(self, block_id: str) -> Block:
        try:
            return self.blocks[block_id]
        except KeyError as exc:
            msg = f"Block not found: {block_id}"
            raise KeyError(msg) from exc

    def blocks_for_page(self, page_number: int) -> list[Block]:
        page = next((item for item in self.pages if item.number == page_number), None)
        if page is None:
            return []
        return [self.get_block(block_id) for block_id in page.block_ids]

    def blocks_for_section(self, section_id: str) -> list[Block]:
        section = next((item for item in self.sections if item.id == section_id), None)
        if section is None:
            return []
        return [self.get_block(block_id) for block_id in section.block_ids]

    def root_sections(self) -> list[Section]:
        return [section for section in self.sections if section.parent_id is None]

    def child_sections(self, parent_id: str) -> list[Section]:
        return [section for section in self.sections if section.parent_id == parent_id]
