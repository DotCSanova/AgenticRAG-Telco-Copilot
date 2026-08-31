from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

    def to_payload(self) -> dict[str, Any]:
        """Serialize this document to a schema 1.0 JSON-ready dict.

        Returns:
            Payload with ``schema_version``, identity/stats split from ``extra``,
            and blocks as a list ordered by ``order``. Irrelevant nulls omitted.
        """
        from RAG_Agent.domain.value_objects.canonical_codec import canonical_to_payload

        return canonical_to_payload(self)

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> CanonicalDocument:
        """Rebuild a document from a schema 1.0 payload.

        Args:
            data: Mapping produced by :meth:`to_payload`. Unknown keys are ignored.

        Returns:
            Canonical document with identity/stats written back into ``extra``.

        Raises:
            ValueError: If ``schema_version`` is missing or not ``1.0``.
        """
        from RAG_Agent.domain.value_objects.canonical_codec import canonical_from_payload

        return canonical_from_payload(data)
