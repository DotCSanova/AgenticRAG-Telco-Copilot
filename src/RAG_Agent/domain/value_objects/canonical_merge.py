from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from RAG_Agent.domain.value_objects._block_utils import renumber_blocks
from RAG_Agent.domain.value_objects.block import Block
from RAG_Agent.domain.value_objects.canonical_document import (
    CanonicalDocument,
    DocumentMetadata,
)
from RAG_Agent.domain.value_objects.page import Page
from RAG_Agent.domain.value_objects.plantuml_groups import merge_plantuml_fragments
from RAG_Agent.domain.value_objects.section import Section
from RAG_Agent.domain.value_objects.table_groups import link_table_continuations


def _shift_block(block: Block, *, page_offset: int, order: int) -> Block:
    page = block.page + page_offset if block.page is not None else None
    metadata = dict(block.metadata)
    if page is not None and metadata.get("page_end"):
        try:
            metadata["page_end"] = str(int(metadata["page_end"]) + page_offset)
        except ValueError:
            pass
    return replace(block, id=f"block_{order}", order=order, page=page, metadata=metadata)


def _build_pages(
    blocks: list[Block],
    page_sizes: dict[int, tuple[float | None, float | None]],
) -> list[Page]:
    by_page: dict[int, list[str]] = {}
    for block in blocks:
        if block.page is None:
            continue
        by_page.setdefault(block.page, []).append(block.id)

    pages: list[Page] = []
    for number in sorted(set(by_page) | set(page_sizes)):
        width = height = None
        if number in page_sizes:
            width, height = page_sizes[number]
        pages.append(
            Page(
                number=number,
                block_ids=by_page.get(number, []),
                width=width,
                height=height,
            )
        )
    return pages


def merge_canonical_shards(
    shards: Sequence[tuple[int, CanonicalDocument]],
    *,
    source_path: Path,
    profile_id: str | None,
    parser_name: str,
    build_sections: Callable[[list[Block]], list[Section]],
    resolve_title: Callable[[list[Section], list[Block], str], str],
    title_hint: str,
    extra: dict[str, str] | None = None,
) -> CanonicalDocument:
    """Fusiona canónicos de shards PDF. ``page_offset`` es el índice 0-based del primer folio."""
    if not shards:
        msg = "merge_canonical_shards requires at least one shard"
        raise ValueError(msg)

    shifted: list[Block] = []
    page_sizes: dict[int, tuple[float | None, float | None]] = {}
    order = 0

    for page_offset, document in shards:
        for block in sorted(document.blocks.values(), key=lambda item: item.order):
            shifted.append(_shift_block(block, page_offset=page_offset, order=order))
            order += 1
        for page in document.pages:
            page_sizes[page.number + page_offset] = (page.width, page.height)

    blocks = renumber_blocks(
        merge_plantuml_fragments(link_table_continuations(shifted))
    )
    pages = _build_pages(blocks, page_sizes)
    sections = build_sections(blocks)
    title = resolve_title(sections, blocks, title_hint)

    meta_extra = dict(extra or {})
    meta_extra["num_pages"] = str(len(pages))
    meta_extra["num_blocks"] = str(len(blocks))
    meta_extra["num_sections"] = str(len(sections))
    meta_extra["num_shards"] = str(len(shards))

    return CanonicalDocument(
        metadata=DocumentMetadata(
            source_path=source_path,
            title=title,
            profile_id=profile_id,
            parser=parser_name,
            extra=meta_extra,
        ),
        blocks={block.id: block for block in blocks},
        pages=pages,
        sections=sections,
    )
