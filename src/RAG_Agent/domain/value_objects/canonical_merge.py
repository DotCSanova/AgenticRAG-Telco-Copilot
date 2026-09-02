from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from RAG_Agent.domain.doc_processing_rules.document_processing import (
    DocumentProcessingRules,
)
from RAG_Agent.domain.value_objects._block_utils import index_block_ids_by_page
from RAG_Agent.domain.value_objects.block import Block
from RAG_Agent.domain.value_objects.block_pipeline import refine_block_sequence
from RAG_Agent.domain.value_objects.canonical_document import (
    CanonicalDocument,
    DocumentMetadata,
)
from RAG_Agent.domain.value_objects.page import Page
from RAG_Agent.domain.value_objects.section import Section


def _shift_block(block: Block, *, page_offset: int, order: int) -> Block:
    page = block.page + page_offset if block.page is not None else None
    metadata = dict(block.metadata)
    if page is not None and metadata.get("page_end"):
        try:
            metadata["page_end"] = str(int(metadata["page_end"]) + page_offset)
        except ValueError:
            pass
    spans = tuple(
        replace(
            span,
            page=span.page + page_offset if span.page is not None else None,
        )
        for span in block.layout_spans
    )
    return replace(
        block,
        id=f"block_{order}",
        order=order,
        page=page,
        metadata=metadata,
        layout_spans=spans,
    )


def _build_pages(
    blocks: list[Block],
    page_sizes: dict[int, tuple[float | None, float | None]],
) -> list[Page]:
    by_page = index_block_ids_by_page(blocks)

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
    rules: DocumentProcessingRules,
    build_sections: Callable[[list[Block]], list[Section]],
    resolve_title: Callable[[list[Section], list[Block], str], str],
    title_hint: str,
    extra: dict[str, str] | None = None,
) -> CanonicalDocument:
    """Merge per-shard canonical documents, then run the shared block pipeline.

    Page numbers are shifted by each shard's 0-based ``page_offset`` before
    ``refine_block_sequence`` (removable sections are dropped after concat).

    Args:
        shards: ``(page_offset, canonical)`` pairs in reading order.
        source_path: Original PDF path stored in metadata.
        profile_id: Family profile id stored in metadata.
        parser_name: Value stored in ``metadata.parser``.
        rules: Family rules used by ``refine_block_sequence``.
        build_sections: Builds sections from the refined block list.
        resolve_title: Resolves the document title.
        title_hint: Fallback title from the profile identity.
        extra: Extra metadata merged into the result.

    Returns:
        Single canonical document covering all shards.
    """
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

    blocks = refine_block_sequence(shifted, rules=rules)
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
