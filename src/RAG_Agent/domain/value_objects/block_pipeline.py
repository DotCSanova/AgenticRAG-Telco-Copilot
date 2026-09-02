from __future__ import annotations

from RAG_Agent.domain.doc_processing_rules.document_processing import (
    DocumentProcessingRules,
)
from RAG_Agent.domain.value_objects._block_utils import renumber_blocks
from RAG_Agent.domain.value_objects.block import Block, BlockType
from RAG_Agent.domain.value_objects.figure_groups import attach_figure_captions
from RAG_Agent.domain.value_objects.layout_dedup import drop_overlapping_images_on_tables
from RAG_Agent.domain.value_objects.table_groups import refine_table_blocks


def drop_removable_sections(
    blocks: list[Block],
    *,
    rules: DocumentProcessingRules,
) -> list[Block]:
    """Drop a removable heading and the blocks that follow it until a peer heading.

    A heading matches when ``rules.is_removable_section`` is true. Following
    blocks are discarded until a heading whose ``level`` is less than or equal
    to that heading's level (that heading is kept).

    Args:
        blocks: Blocks in reading order, after family ``refine_blocks``.
        rules: Family profile (front-matter titles).

    Returns:
        New list without the removable section bodies.
    """
    result: list[Block] = []
    skip_until_level: int | None = None

    for block in blocks:
        if block.type == BlockType.HEADING:
            title = (block.text or "").strip()
            level = block.level or 1
            if rules.is_removable_section(title):
                skip_until_level = level
                continue
            if skip_until_level is not None:
                if level <= skip_until_level:
                    skip_until_level = None
                else:
                    continue
            result.append(block)
            continue

        if skip_until_level is not None:
            continue
        result.append(block)

    return result


def drop_cover_page_boilerplate(
    blocks: list[Block],
    *,
    rules: DocumentProcessingRules,
) -> list[Block]:
    """Remove cover-page blocks whose text is title boilerplate.

    Args:
        blocks: Blocks in reading order.
        rules: Family profile (cover page number and boilerplate predicate).

    Returns:
        New list without cover-page VAT, copyright, or register lines.
    """
    cover = rules.cover_page_number
    return [
        block
        for block in blocks
        if not (
            block.page == cover
            and block.text
            and rules.is_title_boilerplate(block.text)
        )
    ]


def refine_block_sequence(
    blocks: list[Block],
    *,
    rules: DocumentProcessingRules,
) -> list[Block]:
    """Apply family and structural refiners to a mapped block sequence.

    Order: ``refine_blocks`` → drop removable sections → drop cover
    boilerplate → overlapping picture/table → family diagrams → figure
    captions → table captions/merge → renumber once.

    Args:
        blocks: Blocks already mapped from the parser, in reading order.
        rules: Family profile (headings, front-matter, extra refiners).

    Returns:
        New list, re-numbered (``block_{i}`` / ``order`` sequential).
    """
    refined = rules.refine_blocks(blocks)
    refined = drop_removable_sections(refined, rules=rules)
    refined = drop_cover_page_boilerplate(refined, rules=rules)
    refined = drop_overlapping_images_on_tables(refined)
    refined = rules.merge_diagram_fragments(refined)
    refined = attach_figure_captions(refined)
    refined = refine_table_blocks(refined)
    return renumber_blocks(refined)
