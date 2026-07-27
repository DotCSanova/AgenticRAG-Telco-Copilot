from __future__ import annotations

from collections.abc import Iterable

from RAG_Agent.domain.value_objects.block import Block, BlockType, TableData


def render_blocks(blocks: Iterable[Block]) -> tuple[str, tuple[str, ...]]:
    parts: list[str] = []
    used_ids: list[str] = []
    for block in blocks:
        piece = block_text(block)
        if not piece:
            continue
        parts.append(piece)
        used_ids.append(block.id)
    return "\n\n".join(parts), tuple(used_ids)


def block_text(block: Block) -> str:
    if block.type == BlockType.TABLE:
        return table_text(block.table)
    if block.type == BlockType.IMAGE:
        if block.image is None:
            return ""
        caption = (block.image.caption or "").strip()
        if caption:
            return caption
        return (block.image.alt or "").strip()
    return (block.text or "").strip()


def table_text(table: TableData | None) -> str:
    if table is None:
        return ""
    lines: list[str] = []
    if table.caption:
        lines.append(table.caption.strip())
    if table.headers:
        lines.append(" | ".join(header.strip() for header in table.headers))
    for row in table.rows:
        lines.append(" | ".join(cell.strip() for cell in row))
    return "\n".join(line for line in lines if line)
