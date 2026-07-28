from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

from RAG_Agent.domain.value_objects.block import Block, BlockType, TableData


class BlockTextFormat(str, Enum):
    PLAIN = "plain"
    MARKDOWN = "markdown"


def plain_text(block: Block) -> str:
    """Texto plano indexable de un block (sin markup)."""
    if block.type == BlockType.TABLE:
        return _table_plain(block.table)
    if block.type == BlockType.IMAGE:
        return _image_plain(block)
    return (block.text or "").strip()


def markdown_text(block: Block) -> str:
    """Representación Markdown de un block para chunking / export."""
    if block.type == BlockType.HEADING:
        title = (block.text or "").strip()
        if not title:
            return ""
        level = max(1, min(block.level or 1, 6))
        return f"{'#' * level} {title}"

    if block.type == BlockType.LIST_ITEM:
        text = (block.text or "").strip()
        if not text:
            return ""
        depth = max(1, block.level or 1)
        indent = "  " * (depth - 1)
        return f"{indent}- {text}"

    if block.type == BlockType.TABLE:
        return _table_markdown(block.table)

    if block.type == BlockType.IMAGE:
        return _image_markdown(block)

    if block.type == BlockType.CODE:
        text = (block.text or "").strip()
        if not text:
            return ""
        language = (block.metadata.get("language") or "").strip()
        fence = f"```{language}" if language else "```"
        return f"{fence}\n{text}\n```"

    if block.type == BlockType.FORMULA:
        text = (block.text or "").strip()
        return f"$$\n{text}\n$$" if text else ""

    return (block.text or "").strip()


def block_text(block: Block, *, fmt: BlockTextFormat = BlockTextFormat.MARKDOWN) -> str:
    if fmt is BlockTextFormat.PLAIN:
        return plain_text(block)
    return markdown_text(block)


def render_blocks(
    blocks: Iterable[Block],
    *,
    fmt: BlockTextFormat = BlockTextFormat.MARKDOWN,
) -> tuple[str, tuple[str, ...]]:
    """Concatena blocks en el formato indicado. Devuelve texto + ids usados."""
    parts: list[str] = []
    used_ids: list[str] = []
    for block in blocks:
        piece = block_text(block, fmt=fmt)
        if not piece:
            continue
        parts.append(piece)
        used_ids.append(block.id)
    return "\n\n".join(parts), tuple(used_ids)


def _image_plain(block: Block) -> str:
    if block.image is None:
        return ""
    caption = (block.image.caption or "").strip()
    if caption:
        return caption
    return (block.image.alt or "").strip()


def _image_markdown(block: Block) -> str:
    plain = _image_plain(block)
    if not plain:
        return ""
    uri = (block.image.uri if block.image is not None else None) or ""
    if uri:
        return f"![{plain}]({uri})"
    # Sin URI: el caption/alt es la representación textual de la figura.
    return plain


def _table_plain(table: TableData | None) -> str:
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


def _table_markdown(table: TableData | None) -> str:
    if table is None:
        return ""

    lines: list[str] = []
    if table.caption:
        lines.append(f"**{table.caption.strip()}**")

    headers = [header.strip() for header in table.headers]
    rows = [[cell.strip() for cell in row] for row in table.rows]

    if headers:
        width = len(headers)
        normalized_rows = [_pad_row(row, width) for row in rows]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in normalized_rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    if not rows:
        return "\n".join(lines)

    width = max(len(row) for row in rows)
    for row in rows:
        lines.append("| " + " | ".join(_pad_row(row, width)) + " |")
    return "\n".join(lines)


def _pad_row(row: list[str], width: int) -> list[str]:
    if len(row) >= width:
        return row[:width]
    return row + [""] * (width - len(row))
