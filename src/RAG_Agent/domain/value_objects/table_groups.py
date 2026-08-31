from __future__ import annotations

import re
from dataclasses import replace

from RAG_Agent.domain.value_objects.block import Block, BlockType, BoundingBox, LayoutSpan, TableData

_TABLE_CAPTION_RE = re.compile(r"^table\s+[\d.]+-?\d*", re.IGNORECASE)


def is_placeholder_headers(headers: list[str]) -> bool:
    """True si los headers parecen índices artificiales (0,1,2… / col0…)."""
    if not headers:
        return True
    placeholder_count = 0
    for index, header in enumerate(headers):
        text = str(header).strip()
        if not text:
            placeholder_count += 1
            continue
        if text.isdigit() and int(text) == index:
            placeholder_count += 1
            continue
        lowered = text.lower()
        if lowered.startswith("col") and lowered[3:].isdigit():
            placeholder_count += 1
            continue
    return placeholder_count == len(headers)


def table_column_count(table: TableData) -> int:
    if table.headers and not is_placeholder_headers(table.headers):
        return len(table.headers)
    if table.rows:
        return max(len(row) for row in table.rows)
    return len(table.headers)


def looks_like_table_caption(text: str) -> bool:
    return bool(_TABLE_CAPTION_RE.match(text.strip()))


def _table_caption_text(block: Block) -> str | None:
    if block.type not in {BlockType.PARAGRAPH, BlockType.HEADING}:
        return None
    text = (block.text or "").strip()
    if text and looks_like_table_caption(text):
        return text
    return None


def _bboxes_overlap_x(left: BoundingBox | None, right: BoundingBox | None) -> bool:
    if left is None or right is None:
        return True
    return left.x0 < right.x1 and right.x0 < left.x1


def is_table_continuation(previous: Block, current: Block) -> bool:
    """True if ``current`` looks like the next fragment of ``previous``.

    Requires similar column counts and either the next PDF page (typical
    Docling split) or matching real headers. Placeholder headers (``0,1,2``)
    alone are not enough unless the fragments are on consecutive pages.
    """
    prev_table = previous.table
    curr_table = current.table
    if prev_table is None or curr_table is None:
        return False

    prev_cols = table_column_count(prev_table)
    curr_cols = table_column_count(curr_table)
    if prev_cols == 0 or curr_cols == 0:
        return False
    if abs(prev_cols - curr_cols) > 1:
        return False

    next_page = (
        previous.page is not None
        and current.page is not None
        and current.page == previous.page + 1
    )
    overlap_x = _bboxes_overlap_x(previous.bbox, current.bbox)
    matching_headers = bool(
        curr_table.headers
        and prev_table.headers
        and not is_placeholder_headers(curr_table.headers)
        and curr_table.headers == prev_table.headers
    )
    placeholder = is_placeholder_headers(curr_table.headers)

    if matching_headers:
        return next_page or overlap_x
    if placeholder:
        return next_page and overlap_x
    return False


def _continuation_table_indexes(blocks: list[Block], start: int) -> list[int]:
    """Índices TABLE de un run partido por página, saltando captions entre fragmentos."""
    first = blocks[start]
    if first.type != BlockType.TABLE or first.table is None:
        return []

    indexes = [start]
    cursor = start + 1
    while cursor < len(blocks):
        candidate = blocks[cursor]
        if candidate.type == BlockType.TABLE and candidate.table is not None:
            if is_table_continuation(blocks[indexes[-1]], candidate):
                indexes.append(cursor)
                cursor += 1
                continue
            break
        if _table_caption_text(candidate) is not None:
            cursor += 1
            continue
        break
    return indexes


def _is_continuation_fragment(blocks: list[Block], index: int) -> bool:
    block = blocks[index]
    if block.type != BlockType.TABLE or block.table is None:
        return False
    cursor = index - 1
    while cursor >= 0 and _table_caption_text(blocks[cursor]) is not None:
        cursor -= 1
    if cursor < 0:
        return False
    previous = blocks[cursor]
    if previous.type != BlockType.TABLE or previous.table is None:
        return False
    return is_table_continuation(previous, block)


def attach_table_captions(blocks: list[Block]) -> list[Block]:
    """Mueve bloques adyacentes ``Table …`` a ``table.caption`` y los elimina.

    Misma idea que las captions de figura: el rótulo no es prosa del cuerpo.
    Prioridad (un solo caption por tabla lógica / primer fragmento):
    1. Inmediatamente después de la primera parte (típico entre fragmentos).
    2. Inmediatamente después de la última parte (caption bajo la tabla).
    3. Inmediatamente antes de la primera parte (caption sobre la tabla).
    """
    if not blocks:
        return blocks

    caption_by_table: dict[int, str] = {}
    drop: set[int] = set()

    def take_caption(table_index: int, caption_index: int) -> None:
        if caption_index in drop or table_index in caption_by_table:
            return
        text = _table_caption_text(blocks[caption_index])
        if not text:
            return
        caption_by_table[table_index] = text
        drop.add(caption_index)

    for index, block in enumerate(blocks):
        if block.type != BlockType.TABLE or block.table is None:
            continue
        if _is_continuation_fragment(blocks, index):
            continue

        after_first = index + 1
        if after_first < len(blocks) and _table_caption_text(blocks[after_first]):
            take_caption(index, after_first)
            continue

        after_last = _continuation_table_indexes(blocks, index)[-1] + 1
        if after_last < len(blocks) and _table_caption_text(blocks[after_last]):
            take_caption(index, after_last)
            continue

        before = index - 1
        if before >= 0 and _table_caption_text(blocks[before]):
            take_caption(index, before)

    result: list[Block] = []
    for index, block in enumerate(blocks):
        if index in drop:
            continue
        caption = caption_by_table.get(index)
        if caption and block.table is not None and not block.table.caption:
            table = TableData(
                headers=block.table.headers,
                rows=block.table.rows,
                caption=caption,
            )
            result.append(replace(block, table=table))
        else:
            result.append(block)
    return result


def _normalize_row(row: list[str], width: int) -> list[str]:
    cells = [str(cell) for cell in row]
    if len(cells) < width:
        cells = cells + [""] * (width - len(cells))
    return cells[:width]


def _forward_fill_leading_cells(rows: list[list[str]]) -> list[list[str]]:
    """Rellena celdas vacías a la izquierda (rowspan de Docling en saltos de página)."""
    filled: list[list[str]] = []
    previous: list[str] | None = None
    for row in rows:
        current = list(row)
        if previous is not None:
            for index, cell in enumerate(current):
                if cell.strip():
                    break
                if index < len(previous):
                    current[index] = previous[index]
        filled.append(current)
        previous = current
    return filled


def _merge_table_parts(parts: list[Block]) -> TableData:
    """Fusiona partes en una sola TableData (headers de la 1ª parte + rowspan)."""
    table_parts = [block for block in parts if block.type == BlockType.TABLE and block.table]
    if not table_parts:
        return TableData()

    first = table_parts[0].table
    assert first is not None
    headers = list(first.headers)
    width = (
        len(headers)
        if headers and not is_placeholder_headers(headers)
        else table_column_count(first)
    )
    if not headers or is_placeholder_headers(headers):
        headers = [f"col{i}" for i in range(width)]
    rows: list[list[str]] = [_normalize_row(row, width) for row in first.rows]
    caption = first.caption

    for block in table_parts[1:]:
        table = block.table
        assert table is not None
        if not caption and table.caption:
            caption = table.caption
        part_rows = table.rows
        if (
            table.headers
            and not is_placeholder_headers(table.headers)
            and table.headers == headers
            and part_rows
            and [str(cell) for cell in part_rows[0]] == [str(cell) for cell in headers]
        ):
            part_rows = part_rows[1:]
        rows.extend(_normalize_row(row, width) for row in part_rows)

    return TableData(
        headers=headers,
        rows=_forward_fill_leading_cells(rows),
        caption=caption,
    )


def _collapse_parts(parts: list[Block]) -> Block:
    first = parts[0]
    pages = [block.page for block in parts if block.page is not None]
    page_start = pages[0] if pages else first.page
    page_end = pages[-1] if pages else first.page

    metadata = dict(first.metadata)
    metadata["merged_parts"] = str(len(parts))
    if page_end is not None and page_start is not None and page_end != page_start:
        metadata["page_end"] = str(page_end)
        metadata["continued"] = "true"
    elif page_end is not None:
        metadata.setdefault("page_end", str(page_end))

    return replace(
        first,
        page=page_start,
        table=_merge_table_parts(parts),
        bbox=first.bbox,
        source_ref=first.source_ref,
        metadata=metadata,
        layout_spans=tuple(
            LayoutSpan(page=part.page, bbox=part.bbox, source_ref=part.source_ref)
            for part in parts
        ),
    )


def collapse_continued_tables(blocks: list[Block]) -> list[Block]:
    """Sustituye fragmentos de una tabla partida por un único block TABLE.

    ``page`` / ``bbox`` / ``source_ref`` quedan los del primer fragmento (cita).
    ``layout_spans`` guarda un rectángulo por página para overlay PDF completo.
    """
    if not blocks:
        return blocks

    merged_at: dict[int, Block] = {}
    drop: set[int] = set()
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if block.type != BlockType.TABLE or block.table is None:
            index += 1
            continue

        part_indexes = _continuation_table_indexes(blocks, index)
        if len(part_indexes) < 2:
            index += 1
            continue

        merged_at[part_indexes[0]] = _collapse_parts([blocks[i] for i in part_indexes])
        drop.update(part_indexes[1:])
        index = part_indexes[-1] + 1

    return [
        merged_at[index] if index in merged_at else block
        for index, block in enumerate(blocks)
        if index not in drop
    ]


def refine_table_blocks(blocks: list[Block]) -> list[Block]:
    """Captions de tabla + fusión de continuaciones en un block lógico."""
    return collapse_continued_tables(attach_table_captions(blocks))
