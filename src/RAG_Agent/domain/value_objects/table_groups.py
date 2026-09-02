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


def _last_page(block: Block) -> int | None:
    pages = block.page_numbers()
    return pages[-1] if pages else block.page


def _is_next_page(previous: Block, current: Block) -> bool:
    """True if ``current`` starts on the folio after ``previous`` ends."""
    end = _last_page(previous)
    return end is not None and current.page is not None and current.page == end + 1


def _is_new_table_caption(caption: str | None) -> bool:
    """True for ``Table N.N-N`` titles, not footnotes such as ``NOTE 1:``."""
    return bool(caption and looks_like_table_caption(caption))


def is_table_continuation(previous: Block, current: Block) -> bool:
    """True if ``current`` looks like the next fragment of ``previous``.

    A fragment labelled ``Table …`` is a new table. ``NOTE`` footnotes are not.
    Identical real headers still join on the next page or overlapping x-range.
    Otherwise the next folio after the previous block ends, x-overlap, and
    ``current`` no wider than ``previous`` (same grid, header drift, or
    dropped leading columns).
    """
    prev_table = previous.table
    curr_table = current.table
    if prev_table is None or curr_table is None:
        return False
    if _is_new_table_caption(curr_table.caption):
        return False

    prev_cols = table_column_count(prev_table)
    curr_cols = table_column_count(curr_table)
    if prev_cols == 0 or curr_cols == 0:
        return False

    overlap_x = _bboxes_overlap_x(previous.bbox, current.bbox)
    next_page = _is_next_page(previous, current)
    placeholder = is_placeholder_headers(curr_table.headers)
    matching_headers = bool(
        curr_table.headers
        and prev_table.headers
        and not placeholder
        and curr_table.headers == prev_table.headers
    )
    similar_cols = abs(prev_cols - curr_cols) <= 1

    if matching_headers:
        return similar_cols and (next_page or overlap_x)
    if not (next_page and overlap_x):
        return False
    return curr_cols <= prev_cols


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
            owner = blocks[indexes[-1]].table
            if owner is not None and _is_new_table_caption(owner.caption):
                break
            cursor += 1
            continue
        break
    return indexes


def _is_continuation_fragment(blocks: list[Block], index: int) -> bool:
    block = blocks[index]
    if block.type != BlockType.TABLE or block.table is None:
        return False
    cursor = index - 1
    skipped_caption = False
    while cursor >= 0 and _table_caption_text(blocks[cursor]) is not None:
        skipped_caption = True
        cursor -= 1
    if cursor < 0:
        return False
    previous = blocks[cursor]
    if previous.type != BlockType.TABLE or previous.table is None:
        return False
    if skipped_caption and _is_new_table_caption(previous.table.caption):
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
        owner = blocks[table_index].table
        if owner is not None and owner.caption:
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


def _normalize_row(row: list[str], width: int, *, pad_left: bool = False) -> list[str]:
    cells = [str(cell) for cell in row]
    if len(cells) >= width:
        return cells[:width]
    pad = [""] * (width - len(cells))
    return (pad + cells) if pad_left else (cells + pad)


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


_SENTENCE_END = frozenset(".!?;:")
_OPEN_CELL_MIN_CHARS = 20
_STITCH_MAX_ROWS = 2


def _is_open_cell(text: str) -> bool:
    """True if the cell has running text that may continue after a page break.

    A page can cut on any word. Short tokens (IDs, ticks) are skipped so they
    are not treated as wrap targets. A cell that already ends with sentence
    punctuation is treated as complete.
    """
    stripped = text.strip()
    return len(stripped) >= _OPEN_CELL_MIN_CHARS and stripped[-1] not in _SENTENCE_END


def _stitch_targets(last: list[str], n_pieces: int, width: int) -> list[int]:
    """Columns that should receive leftover cells, left to right.

    Open cells are candidates. If there are more candidates than pieces
    (row label plus wrapping prose), keep the rightmost ones: overflow is
    usually in text columns, not the stub on the left.
    """
    if n_pieces <= 0:
        return []
    open_cols = [index for index, cell in enumerate(last) if _is_open_cell(cell)]
    if len(open_cols) < n_pieces:
        return list(range(width - n_pieces, width))
    return open_cols[-n_pieces:]


def _is_header_drift(previous: list[str], current: list[str]) -> bool:
    """True if ``current`` is the same header row with labels shortened or glued."""
    if len(previous) != len(current) or not previous:
        return False
    matched = 0
    for left, right in zip(previous, current, strict=True):
        a, b = left.strip().lower(), right.strip().lower()
        if not a or not b:
            continue
        if a == b or a.startswith(b) or b.startswith(a):
            matched += 1
            continue
        return False
    return matched >= max(1, len(previous) // 2)


def _fragment_data_rows(table: TableData, parent_headers: list[str]) -> list[list[str]]:
    rows = [list(row) for row in table.rows]
    headers = table.headers
    if not headers or is_placeholder_headers(headers):
        return rows
    if headers == parent_headers:
        if rows and [str(cell) for cell in rows[0]] == [str(cell) for cell in headers]:
            return rows[1:]
        return rows
    if _is_header_drift(parent_headers, headers):
        return rows
    return [list(headers), *rows]


def _should_stitch_last_row(last: list[str], incoming: list[list[str]], width: int) -> bool:
    if not last or not incoming or len(incoming) > _STITCH_MAX_ROWS:
        return False
    filled = sum(1 for cell in incoming[0] if cell.strip())
    return 0 < filled < width and any(_is_open_cell(cell) for cell in last)


def _stitch_row(last: list[str], extra: list[str], width: int) -> None:
    pieces = [cell.strip() for cell in extra if cell.strip()]
    if not pieces:
        return
    for index, piece in zip(_stitch_targets(last, len(pieces), width), pieces):
        if 0 <= index < width:
            last[index] = f"{last[index].rstrip()} {piece}".strip()


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
        elif table.caption and not _is_new_table_caption(table.caption):
            if table.caption not in (caption or ""):
                caption = f"{caption} {table.caption.strip()}".strip() if caption else table.caption
        fragment_rows = _fragment_data_rows(table, headers)
        pad_left = any(len(row) < width for row in fragment_rows)
        incoming = [
            _normalize_row(row, width, pad_left=pad_left) for row in fragment_rows
        ]
        if table.headers and _is_header_drift(headers, table.headers):
            headers = list(table.headers)
        if rows and incoming and _should_stitch_last_row(rows[-1], incoming, width):
            _stitch_row(rows[-1], incoming[0], width)
            rows.extend(incoming[1:])
        else:
            rows.extend(incoming)

    return TableData(
        headers=headers,
        rows=_forward_fill_leading_cells(rows),
        caption=caption,
    )


def _layout_spans_from_parts(parts: list[Block]) -> tuple[LayoutSpan, ...]:
    collected: list[LayoutSpan] = []
    for part in parts:
        if part.layout_spans:
            collected.extend(part.layout_spans)
        else:
            collected.append(
                LayoutSpan(page=part.page, bbox=part.bbox, source_ref=part.source_ref)
            )
    return tuple(collected)


def _merged_part_count(parts: list[Block]) -> int:
    total = 0
    for part in parts:
        raw = part.metadata.get("merged_parts")
        total += int(raw) if raw and raw.isdigit() else 1
    return total


def _collapse_parts(parts: list[Block]) -> Block:
    first = parts[0]
    spans = _layout_spans_from_parts(parts)
    pages = [span.page for span in spans if span.page is not None]
    if not pages:
        pages = [block.page for block in parts if block.page is not None]
    page_start = pages[0] if pages else first.page
    page_end = pages[-1] if pages else first.page

    metadata = dict(first.metadata)
    metadata["merged_parts"] = str(_merged_part_count(parts))
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
        layout_spans=spans,
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
