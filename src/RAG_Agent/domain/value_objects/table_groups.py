from __future__ import annotations

import re
from dataclasses import replace

from RAG_Agent.domain.value_objects.block import Block, BlockType, TableData

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


def is_table_continuation(previous: TableData, current: TableData) -> bool:
    prev_cols = table_column_count(previous)
    curr_cols = table_column_count(current)
    if prev_cols == 0 or curr_cols == 0:
        return False

    # Cabecera real repetida en salto de página
    if (
        current.headers
        and not is_placeholder_headers(current.headers)
        and previous.headers
        and current.headers == previous.headers
    ):
        return True

    # Fragmento típico Docling: headers 0,1,2… (el nº de columnas puede drift)
    if is_placeholder_headers(current.headers):
        return True

    return False


def link_table_continuations(blocks: list[Block]) -> list[Block]:
    """Marca grupos de tablas partidas por página (table_group_id, part_index, …)."""
    result = list(blocks)
    group_counter = 0
    index = 0

    while index < len(result):
        block = result[index]
        if block.type != BlockType.TABLE or block.table is None:
            index += 1
            continue

        table_indexes = [index]
        cursor = index + 1
        while cursor < len(result):
            candidate = result[cursor]
            if candidate.type == BlockType.TABLE and candidate.table is not None:
                previous_table = result[table_indexes[-1]].table
                assert previous_table is not None
                if is_table_continuation(previous_table, candidate.table):
                    table_indexes.append(cursor)
                    cursor += 1
                    continue
                break
            if candidate.type == BlockType.PARAGRAPH and looks_like_table_caption(
                candidate.text or ""
            ):
                cursor += 1
                continue
            break

        if len(table_indexes) < 2:
            index += 1
            continue

        group_id = f"table_group_{group_counter}"
        group_counter += 1
        first = result[table_indexes[0]]
        assert first.table is not None

        caption = first.table.caption
        after_first = table_indexes[0] + 1
        if after_first < len(result):
            maybe_caption = result[after_first]
            if maybe_caption.type == BlockType.PARAGRAPH and looks_like_table_caption(
                maybe_caption.text or ""
            ):
                caption = maybe_caption.text

        for part_index, block_index in enumerate(table_indexes):
            old = result[block_index]
            assert old.table is not None
            metadata = dict(old.metadata)
            metadata["table_group_id"] = group_id
            metadata["table_part_index"] = str(part_index)
            metadata["table_part_count"] = str(len(table_indexes))
            if part_index > 0:
                metadata["continues_from"] = first.id

            table = old.table
            if part_index == 0 and caption and not table.caption:
                table = TableData(headers=table.headers, rows=table.rows, caption=caption)
            elif part_index > 0 and is_placeholder_headers(table.headers):
                table = TableData(headers=[], rows=table.rows, caption=None)

            result[block_index] = replace(old, table=table, metadata=metadata)

        index = table_indexes[-1] + 1

    return result


def _normalize_row(row: list[str], width: int) -> list[str]:
    cells = [str(cell) for cell in row]
    if len(cells) < width:
        cells = cells + [""] * (width - len(cells))
    return cells[:width]


def merge_table_group(parts: list[Block]) -> TableData:
    """Fusiona partes de un mismo grupo en una sola TableData (headers de la 1ª parte)."""
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

    return TableData(headers=headers, rows=rows, caption=caption)
