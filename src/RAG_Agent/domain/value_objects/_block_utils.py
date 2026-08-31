from __future__ import annotations

from dataclasses import replace

from RAG_Agent.domain.value_objects.block import Block


def renumber_blocks(blocks: list[Block]) -> list[Block]:
    """Re-numera bloques con ids secuenciales ``block_{index}`` y ``order`` correlativo."""
    return [replace(block, id=f"block_{index}", order=index) for index, block in enumerate(blocks)]


def index_block_ids_by_page(blocks: list[Block]) -> dict[int, list[str]]:
    """Un block que cruza páginas aparece en cada folio de ``page_numbers()``."""
    by_page: dict[int, list[str]] = {}
    for block in blocks:
        for number in block.page_numbers():
            ids = by_page.setdefault(number, [])
            if block.id not in ids:
                ids.append(block.id)
    return by_page
