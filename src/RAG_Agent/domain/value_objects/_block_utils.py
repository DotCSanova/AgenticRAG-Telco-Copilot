from __future__ import annotations

from dataclasses import replace

from RAG_Agent.domain.value_objects.block import Block


def renumber_blocks(blocks: list[Block]) -> list[Block]:
    """Re-numera bloques con ids secuenciales ``block_{index}`` y ``order`` correlativo."""
    return [replace(block, id=f"block_{index}", order=index) for index, block in enumerate(blocks)]
