from __future__ import annotations

import re
from dataclasses import replace

from RAG_Agent.domain.value_objects.block import Block, BlockType

# Figure 4.1-1 | Fig. 2 | Figura 1 | Figure A.1-1
_FIGURE_CAPTION_RE = re.compile(
    r"^(?:figure|fig\.|figura)\s+(?:[A-Za-z](?:\.\d+)+|\d)",
    re.IGNORECASE,
)


def looks_like_figure_caption(text: str) -> bool:
    first = text.strip().split("\n", 1)[0].strip()
    return bool(first and _FIGURE_CAPTION_RE.match(first))


def _figure_caption_text(block: Block) -> str | None:
    if block.type not in {BlockType.PARAGRAPH, BlockType.HEADING}:
        return None
    text = (block.text or "").strip()
    if text and looks_like_figure_caption(text):
        return text
    return None


def attach_figure_captions(blocks: list[Block]) -> list[Block]:
    """Mueve un rótulo ``Figure/Fig./Figura …`` adyacente a ``image.caption`` y lo elimina.

    1. Vecino exclusivo (el otro lado no es IMAGE).
    2. ``IMAGE, CAP, IMAGE`` → el caption es de la primera (debajo).
    3. Resto: caption sin usar inmediatamente después, si no antes.
    Un caption no se reutiliza. Texto que no parece rótulo no se adjunta.
    """
    if not blocks:
        return blocks

    n = len(blocks)
    assigned: dict[int, int] = {}
    used: set[int] = set()

    def is_image(index: int) -> bool:
        return 0 <= index < n and blocks[index].type == BlockType.IMAGE

    def caption_at(index: int) -> str | None:
        if not 0 <= index < n:
            return None
        return _figure_caption_text(blocks[index])

    def take(image_index: int, caption_index: int) -> None:
        if image_index in assigned or caption_index in used:
            return
        if caption_at(caption_index) is None:
            return
        assigned[image_index] = caption_index
        used.add(caption_index)

    for index, block in enumerate(blocks):
        if block.type != BlockType.IMAGE:
            continue
        after, before = index + 1, index - 1
        if caption_at(after) and not is_image(after + 1):
            take(index, after)
            continue
        if caption_at(before) and not is_image(before - 1):
            take(index, before)

    for index, block in enumerate(blocks):
        if block.type != BlockType.IMAGE or index in assigned:
            continue
        after = index + 1
        if caption_at(after) and is_image(after + 1):
            take(index, after)

    for index, block in enumerate(blocks):
        if block.type != BlockType.IMAGE or index in assigned:
            continue
        after, before = index + 1, index - 1
        if caption_at(after):
            take(index, after)
        elif caption_at(before):
            take(index, before)

    result: list[Block] = []
    for index, block in enumerate(blocks):
        if index in used:
            continue
        caption_index = assigned.get(index)
        if caption_index is None or block.image is None:
            result.append(block)
            continue
        image = block.image
        caption = caption_at(caption_index)
        if caption and not image.caption:
            image = replace(image, caption=caption)
        result.append(replace(block, image=image))
    return result
