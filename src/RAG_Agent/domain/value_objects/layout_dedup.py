from __future__ import annotations

from RAG_Agent.domain.value_objects.block import Block, BlockType, BoundingBox

_IOU_KEEP_TABLE = 0.5


def bbox_iou(left: BoundingBox, right: BoundingBox) -> float:
    """Intersection-over-union of two axis-aligned boxes."""
    overlap_x0 = max(left.x0, right.x0)
    overlap_y0 = max(left.y0, right.y0)
    overlap_x1 = min(left.x1, right.x1)
    overlap_y1 = min(left.y1, right.y1)
    width = max(0.0, overlap_x1 - overlap_x0)
    height = max(0.0, overlap_y1 - overlap_y0)
    intersection = width * height
    if intersection <= 0.0:
        return 0.0
    area_left = max(0.0, left.x1 - left.x0) * max(0.0, left.y1 - left.y0)
    area_right = max(0.0, right.x1 - right.x0) * max(0.0, right.y1 - right.y0)
    union = area_left + area_right - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def drop_overlapping_images_on_tables(
    blocks: list[Block],
    *,
    iou_threshold: float = _IOU_KEEP_TABLE,
) -> list[Block]:
    """Drop an IMAGE that shares a bbox with a neighboring TABLE (same page).

    Layout models sometimes emit a picture and a table for the same crop.
    Keep the structured TABLE.

    Args:
        blocks: Blocks in reading order.
        iou_threshold: Minimum IoU to treat the pair as the same object.

    Returns:
        New list without the overlapping IMAGE blocks.
    """
    drop: set[int] = set()
    for index, block in enumerate(blocks):
        if block.type != BlockType.IMAGE or block.bbox is None:
            continue
        for neighbor_index in (index - 1, index + 1):
            if neighbor_index < 0 or neighbor_index >= len(blocks):
                continue
            neighbor = blocks[neighbor_index]
            if neighbor.type != BlockType.TABLE or neighbor.bbox is None:
                continue
            if (
                block.page is not None
                and neighbor.page is not None
                and block.page != neighbor.page
            ):
                continue
            if bbox_iou(block.bbox, neighbor.bbox) >= iou_threshold:
                drop.add(index)
                break
    if not drop:
        return blocks
    return [block for index, block in enumerate(blocks) if index not in drop]
