from RAG_Agent.domain.value_objects.block import Block, BlockType, BoundingBox, TableData
from RAG_Agent.domain.value_objects.layout_dedup import drop_overlapping_images_on_tables


def test_drop_image_overlapping_neighbor_table():
    bbox = BoundingBox(64.5, 490.0, 533.0, 733.0)
    blocks = [
        Block(
            id="img",
            type=BlockType.IMAGE,
            order=0,
            page=8,
            bbox=bbox,
        ),
        Block(
            id="tbl",
            type=BlockType.TABLE,
            order=1,
            page=8,
            table=TableData(headers=["A"], rows=[["1"]]),
            bbox=bbox,
        ),
    ]
    result = drop_overlapping_images_on_tables(blocks)
    assert [block.id for block in result] == ["tbl"]


def test_keep_image_when_iou_is_low():
    blocks = [
        Block(
            id="img",
            type=BlockType.IMAGE,
            order=0,
            page=8,
            bbox=BoundingBox(10, 10, 80, 80),
        ),
        Block(
            id="tbl",
            type=BlockType.TABLE,
            order=1,
            page=8,
            table=TableData(headers=["A"], rows=[["1"]]),
            bbox=BoundingBox(200, 200, 400, 400),
        ),
    ]
    result = drop_overlapping_images_on_tables(blocks)
    assert [block.id for block in result] == ["img", "tbl"]
