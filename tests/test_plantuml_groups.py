from RAG_Agent.domain.value_objects.block import Block, BlockType, BoundingBox
from RAG_Agent.domain.value_objects.plantuml_groups import merge_plantuml_fragments


def _para(block_id: str, text: str, *, order: int, page: int, y0: float, source_ref: str) -> Block:
    return Block(
        id=block_id,
        type=BlockType.PARAGRAPH,
        order=order,
        page=page,
        text=text,
        bbox=BoundingBox(x0=50, y0=y0, x1=540, y1=y0 + 20),
        source_ref=source_ref,
    )


def test_merge_plantuml_keeps_layout_spans_per_fragment():
    bbox_p3 = BoundingBox(x0=50, y0=100, x1=540, y1=120)
    bbox_p4 = BoundingBox(x0=50, y0=400, x1=540, y1=420)
    blocks = [
        _para("a", "@startuml", order=0, page=3, y0=100, source_ref="#/texts/1"),
        Block(
            id="b",
            type=BlockType.PARAGRAPH,
            order=1,
            page=3,
            text="Alice -> Bob: hi",
            bbox=bbox_p3,
            source_ref="#/texts/2",
        ),
        Block(
            id="c",
            type=BlockType.PARAGRAPH,
            order=2,
            page=4,
            text="@enduml",
            bbox=bbox_p4,
            source_ref="#/texts/3",
        ),
        Block(id="d", type=BlockType.PARAGRAPH, order=3, page=4, text="Following prose."),
    ]

    result = merge_plantuml_fragments(blocks)
    assert len(result) == 2
    diagram = result[0]
    assert diagram.type == BlockType.CODE
    assert diagram.page == 3
    assert diagram.metadata["page_end"] == "4"
    assert diagram.page_numbers() == (3, 4)
    spans = diagram.pdf_layout()
    assert [span.page for span in spans] == [3, 3, 4]
    assert spans[-1].bbox == bbox_p4
    assert spans[-1].source_ref == "#/texts/3"
    assert result[1].text == "Following prose."
