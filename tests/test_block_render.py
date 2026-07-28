from RAG_Agent.domain.value_objects.block import Block, BlockType, ImageRef, TableData
from RAG_Agent.domain.value_objects.block_render import (
    BlockTextFormat,
    markdown_text,
    plain_text,
    render_blocks,
)


def test_markdown_heading_and_list_levels():
    heading = Block(id="h", type=BlockType.HEADING, order=0, text="Scope", level=2)
    item = Block(id="l", type=BlockType.LIST_ITEM, order=1, text="Non-RT RIC:", level=1)
    child = Block(id="c", type=BlockType.LIST_ITEM, order=2, text="Retrieve metrics.", level=2)
    assert markdown_text(heading) == "## Scope"
    assert markdown_text(item) == "- Non-RT RIC:"
    assert markdown_text(child) == "  - Retrieve metrics."


def test_markdown_table_and_image_caption():
    table = Block(
        id="t",
        type=BlockType.TABLE,
        order=0,
        table=TableData(headers=["A", "B"], rows=[["1", "2"]], caption="Table 1-1: Demo"),
    )
    image = Block(
        id="i",
        type=BlockType.IMAGE,
        order=1,
        image=ImageRef(caption="Figure 4.1-1: Architecture"),
    )
    md_table = markdown_text(table)
    assert "**Table 1-1: Demo**" in md_table
    assert "| A | B |" in md_table
    assert markdown_text(image) == "Figure 4.1-1: Architecture"
    assert plain_text(image) == "Figure 4.1-1: Architecture"


def test_render_blocks_markdown_joins_and_skips_empty():
    blocks = [
        Block(id="h", type=BlockType.HEADING, order=0, text="Intro", level=1),
        Block(id="empty", type=BlockType.PARAGRAPH, order=1, text="  "),
        Block(id="p", type=BlockType.PARAGRAPH, order=2, text="Hello."),
    ]
    text, ids = render_blocks(blocks, fmt=BlockTextFormat.MARKDOWN)
    assert text == "# Intro\n\nHello."
    assert ids == ("h", "p")
