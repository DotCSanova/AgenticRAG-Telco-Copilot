from pathlib import Path

from RAG_Agent.domain.value_objects.block import Block, BlockType, TableData
from RAG_Agent.domain.value_objects.canonical_document import CanonicalDocument, DocumentMetadata
from RAG_Agent.domain.value_objects.section import Section
from RAG_Agent.infrastructure.indexing.section_chunker import SectionChunker


def _doc(
    *,
    blocks: dict[str, Block],
    sections: list[Section] | None = None,
) -> CanonicalDocument:
    return CanonicalDocument(
        metadata=DocumentMetadata(source_path=Path("data/demo-doc.pdf"), title="Demo"),
        blocks=blocks,
        sections=sections or [],
    )


def test_section_chunker_only_leaf_sections_with_path():
    blocks = {
        "h0": Block(id="h0", type=BlockType.HEADING, order=0, page=1, text="Architecture", level=1),
        "h1": Block(id="h1", type=BlockType.HEADING, order=1, page=1, text="Overview", level=2),
        "p1": Block(id="p1", type=BlockType.PARAGRAPH, order=2, page=1, text="Leaf body A."),
        "h2": Block(id="h2", type=BlockType.HEADING, order=3, page=2, text="Details", level=2),
        "p2": Block(id="p2", type=BlockType.PARAGRAPH, order=4, page=2, text="Leaf body B."),
    }
    sections = [
        Section(
            id="sec_arch",
            title="Architecture",
            level=1,
            order=0,
            block_ids=["h0"],  # parent structural: only heading
            page_start=1,
            page_end=1,
        ),
        Section(
            id="sec_overview",
            title="Overview",
            level=2,
            order=1,
            parent_id="sec_arch",
            block_ids=["h1", "p1"],
            page_start=1,
            page_end=1,
        ),
        Section(
            id="sec_details",
            title="Details",
            level=2,
            order=2,
            parent_id="sec_arch",
            block_ids=["h2", "p2"],
            page_start=2,
            page_end=2,
        ),
    ]
    chunks = SectionChunker().chunk(_doc(blocks=blocks, sections=sections))
    assert len(chunks) == 2
    assert [chunk.section_id for chunk in chunks] == ["sec_overview", "sec_details"]
    assert chunks[0].metadata["section_path"] == "Architecture > Overview"
    assert chunks[1].metadata["section_path"] == "Architecture > Details"
    assert "Leaf body A." in chunks[0].text
    assert "Architecture" not in [chunk.section_id for chunk in chunks]


def test_section_chunker_flat_leaves_still_work():
    blocks = {
        "b0": Block(id="b0", type=BlockType.HEADING, order=0, page=1, text="Intro", level=1),
        "b1": Block(id="b1", type=BlockType.PARAGRAPH, order=1, page=1, text="Hello world."),
    }
    sections = [
        Section(
            id="sec_0",
            title="Intro",
            level=1,
            order=0,
            block_ids=["b0", "b1"],
            page_start=1,
            page_end=1,
        ),
    ]
    chunks = SectionChunker().chunk(_doc(blocks=blocks, sections=sections))
    assert len(chunks) == 1
    assert chunks[0].metadata["section_path"] == "Intro"
    assert chunks[0].metadata["section_title"] == "Intro"


def test_section_chunker_renders_tables_and_skips_empty_blocks():
    blocks = {
        "h": Block(id="h", type=BlockType.HEADING, order=0, page=1, text="Tables", level=1),
        "t": Block(
            id="t",
            type=BlockType.TABLE,
            order=1,
            page=1,
            table=TableData(headers=["A", "B"], rows=[["1", "2"]], caption="Table 1"),
        ),
        "empty": Block(id="empty", type=BlockType.PARAGRAPH, order=2, page=1, text="  "),
        "img": Block(id="img", type=BlockType.IMAGE, order=3, page=1),
    }
    sections = [
        Section(
            id="sec_t",
            title="Tables",
            level=1,
            order=0,
            block_ids=["h", "t", "empty", "img"],
            page_start=1,
            page_end=1,
        )
    ]
    chunks = SectionChunker().chunk(_doc(blocks=blocks, sections=sections))
    assert len(chunks) == 1
    assert chunks[0].block_ids == ("h", "t")
    assert chunks[0].metadata["section_path"] == "Tables"
    assert "A | B" in chunks[0].text


def test_section_chunker_skips_leaf_with_no_text():
    blocks = {"img": Block(id="img", type=BlockType.IMAGE, order=0, page=1)}
    sections = [
        Section(id="sec_empty", title="Empty", level=1, order=0, block_ids=["img"]),
    ]
    assert SectionChunker().chunk(_doc(blocks=blocks, sections=sections)) == []


def test_section_chunker_orphan_blocks_without_sections():
    blocks = {
        "b0": Block(id="b0", type=BlockType.PARAGRAPH, order=0, page=3, text="Loose text"),
    }
    chunks = SectionChunker().chunk(_doc(blocks=blocks, sections=[]))
    assert len(chunks) == 1
    assert chunks[0].id == "demo-doc:orphan"
    assert chunks[0].metadata["section_path"] == ""
