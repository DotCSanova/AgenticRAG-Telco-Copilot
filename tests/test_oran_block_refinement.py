from RAG_Agent.domain.doc_processing_rules.oran_block_refinement import (
    is_enumerated_list_marker,
    is_list_label,
    refine_oran_blocks,
    section_heading_level,
)
from RAG_Agent.domain.doc_processing_rules.oran_document_rules import OranDocumentRules
from RAG_Agent.domain.value_objects.block import Block, BlockType, BoundingBox, ImageRef


def _bbox(x0: float) -> BoundingBox:
    return BoundingBox(x0=x0, y0=0.0, x1=x0 + 100.0, y1=10.0)


def test_section_heading_level_numeric_and_named():
    assert section_heading_level("4.2.2 Entities/resources involved") == 3
    assert section_heading_level("1 Introduction") == 1
    assert section_heading_level("Introduction") == 1
    assert section_heading_level("List of figures") == 1
    assert section_heading_level("Modal verbs terminology") == 1
    assert section_heading_level("A.1 Traffic steering use case") == 2
    assert section_heading_level("1) Non-RT RIC:") is None
    assert section_heading_level("Near-RT RIC:") is None
    assert section_heading_level("Figure 4.21.3.5.2-1: Something") is None


def test_list_label_patterns():
    assert is_enumerated_list_marker("1) Non-RT RIC:")
    assert is_list_label("1) Non-RT RIC:")
    assert is_list_label("Near-RT RIC:")
    assert is_list_label("RAN:")
    assert not is_list_label("4.2.2 Entities/resources involved")
    assert not is_list_label("Retrieve necessary of O-RAN support for aerial vehicles " * 3)


def test_refine_blocks_demotes_enum_heading_and_sets_list_levels():
    blocks = [
        Block(
            id="b0",
            type=BlockType.HEADING,
            order=0,
            page=17,
            text="4.2.2 Entities/resources involved in the use case",
            level=3,
            bbox=_bbox(56.664),
        ),
        Block(
            id="b1",
            type=BlockType.HEADING,
            order=1,
            page=17,
            text="1) Non-RT RIC:",
            level=1,
            bbox=_bbox(56.664),
        ),
        Block(
            id="b2",
            type=BlockType.LIST_ITEM,
            order=2,
            page=17,
            text="Retrieve necessary measurement metrics from the network.",
            bbox=_bbox(77.664),
        ),
        Block(
            id="b3",
            type=BlockType.LIST_ITEM,
            order=3,
            page=17,
            text="Near-RT RIC:",
            bbox=_bbox(56.664),
        ),
        Block(
            id="b4",
            type=BlockType.LIST_ITEM,
            order=4,
            page=17,
            text="Support update of AI/ML models from Non-RT RIC.",
            bbox=_bbox(77.664),
        ),
    ]

    refined = OranDocumentRules(profile_id="oran_default").refine_blocks(blocks)

    assert refined[0].type == BlockType.HEADING
    assert refined[0].level == 3

    assert refined[1].type == BlockType.LIST_ITEM
    assert refined[1].level == 1
    assert refined[1].text == "1) Non-RT RIC:"

    assert refined[2].type == BlockType.LIST_ITEM
    assert refined[2].level == 2

    assert refined[3].type == BlockType.LIST_ITEM
    assert refined[3].level == 1

    assert refined[4].type == BlockType.LIST_ITEM
    assert refined[4].level == 2


def test_refine_blocks_promotes_named_heading_and_demotes_noise_heading():
    blocks = [
        Block(
            id="b0",
            type=BlockType.PARAGRAPH,
            order=0,
            text="Introduction",
            bbox=_bbox(56.0),
        ),
        Block(
            id="b1",
            type=BlockType.HEADING,
            order=1,
            text="Figure 4.21.3.5.2-1: PA power optimization",
            level=1,
            bbox=_bbox(56.0),
        ),
    ]
    refined = refine_oran_blocks(blocks)
    assert refined[0].type == BlockType.HEADING
    assert refined[0].level == 1
    assert refined[1].type == BlockType.PARAGRAPH
    assert refined[1].level is None


def test_refine_blocks_retains_annex_and_history_only_if_already_heading():
    blocks = [
        Block(
            id="b0",
            type=BlockType.HEADING,
            order=0,
            text="Annex A (informative):",
            level=1,
            bbox=_bbox(56.0),
        ),
        Block(
            id="b1",
            type=BlockType.HEADING,
            order=1,
            text="Change history/Change request (history)",
            level=1,
            bbox=_bbox(56.0),
        ),
        Block(
            id="b2",
            type=BlockType.HEADING,
            order=2,
            text="Additional information",
            level=1,
            bbox=_bbox(56.0),
        ),
        Block(
            id="b3",
            type=BlockType.PARAGRAPH,
            order=3,
            text="Revision history",
            bbox=_bbox(56.0),
        ),
        Block(
            id="b4",
            type=BlockType.HEADING,
            order=4,
            text="History",
            level=1,
            bbox=_bbox(56.0),
        ),
    ]
    refined = refine_oran_blocks(blocks)
    assert refined[0].type == BlockType.HEADING
    assert refined[1].type == BlockType.HEADING
    assert refined[2].type == BlockType.HEADING
    assert refined[3].type == BlockType.PARAGRAPH  # no promoción
    assert refined[4].type == BlockType.HEADING


def test_refine_blocks_demotes_figure_heading_and_does_not_attach_caption():
    blocks = [
        Block(
            id="img",
            type=BlockType.IMAGE,
            order=0,
            page=11,
            image=ImageRef(),
            bbox=_bbox(56.0),
        ),
        Block(
            id="cap",
            type=BlockType.HEADING,
            order=1,
            page=11,
            text="Figure 4.1.3-1: Dynamic handover management for V2X use case",
            level=1,
            bbox=_bbox(56.0),
        ),
    ]
    refined = refine_oran_blocks(blocks)
    assert refined[0].type == BlockType.IMAGE
    assert refined[0].image is not None
    assert refined[0].image.caption is None
    assert refined[1].type == BlockType.PARAGRAPH
    assert refined[1].id == "cap"
