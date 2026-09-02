from RAG_Agent.domain.doc_processing_rules.oran_block_refinement import refine_oran_blocks
from RAG_Agent.domain.value_objects.block import Block, BlockType, ImageRef
from RAG_Agent.domain.value_objects.figure_groups import (
    attach_figure_captions,
    looks_like_figure_caption,
)


def _image(block_id: str, *, order: int) -> Block:
    return Block(
        id=block_id,
        type=BlockType.IMAGE,
        order=order,
        image=ImageRef(),
    )


def _para(block_id: str, text: str, *, order: int) -> Block:
    return Block(
        id=block_id,
        type=BlockType.PARAGRAPH,
        order=order,
        text=text,
    )


def test_looks_like_figure_caption_accepts_annex_and_fig():
    assert looks_like_figure_caption("Figure 4.1.3-1: Dynamic handover")
    assert looks_like_figure_caption("Figure A.1-1: Reduction of environmental impact")
    assert looks_like_figure_caption("Fig. 2 Overview")
    assert looks_like_figure_caption("Figura 1: Arquitectura")
    assert not looks_like_figure_caption("See the figure below.")
    assert not looks_like_figure_caption("Table 4.1-1 Priorities")


def test_attach_caption_after_image_and_drops_paragraph():
    caption = "Figure 4.1.3-1: Dynamic handover management for V2X use case"
    result = attach_figure_captions(
        [
            _image("img", order=0),
            _para("cap", caption, order=1),
            _para("p", "Following prose remains.", order=2),
        ]
    )
    assert len(result) == 2
    assert result[0].image is not None
    assert result[0].image.caption == caption
    assert result[1].id == "p"


def test_attach_caption_before_image():
    caption = "Figure A.1-1: Reduction of environmental impact through circular economy"
    result = attach_figure_captions(
        [
            _para("cap", caption, order=0),
            _image("img", order=1),
            _para("p", "Next section.", order=2),
        ]
    )
    assert len(result) == 2
    assert result[0].image is not None
    assert result[0].image.caption == caption
    assert result[1].id == "p"


def test_two_figures_with_captions_below():
    result = attach_figure_captions(
        [
            _image("i1", order=0),
            _para("c1", "Figure 1-1: First", order=1),
            _image("i2", order=2),
            _para("c2", "Figure 1-2: Second", order=3),
        ]
    )
    assert [block.id for block in result] == ["i1", "i2"]
    assert result[0].image is not None and result[0].image.caption == "Figure 1-1: First"
    assert result[1].image is not None and result[1].image.caption == "Figure 1-2: Second"


def test_two_figures_with_captions_above():
    result = attach_figure_captions(
        [
            _para("c1", "Figure 1-1: First", order=0),
            _image("i1", order=1),
            _para("c2", "Figure 1-2: Second", order=2),
            _image("i2", order=3),
        ]
    )
    assert [block.id for block in result] == ["i1", "i2"]
    assert result[0].image is not None and result[0].image.caption == "Figure 1-1: First"
    assert result[1].image is not None and result[1].image.caption == "Figure 1-2: Second"


def test_caption_between_two_images_goes_to_the_first():
    result = attach_figure_captions(
        [
            _image("i1", order=0),
            _para("c", "Figure 1-1: Shared", order=1),
            _image("i2", order=2),
        ]
    )
    assert result[0].image is not None and result[0].image.caption == "Figure 1-1: Shared"
    assert result[1].image is not None and result[1].image.caption is None


def test_does_not_attach_following_prose():
    result = attach_figure_captions(
        [_image("img", order=0), _para("p", "The deployment is shown below.", order=1)]
    )
    assert len(result) == 2
    assert result[0].image is not None
    assert result[0].image.caption is None


def test_attach_after_oran_demotes_figure_heading():
    caption = "Figure 4.1.3-1: Dynamic handover management for V2X use case"
    refined = refine_oran_blocks(
        [
            _image("img", order=0),
            Block(
                id="cap",
                type=BlockType.HEADING,
                order=1,
                text=caption,
                level=1,
            ),
        ]
    )
    result = attach_figure_captions(refined)
    assert len(result) == 1
    assert result[0].image is not None
    assert result[0].image.caption == caption
