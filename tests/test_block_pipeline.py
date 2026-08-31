from RAG_Agent.domain.doc_processing_rules.default_document_rules import (
    DEFAULT_DOCUMENT_RULES,
)
from RAG_Agent.domain.doc_processing_rules.oran_document_rules import ORAN_RULES_REGISTRY
from RAG_Agent.domain.value_objects.block import Block, BlockType
from RAG_Agent.domain.value_objects.block_pipeline import refine_block_sequence

ORAN_RULES = ORAN_RULES_REGISTRY.get("oran_default")


def _block(
    order: int,
    text: str,
    *,
    block_type: BlockType = BlockType.PARAGRAPH,
    page: int = 1,
    level: int | None = None,
) -> Block:
    return Block(
        id=f"block_{order}",
        type=block_type,
        order=order,
        page=page,
        text=text,
        level=level,
    )


def test_refine_promotes_list_of_figures_then_drops_until_scope():
    blocks = [
        _block(0, "List of figures"),
        _block(1, "Figure 1-1 Architecture"),
        _block(2, "Figure 1-2 Interfaces"),
        _block(3, "1 Scope"),
        _block(4, "This document specifies circular economy guidelines."),
    ]
    result = refine_block_sequence(blocks, rules=ORAN_RULES)
    texts = [block.text for block in result]
    assert "List of figures" not in texts
    assert "Figure 1-1 Architecture" not in texts
    assert "Figure 1-2 Interfaces" not in texts
    assert texts[0] == "1 Scope"
    assert result[0].type == BlockType.HEADING
    assert [block.id for block in result] == ["block_0", "block_1"]


def test_default_rules_do_not_drop_list_of_figures_paragraph():
    blocks = [
        _block(0, "List of figures"),
        _block(1, "Figure 1-1 Architecture"),
        _block(2, "1 Scope", block_type=BlockType.HEADING, level=1),
    ]
    result = refine_block_sequence(blocks, rules=DEFAULT_DOCUMENT_RULES)
    texts = [block.text for block in result]
    assert "List of figures" in texts
    assert "Figure 1-1 Architecture" in texts
    assert "1 Scope" in texts
