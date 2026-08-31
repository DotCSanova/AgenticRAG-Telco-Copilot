from pathlib import Path

import pytest

from RAG_Agent.domain.doc_processing_rules.default_document_rules import (
    DEFAULT_DOCUMENT_RULES,
)
from RAG_Agent.domain.doc_processing_rules.oran_document_rules import ORAN_RULES_REGISTRY
from RAG_Agent.domain.value_objects.block import Block, BlockType
from RAG_Agent.domain.value_objects.block_pipeline import refine_block_sequence
from RAG_Agent.infrastructure.ingestion.normalizers.docling_normalizer import (
    DoclingNormalizer,
)

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


def test_cover_vat_does_not_create_preamble_section():
    blocks = [
        _block(0, "VAT ID: DE123456789", page=1),
        _block(1, "Register of Associations, Bonn", page=1),
        _block(2, "1 Scope", page=2),
    ]
    result = refine_block_sequence(blocks, rules=ORAN_RULES)
    sections = DoclingNormalizer().build_sections(result, rules=ORAN_RULES)
    assert not any(section.title.lower() == "preamble" for section in sections)
    assert any(section.title == "1 Scope" for section in sections)


def test_cover_title_joins_two_lines():
    blocks = [
        _block(0, "O-RAN Focus Group (Sustainability Focus Group)", page=1),
        _block(1, "Circular economy guidelines on network equipment", page=1),
        _block(2, "1 Scope", page=4),
    ]
    result = refine_block_sequence(blocks, rules=ORAN_RULES)
    title = DoclingNormalizer.resolve_title([], result, "CE", rules=ORAN_RULES)
    assert "O-RAN Focus Group (Sustainability Focus Group)" in title
    assert "Circular economy guidelines on network equipment" in title


def test_clean_cover_page_keeps_multiline_title_redacts_footer_and_images(tmp_path: Path):
    fitz = pytest.importorskip("fitz")
    from RAG_Agent.infrastructure.ingestion.preprocessing.pymupdf_preprocessor import (
        PyMuPDFPreprocessor,
    )

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(
        (80, 320),
        "O-RAN Focus Group (Sustainability Focus Group)",
        fontsize=14,
    )
    page.insert_text(
        (80, 350),
        "Circular economy guidelines on network equipment",
        fontsize=14,
    )
    page.insert_text((80, 800), "VAT ID DE123456 Register of Associations", fontsize=8)
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 32, 32), 0)
    pixmap.clear_with(180)
    page.insert_image(fitz.Rect(50, 40, 90, 80), pixmap=pixmap)
    # Extra pages so cover lines are not treated as repeated chrome (1-page PDFs
    # match every string at 100% of pages).
    for _ in range(2):
        extra = doc.new_page(width=595, height=842)
        extra.insert_text((80, 200), "1 Scope", fontsize=12)
        extra.insert_text((80, 240), "This document specifies guidelines.", fontsize=11)

    source = tmp_path / "cover.pdf"
    cleaned = tmp_path / "cover_cleaned.pdf"
    doc.save(source)
    doc.close()

    PyMuPDFPreprocessor(ORAN_RULES).preprocess(source, cleaned)
    with fitz.open(cleaned) as out:
        text = out[0].get_text()
        assert "O-RAN Focus Group (Sustainability Focus Group)" in text
        assert "Circular economy guidelines on network equipment" in text
        assert "VAT ID" not in text
        assert "Register of Associations" not in text
        assert out[0].get_images() == []
