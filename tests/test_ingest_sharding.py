from pathlib import Path

import pytest

from RAG_Agent.domain.doc_processing_rules.default_document_rules import (
    DEFAULT_DOCUMENT_RULES,
    DefaultProfileResolver,
)
from RAG_Agent.domain.doc_processing_rules.oran_document_rules import (
    ORAN_RULES_REGISTRY,
    OranDocumentId,
    OranDocumentRules,
    OranProfileResolver,
)
from RAG_Agent.domain.value_objects.block import Block, BlockType, BoundingBox, TableData
from RAG_Agent.domain.value_objects.canonical_document import (
    CanonicalDocument,
    DocumentMetadata,
)
from RAG_Agent.domain.value_objects.canonical_merge import merge_canonical_shards
from RAG_Agent.domain.value_objects.page import Page
from RAG_Agent.domain.value_objects.section import Section
from RAG_Agent.infrastructure.ingestion.cascading_profile_resolver import CascadingProfileResolver
from RAG_Agent.infrastructure.ingestion.ingest_profile import get_ingest_profile
from RAG_Agent.infrastructure.ingestion.normalizers.docling_normalizer import DoclingNormalizer


def _doc(blocks: list[Block], pages: list[Page]) -> CanonicalDocument:
    return CanonicalDocument(
        metadata=DocumentMetadata(source_path=Path("x.pdf"), title="t"),
        blocks={block.id: block for block in blocks},
        pages=pages,
        sections=[],
    )


@pytest.mark.parametrize(
    ("stem", "group", "doc_type", "subject", "release", "version"),
    [
        (
            "O-RAN.WG1.TR.Use-Cases-Analysis-Report-R005-v19.00",
            "WG1",
            "tr",
            "Use-Cases-Analysis-Report",
            "005",
            "19.00",
        ),
        (
            "O-RAN.WG1.TS.Use-Cases-Detailed-Specification-R005-v19.00",
            "WG1",
            "ts",
            "Use-Cases-Detailed-Specification",
            "005",
            "19.00",
        ),
        (
            "O-RAN.WG11.TR.ZTA-R005-v05.00",
            "WG11",
            "tr",
            "ZTA",
            "005",
            "05.00",
        ),
        (
            "O-RAN.SuFG.CE-v01.00",
            "SuFG",
            "ce",
            "CE",
            None,
            "01.00",
        ),
    ],
)
def test_oran_document_id_parses_official_stems(
    stem: str,
    group: str,
    doc_type: str,
    subject: str,
    release: str | None,
    version: str,
):
    parsed = OranDocumentId.from_path(f"{stem}.pdf")
    assert parsed is not None
    assert parsed.group == group
    assert parsed.doc_type == doc_type
    assert parsed.subject == subject
    assert parsed.release == release
    assert parsed.version == version


def test_oran_document_id_rejects_non_oran():
    assert OranDocumentId.from_path("data/some-other-doc.pdf") is None
    assert OranDocumentId.from_path("O-RAN.UNKNOWN.FOO-v01.00.pdf") is None


def test_local_ingest_profile_defaults():
    profile = get_ingest_profile("local")
    assert profile.name == "local"
    assert profile.pages_per_shard == 50
    assert profile.max_file_size_mb == 200


def test_resolve_title_prefers_cover_page_over_introduction():
    normalizer = DoclingNormalizer()
    rules = OranDocumentRules(profile_id="oran_default")
    blocks = [
        Block(
            id="block_0",
            type=BlockType.PARAGRAPH,
            order=0,
            page=1,
            text="O-RAN Work Group 1 (Use Cases and Overall Architecture)",
        ),
        Block(id="block_1", type=BlockType.HEADING, order=1, page=5, text="Introduction", level=1),
    ]
    sections = [
        Section(id="sec_0", title="Preamble", level=1, order=0, block_ids=["block_0"]),
        Section(id="sec_1", title="Introduction", level=1, order=1, block_ids=["block_1"]),
    ]
    title = normalizer.resolve_title(
        sections, blocks, "Use-Cases-Detailed-Specification", rules=rules
    )
    assert title.startswith("O-RAN Work Group 1")


def test_resolve_title_falls_back_to_hint_when_no_cover():
    normalizer = DoclingNormalizer()
    rules = OranDocumentRules(profile_id="oran_default")
    blocks = [
        Block(id="block_0", type=BlockType.HEADING, order=0, page=5, text="Introduction", level=1),
    ]
    sections = [
        Section(id="sec_0", title="Introduction", level=1, order=0, block_ids=["block_0"]),
    ]
    title = normalizer.resolve_title(
        sections, blocks, "Use-Cases-Detailed-Specification", rules=rules
    )
    assert title == "Use Cases Detailed Specification"


def test_default_rules_are_minimal():
    default = DEFAULT_DOCUMENT_RULES
    oran = OranDocumentRules(
        profile_id="oran_default",
        sections_to_remove=("foreword", "contents"),
    )
    assert default.generic_doc_titles == frozenset()
    assert default.title_boilerplate_pattern == ""
    assert default.noise_paragraph_chars == frozenset()
    assert default.sections_to_remove == ()
    assert not default.is_generic_doc_title("Introduction")
    assert not default.is_title_boilerplate("Copyright 2024")
    assert not default.is_noise_paragraph("---")
    assert oran.is_generic_doc_title("Modal verbs terminology")
    assert oran.is_title_boilerplate("VAT ID: DE123")
    assert oran.preprocess_options.clean_cover_page is True
    assert oran.is_removable_section("Foreword")


def test_cascading_resolver_oran_and_sufg():
    resolver = CascadingProfileResolver(
        (OranProfileResolver(), DefaultProfileResolver()),
    )
    wg1 = resolver.resolve(
        Path("data/O-RAN.WG1.TS.Use-Cases-Detailed-Specification-R005-v19.00.pdf")
    )
    assert wg1.rules.profile_id == "oran_default"
    assert wg1.identity.metadata["group"] == "WG1"
    assert wg1.identity.metadata["family"] == "oran"

    sufg = resolver.resolve(Path("data/O-RAN.SuFG.CE-v01.00.pdf"))
    assert sufg.rules.profile_id == "oran_default"
    assert sufg.identity.metadata["group"] == "SuFG"
    assert sufg.identity.title_hint == "CE"

    other = resolver.resolve(Path("data/some-other-doc.pdf"))
    assert other.rules.profile_id == "default"


def test_merge_canonical_shards_page_offset():
    normalizer = DoclingNormalizer()
    rules = DEFAULT_DOCUMENT_RULES
    shard0 = _doc(
        [
            Block(id="block_0", type=BlockType.HEADING, order=0, page=1, text="Intro", level=1),
            Block(id="block_1", type=BlockType.PARAGRAPH, order=1, page=1, text="A"),
        ],
        [Page(number=1, block_ids=["block_0", "block_1"], width=100, height=200)],
    )
    shard1 = _doc(
        [
            Block(id="block_0", type=BlockType.HEADING, order=0, page=1, text="Scope", level=1),
            Block(id="block_1", type=BlockType.PARAGRAPH, order=1, page=2, text="B"),
        ],
        [
            Page(number=1, block_ids=["block_0"], width=100, height=200),
            Page(number=2, block_ids=["block_1"], width=100, height=200),
        ],
    )

    merged = merge_canonical_shards(
        [(0, shard0), (50, shard1)],
        source_path=Path("data/doc.pdf"),
        profile_id="oran_default",
        parser_name="native_pdf_docling",
        rules=rules,
        build_sections=lambda blocks: normalizer.build_sections(blocks, rules=rules),
        resolve_title=lambda sections, blocks, hint: normalizer.resolve_title(
            sections, blocks, hint, rules=rules
        ),
        title_hint="hint",
        extra={"ingest_profile": "local"},
    )

    assert merged.metadata.extra["num_shards"] == "2"
    assert merged.blocks["block_0"].page == 1
    assert merged.blocks["block_2"].page == 51
    assert merged.blocks["block_3"].page == 52
    assert {page.number for page in merged.pages} == {1, 51, 52}
    assert any(section.title == "Scope" for section in merged.sections)


def test_merge_canonical_shards_collapses_split_table_and_indexes_both_pages():
    bbox_a = BoundingBox(x0=50, y0=70, x1=540, y1=230)
    bbox_b = BoundingBox(x0=50, y0=380, x1=540, y1=760)
    shard0 = _doc(
        [
            Block(
                id="block_0",
                type=BlockType.TABLE,
                order=0,
                page=1,
                table=TableData(
                    headers=["A", "B"],
                    rows=[["keep", "1"]],
                    caption="Table 1-1 Demo",
                ),
                bbox=bbox_a,
                source_ref="#/tables/0",
            )
        ],
        [Page(number=1, block_ids=["block_0"], width=100, height=200)],
    )
    shard1 = _doc(
        [
            Block(
                id="block_0",
                type=BlockType.TABLE,
                order=0,
                page=1,
                table=TableData(headers=["0", "1"], rows=[["", "2"]]),
                bbox=bbox_b,
                source_ref="#/tables/1",
            )
        ],
        [Page(number=1, block_ids=["block_0"], width=100, height=200)],
    )

    merged = merge_canonical_shards(
        [(0, shard0), (1, shard1)],
        source_path=Path("data/doc.pdf"),
        profile_id="default",
        parser_name="native_pdf_docling",
        rules=DEFAULT_DOCUMENT_RULES,
        build_sections=lambda blocks: [],
        resolve_title=lambda sections, blocks, hint: hint,
        title_hint="hint",
    )

    tables = [
        block for block in merged.blocks.values() if block.type == BlockType.TABLE
    ]
    assert len(tables) == 1
    table_block = tables[0]
    assert table_block.table is not None
    assert table_block.table.rows == [["keep", "1"], ["keep", "2"]]
    assert table_block.page == 1
    assert table_block.page_numbers() == (1, 2)
    assert [span.page for span in table_block.pdf_layout()] == [1, 2]
    assert table_block.pdf_layout()[1].bbox == bbox_b
    page1 = next(page for page in merged.pages if page.number == 1)
    page2 = next(page for page in merged.pages if page.number == 2)
    assert table_block.id in page1.block_ids
    assert table_block.id in page2.block_ids


def test_merge_canonical_shards_drops_list_of_figures_split_across_shards():
    oran = ORAN_RULES_REGISTRY.get("oran_default")
    normalizer = DoclingNormalizer()
    shard0 = _doc(
        [
            Block(
                id="block_0",
                type=BlockType.PARAGRAPH,
                order=0,
                page=1,
                text="List of figures",
            ),
            Block(
                id="block_1",
                type=BlockType.PARAGRAPH,
                order=1,
                page=1,
                text="Figure 1-1 Architecture",
            ),
        ],
        [Page(number=1, block_ids=["block_0", "block_1"])],
    )
    shard1 = _doc(
        [
            Block(
                id="block_0",
                type=BlockType.PARAGRAPH,
                order=0,
                page=1,
                text="Figure 1-2 Interfaces",
            ),
            Block(
                id="block_1",
                type=BlockType.PARAGRAPH,
                order=1,
                page=1,
                text="1 Scope",
            ),
            Block(
                id="block_2",
                type=BlockType.PARAGRAPH,
                order=2,
                page=1,
                text="This document specifies scope.",
            ),
        ],
        [Page(number=1, block_ids=["block_0", "block_1", "block_2"])],
    )

    merged = merge_canonical_shards(
        [(0, shard0), (1, shard1)],
        source_path=Path("data/doc.pdf"),
        profile_id="oran_default",
        parser_name="native_pdf_docling",
        rules=oran,
        build_sections=lambda blocks: normalizer.build_sections(blocks, rules=oran),
        resolve_title=lambda sections, blocks, hint: normalizer.resolve_title(
            sections, blocks, hint, rules=oran
        ),
        title_hint="CE",
    )

    texts = [
        block.text
        for block in sorted(merged.blocks.values(), key=lambda item: item.order)
    ]
    assert "List of figures" not in texts
    assert "Figure 1-1 Architecture" not in texts
    assert "Figure 1-2 Interfaces" not in texts
    assert "1 Scope" in texts
    assert any(section.title == "1 Scope" for section in merged.sections)
