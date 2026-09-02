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
from RAG_Agent.infrastructure.ingestion.ingest_profile import ingest_hardware
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


def test_ingest_hardware_defaults():
    profile = ingest_hardware()
    assert profile.pages_per_shard == 50
    assert profile.max_file_size_mb == 200
    assert profile.layout_batch_size == 4
    assert profile.table_batch_size == 2
    assert profile.table_former_mode == "accurate"


def test_ingest_hardware_reads_settings(monkeypatch):
    from RAG_Agent.config import settings

    monkeypatch.setattr(settings, "ingest_pages_per_shard", 10)
    monkeypatch.setattr(settings, "ingest_layout_batch_size", 3)
    monkeypatch.setattr(settings, "ingest_table_batch_size", 1)
    profile = ingest_hardware()
    assert profile.pages_per_shard == 10
    assert profile.layout_batch_size == 3
    assert profile.table_batch_size == 1
    assert profile.table_former_mode == "accurate"
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


def test_oran_default_removes_history_back_matter():
    rules = ORAN_RULES_REGISTRY.get("oran_default")
    assert rules.is_removable_section("Revision history")
    assert rules.is_removable_section("History")
    assert rules.is_removable_section("Change history")
    assert rules.is_removable_section("Change history/Change request (history)")
    assert not rules.is_removable_section("1 Scope")


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


def _write_n_page_pdf(path: Path, page_count: int) -> Path:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for index in range(page_count):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), f"Unique body text for folio {index + 1}")
    doc.save(path)
    doc.close()
    return path


class _LimitsExtractor:
    """DoclingExtractor stand-in that only runs ``_validate_pdf`` / ``extract`` wiring."""

    def __init__(self, max_pages: int = 2, max_file_size_mb: int = 200) -> None:
        from RAG_Agent.infrastructure.ingestion.extractors.docling_extractor import (
            DoclingExtractor,
        )

        self._impl = DoclingExtractor.__new__(DoclingExtractor)
        self._impl._max_pages = max_pages
        self._impl._max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.convert_calls: list[tuple[Path, tuple[int, int] | None, int]] = []

        def _convert_file(
            pdf_path: Path,
            *,
            page_range: tuple[int, int] | None = None,
            page_count: int,
        ):
            self.convert_calls.append((Path(pdf_path), page_range, page_count))
            return object()

        self._impl._convert_file = _convert_file  # type: ignore[method-assign]

    def _validate_pdf(self, pdf_path: Path, page_range: tuple[int, int] | None = None) -> int:
        return self._impl._validate_pdf(pdf_path, page_range=page_range)

    def extract(self, pdf_path: Path, *, page_range: tuple[int, int] | None = None):
        return self._impl.extract(pdf_path, page_range=page_range)


def test_validate_pdf_rejects_page_count_without_range(tmp_path: Path):
    pytest.importorskip("docling")
    from RAG_Agent.infrastructure.ingestion.exceptions import PDFValidationError

    pdf = _write_n_page_pdf(tmp_path / "three.pdf", 3)
    extractor = _LimitsExtractor(max_pages=2)
    with pytest.raises(PDFValidationError, match="too large"):
        extractor._validate_pdf(pdf)


def test_validate_pdf_accepts_range_within_max_pages(tmp_path: Path):
    pytest.importorskip("docling")

    pdf = _write_n_page_pdf(tmp_path / "three.pdf", 3)
    extractor = _LimitsExtractor(max_pages=2)
    assert extractor._validate_pdf(pdf, page_range=(1, 2)) == 3
    result = extractor.extract(pdf, page_range=(1, 2))
    assert result is not None
    assert extractor.convert_calls == [(pdf, (1, 2), 3)]


def test_validate_pdf_rejects_inverted_and_oob_range(tmp_path: Path):
    pytest.importorskip("docling")
    from RAG_Agent.infrastructure.ingestion.exceptions import PDFValidationError

    pdf = _write_n_page_pdf(tmp_path / "three.pdf", 3)
    extractor = _LimitsExtractor(max_pages=2)
    with pytest.raises(PDFValidationError, match="Invalid page_range"):
        extractor._validate_pdf(pdf, page_range=(2, 1))
    with pytest.raises(PDFValidationError, match="Invalid page_range"):
        extractor._validate_pdf(pdf, page_range=(1, 99))


class _RecordingExtractor:
    def __init__(self, fail_on: set[tuple[int, int] | None] | None = None) -> None:
        self.calls: list[tuple[Path, tuple[int, int] | None]] = []
        self.fail_on = fail_on or set()

    def extract(self, pdf_path: Path, *, page_range: tuple[int, int] | None = None):
        from RAG_Agent.infrastructure.ingestion.exceptions import PDFParsingException

        path = Path(pdf_path)
        self.calls.append((path, page_range))
        siblings = sorted(child.name for child in path.parent.glob("*.pdf"))
        assert siblings == [path.name], siblings
        if page_range in self.fail_on:
            raise PDFParsingException(f"failed range {page_range}")
        return page_range


class _RangeNormalizer:
    def normalize(self, docling_document, *, source_path, profile, parser_name):
        lo, _hi = docling_document if isinstance(docling_document, tuple) else (1, 1)
        heading = Block(
            id="block_0",
            type=BlockType.HEADING,
            order=0,
            page=lo,
            text=f"Heading {lo}",
            level=1,
        )
        paragraph = Block(
            id="block_1",
            type=BlockType.PARAGRAPH,
            order=1,
            page=lo,
            text=f"Body {lo}",
        )
        return CanonicalDocument(
            metadata=DocumentMetadata(source_path=source_path, title=f"t{lo}"),
            blocks={heading.id: heading, paragraph.id: paragraph},
            pages=[Page(number=lo, block_ids=[heading.id, paragraph.id])],
            sections=[],
        )

    def build_sections(self, blocks, rules=None):
        return []

    def resolve_title(self, sections, blocks, hint, rules=None):
        return hint


def _pipeline(extractor, normalizer=None, pages_per_shard: int = 2):
    pytest.importorskip("docling")
    from RAG_Agent.infrastructure.ingestion.ingest_profile import IngestHardwareProfile
    from RAG_Agent.infrastructure.ingestion.native_pdf_pipeline import NativePdfPipeline

    return NativePdfPipeline(
        DefaultProfileResolver(),
        extractor=extractor,
        normalizer=normalizer or _RangeNormalizer(),
        hardware=IngestHardwareProfile(
            pages_per_shard=pages_per_shard,
            max_file_size_mb=200,
        ),
    )


def test_pipeline_page_ranges_on_cleaned_pdf_no_sub_pdfs(tmp_path: Path):
    pytest.importorskip("fitz")

    pdf = _write_n_page_pdf(tmp_path / "five.pdf", 5)
    extractor = _RecordingExtractor()
    result = _pipeline(extractor).parse(pdf)

    ranges = [page_range for _path, page_range in extractor.calls]
    assert ranges == [(1, 2), (3, 4), (5, 5)]
    paths = [path for path, _page_range in extractor.calls]
    assert len(set(paths)) == 1
    assert paths[0].name == "five_cleaned.pdf"
    assert {block.page for block in result.blocks.values()} == {1, 3, 5}
    assert "failed_shards" not in result.metadata.extra


def test_pipeline_fail_forward_skips_one_range(tmp_path: Path):
    pytest.importorskip("fitz")

    pdf = _write_n_page_pdf(tmp_path / "five.pdf", 5)
    extractor = _RecordingExtractor(fail_on={(3, 4)})
    result = _pipeline(extractor).parse(pdf)

    ranges = [page_range for _path, page_range in extractor.calls]
    assert ranges == [(1, 2), (3, 4), (5, 5)]
    assert result.metadata.extra["failed_shards"] == "1"
    assert {block.page for block in result.blocks.values()} == {1, 5}


def test_pipeline_all_ranges_fail_raises(tmp_path: Path):
    pytest.importorskip("fitz")
    from RAG_Agent.infrastructure.ingestion.exceptions import PDFParsingException

    pdf = _write_n_page_pdf(tmp_path / "five.pdf", 5)
    extractor = _RecordingExtractor(fail_on={(1, 2), (3, 4), (5, 5)})
    with pytest.raises(PDFParsingException, match="All 3 shards failed"):
        _pipeline(extractor).parse(pdf)


def test_pipeline_does_not_swallow_normalizer_type_error(tmp_path: Path):
    pytest.importorskip("fitz")

    class _BoomNormalizer(_RangeNormalizer):
        def normalize(self, docling_document, *, source_path, profile, parser_name):
            raise TypeError("mapper bug")

    pdf = _write_n_page_pdf(tmp_path / "five.pdf", 5)
    with pytest.raises(TypeError, match="mapper bug"):
        _pipeline(_RecordingExtractor(), normalizer=_BoomNormalizer()).parse(pdf)


@pytest.mark.integration
def test_docling_page_range_keeps_original_folio(tmp_path: Path):
    pytest.importorskip("docling")
    pytest.importorskip("fitz")
    from RAG_Agent.infrastructure.ingestion.extractors.docling_extractor import (
        DoclingExtractor,
    )
    from RAG_Agent.infrastructure.ingestion.normalizers.docling_normalizer import (
        DoclingNormalizer,
    )

    pdf = _write_n_page_pdf(tmp_path / "two.pdf", 2)
    extractor = DoclingExtractor(
        max_pages=2,
        max_file_size_mb=200,
        layout_batch_size=4,
        table_batch_size=2,
    )
    docling_doc = extractor.extract(pdf, page_range=(2, 2))

    page_nos: set[int] = set()
    doc_pages = getattr(docling_doc, "pages", {}) or {}
    page_nos.update(int(key) for key in doc_pages)
    for item, _depth in docling_doc.iterate_items():
        for prov in getattr(item, "prov", None) or []:
            page = getattr(prov, "page_no", None)
            if page is not None:
                page_nos.add(int(page))

    assert page_nos, "Docling returned no page numbers for page_range=(2, 2)"
    assert page_nos == {2}, (
        f"Docling page_range=(2, 2) numbered {page_nos}; "
        "if this is {1}, NativePdfPipeline._page_offset_for_range must return lo - 1"
    )

    profile = DefaultProfileResolver().resolve(pdf)
    canonical = DoclingNormalizer().normalize(
        docling_doc,
        source_path=pdf,
        profile=profile,
        parser_name="native_pdf_docling",
    )
    block_pages = {block.page for block in canonical.blocks.values() if block.page is not None}
    if block_pages:
        assert block_pages == {2}


def test_extractor_pipeline_options_come_from_constructor_knobs():
    pytest.importorskip("docling")
    from docling.datamodel.pipeline_options import TableFormerMode

    from RAG_Agent.infrastructure.ingestion.extractors.docling_extractor import (
        DoclingExtractor,
    )

    options = DoclingExtractor(
        max_pages=50,
        max_file_size_mb=200,
        layout_batch_size=3,
        table_batch_size=1,
        table_former_mode="accurate",
    )._pipeline_options
    assert options.layout_batch_size == 3
    assert options.table_batch_size == 1
    assert options.table_structure_options.mode == TableFormerMode.ACCURATE


def test_pipeline_wires_hardware_knobs_into_extractor():
    pytest.importorskip("docling")
    from docling.datamodel.pipeline_options import TableFormerMode

    from RAG_Agent.infrastructure.ingestion.ingest_profile import IngestHardwareProfile
    from RAG_Agent.infrastructure.ingestion.native_pdf_pipeline import NativePdfPipeline

    hardware = IngestHardwareProfile(
        pages_per_shard=7,
        max_file_size_mb=12,
        layout_batch_size=3,
        table_batch_size=1,
        table_former_mode="accurate",
    )
    pipeline = NativePdfPipeline(DefaultProfileResolver(), hardware=hardware)
    extractor = pipeline._extractor
    options = extractor._pipeline_options
    assert extractor._max_pages == 7
    assert extractor._max_file_size_bytes == 12 * 1024 * 1024
    assert options.layout_batch_size == 3
    assert options.table_batch_size == 1
    assert options.table_structure_options.mode == TableFormerMode.ACCURATE
