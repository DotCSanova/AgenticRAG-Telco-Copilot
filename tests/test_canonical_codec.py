from pathlib import Path

import pytest

from RAG_Agent.domain.value_objects.block import (
    Block,
    BlockType,
    BoundingBox,
    LayoutSpan,
    TableData,
    coord_origin_name,
)
from RAG_Agent.domain.value_objects.canonical_document import (
    CanonicalDocument,
    DocumentMetadata,
)
from RAG_Agent.domain.value_objects.page import Page
from RAG_Agent.domain.value_objects.section import Section


def _heading(**kwargs: object) -> Block:
    defaults: dict = {
        "id": "block_2",
        "type": BlockType.HEADING,
        "order": 2,
        "page": 4,
        "text": "Executive summary",
        "level": 1,
        "bbox": BoundingBox(56.64, 549.39, 223.69, 565.53, coord_origin="BOTTOMLEFT"),
        "source_ref": "#/texts/38",
    }
    defaults.update(kwargs)
    return Block(**defaults)


def _document(*, blocks: list[Block] | None = None, extra: dict[str, str] | None = None) -> CanonicalDocument:
    heading = (blocks or [_heading()])[0]
    return CanonicalDocument(
        metadata=DocumentMetadata(
            source_path=Path("data/O-RAN.SuFG.CE-v01.00.pdf"),
            title="Circular economy guidelines",
            profile_id="oran_default",
            parser="native_pdf_docling",
            extra=extra
            or {
                "family": "oran",
                "group": "SuFG",
                "doc_type": "ce",
                "subject": "CE",
                "version": "01.00",
                "segments": "CE",
                "num_pages": "17",
                "num_blocks": "171",
                "num_sections": "17",
                "ingest_profile": "LOCAL",
                "pages_per_shard": "50",
            },
        ),
        blocks={block.id: block for block in (blocks or [heading])},
        pages=[
            Page(number=4, block_ids=[heading.id], width=595.32, height=842.04),
        ],
        sections=[
            Section(
                id="sec_0001_executive-summary",
                title="Executive summary",
                level=1,
                order=1,
                parent_id=None,
                block_ids=[heading.id],
                page_start=4,
                page_end=4,
            )
        ],
    )


def test_heading_omits_null_payloads_and_has_schema_version():
    payload = _document().to_payload()
    block = payload["blocks"][0]

    assert payload["schema_version"] == "1.0"
    assert "table" not in block
    assert "image" not in block
    assert "layout_spans" not in block
    assert "metadata" not in block
    assert block["bbox"]["coord_origin"] == "BOTTOMLEFT"
    assert "extra" not in payload["metadata"]
    assert payload["metadata"]["identity"]["group"] == "SuFG"
    assert payload["metadata"]["family"] == "oran"
    assert payload["metadata"]["stats"]["num_pages"] == 17
    assert isinstance(payload["metadata"]["stats"]["num_pages"], int)
    assert payload["metadata"]["ingest"]["ingest_profile"] == "LOCAL"


def test_coord_origin_strips_enum_class_prefix():
    assert coord_origin_name("CoordOrigin.BOTTOMLEFT") == "BOTTOMLEFT"
    bbox = BoundingBox(1, 2, 3, 4, coord_origin="CoordOrigin.BOTTOMLEFT")
    payload = _document(blocks=[_heading(bbox=bbox)]).to_payload()
    assert payload["blocks"][0]["bbox"]["coord_origin"] == "BOTTOMLEFT"


def test_root_section_keeps_null_parent_id():
    payload = _document().to_payload()
    assert payload["sections"][0]["parent_id"] is None


def test_roundtrip_restores_extra_strings():
    original = _document()
    restored = CanonicalDocument.from_payload(original.to_payload())

    assert restored.blocks["block_2"].text == "Executive summary"
    assert restored.blocks["block_2"].type is BlockType.HEADING
    assert restored.metadata.extra["num_pages"] == "17"
    assert restored.metadata.extra["group"] == "SuFG"
    assert restored.metadata.extra["family"] == "oran"
    assert restored.metadata.extra["ingest_profile"] == "LOCAL"
    assert restored.metadata.source_path == Path("data/O-RAN.SuFG.CE-v01.00.pdf")
    assert restored.sections[0].parent_id is None


def test_from_payload_rejects_other_schema_version():
    payload = _document().to_payload()
    payload["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="schema_version"):
        CanonicalDocument.from_payload(payload)


def test_from_payload_ignores_unknown_metadata_keys():
    payload = _document().to_payload()
    payload["metadata"]["source_sha256"] = "abc"
    restored = CanonicalDocument.from_payload(payload)
    assert "source_sha256" not in restored.metadata.extra


def test_table_block_keeps_table_omits_null_text():
    table = Block(
        id="block_75",
        type=BlockType.TABLE,
        order=75,
        page=8,
        table=TableData(headers=["A", "B"], rows=[["1", "2"]]),
        bbox=BoundingBox(1, 2, 3, 4),
        source_ref="#/tables/1",
    )
    payload = _document(blocks=[table]).to_payload()
    block = payload["blocks"][0]
    assert "text" not in block
    assert "level" not in block
    assert block["table"]["headers"] == ["A", "B"]
    assert "caption" not in block["table"]


def test_redundant_layout_span_omitted():
    bbox = BoundingBox(1, 2, 3, 4)
    block = _heading(
        bbox=bbox,
        layout_spans=(LayoutSpan(page=4, bbox=bbox, source_ref="#/texts/38"),),
    )
    payload = _document(blocks=[block]).to_payload()
    assert "layout_spans" not in payload["blocks"][0]


def test_split_layout_spans_are_kept():
    bbox = BoundingBox(1, 2, 3, 4)
    other = BoundingBox(5, 6, 7, 8)
    block = _heading(
        bbox=bbox,
        layout_spans=(
            LayoutSpan(page=4, bbox=bbox, source_ref="#/texts/38"),
            LayoutSpan(page=5, bbox=other, source_ref="#/texts/39"),
        ),
    )
    payload = _document(blocks=[block]).to_payload()
    assert len(payload["blocks"][0]["layout_spans"]) == 2
    restored = CanonicalDocument.from_payload(payload)
    assert len(restored.blocks["block_2"].layout_spans) == 2
