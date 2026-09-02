from dataclasses import replace

from RAG_Agent.domain.value_objects.block import Block, BlockType, BoundingBox, LayoutSpan, TableData
from RAG_Agent.domain.value_objects.table_groups import (
    attach_table_captions,
    looks_like_table_caption,
    refine_table_blocks,
)


def _bbox(*, y0: float, y1: float, x0: float = 50.0, x1: float = 540.0) -> BoundingBox:
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _table(
    block_id: str,
    *,
    headers: list[str],
    rows: list[list[str]],
    order: int,
    page: int = 1,
    bbox: BoundingBox | None = None,
    source_ref: str | None = None,
    caption: str | None = None,
) -> Block:
    return Block(
        id=block_id,
        type=BlockType.TABLE,
        order=order,
        page=page,
        table=TableData(headers=headers, rows=rows, caption=caption),
        bbox=bbox,
        source_ref=source_ref,
    )


def _paragraph(block_id: str, text: str, *, order: int, page: int = 1) -> Block:
    return Block(
        id=block_id,
        type=BlockType.PARAGRAPH,
        order=order,
        page=page,
        text=text,
    )


def test_looks_like_table_caption_matches_numbered_labels():
    assert looks_like_table_caption(
        "Table 4.1-1 O-CU/O-DU priority definition based on circularity economy effect"
    )
    assert looks_like_table_caption("Table 4-2 Priority level definition")
    assert looks_like_table_caption("Table 4.2-1 O-RU priority")
    assert not looks_like_table_caption("Table of contents")
    assert not looks_like_table_caption("See the table below.")


def test_attach_drops_caption_between_split_table_parts():
    """Docling often emits: first fragment, caption, continuation on next page."""
    headers = ["Circularity aspects", "Circularity indicator", "Priority"]
    caption = "Table 4.1-1 O-CU/O-DU priority definition based on circularity economy effect"
    blocks = [
        _table(
            "t0",
            headers=headers,
            rows=[["Product Durability", "Software and data support", "1"]],
            order=0,
            page=13,
        ),
        _paragraph("cap", caption, order=1, page=13),
        _table(
            "t1",
            headers=["0", "1", "2"],
            rows=[["", "Recycled metals", "1"]],
            order=2,
            page=14,
        ),
        _paragraph("prose", "Following section prose.", order=3, page=14),
    ]

    result = attach_table_captions(blocks)

    tables = [block for block in result if block.type == BlockType.TABLE]
    texts = [block.text for block in result if block.type == BlockType.PARAGRAPH]
    assert len(tables) == 2
    assert tables[0].table is not None
    assert tables[0].table.caption == caption
    assert tables[1].table is not None
    assert tables[1].table.caption is None
    assert caption not in texts
    assert texts == ["Following section prose."]


def test_attach_caption_immediately_before_first_fragment():
    headers = ["Circularity aspects", "Circularity indicator", "Priority"]
    caption = "Table 4.2-1 O-RU priority definition based on circularity economy effect"
    blocks = [
        _paragraph("cap", caption, order=0, page=14),
        _table(
            "t0",
            headers=headers,
            rows=[["Product Durability", "Scratch resistance", "4"]],
            order=1,
            page=14,
        ),
        _table(
            "t1",
            headers=["0", "1", "2"],
            rows=[["", "Packaging recycling", "2"]],
            order=2,
            page=15,
        ),
    ]

    result = attach_table_captions(blocks)

    assert all(block.type != BlockType.PARAGRAPH for block in result)
    assert result[0].type == BlockType.TABLE
    assert result[0].table is not None
    assert result[0].table.caption == caption


def test_attach_caption_after_standalone_table_prefers_following_paragraph():
    """If both a junk 'Table …' line and the real caption exist, keep the one after."""
    real = "Table 4-2 Priority level definition based on contribution effect to circular economy"
    blocks = [
        _paragraph("junk", "Table 2 -Priority level -consist of a level of relevance", order=0),
        _table(
            "t0",
            headers=["Level", "Priority Level", "Description"],
            rows=[["1", "Very HIGH", "…"]],
            order=1,
        ),
        _paragraph("cap", real, order=2),
        _paragraph("prose", "Next heading follows.", order=3),
    ]

    result = attach_table_captions(blocks)

    assert result[0].id == "junk"
    assert result[1].type == BlockType.TABLE
    assert result[1].table is not None
    assert result[1].table.caption == real
    assert result[2].text == "Next heading follows."


def test_attach_does_not_consume_following_prose():
    blocks = [
        _table("t0", headers=["A", "B"], rows=[["1", "2"]], order=0),
        _paragraph("p", "In terms of network equipment circularity aspects…", order=1),
    ]

    result = attach_table_captions(blocks)

    assert len(result) == 2
    assert result[0].table is not None
    assert result[0].table.caption is None
    assert result[1].id == "p"


def test_attach_drops_heading_misclassified_as_caption():
    caption = "Table 4.1-1 O-CU/O-DU priority definition"
    blocks = [
        _table("t0", headers=["A", "B"], rows=[["1", "2"]], order=0),
        Block(
            id="cap",
            type=BlockType.HEADING,
            order=1,
            page=1,
            text=caption,
            level=1,
        ),
        _table(
            "t1",
            headers=["0", "1"],
            rows=[["3", "4"]],
            order=2,
            page=2,
        ),
    ]

    result = attach_table_captions(blocks)

    assert all(block.id != "cap" for block in result)
    assert result[0].table is not None
    assert result[0].table.caption == caption
    assert len([block for block in result if block.type == BlockType.TABLE]) == 2


def test_refine_merges_split_table_into_one_block_with_layout_spans():
    headers = ["Circularity aspects", "Circularity indicator", "Priority"]
    caption = "Table 4.1-1 O-CU/O-DU priority definition based on circularity economy effect"
    bbox_p13 = _bbox(y0=74.0, y1=237.0)
    bbox_p14 = _bbox(y0=388.0, y1=763.0)
    blocks = [
        _table(
            "t0",
            headers=headers,
            rows=[
                ["Product Durability", "Software and data support", "1"],
                ["Product Durability", "Diagnostic support", "2"],
            ],
            order=0,
            page=13,
            bbox=bbox_p13,
            source_ref="#/tables/3",
        ),
        _paragraph("cap", caption, order=1, page=13),
        _table(
            "t1",
            headers=["0", "1", "2"],
            rows=[
                ["", "Recycled metals", "1"],
                ["Recycle, repair, reuse, upgrade - Manufacturer level", "Service offered", "1"],
                ["", "Spare parts availability", "1"],
            ],
            order=2,
            page=14,
            bbox=bbox_p14,
            source_ref="#/tables/4",
        ),
        _paragraph("prose", "Following section prose.", order=3, page=14),
    ]

    result = refine_table_blocks(blocks)
    tables = [block for block in result if block.type == BlockType.TABLE]

    assert len(tables) == 1
    table_block = tables[0]
    assert table_block.table is not None
    assert table_block.table.caption == caption
    assert table_block.page == 13
    assert table_block.bbox == bbox_p13
    assert table_block.source_ref == "#/tables/3"
    assert table_block.metadata["page_end"] == "14"
    assert table_block.metadata["merged_parts"] == "2"
    assert table_block.table.rows == [
        ["Product Durability", "Software and data support", "1"],
        ["Product Durability", "Diagnostic support", "2"],
        ["Product Durability", "Recycled metals", "1"],
        ["Recycle, repair, reuse, upgrade - Manufacturer level", "Service offered", "1"],
        ["Recycle, repair, reuse, upgrade - Manufacturer level", "Spare parts availability", "1"],
    ]

    spans = table_block.pdf_layout()
    assert len(spans) == 2
    assert spans[0].page == 13
    assert spans[0].bbox == bbox_p13
    assert spans[0].source_ref == "#/tables/3"
    assert spans[1].page == 14
    assert spans[1].bbox == bbox_p14
    assert spans[1].source_ref == "#/tables/4"
    assert table_block.page_numbers() == (13, 14)
    assert [block.text for block in result if block.type == BlockType.PARAGRAPH] == [
        "Following section prose."
    ]


def test_refine_merges_caption_before_first_fragment():
    caption = "Table 4.2-1 O-RU priority definition based on circularity economy effect"
    result = refine_table_blocks(
        [
            _paragraph("cap", caption, order=0, page=14),
            _table(
                "t0",
                headers=["A", "B", "C"],
                rows=[["Product Durability", "Scratch resistance", "4"]],
                order=1,
                page=14,
                bbox=_bbox(y0=60.0, y1=270.0),
            ),
            _table(
                "t1",
                headers=["0", "1", "2"],
                rows=[["", "Packaging recycling", "2"]],
                order=2,
                page=15,
                bbox=_bbox(y0=430.0, y1=760.0),
            ),
        ]
    )

    assert len(result) == 1
    assert result[0].type == BlockType.TABLE
    assert result[0].table is not None
    assert result[0].table.caption == caption
    assert result[0].table.rows[1][0] == "Product Durability"
    assert result[0].page_numbers() == (14, 15)


def test_refine_leaves_standalone_table_without_layout_spans():
    blocks = [
        _table("t0", headers=["A", "B"], rows=[["1", "2"]], order=0, page=8),
        _paragraph("p", "In terms of network equipment circularity aspects…", order=1),
    ]

    result = refine_table_blocks(blocks)

    assert len(result) == 2
    assert result[0].layout_spans == ()
    assert result[0].page_numbers() == (8,)
    assert result[0].pdf_layout()[0].page == 8


def test_page_numbers_fills_metadata_page_end_without_spans():
    block = Block(
        id="c",
        type=BlockType.CODE,
        order=0,
        page=3,
        text="@startuml",
        metadata={"page_end": "5", "continued": "true"},
    )
    assert block.page_numbers() == (3, 4, 5)


def test_refine_does_not_merge_placeholder_tables_on_same_page():
    blocks = [
        _table(
            "t0",
            headers=["0", "1"],
            rows=[["a", "b"]],
            order=0,
            page=8,
            bbox=_bbox(y0=100, y1=200),
        ),
        _table(
            "t1",
            headers=["0", "1"],
            rows=[["c", "d"]],
            order=1,
            page=8,
            bbox=_bbox(y0=400, y1=500),
        ),
    ]
    tables = [block for block in refine_table_blocks(blocks) if block.type == BlockType.TABLE]
    assert len(tables) == 2


def test_refine_does_not_merge_when_column_count_drifts():
    blocks = [
        _table(
            "t0",
            headers=["A", "B", "C"],
            rows=[["1", "2", "3"]],
            order=0,
            page=13,
            bbox=_bbox(y0=70, y1=200),
        ),
        _table(
            "t1",
            headers=["0", "1", "2", "3", "4"],
            rows=[["a", "b", "c", "d", "e"]],
            order=1,
            page=14,
            bbox=_bbox(y0=380, y1=760),
        ),
    ]
    tables = [block for block in refine_table_blocks(blocks) if block.type == BlockType.TABLE]
    assert len(tables) == 2


def test_refine_stitches_narrow_wrap_into_last_row():
    guidance = (
        "a) Initiate the use of automated methods to manage the configuration "
        "and lifecycle of resources within specific networks or environments. "
        "b) Ensure that all resources have a defined lifetime determined by "
        "policies and telemetry data."
    )
    comments = (
        "Network resource configuration and lifecycle management is out of "
        "scope for O-RAN specifications."
    )
    result = refine_table_blocks(
        [
            _table(
                "t0",
                headers=["CISA ZT Function", "Guidance", "Asset(s)", "Comments"],
                rows=[
                    [
                        "Automation and Orchestration Capability",
                        "a) Initiate the use of automated methods",
                        "ASSET-C-40",
                        "Network resource configuration",
                    ]
                ],
                order=0,
                page=184,
                bbox=_bbox(y0=56, y1=556),
                caption="Table 11.2.3-1: E2 Interface Identified Gaps",
            ),
            _table(
                "t1",
                headers=[],
                rows=[
                    [
                        "to manage the configuration and lifecycle of resources "
                        "within specific networks or environments. b) Ensure "
                        "that all resources have a defined lifetime determined "
                        "by policies and telemetry data.",
                        "and lifecycle management is out of scope for O-RAN "
                        "specifications.",
                    ]
                ],
                order=1,
                page=185,
                bbox=_bbox(y0=634, y1=764),
            ),
        ]
    )
    tables = [block for block in result if block.type == BlockType.TABLE]
    assert len(tables) == 1
    table = tables[0].table
    assert table is not None
    assert table.caption == "Table 11.2.3-1: E2 Interface Identified Gaps"
    assert table.rows[-1][0] == "Automation and Orchestration Capability"
    assert table.rows[-1][1] == guidance
    assert table.rows[-1][2] == "ASSET-C-40"
    assert table.rows[-1][3] == comments
    assert tables[0].metadata["merged_parts"] == "2"


def test_refine_merges_header_drift_on_next_page():
    result = refine_table_blocks(
        [
            _table(
                "t0",
                headers=[
                    "Asset ID Data & Interfaces",
                    "Asset Description Data & Interfaces",
                ],
                rows=[["ASSET-D-X", "The F1-c interface reuses the"]],
                order=0,
                page=30,
                bbox=_bbox(y0=64, y1=200),
                caption="Table 5.3.4-1: F1 Assets",
            ),
            _table(
                "t1",
                headers=["Asset ID", "Asset Description"],
                rows=[["", "principles and protocol stack defined by 3GPP."]],
                order=1,
                page=31,
                bbox=_bbox(y0=400, y1=760),
            ),
        ]
    )
    tables = [block for block in result if block.type == BlockType.TABLE]
    assert len(tables) == 1
    table = tables[0].table
    assert table is not None
    assert table.headers == ["Asset ID", "Asset Description"]
    assert table.rows == [
        [
            "ASSET-D-X",
            "The F1-c interface reuses the principles and protocol stack defined by 3GPP.",
        ]
    ]
    assert table.caption == "Table 5.3.4-1: F1 Assets"


def test_refine_does_not_merge_when_next_table_has_caption():
    blocks = [
        _table(
            "t0",
            headers=["A", "B"],
            rows=[["1", "2"]],
            order=0,
            page=10,
            bbox=_bbox(y0=70, y1=200),
            caption="Table 1-1: First",
        ),
        _table(
            "t1",
            headers=["A", "B"],
            rows=[["3", "4"]],
            order=1,
            page=11,
            bbox=_bbox(y0=400, y1=700),
            caption="Table 1-2: Second",
        ),
    ]
    tables = [block for block in refine_table_blocks(blocks) if block.type == BlockType.TABLE]
    assert len(tables) == 2
    assert tables[1].table is not None
    assert tables[1].table.caption == "Table 1-2: Second"


def test_refine_merges_after_already_spanned_table():
    bbox_p64 = _bbox(y0=70, y1=200)
    bbox_p65 = _bbox(y0=400, y1=760)
    bbox_p66 = _bbox(y0=500, y1=760)
    first = replace(
        _table(
            "t0",
            headers=["A", "B", "C"],
            rows=[["keep", "long running guidance that does not finish", "1"]],
            order=0,
            page=64,
            bbox=bbox_p64,
            caption="Table 7.3.3-1: xApp Data Pillar Gap Analysis",
        ),
        metadata={"merged_parts": "2", "page_end": "65", "continued": "true"},
        layout_spans=(
            LayoutSpan(page=64, bbox=bbox_p64, source_ref="#/tables/1"),
            LayoutSpan(page=65, bbox=bbox_p65, source_ref="#/tables/2"),
        ),
    )
    result = refine_table_blocks(
        [
            first,
            _table(
                "t1",
                headers=["0", "1", "2"],
                rows=[["", "on the next folio.", "1"]],
                order=1,
                page=66,
                bbox=bbox_p66,
                source_ref="#/tables/3",
            ),
        ]
    )
    tables = [block for block in result if block.type == BlockType.TABLE]
    assert len(tables) == 1
    assert [span.page for span in tables[0].layout_spans] == [64, 65, 66]
    assert tables[0].metadata["page_end"] == "66"
    assert tables[0].metadata["merged_parts"] == "3"


def test_refine_merges_when_continuation_caption_is_note():
    result = refine_table_blocks(
        [
            _table(
                "t0",
                headers=["Function", "Guidance", "Comments"],
                rows=[
                    [
                        "Visibility",
                        "b) Commence correlating telemetry across",
                        "SIEM collection is",
                    ]
                ],
                order=0,
                page=133,
                bbox=_bbox(y0=70, y1=200),
                caption="Table 7.3.11-4: O-Cloud Networks Pillar Gap Analysis",
            ),
            _table(
                "t1",
                headers=[],
                rows=[
                    [
                        "traffic types and environments to facilitate analysis.",
                        "considered out of scope for O-RAN.",
                    ]
                ],
                order=1,
                page=134,
                bbox=_bbox(y0=500, y1=760),
                caption=(
                    "NOTE 1: Encryption of data in use is considered as Optimal stage "
                    "according to O-RAN ZTA Security Requirements for Data Encryption."
                ),
            ),
        ]
    )
    tables = [block for block in result if block.type == BlockType.TABLE]
    assert len(tables) == 1
    table = tables[0].table
    assert table is not None
    assert table.caption is not None
    assert table.caption.startswith("Table 7.3.11-4")
    assert "NOTE 1:" in table.caption
    assert "traffic types and environments" in table.rows[-1][1]


def test_attach_leaves_following_caption_when_table_already_labelled():
    first_caption = "Table 11.2.3-1: E2 Interface"
    second_caption = "Table 11.2.4-1: Y1 Interface"
    result = attach_table_captions(
        [
            _table(
                "t0",
                headers=["A", "B"],
                rows=[["1", "2"]],
                order=0,
                page=184,
                caption=first_caption,
            ),
            _paragraph("cap", second_caption, order=1, page=185),
            _table(
                "t1",
                headers=["A", "B"],
                rows=[["3", "4"]],
                order=2,
                page=185,
            ),
        ]
    )
    tables = [block for block in result if block.type == BlockType.TABLE]
    assert len(tables) == 2
    assert tables[0].table is not None
    assert tables[0].table.caption == first_caption
    assert tables[1].table is not None
    assert tables[1].table.caption == second_caption
    assert all(block.text != second_caption for block in result if block.type == BlockType.PARAGRAPH)


def test_refine_merges_when_real_headers_match_on_next_page():
    result = refine_table_blocks(
        [
            _table(
                "t0",
                headers=["Asset ID", "Description"],
                rows=[["A-1", "First row."]],
                order=0,
                page=30,
                bbox=_bbox(y0=70, y1=200),
                caption="Table 5.3.4-1: F1 Assets",
            ),
            _table(
                "t1",
                headers=["Asset ID", "Description"],
                rows=[["A-2", "Second row."]],
                order=1,
                page=31,
                bbox=_bbox(y0=400, y1=760),
            ),
        ]
    )
    tables = [block for block in result if block.type == BlockType.TABLE]
    assert len(tables) == 1
    assert tables[0].table is not None
    assert tables[0].table.rows == [["A-1", "First row."], ["A-2", "Second row."]]


def test_refine_merges_narrower_fragment_with_tail_headers():
    result = refine_table_blocks(
        [
            _table(
                "t0",
                headers=["Function", "Guidance", "Asset(s)", "Req", "Gap", "Comments"],
                rows=[
                    [
                        "Automation and Orchestration Capability",
                        "Data lifecycle and security policies are automated.",
                        "ASSET-D-06",
                        "None",
                        "Gap",
                        "Out of scope, as data lifecycle and security",
                    ]
                ],
                order=0,
                page=162,
                bbox=_bbox(y0=76, y1=760),
                caption="Table 7.3.15-1: O-RU Data Pillar Gap Analysis",
            ),
            _table(
                "t1",
                headers=["", "", "", "policies fall under the operator's responsibilities."],
                rows=[
                    [
                        "ASSET-D-16",
                        "None",
                        "Gap",
                        "Out of scope, as data lifecycle and security policies fall under the operator's responsibilities.",
                    ]
                ],
                order=1,
                page=163,
                bbox=_bbox(y0=305, y1=764),
            ),
        ]
    )
    tables = [block for block in result if block.type == BlockType.TABLE]
    assert len(tables) == 1
    table = tables[0].table
    assert table is not None
    assert table.rows[0][5].endswith("operator's responsibilities.")
    assert table.rows[1][2] == "ASSET-D-16"
    assert table.rows[1][0] == "Automation and Orchestration Capability"
