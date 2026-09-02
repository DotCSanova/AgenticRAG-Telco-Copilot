"""Docling canaries. Skip when the PDF is not in the working tree.

Smoke (SuFG.CE, 17 pages) is ``integration`` so default fast CI without the
file stays green. The 468-page WG1 TS canary also needs ``RAG_INGEST_CANARY=1``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from RAG_Agent.infrastructure.ingestion.cascading_profile_resolver import (
    CascadingProfileResolver,
)
from RAG_Agent.infrastructure.ingestion.ingest_stats import collect_ingest_stats
from RAG_Agent.infrastructure.ingestion.native_pdf_pipeline import NativePdfPipeline

_REPO = Path(__file__).resolve().parents[1]
_SMOKE_PDF = _REPO / "data" / "O-RAN.SuFG.CE-v01.00.pdf"
_CANARY_PDF = _REPO / "data" / "O-RAN.WG1.TS.Use-Cases-Detailed-Specification-R005-v19.00.pdf"

_SMOKE_TITLE = (
    "Technical Report O-RAN Focus Group (Sustainability Focus Group) "
    "Circular economy guidelines on network equipment"
)
_SMOKE_PAGES = 17
_SMOKE_BLOCKS = 199
_SMOKE_SECTIONS = 17


def _within_percent(actual: int, expected: int, *, rel: float = 0.05) -> bool:
    tolerance = max(1, round(expected * rel))
    return abs(actual - expected) <= tolerance


def _parse(path: Path):
    pytest.importorskip("docling")
    pytest.importorskip("fitz")
    return NativePdfPipeline(CascadingProfileResolver()).parse(path)


@pytest.mark.integration
def test_smoke_sufg_ce_canonical_baseline():
    if not _SMOKE_PDF.is_file():
        pytest.skip(f"smoke PDF not present: {_SMOKE_PDF}")

    document = _parse(_SMOKE_PDF)
    stats = collect_ingest_stats(document)

    assert stats.title == _SMOKE_TITLE
    assert stats.num_pages == _SMOKE_PAGES
    assert stats.docling_version
    assert stats.failed_shards == 0
    assert _within_percent(stats.num_blocks, _SMOKE_BLOCKS), (
        f"blocks {stats.num_blocks} outside ±5% of {_SMOKE_BLOCKS}"
    )
    assert _within_percent(stats.num_sections, _SMOKE_SECTIONS), (
        f"sections {stats.num_sections} outside ±5% of {_SMOKE_SECTIONS}"
    )


@pytest.mark.integration
def test_canary_wg1_ts_use_cases_detailed():
    if os.environ.get("RAG_INGEST_CANARY") != "1":
        pytest.skip("set RAG_INGEST_CANARY=1 to run the 468-page WG1 TS canary")
    if not _CANARY_PDF.is_file():
        pytest.skip(f"canary PDF not present: {_CANARY_PDF}")

    document = _parse(_CANARY_PDF)
    stats = collect_ingest_stats(document)

    assert stats.num_pages == 468
    assert stats.title
    assert stats.docling_version
    assert stats.num_blocks > 0
    assert stats.num_sections > 0
    assert stats.failed_shards == 0
