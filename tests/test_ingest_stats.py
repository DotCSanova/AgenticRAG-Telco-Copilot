import logging
from pathlib import Path

from RAG_Agent.domain.value_objects.block import Block, BlockType, TableData
from RAG_Agent.domain.value_objects.canonical_document import (
    CanonicalDocument,
    DocumentMetadata,
)
from RAG_Agent.domain.value_objects.page import Page
from RAG_Agent.domain.value_objects.section import Section
from RAG_Agent.infrastructure.ingestion.ingest_stats import (
    collect_ingest_stats,
    log_ingest_stats,
)


def test_collect_ingest_stats_counts_types_and_degraded_tables():
    document = CanonicalDocument(
        metadata=DocumentMetadata(
            source_path=Path("doc.pdf"),
            title="Demo",
            extra={"docling_version": "2.93.0", "failed_shards": "1,4"},
        ),
        blocks={
            "h": Block(id="h", type=BlockType.HEADING, order=0, page=1, text="Scope", level=1),
            "t": Block(
                id="t",
                type=BlockType.TABLE,
                order=1,
                page=1,
                table=TableData(headers=["A"], rows=[["1"]]),
            ),
            "bad": Block(
                id="bad",
                type=BlockType.TABLE,
                order=2,
                page=2,
                table=TableData(headers=[], rows=[]),
            ),
        },
        pages=[
            Page(number=1, block_ids=["h", "t"]),
            Page(number=2, block_ids=["bad"]),
            Page(number=3, block_ids=[]),
        ],
        sections=[Section(id="s0", title="Scope", level=1, order=0, block_ids=["h", "t"])],
    )
    stats = collect_ingest_stats(document)
    assert stats.num_pages == 3
    assert stats.num_blocks == 3
    assert stats.num_sections == 1
    assert stats.num_tables == 2
    assert stats.empty_pages == 1
    assert stats.degraded_blocks == 1
    assert stats.failed_shards == 2
    assert stats.blocks_by_type == {"heading": 1, "table": 2}
    assert stats.docling_version == "2.93.0"
    assert stats.title == "Demo"


def test_log_ingest_stats_emits_ingest_stats_line(caplog):
    document = CanonicalDocument(
        metadata=DocumentMetadata(source_path=Path("doc.pdf"), title="Demo"),
        blocks={
            "h": Block(id="h", type=BlockType.HEADING, order=0, page=1, text="Scope", level=1),
        },
        pages=[Page(number=1, block_ids=["h"])],
    )
    stats_logger = logging.getLogger("RAG_Agent.infrastructure.ingestion.ingest_stats")
    stats_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger=stats_logger.name):
            log_ingest_stats(document, source_path=Path("doc.pdf"), elapsed_s=1.25)
    finally:
        stats_logger.removeHandler(caplog.handler)
    assert "ingest_stats" in caplog.text
    assert "file=doc.pdf" in caplog.text
    assert "blocks=1" in caplog.text
