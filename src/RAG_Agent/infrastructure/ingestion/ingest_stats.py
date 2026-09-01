"""Parsing-quality stats for ingest logs and canary tests."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from RAG_Agent.domain.value_objects.block import BlockType
from RAG_Agent.domain.value_objects.canonical_document import CanonicalDocument

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestStats:
    """Counts taken from a canonical document after parse (before chunk/index)."""

    num_pages: int
    num_blocks: int
    num_sections: int
    num_tables: int
    empty_pages: int
    degraded_blocks: int
    failed_shards: int
    blocks_by_type: dict[str, int]
    docling_version: str | None
    title: str | None


def collect_ingest_stats(document: CanonicalDocument) -> IngestStats:
    """Derive parse-quality counts from ``document``.

    A TABLE with no rows is counted as degraded (mapper warning path). Pages
    listed with no ``block_ids`` are empty pages.

    Args:
        document: Canonical aggregate after normalize/merge.

    Returns:
        Frozen counts suitable for logs and canary assertions.
    """
    extra = document.metadata.extra
    type_counts = Counter(block.type.value for block in document.blocks.values())
    degraded = 0
    for block in document.blocks.values():
        if block.type is BlockType.TABLE and (block.table is None or not block.table.rows):
            degraded += 1

    failed_raw = extra.get("failed_shards") or ""
    failed_shards = len([part for part in failed_raw.split(",") if part.strip()]) if failed_raw else 0

    return IngestStats(
        num_pages=len(document.pages),
        num_blocks=len(document.blocks),
        num_sections=len(document.sections),
        num_tables=type_counts.get(BlockType.TABLE.value, 0),
        empty_pages=sum(1 for page in document.pages if not page.block_ids),
        degraded_blocks=degraded,
        failed_shards=failed_shards,
        blocks_by_type=dict(sorted(type_counts.items())),
        docling_version=extra.get("docling_version"),
        title=document.metadata.title,
    )


def log_ingest_stats(
    document: CanonicalDocument,
    *,
    source_path: Path,
    elapsed_s: float,
) -> IngestStats:
    """Emit one structured INFO line ``ingest_stats ...`` for Cloud Logging.

    Args:
        document: Canonical aggregate after parse.
        source_path: Original PDF path (logged as name only).
        elapsed_s: Wall time of ``parse``.

    Returns:
        The same counts that were logged.
    """
    stats = collect_ingest_stats(document)
    by_type = ",".join(f"{kind}={count}" for kind, count in stats.blocks_by_type.items())
    logger.info(
        "ingest_stats file=%s title=%r pages=%d blocks=%d sections=%d tables=%d "
        "empty_pages=%d degraded_blocks=%d failed_shards=%d by_type=%s "
        "docling_version=%s elapsed_s=%.1f",
        source_path.name,
        stats.title,
        stats.num_pages,
        stats.num_blocks,
        stats.num_sections,
        stats.num_tables,
        stats.empty_pages,
        stats.degraded_blocks,
        stats.failed_shards,
        by_type or "-",
        stats.docling_version or "-",
        elapsed_s,
    )
    return stats
