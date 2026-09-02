from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from RAG_Agent.config import settings

TableFormerModeName = Literal["accurate", "fast"]

MAX_FILE_SIZE_MB = 200
TABLE_FORMER_MODE: TableFormerModeName = "accurate"


@dataclass(frozen=True)
class IngestHardwareProfile:
    """Operational parse limits (hardware), not document-family rules.

    Args:
        pages_per_shard: Max pages per Docling ``convert`` call.
        max_file_size_mb: Whole-file reject ceiling, not a shard slicer.
        layout_batch_size: Pages per layout-model forward pass.
        table_batch_size: Tables per TableFormer forward pass.
        table_former_mode: Docling ``TableFormerMode`` value.
    """

    pages_per_shard: int
    max_file_size_mb: int
    layout_batch_size: int = 4
    table_batch_size: int = 2
    table_former_mode: TableFormerModeName = "accurate"


def ingest_hardware() -> IngestHardwareProfile:
    """Docling knobs from ``Settings`` (env), plus fixed file-size / table mode."""
    return IngestHardwareProfile(
        pages_per_shard=settings.ingest_pages_per_shard,
        max_file_size_mb=MAX_FILE_SIZE_MB,
        layout_batch_size=settings.ingest_layout_batch_size,
        table_batch_size=settings.ingest_table_batch_size,
        table_former_mode=TABLE_FORMER_MODE,
    )
