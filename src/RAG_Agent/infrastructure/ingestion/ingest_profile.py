from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from RAG_Agent.config import settings

TableFormerModeName = Literal["accurate", "fast"]


@dataclass(frozen=True)
class IngestHardwareProfile:
    """Operational parse limits (hardware), not document-family rules.

    Args:
        name: Profile key (``local``, ``cloud``).
        pages_per_shard: Max pages per Docling ``convert`` call.
        max_file_size_mb: Whole-file reject ceiling, not a shard slicer.
        layout_batch_size: Pages per layout-model forward pass.
        table_batch_size: Tables per TableFormer forward pass.
        table_former_mode: Docling ``TableFormerMode`` value.
    """

    name: str
    pages_per_shard: int
    max_file_size_mb: int
    layout_batch_size: int = 4
    table_batch_size: int = 2
    table_former_mode: TableFormerModeName = "accurate"


LOCAL = IngestHardwareProfile(
    name="local",
    pages_per_shard=50,
    max_file_size_mb=200,
    layout_batch_size=4,
    table_batch_size=2,
    table_former_mode="accurate",
)

# Same page/file ceilings as local; smaller model batches for 8 Gi Cloud Run Jobs.
CLOUD = IngestHardwareProfile(
    name="cloud",
    pages_per_shard=50,
    max_file_size_mb=200,
    layout_batch_size=2,
    table_batch_size=1,
    table_former_mode="accurate",
)

_PROFILES: dict[str, IngestHardwareProfile] = {
    "local": LOCAL,
    "cloud": CLOUD,
}


def get_ingest_profile(
    name: str | None = None,
    *,
    pages_per_shard: int | None = None,
) -> IngestHardwareProfile:
    """Resolve a named hardware profile, then apply the optional page override.

    Args:
        name: Profile key. Defaults to ``settings.ingest_profile``.
        pages_per_shard: If set (or ``settings.ingest_pages_per_shard``), replaces
            ``pages_per_shard`` only. Batch sizes and table mode stay on the base
            profile.

    Returns:
        Frozen hardware limits for Docling.

    Raises:
        ValueError: Unknown name, or page override below 1.
    """
    profile_name = (name or settings.ingest_profile).strip().lower()
    try:
        base = _PROFILES[profile_name]
    except KeyError as exc:
        known = ", ".join(sorted(_PROFILES))
        msg = f"Unknown ingest profile {profile_name!r}. Known: {known}"
        raise ValueError(msg) from exc

    override = pages_per_shard if pages_per_shard is not None else settings.ingest_pages_per_shard
    if override is not None and override != base.pages_per_shard:
        if override < 1:
            msg = f"ingest_pages_per_shard must be >= 1, got {override}"
            raise ValueError(msg)
        return replace(base, pages_per_shard=override)
    return base
