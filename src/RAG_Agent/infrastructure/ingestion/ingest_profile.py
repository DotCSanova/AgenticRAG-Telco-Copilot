from __future__ import annotations

from dataclasses import dataclass, replace

from RAG_Agent.config import settings


@dataclass(frozen=True)
class IngestHardwareProfile:
    """Límites operativos del parse (hardware), no reglas de documento."""

    name: str
    pages_per_shard: int
    max_file_size_mb: int


LOCAL = IngestHardwareProfile(
    name="local",
    pages_per_shard=50,
    max_file_size_mb=200,
)

_PROFILES: dict[str, IngestHardwareProfile] = {
    "local": LOCAL,
}


def get_ingest_profile(
    name: str | None = None,
    *,
    pages_per_shard: int | None = None,
) -> IngestHardwareProfile:
    """Resuelve el perfil de ingest. Hoy solo ``local``; cloud se añade aquí."""
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
