"""ADK session service: Postgres (local / Cloud SQL) or InMemory fallback."""

from __future__ import annotations

import logging
from urllib.parse import quote_plus

from google.adk.sessions import InMemorySessionService
from google.adk.sessions.base_session_service import BaseSessionService

from RAG_Agent.config import settings

logger = logging.getLogger(__name__)


def cloudsql_connection_name() -> str | None:
    """Return ``project:region:instance`` when all three pieces are set."""
    project = (settings.google_cloud_project or "").strip()
    region = (settings.region or "").strip()
    instance = (settings.instance_name or "").strip()
    if not project or not region or not instance:
        return None
    return f"{project}:{region}:{instance}"


def resolve_sessions_db_url() -> str | None:
    """SQLAlchemy URL for ADK ``DatabaseSessionService``.

    Two paths, one helper:
    - Local / Compose: ``SESSIONS_DB_URL`` set → use as-is.
    - Cloud Run: URL not set; build from ``DB_PASS`` +
      ``GOOGLE_CLOUD_PROJECT``:``REGION``:``INSTANCE_NAME``
      (Unix socket via Cloud SQL Auth Proxy). Password is URL-encoded.
    """
    if settings.sessions_db_url:
        return settings.sessions_db_url.strip()

    connection = cloudsql_connection_name()
    if not connection:
        return None
    if not settings.db_pass:
        raise ValueError(
            "DB_PASS is required when GOOGLE_CLOUD_PROJECT, REGION, and "
            "INSTANCE_NAME are set and SESSIONS_DB_URL is empty"
        )

    db_user = (settings.db_user or "app").strip()
    db_pass = quote_plus(settings.db_pass.strip())
    return f"postgresql+asyncpg://{db_user}:{db_pass}@/sessions?host=/cloudsql/{connection}"


def build_session_service() -> BaseSessionService:
    """Postgres via ADK DatabaseSessionService, or InMemory if no URL resolved."""
    url = resolve_sessions_db_url()
    if url is None:
        logger.info("Session backend=InMemorySessionService (no SESSIONS_DB_URL / Cloud SQL)")
        return InMemorySessionService()

    from google.adk.sessions.database_session_service import DatabaseSessionService

    redacted = url.split("@")[-1] if "@" in url else url
    logger.info("Session backend=DatabaseSessionService (%s)", redacted)
    return DatabaseSessionService(db_url=url)
