"""ADK session service: Postgres (local / Cloud SQL) or InMemory fallback."""

from __future__ import annotations

import logging
from urllib.parse import quote_plus

from google.adk.sessions import InMemorySessionService
from google.adk.sessions.base_session_service import BaseSessionService

from RAG_Agent.config import settings

logger = logging.getLogger(__name__)


def resolve_sessions_db_url() -> str | None:
    """SQLAlchemy URL for ADK ``DatabaseSessionService``.

    Two paths, one helper:
    - Local / Compose: ``SESSIONS_DB_URL`` set → use as-is.
    - Cloud Run: URL not set; build from ``DB_PASS`` + ``CLOUDSQL_INSTANCE``
      (Unix socket via Cloud SQL Auth Proxy). Password is URL-encoded.
    """
    if settings.sessions_db_url:
        return settings.sessions_db_url.strip()

    if not settings.cloudsql_instance:
        return None
    if not settings.db_pass:
        raise ValueError(
            "DB_PASS is required when CLOUDSQL_INSTANCE is set and SESSIONS_DB_URL is empty"
        )

    db_user = (settings.db_user or "app").strip()
    db_pass = quote_plus(settings.db_pass.strip())
    db_name = (settings.sessions_db_name or "sessions").strip()
    instance = settings.cloudsql_instance.strip()
    return f"postgresql+asyncpg://{db_user}:{db_pass}@/{db_name}?host=/cloudsql/{instance}"


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
