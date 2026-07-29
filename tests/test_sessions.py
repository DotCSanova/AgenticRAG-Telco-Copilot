from urllib.parse import quote_plus

from google.adk.sessions import InMemorySessionService

from RAG_Agent.config import settings
from RAG_Agent.infrastructure.agent.sessions import build_session_service, resolve_sessions_db_url


def test_resolve_sessions_db_url_prefers_explicit(monkeypatch):
    monkeypatch.setattr(settings, "sessions_db_url", "postgresql+asyncpg://u:p@h/db")
    monkeypatch.setattr(settings, "cloudsql_instance", "proj:eu:inst")
    monkeypatch.setattr(settings, "db_pass", "ignored")
    assert resolve_sessions_db_url() == "postgresql+asyncpg://u:p@h/db"


def test_resolve_sessions_db_url_cloud_sql_encodes_password(monkeypatch):
    monkeypatch.setattr(settings, "sessions_db_url", None)
    monkeypatch.setattr(settings, "db_user", "app")
    monkeypatch.setattr(settings, "db_pass", "a/b+c=d")
    monkeypatch.setattr(settings, "sessions_db_name", "sessions")
    monkeypatch.setattr(settings, "cloudsql_instance", "proj:region:agent-sessions")
    url = resolve_sessions_db_url()
    assert url is not None
    assert quote_plus("a/b+c=d") in url
    assert url.endswith("?host=/cloudsql/proj:region:agent-sessions")
    assert url.startswith("postgresql+asyncpg://app:")


def test_resolve_sessions_db_url_none_without_config(monkeypatch):
    monkeypatch.setattr(settings, "sessions_db_url", None)
    monkeypatch.setattr(settings, "cloudsql_instance", None)
    monkeypatch.setattr(settings, "db_pass", None)
    assert resolve_sessions_db_url() is None


def test_build_session_service_falls_back_to_memory(monkeypatch):
    monkeypatch.setattr(settings, "sessions_db_url", None)
    monkeypatch.setattr(settings, "cloudsql_instance", None)
    service = build_session_service()
    assert isinstance(service, InMemorySessionService)
