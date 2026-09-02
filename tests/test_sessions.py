from urllib.parse import quote_plus

import pytest
from google.adk.sessions import InMemorySessionService

from RAG_Agent.config import settings
from RAG_Agent.infrastructure.agent.sessions import (
    build_session_service,
    cloudsql_connection_name,
    resolve_sessions_db_url,
)


def test_resolve_sessions_db_url_prefers_explicit(monkeypatch):
    monkeypatch.setattr(settings, "sessions_db_url", "postgresql+asyncpg://u:p@h/db")
    monkeypatch.setattr(settings, "google_cloud_project", "proj")
    monkeypatch.setattr(settings, "region", "eu")
    monkeypatch.setattr(settings, "instance_name", "inst")
    monkeypatch.setattr(settings, "db_pass", "ignored")
    assert resolve_sessions_db_url() == "postgresql+asyncpg://u:p@h/db"


def test_cloudsql_connection_name_joins_parts(monkeypatch):
    monkeypatch.setattr(settings, "google_cloud_project", "proj")
    monkeypatch.setattr(settings, "region", "us-central1")
    monkeypatch.setattr(settings, "instance_name", "agent-sessions")
    assert cloudsql_connection_name() == "proj:us-central1:agent-sessions"


def test_cloudsql_connection_name_none_if_any_part_missing(monkeypatch):
    monkeypatch.setattr(settings, "google_cloud_project", "proj")
    monkeypatch.setattr(settings, "region", "us-central1")
    monkeypatch.setattr(settings, "instance_name", None)
    assert cloudsql_connection_name() is None


def test_resolve_sessions_db_url_cloud_sql_encodes_password(monkeypatch):
    monkeypatch.setattr(settings, "sessions_db_url", None)
    monkeypatch.setattr(settings, "google_cloud_project", "proj")
    monkeypatch.setattr(settings, "region", "region")
    monkeypatch.setattr(settings, "instance_name", "agent-sessions")
    monkeypatch.setattr(settings, "db_user", "app")
    monkeypatch.setattr(settings, "db_pass", "a/b+c=d")
    monkeypatch.setattr(settings, "sessions_db_name", "sessions")
    url = resolve_sessions_db_url()
    assert url is not None
    assert quote_plus("a/b+c=d") in url
    assert url.endswith("?host=/cloudsql/proj:region:agent-sessions")
    assert url.startswith("postgresql+asyncpg://app:")


def test_resolve_sessions_db_url_requires_password_when_instance_set(monkeypatch):
    monkeypatch.setattr(settings, "sessions_db_url", None)
    monkeypatch.setattr(settings, "google_cloud_project", "proj")
    monkeypatch.setattr(settings, "region", "region")
    monkeypatch.setattr(settings, "instance_name", "agent-sessions")
    monkeypatch.setattr(settings, "db_pass", None)
    with pytest.raises(ValueError, match="DB_PASS"):
        resolve_sessions_db_url()


def test_resolve_sessions_db_url_none_without_config(monkeypatch):
    monkeypatch.setattr(settings, "sessions_db_url", None)
    monkeypatch.setattr(settings, "instance_name", None)
    monkeypatch.setattr(settings, "db_pass", None)
    assert resolve_sessions_db_url() is None


def test_build_session_service_falls_back_to_memory(monkeypatch):
    monkeypatch.setattr(settings, "sessions_db_url", None)
    monkeypatch.setattr(settings, "instance_name", None)
    service = build_session_service()
    assert isinstance(service, InMemorySessionService)
