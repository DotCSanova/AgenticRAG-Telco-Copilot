from pathlib import Path

from fastapi.testclient import TestClient

from RAG_Agent.config import settings
from RAG_Agent.infrastructure.api.main import app
from RAG_Agent.infrastructure.ingestion.cascading_profile_resolver import CascadingProfileResolver


def test_cascading_resolver_oran():
    profile = CascadingProfileResolver().resolve(
        Path("data/O-RAN.WG1.TS.Use-Cases-Detailed-Specification-R005-v19.00.pdf")
    )
    assert profile.rules.profile_id == "oran_default"
    assert profile.identity.metadata["group"] == "WG1"


def test_cascading_resolver_sufg():
    profile = CascadingProfileResolver().resolve(Path("data/O-RAN.SuFG.CE-v01.00.pdf"))
    assert profile.rules.profile_id == "oran_default"
    assert profile.identity.metadata["group"] == "SuFG"


def test_cascading_resolver_default():
    profile = CascadingProfileResolver().resolve(Path("data/some-other-doc.pdf"))
    assert profile.rules.profile_id == "default"


def test_ingest_endpoint_missing_file(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-dummy-key")
    monkeypatch.setattr(settings, "qdrant_enable_sparse", False)
    monkeypatch.setattr(settings, "qdrant_mode", "memory")
    with TestClient(app) as client:
        response = client.post("/ingest", json={"path": "data/does-not-exist.pdf"})
    assert response.status_code == 404
