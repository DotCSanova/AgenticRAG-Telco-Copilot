from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from RAG_Agent.application.ingest_documents_service.ingest_document import IngestResult
from RAG_Agent.config import settings
from RAG_Agent.domain.value_objects.canonical_document import CanonicalDocument, DocumentMetadata
from RAG_Agent.infrastructure.api import main_ingest


class _FakeIngestService:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def execute(self, path, *, index: bool = False) -> IngestResult:
        path = Path(path)
        self.paths.append(path)
        return IngestResult(
            canonical=CanonicalDocument(metadata=DocumentMetadata(source_path=path)),
            chunk_count=2,
            indexed=index,
            extra={"doc_id": path.stem, "deleted": "0", "upserted": "2"},
        )


@pytest.fixture
def ingest_client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "use_secret_manager", False)
    fake_service = _FakeIngestService()

    @contextmanager
    def _fake_download(bucket: str, object_name: str, *, client=None):
        assert bucket == "my-bucket"
        tmp = tmp_path / Path(object_name).name
        tmp.write_bytes(b"%PDF-1.4")
        yield tmp

    monkeypatch.setattr(main_ingest, "build_ingest_service", lambda: fake_service)
    monkeypatch.setattr(main_ingest, "download_gcs_object", _fake_download)
    monkeypatch.setattr(main_ingest, "apply_ingest_secrets_from_secret_manager", lambda: None)

    with TestClient(main_ingest.app) as client:
        client.fake_service = fake_service  # type: ignore[attr-defined]
        yield client


def _finalize_envelope(object_id: str = "docs/WG1.pdf", **attrs):
    attributes = {
        "eventType": "OBJECT_FINALIZE",
        "bucketId": "my-bucket",
        "objectId": object_id,
        "objectGeneration": "99",
        **attrs,
    }
    return {"message": {"attributes": attributes}}


def test_pubsub_push_indexes_pdf(ingest_client):
    response = ingest_client.post("/", json=_finalize_envelope())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["doc_id"] == "WG1"
    assert body["chunk_count"] == 2
    assert ingest_client.fake_service.paths[0].name == "WG1.pdf"


def test_pubsub_push_rejects_word(ingest_client):
    response = ingest_client.post("/", json=_finalize_envelope("report.docx"))
    assert response.status_code == 400
    assert "not supported yet" in response.json()["detail"]


def test_pubsub_push_rejects_unsupported(ingest_client):
    response = ingest_client.post("/", json=_finalize_envelope("notes.txt"))
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"]


def test_pubsub_push_ignores_non_finalize(ingest_client):
    response = ingest_client.post(
        "/",
        json=_finalize_envelope(eventType="OBJECT_DELETE"),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ignored"
    assert ingest_client.fake_service.paths == []


def test_pubsub_push_bad_envelope(ingest_client):
    response = ingest_client.post("/", json={"message": {}})
    assert response.status_code == 400


def test_pubsub_push_gcs_not_found(ingest_client, monkeypatch):
    class NotFound(Exception):
        pass

    NotFound.__module__ = "google.cloud.exceptions"

    @contextmanager
    def _missing(*_a, **_k):
        raise NotFound("404")
        yield  # pragma: no cover

    monkeypatch.setattr(main_ingest, "download_gcs_object", _missing)
    response = ingest_client.post("/", json=_finalize_envelope())
    assert response.status_code == 404
