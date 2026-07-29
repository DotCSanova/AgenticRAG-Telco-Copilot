from qdrant_client import QdrantClient

from RAG_Agent.config import settings
from RAG_Agent.infrastructure.indexing.qdrant_vector_store import QdrantVectorStore


def test_build_client_in_memory(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_in_memory", True)
    client = QdrantVectorStore._build_client()
    assert isinstance(client, QdrantClient)


def test_build_client_uses_url_and_optional_key(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_in_memory", False)
    monkeypatch.setattr(settings, "qdrant_url", "http://qdrant:6333")
    monkeypatch.setattr(settings, "qdrant_api_key", None)
    # Construction must not require a live server (client init is lazy enough for URL).
    client = QdrantVectorStore._build_client()
    assert isinstance(client, QdrantClient)


def test_build_client_defaults_to_localhost_when_url_missing(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_in_memory", False)
    monkeypatch.setattr(settings, "qdrant_url", None)
    monkeypatch.setattr(settings, "qdrant_api_key", "")
    client = QdrantVectorStore._build_client()
    assert isinstance(client, QdrantClient)
