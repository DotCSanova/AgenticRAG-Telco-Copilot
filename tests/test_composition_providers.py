import pytest

from RAG_Agent.config import settings
from RAG_Agent.infrastructure.composition import shared, serving


def test_build_dense_embedder_rejects_unknown(monkeypatch):
    monkeypatch.setattr(settings, "dense_embedder", "openai")
    with pytest.raises(ValueError, match="Unknown dense_embedder"):
        shared.build_dense_embedder()


def test_build_reranker_rejects_unknown(monkeypatch):
    monkeypatch.setattr(settings, "reranker_provider", "bge")
    with pytest.raises(ValueError, match="Unknown reranker_provider"):
        serving.build_reranker()


def test_build_vector_store_rejects_unknown(monkeypatch):
    monkeypatch.setattr(settings, "vector_store_provider", "pgvector")
    with pytest.raises(ValueError, match="Unknown vector_store_provider"):
        shared.build_vector_store()
