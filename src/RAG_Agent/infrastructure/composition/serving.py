"""Composition root for chat / retrieval serving (no Docling / chunkers)."""

from __future__ import annotations

from RAG_Agent.application.search_documents_service.search_documents import SearchDocumentsService
from RAG_Agent.config import settings
from RAG_Agent.domain.ports.embedder import Embedder
from RAG_Agent.domain.ports.reranker import Reranker
from RAG_Agent.domain.ports.vector_store import VectorStore
from RAG_Agent.infrastructure.composition.shared import build_embedder, build_vector_store
from RAG_Agent.infrastructure.indexing.cohere_reranker import CohereReranker

__all__ = [
    "build_embedder",
    "build_reranker",
    "build_search_service",
    "build_vector_store",
]


def build_reranker() -> Reranker:
    name = settings.reranker_provider.lower().strip()
    if name == "cohere":
        return CohereReranker()
    raise ValueError(f"Unknown reranker_provider={settings.reranker_provider!r} (expected cohere)")


def build_search_service(
    *,
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
    reranker: Reranker | None = None,
    candidate_limit: int | None = None,
    rerank_top_n: int | None = None,
) -> SearchDocumentsService:
    return SearchDocumentsService(
        embedder=embedder or build_embedder(),
        vector_store=vector_store or build_vector_store(),
        reranker=reranker or build_reranker(),
        candidate_limit=candidate_limit or settings.retrieval_candidate_limit,
        rerank_top_n=rerank_top_n or settings.rerank_top_n,
    )
