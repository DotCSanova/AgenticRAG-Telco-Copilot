"""Composition helpers compartidos por API de producto y Dev UI ADK."""

from __future__ import annotations

from RAG_Agent.application.search_documents_service.search_documents import SearchDocumentsService
from RAG_Agent.config import settings
from RAG_Agent.infrastructure.indexing.bm25_embedder import BM25Embedder
from RAG_Agent.infrastructure.indexing.cohere_embedder import CohereEmbedder
from RAG_Agent.infrastructure.indexing.cohere_reranker import CohereReranker
from RAG_Agent.infrastructure.indexing.hybrid_embedder import HybridEmbedder
from RAG_Agent.infrastructure.indexing.qdrant_vector_store import QdrantVectorStore
from RAG_Agent.infrastructure.indexing.section_chunker import SectionChunker
from RAG_Agent.infrastructure.indexing.semantic_chunker import SemanticChunker


def build_embedder():
    dense = CohereEmbedder()
    if settings.qdrant_enable_sparse:
        return HybridEmbedder(dense=dense, sparse=BM25Embedder())
    return dense


def build_chunker():
    name = settings.chunker.lower().strip()
    if name == "section":
        return SectionChunker()
    if name == "semantic":
        return SemanticChunker(
            model_name=settings.semantic_chunk_model,
            threshold=settings.semantic_chunk_threshold,
            min_tokens=settings.semantic_chunk_min_tokens,
            max_tokens=settings.semantic_chunk_max_tokens,
        )
    raise ValueError(f"Unknown chunker={settings.chunker!r} (expected section|semantic)")


def build_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore()


def build_search_service(
    *,
    embedder=None,
    vector_store: QdrantVectorStore | None = None,
) -> SearchDocumentsService:
    return SearchDocumentsService(
        embedder=embedder or build_embedder(),
        vector_store=vector_store or build_vector_store(),
        reranker=CohereReranker(),
    )
