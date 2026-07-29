"""Composition helpers compartidos por API de producto y Dev UI ADK."""

from __future__ import annotations

from RAG_Agent.application.search_documents_service.search_documents import SearchDocumentsService
from RAG_Agent.config import settings
from RAG_Agent.domain.ports.embedder import Embedder
from RAG_Agent.domain.ports.reranker import Reranker
from RAG_Agent.domain.ports.vector_store import VectorStore
from RAG_Agent.infrastructure.indexing.bm25_embedder import BM25Embedder
from RAG_Agent.infrastructure.indexing.cohere_embedder import CohereEmbedder
from RAG_Agent.infrastructure.indexing.cohere_reranker import CohereReranker
from RAG_Agent.infrastructure.indexing.hybrid_embedder import HybridEmbedder
from RAG_Agent.infrastructure.indexing.qdrant_vector_store import QdrantVectorStore
from RAG_Agent.infrastructure.indexing.section_chunker import SectionChunker
from RAG_Agent.infrastructure.indexing.semantic_chunker import SemanticChunker


def build_dense_embedder():
    name = settings.dense_embedder.lower().strip()
    if name == "cohere":
        # Smaller batches + pacing for trial TPM limits on large docs.
        return CohereEmbedder(batch_size=32, inter_batch_sleep_s=2.0)
    raise ValueError(f"Unknown dense_embedder={settings.dense_embedder!r} (expected cohere)")


def build_embedder() -> Embedder:
    dense = build_dense_embedder()
    if settings.qdrant_enable_sparse:
        return HybridEmbedder(dense=dense, sparse=BM25Embedder())
    return dense


def build_reranker() -> Reranker:
    name = settings.reranker_provider.lower().strip()
    if name == "cohere":
        return CohereReranker()
    raise ValueError(f"Unknown reranker_provider={settings.reranker_provider!r} (expected cohere)")


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


def build_vector_store() -> VectorStore:
    name = settings.vector_store_provider.lower().strip()
    if name == "qdrant":
        return QdrantVectorStore(prefetch_limit=settings.retrieval_prefetch_limit)
    raise ValueError(
        f"Unknown vector_store_provider={settings.vector_store_provider!r} (expected qdrant)"
    )


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
