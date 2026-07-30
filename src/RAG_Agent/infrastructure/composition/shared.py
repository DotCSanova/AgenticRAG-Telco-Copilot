"""Shared composition builders: embedder and vector store (no Docling/chunkers)."""

from __future__ import annotations

from RAG_Agent.config import settings
from RAG_Agent.domain.ports.embedder import Embedder
from RAG_Agent.domain.ports.vector_store import VectorStore
from RAG_Agent.infrastructure.indexing.bm25_embedder import BM25Embedder
from RAG_Agent.infrastructure.indexing.cohere_embedder import CohereEmbedder
from RAG_Agent.infrastructure.indexing.hybrid_embedder import HybridEmbedder
from RAG_Agent.infrastructure.indexing.qdrant_vector_store import QdrantVectorStore


def build_dense_embedder() -> Embedder:
    """Build the dense embedder selected by ``settings.dense_embedder``."""
    name = settings.dense_embedder.lower().strip()
    if name == "cohere":
        # Smaller batches + pacing for trial TPM limits on large docs.
        return CohereEmbedder(batch_size=32, inter_batch_sleep_s=2.0)
    raise ValueError(f"Unknown dense_embedder={settings.dense_embedder!r} (expected cohere)")


def build_embedder() -> Embedder:
    """Build dense or hybrid (dense + BM25) embedder from settings."""
    dense = build_dense_embedder()
    if settings.qdrant_enable_sparse:
        return HybridEmbedder(dense=dense, sparse=BM25Embedder())
    return dense


def build_vector_store() -> VectorStore:
    """Build the vector store selected by ``settings.vector_store_provider``."""
    name = settings.vector_store_provider.lower().strip()
    if name == "qdrant":
        return QdrantVectorStore(prefetch_limit=settings.retrieval_prefetch_limit)
    raise ValueError(
        f"Unknown vector_store_provider={settings.vector_store_provider!r} (expected qdrant)"
    )
