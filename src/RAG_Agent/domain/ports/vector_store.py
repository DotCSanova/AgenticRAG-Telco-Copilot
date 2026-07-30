from __future__ import annotations

from typing import Protocol

from RAG_Agent.domain.value_objects.chunk import Chunk
from RAG_Agent.domain.value_objects.embedding import TextEmbedding
from RAG_Agent.domain.value_objects.search_hit import RetrievedChunk


class VectorStore(Protocol):
    """Persistence for chunks and embeddings (dense / hybrid)."""

    def upsert(self, chunks: list[Chunk], embeddings: list[TextEmbedding]) -> int:
        """Upsert chunk/embedding pairs. Returns how many points were written."""

    def search(self, embedding: TextEmbedding, *, limit: int) -> list[RetrievedChunk]:
        """Retrieve candidates by similarity (dense or hybrid RRF)."""

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete points whose payload ``doc_id`` matches. Returns prior count (0 if none)."""
