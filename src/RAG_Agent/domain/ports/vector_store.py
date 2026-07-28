from __future__ import annotations

from typing import Protocol

from RAG_Agent.domain.value_objects.chunk import Chunk
from RAG_Agent.domain.value_objects.embedding import TextEmbedding
from RAG_Agent.domain.value_objects.search_hit import RetrievedChunk


class VectorStore(Protocol):
    """Persistencia de chunks + embeddings (dense / hybrid)."""

    def upsert(self, chunks: list[Chunk], embeddings: list[TextEmbedding]) -> int:
        """Indexa pares chunk/embedding. Devuelve cuántos se upsertaron."""

    def search(self, embedding: TextEmbedding, *, limit: int) -> list[RetrievedChunk]:
        """Recupera candidatos por similitud (dense o hybrid RRF)."""
