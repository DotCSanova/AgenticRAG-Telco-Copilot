from __future__ import annotations

from typing import Protocol

from RAG_Agent.domain.value_objects.chunk import Chunk
from RAG_Agent.domain.value_objects.embedding import TextEmbedding


class VectorStore(Protocol):
    """Persistencia de chunks + embeddings (dense / hybrid)."""

    def upsert(self, chunks: list[Chunk], embeddings: list[TextEmbedding]) -> int:
        """Indexa pares chunk/embedding. Devuelve cuántos se upsertaron."""
