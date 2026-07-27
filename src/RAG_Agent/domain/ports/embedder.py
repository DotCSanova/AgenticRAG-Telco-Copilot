from __future__ import annotations

from typing import Protocol

from RAG_Agent.domain.value_objects.embedding import SparseEmbedding, TextEmbedding


class Embedder(Protocol):
    """Puerto de indexado/búsqueda: dense (+ sparse opcional) como ``TextEmbedding``."""

    def embed_doc(self, texts: list[str]) -> list[TextEmbedding]:
        """Un ``TextEmbedding`` por documento/chunk, mismo orden que ``texts``."""

    def embed_query(self, query: str) -> TextEmbedding:
        """Embedding de una query de búsqueda."""


class DenseEmbedder(Protocol):
    """Solo vectores densos (p. ej. Cohere)."""

    def embed_doc(self, texts: list[str]) -> list[TextEmbedding]: ...

    def embed_query(self, query: str) -> TextEmbedding: ...


class SparseEmbedder(Protocol):
    """Solo vectores sparse (p. ej. BM25)."""

    def embed_doc(self, texts: list[str]) -> list[SparseEmbedding]: ...

    def embed_query(self, query: str) -> SparseEmbedding: ...
