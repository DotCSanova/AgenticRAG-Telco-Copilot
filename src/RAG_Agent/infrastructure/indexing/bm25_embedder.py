from __future__ import annotations

from typing import Any

from fastembed import SparseTextEmbedding

from RAG_Agent.domain.ports.embedder import SparseEmbedder
from RAG_Agent.domain.value_objects.embedding import SparseEmbedding


class BM25Embedder(SparseEmbedder):
    """Sparse embeddings vía FastEmbed ``Qdrant/bm25`` (para hybrid search)."""

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        self._model = SparseTextEmbedding(model_name=model_name)

    def embed_doc(self, texts: list[str]) -> list[SparseEmbedding]:
        if not texts:
            return []
        return [self._to_sparse(sv) for sv in self._model.embed(texts)]

    def embed_query(self, query: str) -> SparseEmbedding:
        """BM25 query: pesos 1.0; el IDF lo aplica Qdrant (``Modifier.IDF``)."""
        if not query.strip():
            raise ValueError("query must be non-empty")
        sv = next(self._model.query_embed(query))
        return self._to_sparse(sv)

    @staticmethod
    def _to_sparse(sv: Any) -> SparseEmbedding:
        return SparseEmbedding(
            indices=tuple(int(i) for i in sv.indices.tolist()),
            values=tuple(float(v) for v in sv.values.tolist()),
        )
