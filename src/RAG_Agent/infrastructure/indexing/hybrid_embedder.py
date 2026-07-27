from __future__ import annotations

from RAG_Agent.domain.ports.embedder import DenseEmbedder, Embedder, SparseEmbedder
from RAG_Agent.domain.value_objects.embedding import TextEmbedding


class HybridEmbedder(Embedder):
    """Combina dense + sparse en un ``TextEmbedding`` (hybrid search)."""

    def __init__(self, dense: DenseEmbedder, sparse: SparseEmbedder) -> None:
        self._dense = dense
        self._sparse = sparse

    def embed_doc(self, texts: list[str]) -> list[TextEmbedding]:
        if not texts:
            return []

        dense_list = self._dense.embed_doc(texts)
        sparse_list = self._sparse.embed_doc(texts)
        if len(dense_list) != len(texts) or len(sparse_list) != len(texts):
            raise RuntimeError(
                f"hybrid embed_doc length mismatch: texts={len(texts)} "
                f"dense={len(dense_list)} sparse={len(sparse_list)}"
            )

        return [
            TextEmbedding(dense=dense.dense, sparse=sparse)
            for dense, sparse in zip(dense_list, sparse_list, strict=True)
        ]

    def embed_query(self, query: str) -> TextEmbedding:
        dense = self._dense.embed_query(query)
        sparse = self._sparse.embed_query(query)
        return TextEmbedding(dense=dense.dense, sparse=sparse)
