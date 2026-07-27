from RAG_Agent.domain.value_objects.embedding import SparseEmbedding, TextEmbedding
from RAG_Agent.infrastructure.indexing.hybrid_embedder import HybridEmbedder


class _FakeDense:
    def embed_doc(self, texts: list[str]) -> list[TextEmbedding]:
        return [TextEmbedding(dense=(float(i), 0.1)) for i, _ in enumerate(texts)]

    def embed_query(self, query: str) -> TextEmbedding:
        return TextEmbedding(dense=(1.0, 2.0))


class _FakeSparse:
    def embed_doc(self, texts: list[str]) -> list[SparseEmbedding]:
        return [
            SparseEmbedding(indices=(i, i + 1), values=(0.5, 0.25))
            for i, _ in enumerate(texts)
        ]

    def embed_query(self, query: str) -> SparseEmbedding:
        return SparseEmbedding(indices=(9,), values=(1.0,))


def test_hybrid_embed_doc_merges_dense_and_sparse():
    hybrid = HybridEmbedder(dense=_FakeDense(), sparse=_FakeSparse())
    out = hybrid.embed_doc(["a", "b"])
    assert len(out) == 2
    assert out[0].dense == (0.0, 0.1)
    assert out[0].sparse == SparseEmbedding(indices=(0, 1), values=(0.5, 0.25))
    assert out[1].dense == (1.0, 0.1)
    assert out[1].sparse == SparseEmbedding(indices=(1, 2), values=(0.5, 0.25))


def test_hybrid_embed_query_merges_dense_and_sparse():
    hybrid = HybridEmbedder(dense=_FakeDense(), sparse=_FakeSparse())
    out = hybrid.embed_query("what is smo?")
    assert out.dense == (1.0, 2.0)
    assert out.sparse == SparseEmbedding(indices=(9,), values=(1.0,))


def test_hybrid_embed_doc_empty():
    hybrid = HybridEmbedder(dense=_FakeDense(), sparse=_FakeSparse())
    assert hybrid.embed_doc([]) == []
