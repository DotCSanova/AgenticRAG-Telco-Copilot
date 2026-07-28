from types import SimpleNamespace

from RAG_Agent.infrastructure.indexing.cohere_reranker import CohereReranker


class _FakeCohereClient:
    def rerank(self, **kwargs):
        docs = kwargs["documents"]
        top_n = kwargs["top_n"]
        # Prefer last documents.
        order = list(range(len(docs)))[::-1][:top_n]
        return SimpleNamespace(
            results=[
                SimpleNamespace(index=i, relevance_score=0.9 - 0.1 * rank)
                for rank, i in enumerate(order)
            ]
        )


def test_cohere_reranker_maps_results():
    reranker = CohereReranker(client=_FakeCohereClient(), model="rerank-v3.5")
    results = reranker.rerank("q", ["a", "b", "c"], top_n=2)
    assert [r.index for r in results] == [2, 1]
    assert results[0].score == 0.9


def test_cohere_reranker_empty_documents():
    reranker = CohereReranker(client=_FakeCohereClient())
    assert reranker.rerank("q", [], top_n=5) == []
