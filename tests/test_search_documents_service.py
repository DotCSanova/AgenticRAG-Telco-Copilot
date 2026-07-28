from RAG_Agent.application.search_documents_service.search_documents import SearchDocumentsService
from RAG_Agent.domain.ports.reranker import RerankResult
from RAG_Agent.domain.value_objects.embedding import TextEmbedding
from RAG_Agent.domain.value_objects.search_hit import RetrievedChunk


class _FakeEmbedder:
    def embed_doc(self, texts: list[str]) -> list[TextEmbedding]:
        return [TextEmbedding(dense=(0.1, 0.2)) for _ in texts]

    def embed_query(self, query: str) -> TextEmbedding:
        return TextEmbedding(dense=(0.1, 0.2))


class _FakeStore:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def upsert(self, chunks, embeddings):
        return 0

    def search(self, embedding, *, limit: int) -> list[RetrievedChunk]:
        return self._chunks[:limit]


class _FakeReranker:
    def rerank(self, query: str, documents: list[str], *, top_n: int) -> list[RerankResult]:
        # Reverse order to prove rerank is applied.
        indices = list(range(len(documents)))[::-1][:top_n]
        return [RerankResult(index=i, score=1.0 - (0.1 * rank)) for rank, i in enumerate(indices)]


def test_search_documents_service_reranks_candidates():
    candidates = [
        RetrievedChunk(text="first", doc_id="docA", section_path="A > 1"),
        RetrievedChunk(text="second", doc_id="docA", section_path="A > 2"),
        RetrievedChunk(text="third", doc_id="docB", section_path="B"),
    ]
    service = SearchDocumentsService(
        embedder=_FakeEmbedder(),
        vector_store=_FakeStore(candidates),
        reranker=_FakeReranker(),
        candidate_limit=10,
        rerank_top_n=2,
    )
    hits = service.execute("what is second?")
    assert len(hits) == 2
    assert hits[0].text == "third"
    assert hits[0].doc_id == "docB"
    assert hits[0].section_path == "B"
    assert hits[1].text == "second"
    assert hits[1].section_path == "A > 2"


def test_search_documents_service_empty_store():
    service = SearchDocumentsService(
        embedder=_FakeEmbedder(),
        vector_store=_FakeStore([]),
        reranker=_FakeReranker(),
    )
    assert service.execute("anything") == []
