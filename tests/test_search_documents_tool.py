from RAG_Agent.application.search_documents_service.search_documents import SearchDocumentsService
from RAG_Agent.domain.ports.reranker import RerankResult
from RAG_Agent.domain.tools.search_documents import make_search_documents_tool
from RAG_Agent.domain.value_objects.embedding import TextEmbedding
from RAG_Agent.domain.value_objects.search_hit import RetrievedChunk


class _Embedder:
    def embed_doc(self, texts):
        return [TextEmbedding(dense=(0.1,)) for _ in texts]

    def embed_query(self, query):
        return TextEmbedding(dense=(0.1,))


class _Store:
    def upsert(self, chunks, embeddings):
        return 0

    def search(self, embedding, *, limit):
        return [
            RetrievedChunk(text="passage", doc_id="doc1", section_path="1 Scope"),
        ]


class _Reranker:
    def rerank(self, query, documents, *, top_n):
        return [RerankResult(index=0, score=0.88)]


def test_make_search_documents_tool_formats_hits():
    service = SearchDocumentsService(
        embedder=_Embedder(),
        vector_store=_Store(),
        reranker=_Reranker(),
        candidate_limit=10,
        rerank_top_n=5,
    )
    tool = make_search_documents_tool(service.execute)
    payload = tool("scope?")
    assert payload == {
        "results": [
            {
                "text": "passage",
                "score": 0.88,
                "doc_id": "doc1",
                "section_path": "1 Scope",
            }
        ]
    }
