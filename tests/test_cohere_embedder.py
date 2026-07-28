from types import SimpleNamespace

from RAG_Agent.infrastructure.indexing.cohere_embedder import CohereEmbedder


class _FakeCohereClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def embed(self, **kwargs):
        self.calls.append(kwargs)
        texts = kwargs["texts"]
        return SimpleNamespace(
            embeddings=SimpleNamespace(float_=[[float(i), 0.0] for i, _ in enumerate(texts)])
        )


def test_cohere_embedder_accepts_injected_client_and_batches():
    client = _FakeCohereClient()
    embedder = CohereEmbedder(client=client, batch_size=2)

    docs = embedder.embed_doc(["a", "b", "c"])
    assert len(docs) == 3
    assert docs[0].dense == (0.0, 0.0)
    assert docs[2].dense == (0.0, 0.0)
    assert len(client.calls) == 2
    assert client.calls[0]["input_type"] == "search_document"
    assert client.calls[0]["texts"] == ["a", "b"]
    assert client.calls[1]["texts"] == ["c"]


def test_cohere_embedder_query_uses_search_query():
    client = _FakeCohereClient()
    embedder = CohereEmbedder(client=client)

    query = embedder.embed_query("hello")
    assert query.dense == (0.0, 0.0)
    assert client.calls[0]["input_type"] == "search_query"
    assert client.calls[0]["texts"] == ["hello"]
