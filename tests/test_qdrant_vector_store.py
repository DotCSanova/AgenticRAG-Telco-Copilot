from qdrant_client import QdrantClient

from RAG_Agent.domain.value_objects.chunk import Chunk
from RAG_Agent.domain.value_objects.embedding import TextEmbedding, SparseEmbedding
from RAG_Agent.infrastructure.indexing.qdrant_vector_store import QdrantVectorStore


def test_qdrant_upsert_dense_only():
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(collection_name="test_dense", client=client, enable_sparse=False)
    chunks = [
        Chunk(id="doc1:c0", doc_id="doc1", text="hello", page_start=1),
        Chunk(id="doc1:c1", doc_id="doc1", text="world", page_start=2),
    ]
    embeddings = [
        TextEmbedding(dense=(0.1, 0.2, 0.3)),
        TextEmbedding(dense=(0.4, 0.5, 0.6)),
    ]
    assert store.upsert(chunks, embeddings) == 2
    assert client.get_collection("test_dense").points_count == 2


def test_qdrant_upsert_hybrid_sparse():
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(collection_name="test_hybrid", client=client, enable_sparse=True)
    chunks = [Chunk(id="doc1:c0", doc_id="doc1", text="ran ric")]
    embeddings = [
        TextEmbedding(
            dense=(0.1, 0.2, 0.3),
            sparse=SparseEmbedding(indices=(1, 7), values=(0.4, 0.9)),
        )
    ]
    assert store.upsert(chunks, embeddings) == 1
    assert client.get_collection("test_hybrid").points_count == 1


def test_qdrant_upsert_length_mismatch():
    store = QdrantVectorStore(
        client=QdrantClient(location=":memory:"),
        enable_sparse=False,
    )
    try:
        store.upsert(
            [Chunk(id="a", doc_id="d", text="t")],
            [
                TextEmbedding(dense=(0.1, 0.2)),
                TextEmbedding(dense=(0.3, 0.4)),
            ],
        )
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_qdrant_search_dense_returns_payload_fields():
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(collection_name="test_search", client=client, enable_sparse=False)
    chunks = [
        Chunk(
            id="doc1:c0",
            doc_id="doc1",
            text="Near-RT RIC handover",
            metadata={"section_path": "4 Use cases > 4.1 HO"},
        ),
        Chunk(
            id="doc1:c1",
            doc_id="doc1",
            text="unrelated energy saving",
            metadata={"section_path": "4 Use cases > 4.9"},
        ),
    ]
    embeddings = [
        TextEmbedding(dense=(1.0, 0.0, 0.0)),
        TextEmbedding(dense=(0.0, 1.0, 0.0)),
    ]
    store.upsert(chunks, embeddings)

    hits = store.search(TextEmbedding(dense=(0.95, 0.05, 0.0)), limit=1)
    assert len(hits) == 1
    assert hits[0].doc_id == "doc1"
    assert hits[0].text == "Near-RT RIC handover"
    assert hits[0].section_path == "4 Use cases > 4.1 HO"


def test_qdrant_search_hybrid_rrf():
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(collection_name="test_hybrid_search", client=client, enable_sparse=True)
    chunks = [
        Chunk(
            id="doc1:c0",
            doc_id="doc1",
            text="traffic steering",
            metadata={"section_path": "4.5 Traffic steering"},
        )
    ]
    embeddings = [
        TextEmbedding(
            dense=(0.2, 0.3, 0.4),
            sparse=SparseEmbedding(indices=(3, 9), values=(0.5, 0.8)),
        )
    ]
    store.upsert(chunks, embeddings)
    hits = store.search(
        TextEmbedding(
            dense=(0.2, 0.3, 0.4),
            sparse=SparseEmbedding(indices=(3, 9), values=(0.5, 0.8)),
        ),
        limit=5,
    )
    assert len(hits) == 1
    assert hits[0].section_path == "4.5 Traffic steering"


def test_qdrant_delete_by_doc_id_removes_only_matching_points():
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(collection_name="test_delete", client=client, enable_sparse=False)
    chunks = [
        Chunk(id="doc1:c0", doc_id="doc1", text="keep-me-out"),
        Chunk(id="doc1:c1", doc_id="doc1", text="keep-me-out-too"),
        Chunk(id="doc2:c0", doc_id="doc2", text="stay"),
    ]
    embeddings = [
        TextEmbedding(dense=(0.1, 0.0, 0.0)),
        TextEmbedding(dense=(0.0, 0.1, 0.0)),
        TextEmbedding(dense=(0.0, 0.0, 0.1)),
    ]
    store.upsert(chunks, embeddings)

    assert store.delete_by_doc_id("doc1") == 2
    assert client.get_collection("test_delete").points_count == 1
    remaining = store.search(TextEmbedding(dense=(0.0, 0.0, 1.0)), limit=5)
    assert len(remaining) == 1
    assert remaining[0].doc_id == "doc2"


def test_qdrant_delete_by_doc_id_missing_is_zero():
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(collection_name="test_delete_miss", client=client, enable_sparse=False)
    store.upsert(
        [Chunk(id="doc1:c0", doc_id="doc1", text="x")],
        [TextEmbedding(dense=(0.1, 0.2, 0.3))],
    )
    assert store.delete_by_doc_id("does-not-exist") == 0
    assert client.get_collection("test_delete_miss").points_count == 1


def test_qdrant_delete_by_doc_id_no_collection_is_zero():
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(
        collection_name="never_created",
        client=client,
        enable_sparse=False,
    )
    assert store.delete_by_doc_id("doc1") == 0
