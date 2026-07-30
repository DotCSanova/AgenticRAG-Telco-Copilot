from pathlib import Path

from RAG_Agent.application.ingest_documents_service.ingest_document import IngestDocumentService
from RAG_Agent.domain.value_objects.canonical_document import CanonicalDocument, DocumentMetadata
from RAG_Agent.domain.value_objects.chunk import Chunk
from RAG_Agent.domain.value_objects.embedding import TextEmbedding


class _Parser:
    def __init__(self, canonical: CanonicalDocument) -> None:
        self.canonical = canonical
        self.calls: list[Path] = []

    def parse(self, path: Path) -> CanonicalDocument:
        self.calls.append(path)
        return self.canonical


class _Chunker:
    def chunk(self, document: CanonicalDocument) -> list[Chunk]:
        doc_id = document.metadata.source_path.stem
        return [Chunk(id=f"{doc_id}:c0", doc_id=doc_id, text="passage")]


class _Embedder:
    def embed_doc(self, texts: list[str]) -> list[TextEmbedding]:
        return [TextEmbedding(dense=(0.1, 0.2)) for _ in texts]

    def embed_query(self, query: str) -> TextEmbedding:
        return TextEmbedding(dense=(0.1, 0.2))


class _Store:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.upserts: list[tuple[list[Chunk], list[TextEmbedding]]] = []

    def upsert(self, chunks, embeddings):
        self.upserts.append((chunks, embeddings))
        return len(chunks)

    def search(self, embedding, *, limit):
        return []

    def delete_by_doc_id(self, doc_id: str) -> int:
        self.deleted.append(doc_id)
        return 3 if doc_id == "sample-doc" else 0


def _canonical(path: Path) -> CanonicalDocument:
    return CanonicalDocument(metadata=DocumentMetadata(source_path=path))


def test_ingest_index_false_skips_delete_and_upsert():
    path = Path("data/sample-doc.pdf")
    store = _Store()
    service = IngestDocumentService(
        parser=_Parser(_canonical(path)),
        chunker=_Chunker(),
        embedder=_Embedder(),
        vector_store=store,
    )

    result = service.execute(path, index=False)

    assert result.indexed is False
    assert result.chunk_count == 0
    assert store.deleted == []
    assert store.upserts == []


def test_ingest_index_true_deletes_by_stem_then_upserts():
    path = Path("data/sample-doc.pdf")
    store = _Store()
    service = IngestDocumentService(
        parser=_Parser(_canonical(path)),
        chunker=_Chunker(),
        embedder=_Embedder(),
        vector_store=store,
    )

    result = service.execute(path, index=True)

    assert result.indexed is True
    assert result.chunk_count == 1
    assert store.deleted == ["sample-doc"]
    assert len(store.upserts) == 1
    assert store.upserts[0][0][0].doc_id == "sample-doc"
    assert result.extra == {"upserted": "1", "deleted": "3", "doc_id": "sample-doc"}
