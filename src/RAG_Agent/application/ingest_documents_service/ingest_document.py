from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from RAG_Agent.domain.ports.chunker import Chunker
from RAG_Agent.domain.ports.document_parser import DocumentParser
from RAG_Agent.domain.ports.embedder import Embedder
from RAG_Agent.domain.ports.vector_store import VectorStore
from RAG_Agent.domain.value_objects.canonical_document import CanonicalDocument


@dataclass(frozen=True)
class IngestResult:
    """Outcome of the ingest use case."""

    canonical: CanonicalDocument
    chunk_count: int = 0
    indexed: bool = False
    extra: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestDocumentService:
    """Parse a PDF, optionally chunk → embed → upsert (idempotent by doc stem)."""

    parser: DocumentParser
    chunker: Chunker | None = None
    embedder: Embedder | None = None
    vector_store: VectorStore | None = None

    def execute(self, path: Path | str, *, index: bool = False) -> IngestResult:
        """Parse ``path``. With ``index=True``, delete existing stem chunks then upsert."""
        canonical = self.parser.parse(Path(path))

        if not index:
            return IngestResult(canonical=canonical, indexed=False)

        if self.chunker is None or self.embedder is None or self.vector_store is None:
            msg = "index=True requires chunker, embedder, and vector_store to be injected"
            raise RuntimeError(msg)

        doc_id = canonical.metadata.source_path.stem
        deleted = self.vector_store.delete_by_doc_id(doc_id)

        chunks = self.chunker.chunk(canonical)
        embeddings = self.embedder.embed_doc([chunk.text for chunk in chunks])
        if len(embeddings) != len(chunks):
            msg = f"embedder returned {len(embeddings)} embeddings for {len(chunks)} chunks"
            raise RuntimeError(msg)

        upserted = self.vector_store.upsert(chunks, embeddings)
        return IngestResult(
            canonical=canonical,
            chunk_count=len(chunks),
            indexed=True,
            extra={"upserted": str(upserted), "deleted": str(deleted), "doc_id": doc_id},
        )
