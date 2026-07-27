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
    """Resultado del caso de uso de ingesta."""

    canonical: CanonicalDocument
    chunk_count: int = 0
    indexed: bool = False
    extra: dict[str, str] = field(default_factory=dict)


class IngestDocumentService:
    """Caso de uso: parse → (opcional) chunk → embed → vector store."""

    def __init__(
        self,
        parser: DocumentParser,
        chunker: Chunker | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store

    def execute(self, path: Path | str, *, index: bool = False) -> IngestResult:
        canonical = self._parser.parse(Path(path))

        if not index:
            return IngestResult(canonical=canonical, indexed=False)

        if self._chunker is None or self._embedder is None or self._vector_store is None:
            msg = "index=True requiere chunker, embedder y vector_store inyectados"
            raise RuntimeError(msg)

        chunks = self._chunker.chunk(canonical)
        embeddings = self._embedder.embed_doc([chunk.text for chunk in chunks])
        if len(embeddings) != len(chunks):
            msg = f"embedder returned {len(embeddings)} embeddings for {len(chunks)} chunks"
            raise RuntimeError(msg)

        upserted = self._vector_store.upsert(chunks, embeddings)
        return IngestResult(
            canonical=canonical,
            chunk_count=len(chunks),
            indexed=True,
            extra={"upserted": str(upserted)},
        )
