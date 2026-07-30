"""Composition root for local/job ingest (Docling + chunkers; no ADK)."""

from __future__ import annotations

from RAG_Agent.application.ingest_documents_service.ingest_document import IngestDocumentService
from RAG_Agent.config import settings
from RAG_Agent.domain.ports.chunker import Chunker
from RAG_Agent.domain.ports.document_parser import DocumentParser
from RAG_Agent.domain.ports.embedder import Embedder
from RAG_Agent.domain.ports.vector_store import VectorStore
from RAG_Agent.infrastructure.composition.shared import build_embedder, build_vector_store
from RAG_Agent.infrastructure.indexing.section_chunker import SectionChunker
from RAG_Agent.infrastructure.indexing.semantic_chunker import SemanticChunker
from RAG_Agent.infrastructure.ingestion.cascading_profile_resolver import CascadingProfileResolver
from RAG_Agent.infrastructure.ingestion.native_pdf_pipeline import NativePdfPipeline

__all__ = [
    "build_chunker",
    "build_embedder",
    "build_ingest_service",
    "build_parser",
    "build_vector_store",
]


def build_parser() -> DocumentParser:
    return NativePdfPipeline(CascadingProfileResolver())


def build_chunker() -> Chunker:
    name = settings.chunker.lower().strip()
    if name == "section":
        return SectionChunker()
    if name == "semantic":
        return SemanticChunker(
            model_name=settings.semantic_chunk_model,
            threshold=settings.semantic_chunk_threshold,
            min_tokens=settings.semantic_chunk_min_tokens,
            max_tokens=settings.semantic_chunk_max_tokens,
        )
    raise ValueError(f"Unknown chunker={settings.chunker!r} (expected section|semantic)")


def build_ingest_service(
    *,
    parser: DocumentParser | None = None,
    chunker: Chunker | None = None,
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
) -> IngestDocumentService:
    return IngestDocumentService(
        parser=parser or build_parser(),
        chunker=chunker or build_chunker(),
        embedder=embedder or build_embedder(),
        vector_store=vector_store or build_vector_store(),
    )
