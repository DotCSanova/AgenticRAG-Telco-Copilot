"""Delete Qdrant collection points, then ingest+index the TR PDF with SectionChunker."""
from __future__ import annotations

import logging
from pathlib import Path

from RAG_Agent.application.ingest_documents_service.ingest_document import IngestDocumentService
from RAG_Agent.config import settings
from RAG_Agent.infrastructure.indexing.bm25_embedder import BM25Embedder
from RAG_Agent.infrastructure.indexing.cohere_embedder import CohereEmbedder
from RAG_Agent.infrastructure.indexing.hybrid_embedder import HybridEmbedder
from RAG_Agent.infrastructure.indexing.qdrant_vector_store import QdrantVectorStore
from RAG_Agent.infrastructure.indexing.section_chunker import SectionChunker
from RAG_Agent.infrastructure.ingestion.cascading_profile_resolver import CascadingProfileResolver
from RAG_Agent.infrastructure.ingestion.native_pdf_pipeline import NativePdfPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("reindex_tr")

PDF = Path("data/O-RAN.WG1.TR.Use-Cases-Analysis-Report-R005-v19.00.pdf")


def main() -> None:
    if not PDF.is_file():
        raise SystemExit(f"PDF missing: {PDF}")

    store = QdrantVectorStore()
    client = store._client
    collection = store._collection

    if client.collection_exists(collection):
        info = client.get_collection(collection)
        logger.info(
            "Deleting collection %s (points=%s)",
            collection,
            getattr(info, "points_count", "?"),
        )
        client.delete_collection(collection)
        logger.info("Collection deleted")
    else:
        logger.info("Collection %s did not exist", collection)

    dense = CohereEmbedder()
    embedder = (
        HybridEmbedder(dense=dense, sparse=BM25Embedder())
        if settings.qdrant_enable_sparse
        else dense
    )
    service = IngestDocumentService(
        parser=NativePdfPipeline(CascadingProfileResolver()),
        chunker=SectionChunker(),
        embedder=embedder,
        vector_store=QdrantVectorStore(),
    )
    logger.info("Parsing + indexing %s (section chunker)", PDF)
    result = service.execute(PDF, index=True)
    logger.info(
        "Done: title=%r pages=%s blocks=%s sections=%s chunks=%s upserted=%s",
        result.canonical.metadata.title,
        result.canonical.metadata.extra.get("num_pages"),
        result.canonical.metadata.extra.get("num_blocks"),
        result.canonical.metadata.extra.get("num_sections"),
        result.chunk_count,
        result.extra.get("upserted"),
    )


if __name__ == "__main__":
    main()
