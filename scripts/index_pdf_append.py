"""Index a PDF into the existing Qdrant collection (delete+upsert by doc stem)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from RAG_Agent.infrastructure.composition.ingest import build_ingest_service
from RAG_Agent.infrastructure.composition.shared import build_vector_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("index_pdf")

DEFAULT_PDF = Path("data/O-RAN.WG1.TS.Use-Cases-Detailed-Specification-R005-v19.00.pdf")


def main() -> None:
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
    if not pdf.is_file():
        raise SystemExit(f"PDF missing: {pdf}")

    store = build_vector_store()
    client = store._client  # noqa: SLF001
    collection = store._collection  # noqa: SLF001
    before = 0
    if client.collection_exists(collection):
        before = int(client.get_collection(collection).points_count or 0)
    logger.info("Collection %s points before=%s", collection, before)

    service = build_ingest_service(vector_store=store)
    logger.info("Parsing + indexing %s", pdf)
    result = service.execute(pdf, index=True)

    after = int(client.get_collection(collection).points_count or 0)
    logger.info(
        "Done: title=%r pages=%s blocks=%s sections=%s chunks=%s upserted=%s deleted=%s points=%s→%s",
        result.canonical.metadata.title,
        result.canonical.metadata.extra.get("num_pages"),
        result.canonical.metadata.extra.get("num_blocks"),
        result.canonical.metadata.extra.get("num_sections"),
        result.chunk_count,
        result.extra.get("upserted"),
        result.extra.get("deleted"),
        before,
        after,
    )


if __name__ == "__main__":
    main()
