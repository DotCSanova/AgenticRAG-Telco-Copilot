"""Local PDF ingest entrypoint (host or ingest container).

Examples::

    uv run --group ingest python scripts/ingest_local.py path/to/doc.pdf
    docker compose run --rm agent-ingest python scripts/ingest_local.py /data/doc.pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from RAG_Agent.infrastructure.composition.ingest import build_ingest_service

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ingest_local")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse and optionally index a PDF into Qdrant.")
    parser.add_argument("path", type=Path, help="Path to the PDF file")
    parser.add_argument(
        "--index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Chunk/embed/upsert after parse (default: true)",
    )
    args = parser.parse_args(argv)

    path = args.path.expanduser()
    if not path.is_file():
        logger.error("PDF not found: %s", path)
        return 1

    service = build_ingest_service()
    result = service.execute(path, index=args.index)
    logger.info(
        "Done: doc_id=%s indexed=%s chunk_count=%s deleted=%s upserted=%s title=%r",
        result.extra.get("doc_id", path.stem),
        result.indexed,
        result.chunk_count,
        result.extra.get("deleted", "0"),
        result.extra.get("upserted", "0"),
        result.canonical.metadata.title,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
