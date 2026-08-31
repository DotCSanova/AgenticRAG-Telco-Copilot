"""Local PDF ingest entrypoint (host or ingest container).

Examples::

    uv run --group ingest python scripts/ingest_local.py path/to/doc.pdf
    docker compose run --rm agent-ingest python scripts/ingest_local.py /data/doc.pdf
    uv run --group ingest python scripts/ingest_local.py data/doc.pdf --no-index `
        --canonical-out data/out/doc.canonical.json `
        --canonical-md-out data/out/doc.canonical.md `
        --json-out data/out/doc.chunks.json --md-out data/out/doc.chunks.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from RAG_Agent.config import settings
from RAG_Agent.domain.value_objects.block_render import BlockTextFormat, render_blocks
from RAG_Agent.domain.value_objects.canonical_document import CanonicalDocument
from RAG_Agent.domain.value_objects.chunk import Chunk
from RAG_Agent.infrastructure.composition.ingest import build_chunker, run_ingest
from RAG_Agent.infrastructure.logging_config import configure_app_logging

configure_app_logging()
logger = logging.getLogger("RAG_Agent.cli.ingest_local")


def _canonical_markdown(canonical: CanonicalDocument) -> str:
    ordered = sorted(canonical.blocks.values(), key=lambda block: block.order)
    text, _ = render_blocks(ordered, fmt=BlockTextFormat.MARKDOWN)
    title = canonical.metadata.title or canonical.metadata.source_path.stem
    extra = canonical.metadata.extra
    header = (
        f"# {title}\n\n"
        f"parser=`{canonical.metadata.parser}` · "
        f"blocks={extra.get('num_blocks')} · "
        f"sections={extra.get('num_sections')} · "
        f"pages={extra.get('num_pages')}\n\n"
    )
    return header + text


def _section_outline(canonical: CanonicalDocument) -> list[dict]:
    return [
        {
            "id": section.id,
            "title": section.title,
            "level": section.level,
            "parent_id": section.parent_id,
            "page_start": section.page_start,
            "page_end": section.page_end,
            "n_blocks": len(section.block_ids),
            "block_ids": list(section.block_ids),
        }
        for section in sorted(canonical.sections, key=lambda item: item.order)
    ]


def _chunks_payload(
    source: Path, canonical: CanonicalDocument, chunks: list[Chunk]
) -> dict:
    extra = canonical.metadata.extra
    return {
        "source": str(source),
        "title": canonical.metadata.title,
        "profile_id": canonical.metadata.profile_id,
        "parser": canonical.metadata.parser,
        "chunker": settings.chunker,
        "num_pages": extra.get("num_pages"),
        "num_blocks": extra.get("num_blocks"),
        "num_sections": extra.get("num_sections"),
        "chunk_count": len(chunks),
        "sections": _section_outline(canonical),
        "chunks": [
            {
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_id": chunk.section_id,
                "block_ids": list(chunk.block_ids),
                "metadata": chunk.metadata,
                "text": chunk.text,
            }
            for chunk in chunks
        ],
    }


def _chunks_markdown(source: Path, chunks: list[Chunk]) -> str:
    lines = [f"# {source.stem}", "", f"chunker=`{settings.chunker}` · chunks={len(chunks)}", ""]
    for chunk in chunks:
        heading = chunk.metadata.get("section_path") or chunk.metadata.get("section_title") or chunk.id
        pages = f"{chunk.page_start}–{chunk.page_end}" if chunk.page_start is not None else "?"
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(f"*id=`{chunk.id}` · pages {pages}*")
        lines.append("")
        lines.append(chunk.text.strip())
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse and optionally index a PDF into Qdrant.")
    parser.add_argument("path", type=Path, help="Path to the PDF file")
    parser.add_argument(
        "--index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Chunk/embed/upsert after parse (default: true)",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Write chunk JSON (post-chunker)")
    parser.add_argument("--md-out", type=Path, default=None, help="Write chunk Markdown preview")
    parser.add_argument(
        "--canonical-out",
        type=Path,
        default=None,
        help="Write CanonicalDocument JSON (parser output, before chunker)",
    )
    parser.add_argument(
        "--canonical-md-out",
        type=Path,
        default=None,
        help="Write CanonicalDocument as concatenated Markdown",
    )
    args = parser.parse_args(argv)

    path = args.path.expanduser()
    if not path.is_file():
        logger.error("PDF not found: %s", path)
        return 1

    result = run_ingest(path, index=args.index)
    logger.info(
        "Done: doc_id=%s indexed=%s chunk_count=%s deleted=%s upserted=%s title=%r",
        result.extra.get("doc_id", path.stem),
        result.indexed,
        result.chunk_count,
        result.extra.get("deleted", "0"),
        result.extra.get("upserted", "0"),
        result.canonical.metadata.title,
    )

    if args.canonical_out or args.canonical_md_out:
        if args.canonical_out:
            args.canonical_out.parent.mkdir(parents=True, exist_ok=True)
            args.canonical_out.write_text(
                json.dumps(
                    result.canonical.to_payload(),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info(
                "Wrote canonical JSON %s (blocks=%s sections=%s)",
                args.canonical_out,
                result.canonical.metadata.extra.get("num_blocks"),
                result.canonical.metadata.extra.get("num_sections"),
            )
        if args.canonical_md_out:
            args.canonical_md_out.parent.mkdir(parents=True, exist_ok=True)
            args.canonical_md_out.write_text(
                _canonical_markdown(result.canonical),
                encoding="utf-8",
            )
            logger.info("Wrote canonical Markdown %s", args.canonical_md_out)

    if args.json_out or args.md_out:
        chunks = build_chunker().chunk(result.canonical)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(
                    _chunks_payload(path, result.canonical, chunks),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info("Wrote JSON %s (%d chunks, %d sections)", args.json_out, len(chunks), len(result.canonical.sections))
        if args.md_out:
            args.md_out.parent.mkdir(parents=True, exist_ok=True)
            args.md_out.write_text(_chunks_markdown(path, chunks), encoding="utf-8")
            logger.info("Wrote Markdown %s", args.md_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
