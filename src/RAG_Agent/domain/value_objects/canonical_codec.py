"""JSON codec for CanonicalDocument (schema 1.0).

In-memory ``DocumentMetadata.extra`` stays a string bag. The dump splits identity,
stats, and ingest leftovers; ``canonical_from_payload`` writes them back into
``extra`` so merge, CLI markdown, and sharding tests keep working.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from RAG_Agent.domain.value_objects.block import (
    Block,
    BlockType,
    BoundingBox,
    ImageRef,
    LayoutSpan,
    TableData,
    coord_origin_name,
)
from RAG_Agent.domain.value_objects.canonical_document import (
    CanonicalDocument,
    DocumentMetadata,
)
from RAG_Agent.domain.value_objects.page import Page
from RAG_Agent.domain.value_objects.section import Section

SCHEMA_VERSION = "1.0"

_IDENTITY_KEYS = ("group", "doc_type", "subject", "version", "segments", "release")
_STATS_KEYS = ("num_pages", "num_blocks", "num_sections", "num_shards")
_INGEST_KEYS = ("ingest_profile", "pages_per_shard", "failed_shards")
_FAMILY_KEY = "family"
_DOCLING_VERSION_KEY = "docling_version"


def canonical_to_payload(document: CanonicalDocument) -> dict[str, Any]:
    """Serialize ``document`` to a schema 1.0 JSON-ready dict.

    Args:
        document: In-memory canonical aggregate.

    Returns:
        Payload with ``schema_version``, typed stats, and blocks as an ordered list.
        Irrelevant nulls on blocks are omitted; section ``parent_id`` is always set.
    """
    extra = document.metadata.extra
    metadata: dict[str, Any] = {
        "source_path": document.metadata.source_path.as_posix(),
    }
    if document.metadata.title:
        metadata["title"] = document.metadata.title
    if document.metadata.parser:
        metadata["parser"] = document.metadata.parser
    docling_version = extra.get(_DOCLING_VERSION_KEY)
    if docling_version:
        metadata["docling_version"] = docling_version
    if document.metadata.profile_id:
        metadata["profile_id"] = document.metadata.profile_id
    family = extra.get(_FAMILY_KEY)
    if family:
        metadata["family"] = family
    identity = {key: extra[key] for key in _IDENTITY_KEYS if extra.get(key)}
    if identity:
        metadata["identity"] = identity
    stats = _stats_payload(extra)
    if stats:
        metadata["stats"] = stats
    ingest = {key: extra[key] for key in _INGEST_KEYS if extra.get(key)}
    if ingest:
        metadata["ingest"] = ingest
    leftover = _leftover_extra(extra)
    if leftover:
        metadata["extra"] = leftover

    ordered_blocks = sorted(document.blocks.values(), key=lambda block: block.order)
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "pages": [_page_payload(page) for page in document.pages],
        "sections": [
            _section_payload(section)
            for section in sorted(document.sections, key=lambda item: item.order)
        ],
        "blocks": [_block_payload(block) for block in ordered_blocks],
    }


def canonical_from_payload(data: Mapping[str, Any]) -> CanonicalDocument:
    """Rebuild a ``CanonicalDocument`` from a schema 1.0 payload.

    Args:
        data: Mapping produced by ``canonical_to_payload`` (or an equivalent dump).
            Unknown keys are ignored.

    Returns:
        Aggregate whose ``extra`` bag is restored (stats as strings).

    Raises:
        ValueError: If ``schema_version`` is missing or not ``1.0``.
        KeyError: If required nested fields are missing.
    """
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"canonical payload requires schema_version {SCHEMA_VERSION!r}, got {version!r}"
        )
    meta = data.get("metadata")
    if not isinstance(meta, Mapping):
        raise ValueError("canonical payload metadata must be an object")
    source_path = meta.get("source_path")
    if not source_path:
        raise ValueError("canonical payload metadata.source_path is required")

    extra = _extra_from_metadata(meta)
    blocks = {
        block.id: block
        for block in (_block_from_payload(item) for item in data.get("blocks") or [])
    }
    return CanonicalDocument(
        metadata=DocumentMetadata(
            source_path=Path(str(source_path)),
            title=str(meta["title"]) if meta.get("title") else None,
            profile_id=str(meta["profile_id"]) if meta.get("profile_id") else None,
            parser=str(meta["parser"]) if meta.get("parser") else None,
            extra=extra,
        ),
        blocks=blocks,
        pages=[_page_from_payload(item) for item in data.get("pages") or []],
        sections=[_section_from_payload(item) for item in data.get("sections") or []],
    )


def _stats_payload(extra: Mapping[str, str]) -> dict[str, int]:
    stats: dict[str, int] = {}
    for key in _STATS_KEYS:
        raw = extra.get(key)
        if raw is None or raw == "":
            continue
        stats[key] = int(raw)
    return stats


def _leftover_extra(extra: Mapping[str, str]) -> dict[str, str]:
    reserved = {_FAMILY_KEY, _DOCLING_VERSION_KEY, *_IDENTITY_KEYS, *_STATS_KEYS, *_INGEST_KEYS}
    return {key: value for key, value in extra.items() if key not in reserved and value}


def _extra_from_metadata(meta: Mapping[str, Any]) -> dict[str, str]:
    extra: dict[str, str] = {}
    family = meta.get(_FAMILY_KEY)
    if family:
        extra[_FAMILY_KEY] = str(family)
    identity = meta.get("identity")
    if isinstance(identity, Mapping):
        for key, value in identity.items():
            extra[str(key)] = str(value)
    stats = meta.get("stats")
    if isinstance(stats, Mapping):
        for key, value in stats.items():
            extra[str(key)] = str(value)
    docling_version = meta.get(_DOCLING_VERSION_KEY)
    if docling_version:
        extra[_DOCLING_VERSION_KEY] = str(docling_version)
    ingest = meta.get("ingest")
    if isinstance(ingest, Mapping):
        for key, value in ingest.items():
            extra[str(key)] = str(value)
    leftover = meta.get("extra")
    if isinstance(leftover, Mapping):
        for key, value in leftover.items():
            extra[str(key)] = str(value)
    return extra


def _page_payload(page: Page) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "number": page.number,
        "block_ids": list(page.block_ids),
    }
    if page.width is not None:
        payload["width"] = page.width
    if page.height is not None:
        payload["height"] = page.height
    return payload


def _page_from_payload(data: Mapping[str, Any]) -> Page:
    return Page(
        number=int(data["number"]),
        block_ids=[str(item) for item in data.get("block_ids") or []],
        width=float(data["width"]) if data.get("width") is not None else None,
        height=float(data["height"]) if data.get("height") is not None else None,
    )


def _section_payload(section: Section) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": section.id,
        "title": section.title,
        "level": section.level,
        "order": section.order,
        "parent_id": section.parent_id,
        "block_ids": list(section.block_ids),
    }
    if section.page_start is not None:
        payload["page_start"] = section.page_start
    if section.page_end is not None:
        payload["page_end"] = section.page_end
    if section.metadata:
        payload["metadata"] = dict(section.metadata)
    return payload


def _section_from_payload(data: Mapping[str, Any]) -> Section:
    parent_id = data.get("parent_id")
    return Section(
        id=str(data["id"]),
        title=str(data["title"]),
        level=int(data["level"]),
        order=int(data["order"]),
        parent_id=str(parent_id) if parent_id is not None else None,
        block_ids=[str(item) for item in data.get("block_ids") or []],
        page_start=int(data["page_start"]) if data.get("page_start") is not None else None,
        page_end=int(data["page_end"]) if data.get("page_end") is not None else None,
        metadata={str(key): str(value) for key, value in (data.get("metadata") or {}).items()},
    )


def _block_payload(block: Block) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": block.id,
        "type": block.type.value,
        "order": block.order,
    }
    if block.page is not None:
        payload["page"] = block.page
    if block.text is not None:
        payload["text"] = block.text
    if block.level is not None:
        payload["level"] = block.level
    if block.table is not None:
        payload["table"] = _table_payload(block.table)
    if block.image is not None:
        image = _image_payload(block.image)
        if image:
            payload["image"] = image
    if block.bbox is not None:
        payload["bbox"] = _bbox_payload(block.bbox)
    if block.source_ref:
        payload["source_ref"] = block.source_ref
    if block.metadata:
        payload["metadata"] = dict(block.metadata)
    spans = _layout_spans_payload(block)
    if spans:
        payload["layout_spans"] = spans
    return payload


def _block_from_payload(data: Mapping[str, Any]) -> Block:
    table_data = data.get("table")
    image_data = data.get("image")
    bbox_data = data.get("bbox")
    spans_data = data.get("layout_spans")
    return Block(
        id=str(data["id"]),
        type=BlockType(str(data["type"])),
        order=int(data["order"]),
        page=int(data["page"]) if data.get("page") is not None else None,
        text=str(data["text"]) if data.get("text") is not None else None,
        level=int(data["level"]) if data.get("level") is not None else None,
        table=_table_from_payload(table_data) if isinstance(table_data, Mapping) else None,
        image=_image_from_payload(image_data) if isinstance(image_data, Mapping) else None,
        bbox=_bbox_from_payload(bbox_data) if isinstance(bbox_data, Mapping) else None,
        source_ref=str(data["source_ref"]) if data.get("source_ref") else None,
        metadata={str(key): str(value) for key, value in (data.get("metadata") or {}).items()},
        layout_spans=tuple(_span_from_payload(item) for item in spans_data or ()),
    )


def _table_payload(table: TableData) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "headers": list(table.headers),
        "rows": [list(row) for row in table.rows],
    }
    if table.caption:
        payload["caption"] = table.caption
    return payload


def _table_from_payload(data: Mapping[str, Any]) -> TableData:
    return TableData(
        headers=[str(item) for item in data.get("headers") or []],
        rows=[[str(cell) for cell in row] for row in data.get("rows") or []],
        caption=str(data["caption"]) if data.get("caption") else None,
    )


def _image_payload(image: ImageRef) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if image.uri:
        payload["uri"] = image.uri
    if image.alt:
        payload["alt"] = image.alt
    if image.caption:
        payload["caption"] = image.caption
    return payload


def _image_from_payload(data: Mapping[str, Any]) -> ImageRef:
    return ImageRef(
        uri=str(data["uri"]) if data.get("uri") else None,
        alt=str(data["alt"]) if data.get("alt") else None,
        caption=str(data["caption"]) if data.get("caption") else None,
    )


def _bbox_payload(bbox: BoundingBox) -> dict[str, Any]:
    return {
        "x0": bbox.x0,
        "y0": bbox.y0,
        "x1": bbox.x1,
        "y1": bbox.y1,
        "coord_origin": coord_origin_name(bbox.coord_origin),
    }


def _bbox_from_payload(data: Mapping[str, Any]) -> BoundingBox:
    return BoundingBox(
        x0=float(data["x0"]),
        y0=float(data["y0"]),
        x1=float(data["x1"]),
        y1=float(data["y1"]),
        coord_origin=coord_origin_name(data.get("coord_origin")),
    )


def _layout_spans_payload(block: Block) -> list[dict[str, Any]]:
    if not block.layout_spans:
        return []
    if len(block.layout_spans) == 1 and _span_matches_block(block, block.layout_spans[0]):
        return []
    return [_span_payload(span) for span in block.layout_spans]


def _span_matches_block(block: Block, span: LayoutSpan) -> bool:
    return (
        span.page == block.page
        and span.source_ref == block.source_ref
        and span.bbox == block.bbox
    )


def _span_payload(span: LayoutSpan) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if span.page is not None:
        payload["page"] = span.page
    if span.bbox is not None:
        payload["bbox"] = _bbox_payload(span.bbox)
    if span.source_ref:
        payload["source_ref"] = span.source_ref
    return payload


def _span_from_payload(data: Mapping[str, Any]) -> LayoutSpan:
    bbox_data = data.get("bbox")
    return LayoutSpan(
        page=int(data["page"]) if data.get("page") is not None else None,
        bbox=_bbox_from_payload(bbox_data) if isinstance(bbox_data, Mapping) else None,
        source_ref=str(data["source_ref"]) if data.get("source_ref") else None,
    )
