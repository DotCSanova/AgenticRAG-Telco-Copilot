"""Parse Pub/Sub push envelopes for GCS object notifications (no GCP SDK)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ObjectKind = Literal["pdf", "word", "unsupported"]

_PDF_SUFFIXES = frozenset({".pdf"})
_WORD_SUFFIXES = frozenset({".doc", ".docx"})


class EnvelopeError(ValueError):
    """Invalid or incomplete Pub/Sub push body."""


@dataclass(frozen=True)
class GcsObjectNotification:
    bucket: str
    object_id: str
    generation: str | None = None
    event_type: str | None = None


def parse_pubsub_gcs_envelope(body: Any) -> GcsObjectNotification:
    """Extract ``bucketId`` / ``objectId`` from a Pub/Sub push JSON body."""
    if not isinstance(body, dict):
        raise EnvelopeError("body must be a JSON object")

    message = body.get("message")
    if not isinstance(message, dict):
        raise EnvelopeError("missing message")

    attributes = message.get("attributes")
    if attributes is None:
        attributes = {}
    if not isinstance(attributes, dict):
        raise EnvelopeError("message.attributes must be an object")

    bucket = attributes.get("bucketId")
    object_id = attributes.get("objectId")
    if not bucket or not object_id:
        raise EnvelopeError("missing attributes.bucketId or attributes.objectId")
    if not isinstance(bucket, str) or not isinstance(object_id, str):
        raise EnvelopeError("bucketId and objectId must be strings")

    generation = attributes.get("objectGeneration")
    event_type = attributes.get("eventType")
    return GcsObjectNotification(
        bucket=bucket,
        object_id=object_id,
        generation=generation if isinstance(generation, str) else None,
        event_type=event_type if isinstance(event_type, str) else None,
    )


def classify_object(object_id: str) -> ObjectKind:
    """Classify by file extension (lowercase)."""
    suffix = Path(object_id).suffix.lower()
    if suffix in _PDF_SUFFIXES:
        return "pdf"
    if suffix in _WORD_SUFFIXES:
        return "word"
    return "unsupported"
