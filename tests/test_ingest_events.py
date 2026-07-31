import pytest

from RAG_Agent.infrastructure.api.ingest_events import (
    EnvelopeError,
    classify_object,
    parse_pubsub_gcs_envelope,
)


def test_parse_pubsub_gcs_envelope_reads_attributes():
    body = {
        "message": {
            "attributes": {
                "eventType": "OBJECT_FINALIZE",
                "bucketId": "my-bucket",
                "objectId": "docs/WG1.pdf",
                "objectGeneration": "123",
            }
        }
    }
    note = parse_pubsub_gcs_envelope(body)
    assert note.bucket == "my-bucket"
    assert note.object_id == "docs/WG1.pdf"
    assert note.generation == "123"
    assert note.event_type == "OBJECT_FINALIZE"


def test_parse_pubsub_gcs_envelope_missing_message():
    with pytest.raises(EnvelopeError, match="missing message"):
        parse_pubsub_gcs_envelope({})


def test_parse_pubsub_gcs_envelope_missing_ids():
    with pytest.raises(EnvelopeError, match="bucketId"):
        parse_pubsub_gcs_envelope({"message": {"attributes": {"bucketId": "b"}}})


@pytest.mark.parametrize(
    ("object_id", "kind"),
    [
        ("a.pdf", "pdf"),
        ("docs/WG1.PDF", "pdf"),
        ("x.doc", "word"),
        ("y.docx", "word"),
        ("z.txt", "unsupported"),
        ("noext", "unsupported"),
    ],
)
def test_classify_object(object_id: str, kind: str):
    assert classify_object(object_id) == kind
