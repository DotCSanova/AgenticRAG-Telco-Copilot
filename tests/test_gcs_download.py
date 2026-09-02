from pathlib import Path

import pytest

from RAG_Agent.infrastructure.storage.gcs import download_gcs_object


class _FakeBlob:
    def __init__(self, data: bytes, *, fail: bool = False) -> None:
        self._data = data
        self._fail = fail

    def download_to_filename(self, path: str) -> None:
        if self._fail:
            raise RuntimeError("download failed")
        Path(path).write_bytes(self._data)


class _FakeBucket:
    def __init__(self, blob: _FakeBlob) -> None:
        self._blob = blob
        self.requested_name: str | None = None

    def blob(self, name: str) -> _FakeBlob:
        self.requested_name = name
        return self._blob


class _FakeClient:
    def __init__(self, data: bytes = b"%PDF-1.4 fake", *, fail: bool = False) -> None:
        self._blob = _FakeBlob(data, fail=fail)
        self.requested_bucket: str | None = None
        self._bucket = _FakeBucket(self._blob)

    def bucket(self, name: str) -> _FakeBucket:
        self.requested_bucket = name
        return self._bucket


def test_download_gcs_object_preserves_basename_for_doc_id():
    client = _FakeClient(b"%PDF-1.4 content")
    with download_gcs_object("my-bucket", "docs/O-RAN/WG1.pdf", client=client) as path:
        assert client.requested_bucket == "my-bucket"
        assert client._bucket.requested_name == "docs/O-RAN/WG1.pdf"
        assert path.name == "WG1.pdf"
        assert path.stem == "WG1"
        assert path.is_file()
        assert path.read_bytes() == b"%PDF-1.4 content"
        kept = path
        tmp_dir = path.parent

    assert not kept.exists()
    assert not tmp_dir.exists()


def test_download_gcs_object_cleans_up_after_caller_error():
    client = _FakeClient()
    with pytest.raises(ValueError, match="boom"):
        with download_gcs_object("b", "docs/WG1.pdf", client=client) as path:
            assert path.name == "WG1.pdf"
            assert path.is_file()
            kept = path
            tmp_dir = path.parent
            raise ValueError("boom")

    assert not kept.exists()
    assert not tmp_dir.exists()


def test_download_gcs_object_cleans_up_after_download_error():
    client = _FakeClient(fail=True)
    with pytest.raises(RuntimeError, match="download failed"):
        with download_gcs_object("b", "x.pdf", client=client):
            pass
