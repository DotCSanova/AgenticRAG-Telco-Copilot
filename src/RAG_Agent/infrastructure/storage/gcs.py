"""Download GCS objects to temporary local paths (ADC / workload identity)."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def download_gcs_object(
    bucket: str,
    object_name: str,
    *,
    client: Any | None = None,
) -> Iterator[Path]:
    """Download ``gs://bucket/object_name`` to a temp file; delete the temp dir on exit.

    The local filename is the object basename (e.g. ``docs/WG1.pdf`` → ``…/WG1.pdf``)
    so ``run_ingest`` derives ``doc_id`` from the same stem as a local CLI path.

    ``client`` is injectable for tests. When omitted, uses ``google.cloud.storage.Client``
    (Application Default Credentials — no keys in the image).
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="rag_gcs_"))
    local_path = tmp_dir / Path(object_name).name
    try:
        storage_client = client if client is not None else _default_client()
        blob = storage_client.bucket(bucket).blob(object_name)
        blob.download_to_filename(str(local_path))
        yield local_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _default_client() -> Any:
    from google.cloud import storage

    return storage.Client()
