"""Fetch secrets from Google Secret Manager (ingest Cloud Run warm-start)."""

from __future__ import annotations

import logging
import os

from RAG_Agent.config import settings

logger = logging.getLogger(__name__)


def get_secret(secret_id: str, *, project_id: str | None = None) -> str:
    """Return the latest version of ``secret_id`` (strips surrounding quotes)."""
    project = project_id or settings.google_cloud_project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise ValueError("GOOGLE_CLOUD_PROJECT (or settings.google_cloud_project) is required")

    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    secret = response.payload.data.decode("UTF-8").strip()
    if (secret.startswith('"') and secret.endswith('"')) or (
        secret.startswith("'") and secret.endswith("'")
    ):
        secret = secret[1:-1]
    return secret


def apply_ingest_secrets_from_secret_manager() -> None:
    """Hydrate Settings from Secret Manager when ``USE_SECRET_MANAGER=true``."""
    if not settings.use_secret_manager:
        return

    logger.info("Loading ingest secrets from Secret Manager")
    settings.cohere_api_key = get_secret("cohere-api-key")
    settings.qdrant_url = get_secret("qdrant-url")
    settings.qdrant_api_key = get_secret("qdrant-api-key")
