from __future__ import annotations

import logging
import os

import cohere

from RAG_Agent.config import settings
from RAG_Agent.domain.ports.embedder import DenseEmbedder
from RAG_Agent.domain.value_objects.embedding import TextEmbedding

logger = logging.getLogger(__name__)


class CohereEmbedder(DenseEmbedder):
    """Embeddings densos vía Cohere ``embed-v4.0``."""

    def __init__(self, api_key: str | None = None, *, model: str = "embed-v4.0") -> None:
        key = api_key or settings.cohere_api_key or os.environ.get("COHERE_API_KEY")
        if not key:
            raise ValueError("COHERE_API_KEY is required (pass api_key or set env var)")
        self._client = cohere.ClientV2(api_key=key)
        self._model = model

    def embed_doc(self, texts: list[str]) -> list[TextEmbedding]:
        return self._embed(texts, input_type="search_document")

    def embed_query(self, query: str) -> TextEmbedding:
        if not query.strip():
            raise ValueError("query must be non-empty")
        return self._embed([query], input_type="search_query")[0]

    def _embed(self, texts: list[str], *, input_type: str) -> list[TextEmbedding]:
        if not texts:
            return []

        logger.debug(
            "Computing dense embeddings (%s, %s) for %d texts",
            self._model,
            input_type,
            len(texts),
        )
        response = self._client.embed(
            model=self._model,
            input_type=input_type,
            embedding_types=["float"],
            texts=texts,
        )
        dense_vectors = response.embeddings.float_
        if dense_vectors is None:
            raise RuntimeError("Cohere embed response missing float embeddings")
        if len(dense_vectors) != len(texts):
            raise RuntimeError(
                f"Cohere returned {len(dense_vectors)} embeddings for {len(texts)} texts"
            )

        return [TextEmbedding(dense=tuple(vector)) for vector in dense_vectors]
