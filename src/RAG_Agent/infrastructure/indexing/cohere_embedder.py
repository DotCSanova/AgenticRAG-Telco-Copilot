from __future__ import annotations

import logging
from typing import Literal

import cohere

from RAG_Agent.config import settings
from RAG_Agent.domain.ports.embedder import DenseEmbedder
from RAG_Agent.domain.value_objects.embedding import TextEmbedding

logger = logging.getLogger(__name__)

# Cohere v2 embed accepts at most 96 texts per request.
_MAX_TEXTS_PER_REQUEST = 96

CohereInputType = Literal["search_document", "search_query"]


class CohereEmbedder(DenseEmbedder):
    """Embeddings densos vía Cohere ``embed-v4.0``."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "embed-v4.0",
        batch_size: int = _MAX_TEXTS_PER_REQUEST,
        client: cohere.ClientV2 | None = None,
    ) -> None:
        if batch_size < 1 or batch_size > _MAX_TEXTS_PER_REQUEST:
            raise ValueError(
                f"batch_size must be in 1..{_MAX_TEXTS_PER_REQUEST}, got {batch_size}"
            )
        if client is not None:
            self._client = client
        else:
            key = api_key or settings.cohere_api_key
            if not key:
                raise ValueError("COHERE_API_KEY is required (pass api_key or set in .env)")
            self._client = cohere.ClientV2(api_key=key)
        self._model = model
        self._batch_size = batch_size

    def embed_doc(self, texts: list[str]) -> list[TextEmbedding]:
        return self._embed(texts, input_type="search_document")

    def embed_query(self, query: str) -> TextEmbedding:
        if not query.strip():
            raise ValueError("query must be non-empty")
        return self._embed([query], input_type="search_query")[0]

    def _embed(
        self, texts: list[str], *, input_type: CohereInputType
    ) -> list[TextEmbedding]:
        if not texts:
            return []

        logger.debug(
            "Computing dense embeddings (%s, %s) for %d texts (batch_size=%d)",
            self._model,
            input_type,
            len(texts),
            self._batch_size,
        )
        results: list[TextEmbedding] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            results.extend(self._embed_batch(batch, input_type=input_type))
        return results

    def _embed_batch(
        self, texts: list[str], *, input_type: CohereInputType
    ) -> list[TextEmbedding]:
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
