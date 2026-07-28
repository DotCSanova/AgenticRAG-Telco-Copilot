from __future__ import annotations

import logging

import cohere

from RAG_Agent.config import settings
from RAG_Agent.domain.ports.reranker import RerankResult, Reranker

logger = logging.getLogger(__name__)


class CohereReranker(Reranker):
    """Rerank vía Cohere ``rerank-v3.5`` (ClientV2)."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        client: cohere.ClientV2 | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            key = api_key or settings.cohere_api_key
            if not key:
                raise ValueError("COHERE_API_KEY is required (pass api_key or set in .env)")
            self._client = cohere.ClientV2(api_key=key)
        self._model = model or settings.rerank_model

    def rerank(self, query: str, documents: list[str], *, top_n: int) -> list[RerankResult]:
        if not query.strip():
            raise ValueError("query must be non-empty")
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")
        if not documents:
            return []

        n = min(top_n, len(documents))
        logger.debug("Reranking %d docs → top_n=%d (%s)", len(documents), n, self._model)
        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=n,
        )
        return [
            RerankResult(index=item.index, score=float(item.relevance_score))
            for item in response.results
        ]
