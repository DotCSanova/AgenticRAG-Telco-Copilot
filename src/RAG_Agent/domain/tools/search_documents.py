"""Framework-agnostic document search tool (no agent SDK imports)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from RAG_Agent.domain.value_objects.search_hit import SearchHit

SearchFn = Callable[[str], list[SearchHit]]


def make_search_documents_tool(
    search: SearchFn,
) -> Callable[[str], Awaitable[dict[str, Any]]]:
    """Build an async tool that closes over a synchronous ``search`` callable.

    The returned tool runs ``search`` via ``asyncio.to_thread`` so Cohere/Qdrant
    (and any pacing sleeps) do not block the agent event loop. The nested
    ``search_documents`` docstring is written for the LLM; the body stays free
    of ADK, LangChain, and similar frameworks.
    """

    async def search_documents(query: str) -> dict[str, Any]:
        """Search indexed technical documents for a topic.

        Returns up to the configured rerank top-N passages with citations
        (doc_id + section_path). Call multiple times with different queries
        when a question spans several topics.

        Args:
            query: A natural-language search query.
        """
        hits = await asyncio.to_thread(search, query)
        return {"results": [_hit_to_dict(hit) for hit in hits]}

    return search_documents


def _hit_to_dict(hit: SearchHit) -> dict[str, Any]:
    return {
        "text": hit.text,
        "score": hit.score,
        "doc_id": hit.doc_id,
        "section_path": hit.section_path,
    }
