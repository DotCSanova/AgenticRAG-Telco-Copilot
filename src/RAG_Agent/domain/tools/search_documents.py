"""Tool cruda de búsqueda documental (sin frameworks de agente)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from RAG_Agent.domain.value_objects.search_hit import SearchHit

SearchFn = Callable[[str], list[SearchHit]]


def make_search_documents_tool(search: SearchFn) -> Callable[[str], dict[str, Any]]:
    """Factory framework-agnostic: cierra sobre una función de búsqueda.

    La docstring de ``search_documents`` está pensada para el LLM; el cuerpo
    no depende de ADK, LangChain ni similares.
    """

    def search_documents(query: str) -> dict[str, Any]:
        """Search indexed technical documents for a topic.

        Returns up to the configured rerank top-N passages with citations
        (doc_id + section_path). Call multiple times with different queries
        when a question spans several topics.

        Args:
            query: A natural-language search query.
        """
        hits = search(query)
        return {"results": [_hit_to_dict(hit) for hit in hits]}

    return search_documents


def _hit_to_dict(hit: SearchHit) -> dict[str, Any]:
    return {
        "text": hit.text,
        "score": hit.score,
        "doc_id": hit.doc_id,
        "section_path": hit.section_path,
    }
