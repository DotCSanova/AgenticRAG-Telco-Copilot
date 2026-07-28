"""ADK agent package for Dev UI (`get_fast_api_app` / `adk web`).

Expects ``root_agent`` at import time. Wiring is intentional for a
dev-only entrypoint (not the product composition root).
"""

from __future__ import annotations

from RAG_Agent.infrastructure.agent.agent import build_root_agent
from RAG_Agent.infrastructure.agent.tools.search_documents import make_search_documents_tool
from RAG_Agent.infrastructure.composition import build_search_service

_search_service = build_search_service()
root_agent = build_root_agent(tools=[make_search_documents_tool(_search_service)])
