from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from google.adk.agents import LlmAgent

from RAG_Agent.config import settings
from RAG_Agent.infrastructure.agent.lite_llm import LiteLlm

logging.getLogger("LiteLLM").setLevel(logging.ERROR)

INSTRUCTION = (
    "You are a precise assistant for indexed technical documents "
    "(e.g. O-RAN specifications and reports).\n\n"
    "Use `search_documents` to gather facts before answering. Call it multiple "
    "times with different queries for multi-part questions. When you quote or "
    "rely on a passage, cite its `doc_id` and `section_path`. If the indexed "
    "corpus does not cover the answer, say so — do not invent."
)


def build_root_agent(
    tools: Sequence[Callable[..., Any]],
    *,
    model: str | None = None,
    name: str = "tech_docs_assistant",
) -> LlmAgent:
    """Construye el ``LlmAgent`` raíz (Cohere vía LiteLLM + tools)."""
    return LlmAgent(
        name=name,
        model=LiteLlm(model=model or settings.agent_model),
        instruction=INSTRUCTION,
        tools=list(tools),
    )
