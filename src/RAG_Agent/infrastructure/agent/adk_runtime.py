"""ADK adapter: LlmAgent + Runner detrás del puerto AgentRuntime."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.base_session_service import BaseSessionService
from google.genai import types

from RAG_Agent.config import settings
from RAG_Agent.domain.agents.tech_docs_assistant import AGENT_INSTRUCTION, AGENT_NAME
from RAG_Agent.domain.exceptions import AgentEmptyResponseError
from RAG_Agent.infrastructure.agent.lite_llm import LiteLlm

logging.getLogger("LiteLLM").setLevel(logging.ERROR)


def build_root_agent(
    tools: Sequence[Callable[..., Any]],
    *,
    model: str | None = None,
    name: str = AGENT_NAME,
    instruction: str = AGENT_INSTRUCTION,
) -> LlmAgent:
    """Construye el ``LlmAgent`` raíz (modelo vía LiteLLM + tools)."""
    return LlmAgent(
        name=name,
        model=LiteLlm(model=model or settings.agent_model),
        instruction=instruction,
        tools=list(tools),
    )


def _final_text_from_event(event) -> str | None:
    if not event.is_final_response():
        return None
    content = event.content
    if content is None or not content.parts:
        return None
    texts = [part.text for part in content.parts if getattr(part, "text", None)]
    if not texts:
        return None
    return "\n".join(texts)


class AdkAgentRuntime:
    """Implementación de ``AgentRuntime`` sobre ADK ``Runner``."""

    def __init__(self, runner: Runner) -> None:
        self._runner = runner

    async def run_turn(self, *, user_id: str, session_id: str, message: str) -> str:
        user_message = types.Content(
            role="user",
            parts=[types.Part(text=message)],
        )
        final_text = ""
        async for event in self._runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_message,
        ):
            text = _final_text_from_event(event)
            if text is not None:
                final_text = text

        if not final_text.strip():
            raise AgentEmptyResponseError("Agent returned no final response")
        return final_text


def build_adk_runtime(
    *,
    tools: Sequence[Callable[..., Any]],
    session_service: BaseSessionService,
    app_name: str | None = None,
) -> AdkAgentRuntime:
    """Composition helper: agent + runner + runtime."""
    name = app_name or settings.agent_app_name
    root_agent = build_root_agent(tools=tools)
    runner = Runner(
        agent=root_agent,
        app_name=name,
        session_service=session_service,
    )
    return AdkAgentRuntime(runner)
