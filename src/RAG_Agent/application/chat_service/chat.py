from __future__ import annotations

import uuid
from dataclasses import dataclass

from RAG_Agent.domain.exceptions import AgentEmptyResponseError
from RAG_Agent.domain.ports.agent_runtime import AgentRuntime
from RAG_Agent.domain.ports.session_store import SessionStore


@dataclass(frozen=True)
class ChatResult:
    message: str
    user_id: str
    session_id: str


@dataclass(frozen=True)
class ChatService:
    """Caso de uso: asegurar sesión y obtener respuesta del agente."""

    runtime: AgentRuntime
    sessions: SessionStore
    app_name: str

    async def execute(
        self,
        *,
        user_id: str,
        message: str,
        session_id: str | None = None,
    ) -> ChatResult:
        if not message.strip():
            raise ValueError("message must be non-empty")

        session = await self._ensure_session(user_id=user_id, session_id=session_id)
        text = await self.runtime.run_turn(
            user_id=user_id,
            session_id=session.id,
            message=message,
        )
        if not text.strip():
            raise AgentEmptyResponseError("Agent returned no final response")

        return ChatResult(message=text, user_id=user_id, session_id=session.id)

    async def _ensure_session(self, *, user_id: str, session_id: str | None):
        if session_id:
            existing = await self.sessions.get_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id,
            )
            if existing is not None:
                return existing
        return await self.sessions.create_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id or str(uuid.uuid4()),
        )
