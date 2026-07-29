from __future__ import annotations

from dataclasses import dataclass

from RAG_Agent.domain.ports.session_store import SessionStore


@dataclass(frozen=True)
class ResetMemoryResult:
    user_id: str
    session_id: str
    reset: bool = True


@dataclass(frozen=True)
class ResetMemoryService:
    """Caso de uso: borrar y recrear la sesión del agente."""

    sessions: SessionStore
    app_name: str

    async def execute(self, *, user_id: str, session_id: str) -> ResetMemoryResult:
        if not session_id.strip():
            raise ValueError("session_id must be non-empty")

        existing = await self.sessions.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if existing is not None:
            await self.sessions.delete_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id,
            )

        await self.sessions.create_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        return ResetMemoryResult(user_id=user_id, session_id=session_id, reset=True)
