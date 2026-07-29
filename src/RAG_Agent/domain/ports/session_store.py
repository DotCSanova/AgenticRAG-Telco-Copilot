from __future__ import annotations

from typing import Protocol


class ChatSession(Protocol):
    """Sesión de chat mínima (id estable)."""

    @property
    def id(self) -> str: ...


class SessionStore(Protocol):
    """Puerto: persistencia de sesiones del agente (sin acoplar a ADK)."""

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> ChatSession | None: ...

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str | None = None,
    ) -> ChatSession: ...

    async def delete_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> None: ...
