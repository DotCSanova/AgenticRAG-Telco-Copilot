from __future__ import annotations

from typing import Protocol


class AgentRuntime(Protocol):
    """Puerto: ejecuta un turno de conversación del agente."""

    async def run_turn(self, *, user_id: str, session_id: str, message: str) -> str:
        """Devuelve el texto final de la respuesta del agente."""
