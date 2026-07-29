"""Excepciones de dominio / aplicación agentic."""


class AgentEmptyResponseError(Exception):
    """El runtime del agente no produjo una respuesta final con texto."""
