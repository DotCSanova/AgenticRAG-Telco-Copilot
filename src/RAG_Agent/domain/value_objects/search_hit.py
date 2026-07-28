from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    """Candidato de retrieval (pre-rerank) desde el vector store."""

    text: str
    doc_id: str
    section_path: str = ""
    score: float | None = None


@dataclass(frozen=True)
class SearchHit:
    """Hit final tras retrieval (+ rerank opcional) para el agente / API."""

    text: str
    score: float
    doc_id: str
    section_path: str = ""
