from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RerankResult:
    """Resultado de rerank: índice en la lista de documentos + score."""

    index: int
    score: float


class Reranker(Protocol):
    """Reordena documentos candidatos respecto a una query."""

    def rerank(self, query: str, documents: list[str], *, top_n: int) -> list[RerankResult]:
        """Devuelve hasta ``top_n`` resultados ordenados por relevancia."""
