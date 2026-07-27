from __future__ import annotations

from pathlib import Path
from typing import Protocol

from RAG_Agent.domain.value_objects.canonical_document import CanonicalDocument


class DocumentParser(Protocol):
    """Puerto de dominio: convierte una fuente documental en CanonicalDocument."""

    def parse(self, path: Path) -> CanonicalDocument:
        """Parsea el documento en la ruta indicada."""
