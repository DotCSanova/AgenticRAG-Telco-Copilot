from __future__ import annotations

from typing import Protocol

from RAG_Agent.domain.value_objects.canonical_document import CanonicalDocument
from RAG_Agent.domain.value_objects.chunk import Chunk


class Chunker(Protocol):
    """Parte un CanonicalDocument en chunks indexables."""

    def chunk(self, document: CanonicalDocument) -> list[Chunk]:
        """Devuelve chunks ordenados listos para embeber."""
