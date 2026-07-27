from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from RAG_Agent.domain.doc_processing_rules.default_document_rules import DefaultProfileResolver
from RAG_Agent.domain.doc_processing_rules.document_processing import (
    DocumentProfile,
    DocumentProfileResolver,
)
from RAG_Agent.domain.doc_processing_rules.oran_document_rules import OranProfileResolver


def default_profile_resolvers() -> tuple[DocumentProfileResolver, ...]:
    """Orden: familias específicas primero; default siempre al final."""
    return (OranProfileResolver(), DefaultProfileResolver())


class CascadingProfileResolver:
    """Primer ``DocumentProfileResolver`` cuyo ``matches`` gana.

    Para añadir una familia: registrar su resolver antes del default.
    """

    def __init__(
        self,
        resolvers: Sequence[DocumentProfileResolver] | None = None,
    ) -> None:
        chain = tuple(resolvers) if resolvers is not None else default_profile_resolvers()
        if not chain:
            msg = "CascadingProfileResolver requires at least one resolver"
            raise ValueError(msg)
        self._resolvers = chain

    def matches(self, path: Path) -> bool:
        path = Path(path)
        return any(resolver.matches(path) for resolver in self._resolvers)

    def resolve(self, path: Path) -> DocumentProfile:
        path = Path(path)
        for resolver in self._resolvers:
            if resolver.matches(path):
                return resolver.resolve(path)
        msg = f"No profile resolver matched: {path}"
        raise ValueError(msg)
