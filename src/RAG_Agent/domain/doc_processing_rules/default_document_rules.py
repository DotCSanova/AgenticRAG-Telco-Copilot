from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from RAG_Agent.domain.doc_processing_rules.document_processing import (
    BaseDocumentRules,
    DocumentIdentity,
    DocumentProfile,
)


@dataclass(frozen=True)
class DefaultDocumentRules(BaseDocumentRules):
    """Fallback mínimo: sin heurísticas de familia (títulos, boilerplate, ruido, TOC).

    Solo chrome genérico del preprocessor (headers/footers repetidos) vía
    ``PreprocessOptions`` por defecto. El resto lo aportan familias concretas.
    """


DEFAULT_DOCUMENT_RULES = DefaultDocumentRules(profile_id="default")


class DefaultProfileResolver:
    """Fallback: aplica a cualquier PDF. Debe ir al final del cascade."""

    def matches(self, path: Path) -> bool:
        return True

    def resolve(self, path: Path) -> DocumentProfile:
        path = Path(path)
        stem = path.stem
        return DocumentProfile(
            identity=DocumentIdentity(
                title_hint=stem,
                metadata={"source_stem": stem, "family": "default"},
            ),
            rules=DEFAULT_DOCUMENT_RULES,
        )
