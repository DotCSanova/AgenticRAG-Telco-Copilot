from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from RAG_Agent.domain.doc_processing_rules.document_processing import (
    BaseDocumentRules,
    DocumentIdentity,
    DocumentProfile,
    PreprocessOptions,
)
from RAG_Agent.domain.doc_processing_rules.oran_block_refinement import (
    refine_oran_blocks,
    section_heading_level,
)
from RAG_Agent.domain.value_objects.block import Block


# Front matter lo filtra el normalizer (is_removable_section). Preprocess solo chrome/portada.
_ORAN_PREPROCESS = PreprocessOptions(
    clean_repeated_headers_footers=True,
    clean_header_footer_images=True,
    clean_cover_page=True,
)

# Front matter / títulos genéricos comunes en especificaciones O-RAN (estilo ETSI).
_ORAN_GENERIC_DOC_TITLES: frozenset[str] = frozenset(
    {
        "introduction",
        "preamble",
        "scope",
        "contents",
        "foreword",
        "references",
        "abstract",
        "executive summary",
        "modal verbs terminology",
    }
)

# Portada O-RAN Alliance: VAT / registro de asociaciones.
_ORAN_TITLE_BOILERPLATE_PATTERN = (
    r"copyright|all rights reserved|vat\s*id|register of associations|"
    r"confidential|do not distribute|©"
)

_ORAN_NOISE_PARAGRAPH_CHARS: frozenset[str] = frozenset(
    {"_", "—", "-", " ", "\u00a0"}
)

# Estricto: O-RAN.<WG\d+|…FG>.<segmentos>[-R<release>]-vNN.MM
# Segmentos permiten guiones (Use-Cases-…) pero no absorben el -R… antes de -v.
ORAN_DOCUMENT_ID_PATTERN = re.compile(
    r"^O-RAN\."
    r"(?P<group>WG\d+|[A-Za-z]+FG)"
    r"(?P<body>(?:\.[A-Za-z0-9]+(?:-(?!R\d+-v)[A-Za-z0-9]+)*)+)"
    r"(?:-R(?P<release>\d+))?"
    r"-v(?P<version>\d{2}\.\d{2})$",
    re.IGNORECASE,
)

COMMON_ORAN_FRONT_MATTER: tuple[str, ...] = (
    "contents",
    "list of figures",
    "list of tables",
    "foreword",
    "modal verbs terminology",
)


@dataclass(frozen=True)
class OranDocumentId:
    """Identidad de un documento O-RAN extraída del nombre de archivo.

    Nomenclatura: ``O-RAN.<group>.<seg>…[-R<release>]-vNN.MM``
    donde ``group`` es un Work Group (``WG1``) o Focus Group (``SuFG``).
    """

    group: str
    segments: tuple[str, ...]
    doc_type: str
    subject: str
    version: str
    release: str | None = None

    @classmethod
    def from_path(cls, path: Path | str) -> OranDocumentId | None:
        stem = Path(path).stem
        match = ORAN_DOCUMENT_ID_PATTERN.match(stem)
        if match is None:
            return None

        group = match.group("group")
        if re.fullmatch(r"WG\d+", group, re.IGNORECASE):
            group = f"WG{group[2:]}"
        # Focus groups (SuFG, …): conservar casing del nombre de fichero.

        body = match.group("body").lstrip(".")
        segments = tuple(part for part in body.split(".") if part)
        if not segments:
            return None

        doc_type = segments[0].lower()
        subject = "-".join(segments[1:]) if len(segments) > 1 else segments[0]

        return cls(
            group=group,
            segments=segments,
            doc_type=doc_type,
            subject=subject,
            version=match.group("version"),
            release=match.group("release"),
        )

    def to_dict(self) -> dict[str, str]:
        result = {
            "family": "oran",
            "group": self.group,
            "doc_type": self.doc_type,
            "subject": self.subject,
            "version": self.version,
            "segments": ".".join(self.segments),
        }
        if self.release is not None:
            result["release"] = self.release
        return result

    def to_identity(self) -> DocumentIdentity:
        return DocumentIdentity(title_hint=self.subject, metadata=self.to_dict())

    def subject_contains(self, *keywords: str) -> bool:
        subject = self.subject.lower()
        return any(keyword.lower() in subject for keyword in keywords)


@dataclass(frozen=True)
class OranDocumentRules(BaseDocumentRules):
    """Reglas de negocio para procesar un documento O-RAN.

    Extiende BaseDocumentRules con heurísticas de familia (título, boilerplate,
    niveles numerados). Añade aquí nuevas reglas de normalización O-RAN.
    """

    preprocess_options: PreprocessOptions = field(default_factory=lambda: _ORAN_PREPROCESS)
    generic_doc_titles: frozenset[str] = field(default_factory=lambda: _ORAN_GENERIC_DOC_TITLES)
    title_boilerplate_pattern: str = _ORAN_TITLE_BOILERPLATE_PATTERN
    cover_title_joined_max_len: int = 240
    noise_paragraph_chars: frozenset[str] = field(
        default_factory=lambda: _ORAN_NOISE_PARAGRAPH_CHARS
    )

    def infer_heading_level(self, title: str, *, extracted_level: int = 1) -> int:
        """Infiere el nivel jerárquico a partir del prefijo numérico / anexo O-RAN."""
        level = section_heading_level(title)
        return level if level is not None else extracted_level

    def refine_blocks(self, blocks: list[Block]) -> list[Block]:
        """Corrige headings y niveles de lista mal detectados por Docling."""
        return refine_oran_blocks(blocks)


@dataclass(frozen=True)
class OranProfileSpec:
    """Criterios para seleccionar un perfil. ``None`` / vacío = comodín."""

    profile_id: str
    group: str | None = None
    doc_types: frozenset[str] | None = None
    subject_keywords: tuple[str, ...] = ()

    def matches(self, document_id: OranDocumentId) -> bool:
        if self.group is not None and document_id.group.upper() != self.group.upper():
            return False
        if self.doc_types is not None and document_id.doc_type not in self.doc_types:
            return False
        if self.subject_keywords and not document_id.subject_contains(*self.subject_keywords):
            return False
        return True


class OranDocumentRulesRegistry:
    """Registro extensible de perfiles O-RAN. Primer match (orden de registro) gana."""

    def __init__(self) -> None:
        self._rules: dict[str, OranDocumentRules] = {}
        self._specs: list[OranProfileSpec] = []

    def register(self, spec: OranProfileSpec, rules: OranDocumentRules) -> None:
        if spec.profile_id != rules.profile_id:
            msg = f"profile_id mismatch: {spec.profile_id!r} vs {rules.profile_id!r}"
            raise ValueError(msg)
        self._rules[spec.profile_id] = rules
        self._specs = [entry for entry in self._specs if entry.profile_id != spec.profile_id]
        self._specs.append(spec)

    def get(self, profile_id: str) -> OranDocumentRules:
        return self._rules[profile_id]

    def resolve(self, document_id: OranDocumentId) -> OranDocumentRules:
        for spec in self._specs:
            if spec.matches(document_id):
                return self._rules[spec.profile_id]

        msg = (
            f"No hay reglas registradas para O-RAN {document_id.group} "
            f"{document_id.doc_type}"
        )
        raise ValueError(msg)


class OranProfileResolver:
    """DocumentProfileResolver concreto para documentos O-RAN."""

    def __init__(self, registry: OranDocumentRulesRegistry | None = None) -> None:
        self._registry = registry or ORAN_RULES_REGISTRY

    def matches(self, path: Path) -> bool:
        return OranDocumentId.from_path(path) is not None

    def resolve(self, path: Path) -> DocumentProfile:
        path = Path(path)
        document_id = OranDocumentId.from_path(path)
        if document_id is None:
            msg = f"No es un documento O-RAN reconocido: {path}"
            raise ValueError(msg)
        rules = self._registry.resolve(document_id)
        return DocumentProfile(identity=document_id.to_identity(), rules=rules)


def _build_default_registry() -> OranDocumentRulesRegistry:
    registry = OranDocumentRulesRegistry()
    registry.register(
        OranProfileSpec(profile_id="oran_default"),
        OranDocumentRules(
            profile_id="oran_default",
            sections_to_remove=COMMON_ORAN_FRONT_MATTER,
        ),
    )
    return registry


ORAN_RULES_REGISTRY = _build_default_registry()
