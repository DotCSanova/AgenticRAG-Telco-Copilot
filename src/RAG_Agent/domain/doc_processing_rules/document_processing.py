from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from RAG_Agent.domain.value_objects.block import Block


@dataclass(frozen=True)
class PageLayoutThresholds:
    """Umbrales relativos de layout para detectar headers, footers y portada."""

    header_top_ratio: float = 0.15
    footer_bottom_ratio: float = 0.85


@dataclass(frozen=True)
class PreprocessOptions:
    """Pasos del preprocessor físico. Nunca elimina páginas; solo limpia chrome/portada.

    El front matter semántico (TOC, foreword, …) lo filtra el normalizer vía
    ``is_removable_section``.
    """

    clean_repeated_headers_footers: bool = True
    clean_header_footer_images: bool = True
    clean_cover_page: bool = False


@dataclass(frozen=True)
class DocumentIdentity:
    """Identidad canónica de un documento, independiente de la familia (O-RAN, etc.)."""

    title_hint: str
    metadata: dict[str, str] = field(default_factory=dict)


class DocumentProcessingRules(Protocol):
    """Contrato de reglas de limpieza y normalización usado por la infra de ingesta."""

    @property
    def profile_id(self) -> str: ...

    @property
    def page_layout(self) -> PageLayoutThresholds: ...

    @property
    def preprocess_options(self) -> PreprocessOptions: ...

    @property
    def repeated_element_min_ratio(self) -> float: ...

    @property
    def removable_sections(self) -> frozenset[str]: ...

    @property
    def cover_page_number(self) -> int: ...

    @property
    def cover_title_max_paragraph_len(self) -> int: ...

    @property
    def cover_title_joined_max_len(self) -> int: ...

    @property
    def orphan_section_title(self) -> str: ...

    def normalize_text(self, text: str) -> str: ...

    def normalize_section_title(self, text: str) -> str: ...

    def is_removable_section(self, title: str) -> bool: ...

    def infer_heading_level(self, title: str, *, extracted_level: int = 1) -> int: ...

    def is_generic_doc_title(self, title: str) -> bool: ...

    def is_title_boilerplate(self, text: str) -> bool: ...

    def is_noise_paragraph(self, text: str) -> bool: ...

    def refine_blocks(self, blocks: list[Block]) -> list[Block]: ...


@dataclass(frozen=True)
class DocumentProfile:
    """Perfil resuelto para procesar un documento concreto."""

    identity: DocumentIdentity
    rules: DocumentProcessingRules


class DocumentProfileResolver(Protocol):
    """Resuelve el perfil de procesamiento a partir de la ruta del documento."""

    def matches(self, path: Path) -> bool:
        """True si este resolver aplica al documento."""

    def resolve(self, path: Path) -> DocumentProfile:
        """Devuelve el perfil aplicable. Solo llamar si ``matches`` es True."""


@dataclass(frozen=True)
class BaseDocumentRules:
    """Implementación base de DocumentProcessingRules.

    Solo define el mecanismo (campos + predicados). Los valores de heurística
    de contenido los aporta cada familia. Los umbrales numéricos de título son
    knobs del algoritmo de resolución (overrideables por familia).
    """

    profile_id: str
    sections_to_remove: tuple[str, ...] = ()
    repeated_element_min_ratio: float = 0.6
    page_layout: PageLayoutThresholds = field(default_factory=PageLayoutThresholds)
    preprocess_options: PreprocessOptions = field(default_factory=PreprocessOptions)
    generic_doc_titles: frozenset[str] = field(default_factory=frozenset)
    title_boilerplate_pattern: str = ""
    noise_paragraph_chars: frozenset[str] = field(default_factory=frozenset)
    cover_page_number: int = 1
    cover_title_max_paragraph_len: int = 180
    cover_title_joined_max_len: int = 120
    orphan_section_title: str = "Preamble"

    @property
    def removable_sections(self) -> frozenset[str]:
        return frozenset(section.lower() for section in self.sections_to_remove)

    def normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def normalize_section_title(self, text: str) -> str:
        line = text.strip().split("\n")[0].strip()
        line = re.sub(r"[\.\u2026·\s_-]+\d*\s*$", "", line)
        line = re.sub(r"\s+", " ", line).lower().strip()
        return line

    def is_removable_section(self, title: str) -> bool:
        return self.normalize_section_title(title) in self.removable_sections

    def infer_heading_level(self, title: str, *, extracted_level: int = 1) -> int:
        return extracted_level

    def is_generic_doc_title(self, title: str) -> bool:
        return title.strip().lower() in self.generic_doc_titles

    def is_title_boilerplate(self, text: str) -> bool:
        if not self.title_boilerplate_pattern:
            return False
        return bool(re.search(self.title_boilerplate_pattern, text, re.IGNORECASE))

    def is_noise_paragraph(self, text: str) -> bool:
        """True si el texto solo usa caracteres de ``noise_paragraph_chars``."""
        chars = self.noise_paragraph_chars
        if not chars:
            return False
        stripped = text.strip()
        return bool(stripped) and set(stripped) <= chars

    def refine_blocks(self, blocks: list[Block]) -> list[Block]:
        """Hook post-extracción para normalización específica de familia. Default: identidad."""
        return blocks
