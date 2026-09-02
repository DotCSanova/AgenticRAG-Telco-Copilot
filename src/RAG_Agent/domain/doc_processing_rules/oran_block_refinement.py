from __future__ import annotations

import re
from dataclasses import replace

from RAG_Agent.domain.value_objects.block import Block, BlockType
from RAG_Agent.domain.value_objects.figure_groups import looks_like_figure_caption

# Títulos de sección O-RAN/ETSI sin numeración (front matter habitual).
# Se promueven aunque Docling no los marque como heading.
_NAMED_SECTION_HEADINGS: frozenset[str] = frozenset(
    {
        "introduction",
        "contents",
        "list of figures",
        "list of tables",
        "foreword",
        "modal verbs terminology",
        "executive summary",
        "change history",
        "revision history",
        "history",
    }
)

# Títulos sin número que solo se conservan si Docling ya los marcó como heading.
# No se promueven desde paragraph/list. Figuras/captions NO entran aquí.
_RETAIN_HEADING_EXACT: frozenset[str] = frozenset(
    {
        "additional information",
    }
)
_RETAIN_HEADING_PREFIXES: tuple[str, ...] = (
    "annex",
)

# 1 Title | 1.2 Title | 4.2.2 Title
_NUMERIC_SECTION_HEADING = re.compile(
    r"^(?P<num>\d+(?:\.\d+)*)\s+(?P<title>\S.+)$"
)
# A.1 Title | B.2.3 Title (anexos ETSI/O-RAN; misma estructura que 1.1)
_ANNEX_SECTION_HEADING = re.compile(
    r"^(?P<num>[A-Z](?:\.\d+)+)\s+(?P<title>\S.+)$"
)
# 1) Label…  — marcador de lista enumerada, NO sección
_ENUMERATED_LIST_MARKER = re.compile(r"^\d+\)\s*\S")

# Etiquetas cortas tipo "Near-RT RIC:" / "RAN:" usadas como padre de sub-ítems.
_LIST_LABEL_MAX_CHARS = 100
_INDENT_EPSILON_PT = 10.0


def _first_line(text: str) -> str:
    return text.strip().split("\n", 1)[0].strip()


def _normalize_named_title(text: str) -> str:
    line = _first_line(text)
    line = re.sub(r"\s+", " ", line).lower().strip()
    return line


def _named_title_matches(normalized: str, name: str) -> bool:
    """Exact title, or O-RAN compound like ``Change history/Change request``."""
    return normalized == name or normalized.startswith(f"{name}/")


def section_heading_level(text: str) -> int | None:
    """Nivel de heading de sección, o None si el texto no es un heading de sección."""
    line = _first_line(text)
    if not line:
        return None

    if any(
        _named_title_matches(_normalize_named_title(line), name)
        for name in _NAMED_SECTION_HEADINGS
    ):
        return 1

    match = _NUMERIC_SECTION_HEADING.match(line)
    if match:
        return match.group("num").count(".") + 1

    match = _ANNEX_SECTION_HEADING.match(line)
    if match:
        return match.group("num").count(".") + 1

    return None


def is_retained_unnumbered_heading(text: str) -> bool:
    """True para Annex / Additional information.

    Solo debe usarse para no degradar un heading ya detectado (no promoción).
    """
    normalized = _normalize_named_title(text)
    if not normalized or looks_like_figure_caption(normalized):
        return False
    if normalized in _RETAIN_HEADING_EXACT:
        return True
    return any(normalized.startswith(prefix) for prefix in _RETAIN_HEADING_PREFIXES)


def is_section_heading_text(text: str) -> bool:
    return section_heading_level(text) is not None


def is_enumerated_list_marker(text: str) -> bool:
    """True para ``1) Non-RT RIC:`` y similares (no son secciones ``1.2 Title``)."""
    return bool(_ENUMERATED_LIST_MARKER.match(_first_line(text)))


def is_list_label(text: str) -> bool:
    """Padre de sub-lista: marcador ``N)`` o etiqueta corta terminada en ``:``."""
    line = _first_line(text)
    if not line or len(line) > _LIST_LABEL_MAX_CHARS:
        return False
    if is_enumerated_list_marker(line):
        return True
    if line.endswith(":") and len(line) >= 2:
        # Evitar prosa larga disfrazada de etiqueta.
        return line.count(" ") <= 12
    return False


def _block_x0(block: Block) -> float | None:
    if block.bbox is None:
        return None
    return block.bbox.x0


def _assign_list_levels(blocks: list[Block]) -> list[Block]:
    """Asigna level a LIST_ITEM por indentación relativa dentro de cada run.

    Un run es una secuencia de list_items entre headings. El x0 mínimo del run
    es nivel 1; mayor indentación → nivel 2 (+). Las etiquetas (``N)`` / ``…:``)
    fuerzan nivel 1.
    """
    result = list(blocks)
    index = 0
    while index < len(result):
        if result[index].type != BlockType.LIST_ITEM:
            index += 1
            continue

        run_start = index
        while index < len(result) and result[index].type == BlockType.LIST_ITEM:
            index += 1
        run_end = index
        run = result[run_start:run_end]

        x0_values = [x for block in run if (x := _block_x0(block)) is not None]
        base_x0 = min(x0_values) if x0_values else None

        for offset, block in enumerate(run):
            if is_list_label(block.text or ""):
                level = 1
            elif base_x0 is None:
                level = 1
            else:
                x0 = _block_x0(block)
                if x0 is None or x0 <= base_x0 + _INDENT_EPSILON_PT:
                    level = 1
                elif x0 <= base_x0 + _INDENT_EPSILON_PT + 25.0:
                    level = 2
                else:
                    level = 3
            result[run_start + offset] = replace(block, level=level)

    return result


def refine_oran_blocks(blocks: list[Block]) -> list[Block]:
    """Corrige headings/listas mal clasificados por Docling (reglas estructurales O-RAN).

    1. Conserva/promueve headings de sección numerada, anexo ``A.1`` o título nombrado.
    2. Conserva (sin promover) headings ya detectados tipo Annex / Additional information.
    3. Degrada falsos headings: ``N)`` / etiquetas ``…:`` → list_item; figuras y resto → paragraph.
    4. Asigna ``level`` a list_items por indentación + etiquetas padre.
    """
    refined: list[Block] = []

    for block in blocks:
        text = (block.text or "").strip()
        heading_level = section_heading_level(text) if text else None

        if heading_level is not None:
            refined.append(
                replace(
                    block,
                    type=BlockType.HEADING,
                    level=heading_level,
                    text=text or block.text,
                )
            )
            continue

        if block.type == BlockType.HEADING:
            # Antes de tratar ``…:`` como etiqueta de lista (Annex A (informative):).
            if text and is_retained_unnumbered_heading(text):
                refined.append(
                    replace(
                        block,
                        type=BlockType.HEADING,
                        level=block.level or 1,
                        text=text,
                    )
                )
            elif text and (is_enumerated_list_marker(text) or is_list_label(text)):
                refined.append(
                    replace(block, type=BlockType.LIST_ITEM, level=None, text=text)
                )
            else:
                refined.append(
                    replace(block, type=BlockType.PARAGRAPH, level=None, text=text or block.text)
                )
            continue

        refined.append(block)

    return _assign_list_levels(refined)
