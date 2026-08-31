from __future__ import annotations

import re
from dataclasses import replace

from RAG_Agent.domain.value_objects.block import Block, BlockType, LayoutSpan

_START = "@startuml"
_END = "@enduml"

_TEXTUAL = frozenset({BlockType.CODE, BlockType.PARAGRAPH})

# Primer token típico de una línea PlantUML (tras comillas opcionales).
_PLANTUML_START_TOKEN = re.compile(
    r"^(?:"
    r"@startuml|@enduml|"
    r"skinparam|autonumber|title|legend|footer|header|"
    r"box|end|boundary|participant|actor|control|entity|database|collections|queue|"
    r"group|alt|opt|loop|par|break|critical|else|"
    r"note|hnote|rnote|activate|deactivate|ref|nest"
    r")\b",
    re.IGNORECASE,
)

# Flechas de secuencia / actividad
_ARROW_RE = re.compile(r"[\w.\"']+\s*[\-\.]{1,2}>{1,2}\s*[\w.\"']+")

_END_BLOCK_RE = re.compile(
    r"\bend\s+(?:box|group|note|hnote|rnote|legend|title|header|footer|alt|opt|loop|par)\b",
    re.IGNORECASE,
)

_STRONG_KEYWORDS = frozenset(
    {
        "box",
        "boundary",
        "participant",
        "actor",
        "control",
        "entity",
        "database",
        "collections",
        "queue",
        "skinparam",
        "autonumber",
        "group",
        "alt",
        "opt",
        "loop",
        "par",
        "note",
        "hnote",
        "rnote",
        "activate",
        "deactivate",
    }
)


def _text_of(block: Block) -> str:
    return (block.text or "").strip()


def _has_start(text: str) -> bool:
    return _START in text.lower()


def _has_end(text: str) -> bool:
    return _END in text.lower()


def looks_like_plantuml_fragment(text: str) -> bool:
    """Heurística ligera: keyword PlantUML, flecha, o delimitadores @start/@end."""
    cleaned = text.strip()
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if _START in lowered or _END in lowered:
        return True
    if _ARROW_RE.search(cleaned):
        return True
    if _END_BLOCK_RE.search(cleaned):
        return True

    first_line = cleaned.split("\n", 1)[0].strip()
    if _PLANTUML_START_TOKEN.match(first_line):
        # "end" suelto es demasiado genérico en prosa inglesa.
        first_token = re.split(r"\s+", first_line, maxsplit=1)[0].lower().strip("'\"")
        if first_token == "end":
            return bool(_END_BLOCK_RE.match(first_line))
        return True

    tokens = lowered.replace("'", " ").replace('"', " ").split()
    return any(token in _STRONG_KEYWORDS for token in tokens[:2])


def _is_absorbable_prefix(block: Block) -> bool:
    if block.type not in _TEXTUAL:
        return False
    text = _text_of(block)
    if not text or _has_start(text) or _has_end(text):
        return False
    return looks_like_plantuml_fragment(text)


def _is_plantuml_continuation(block: Block) -> bool:
    if block.type not in _TEXTUAL:
        return False
    text = _text_of(block)
    if not text:
        return False
    if _has_start(text) or _has_end(text):
        return True
    return looks_like_plantuml_fragment(text)


def _expand_span(blocks: list[Block], startuml_index: int) -> tuple[int, int]:
    start = startuml_index
    while start > 0 and _is_absorbable_prefix(blocks[start - 1]):
        start -= 1

    end = startuml_index
    while end + 1 < len(blocks) and not _has_end(_text_of(blocks[end])):
        nxt = blocks[end + 1]
        if not _is_plantuml_continuation(nxt):
            break
        end += 1
    return start, end


def _merge_group(group: list[Block]) -> Block:
    first = group[0]
    pages = [block.page for block in group if block.page is not None]
    page_start = pages[0] if pages else first.page
    page_end = pages[-1] if pages else first.page

    text = "\n".join(part for block in group if (part := _text_of(block)))
    metadata = dict(first.metadata)
    metadata["language"] = "plantuml"
    if page_end is not None and page_start is not None and page_end != page_start:
        metadata["page_end"] = str(page_end)
        metadata["continued"] = "true"
    elif page_end is not None:
        metadata.setdefault("page_end", str(page_end))

    if len(group) > 1:
        metadata["merged_parts"] = str(len(group))

    spans = ()
    if len(group) > 1:
        spans = tuple(
            LayoutSpan(page=part.page, bbox=part.bbox, source_ref=part.source_ref)
            for part in group
        )

    return replace(
        first,
        type=BlockType.CODE,
        page=page_start,
        text=text,
        metadata=metadata,
        bbox=first.bbox,
        source_ref=first.source_ref,
        layout_spans=spans,
    )


def merge_plantuml_fragments(blocks: list[Block]) -> list[Block]:
    """Fusiona fragments consecutivos @startuml…@enduml en un único block CODE.

    - ``page`` / ``bbox`` / ``source_ref`` = primer fragmento (cita).
    - ``layout_spans`` = un rectángulo por fragmento si cruza bloques/páginas.
    - ``metadata.page_end`` / ``continued`` si cruza páginas.
    - ``metadata.language`` = ``plantuml``.
    Incluye párrafos PlantUML inmediatamente anteriores al ``@startuml`` (p. ej. ``Box ...``).
    Does not re-number ids; ``refine_block_sequence`` does that once.
    """
    if not blocks:
        return []

    spans: list[tuple[int, int]] = []
    for index, block in enumerate(blocks):
        if block.type not in _TEXTUAL:
            continue
        if _has_start(_text_of(block)):
            spans.append(_expand_span(blocks, index))

    if not spans:
        return blocks

    # Fusionar spans solapados/adjacentes
    spans.sort()
    merged_spans: list[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged_spans[-1]
        if start <= prev_end + 1:
            merged_spans[-1] = (prev_start, max(prev_end, end))
        else:
            merged_spans.append((start, end))

    span_by_start = {start: end for start, end in merged_spans}
    covered: set[int] = set()
    for start, end in merged_spans:
        covered.update(range(start, end + 1))

    result: list[Block] = []
    index = 0
    while index < len(blocks):
        if index in span_by_start:
            end = span_by_start[index]
            result.append(_merge_group(blocks[index : end + 1]))
            index = end + 1
            continue
        if index in covered:
            index += 1
            continue
        result.append(blocks[index])
        index += 1

    return result
