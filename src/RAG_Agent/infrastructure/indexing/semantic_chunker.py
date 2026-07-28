from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import numpy as np
import torch

from RAG_Agent.domain.ports.chunker import Chunker
from RAG_Agent.domain.value_objects.block import Block, BlockType
from RAG_Agent.domain.value_objects.block_render import BlockTextFormat, block_text
from RAG_Agent.domain.value_objects.canonical_document import CanonicalDocument
from RAG_Agent.domain.value_objects.chunk import Chunk

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+")
_ATOMIC_BLOCK_TYPES = frozenset(
    {BlockType.TABLE, BlockType.CODE, BlockType.FORMULA, BlockType.IMAGE}
)


class _Encoder(Protocol):
    def encode(self, texts: list[str], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class _Unit:
    text: str
    block_id: str
    page: int | None


class SemanticChunker(Chunker):
    """Chunking semántico: agrupa frases por similitud con límites min/max."""

    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.7,
        min_tokens: int = 64,
        max_tokens: int = 128,
        model: _Encoder | None = None,
    ) -> None:
        if min_tokens < 1 or max_tokens < min_tokens:
            raise ValueError("require 1 <= min_tokens <= max_tokens")
        self._model_name = model_name
        self._threshold = threshold
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens
        self._model = model

    def chunk(self, document: CanonicalDocument) -> list[Chunk]:
        doc_id = document.metadata.source_path.stem
        units = _units_from_document(document)
        if not units:
            return []

        groups = _group_units(
            units,
            model=self._get_model(),
            threshold=self._threshold,
            min_tokens=self._min_tokens,
            max_tokens=self._max_tokens,
        )

        chunks: list[Chunk] = []
        for index, group in enumerate(groups):
            text = " ".join(unit.text for unit in group).strip()
            if not text:
                continue
            pages = [unit.page for unit in group if unit.page is not None]
            block_ids = tuple(dict.fromkeys(unit.block_id for unit in group))
            chunks.append(
                Chunk(
                    id=f"{doc_id}:sem_{index:04d}",
                    doc_id=doc_id,
                    text=text,
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    block_ids=block_ids,
                    metadata={
                        "chunk_index": str(index),
                        "chunker": "semantic",
                        "n_units": str(len(group)),
                        "text_format": "markdown",
                    },
                )
            )
        return chunks

    def _get_model(self) -> _Encoder:
        if self._model is None:
            self._model = _load_sentence_model(self._model_name)
        return self._model


@lru_cache(maxsize=2)
def _load_sentence_model(model_name: str) -> _Encoder:
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info("Loading semantic chunker model %s on %s", model_name, device)
    return SentenceTransformer(
        model_name,
        trust_remote_code=True,
        device=device,
        model_kwargs={"torch_dtype": dtype},
    )


def _units_from_document(document: CanonicalDocument) -> list[_Unit]:
    units: list[_Unit] = []
    for block in sorted(document.blocks.values(), key=lambda item: item.order):
        units.extend(_units_from_block(block))
    return units


def _units_from_block(block: Block) -> list[_Unit]:
    text = block_text(block, fmt=BlockTextFormat.MARKDOWN)
    if not text:
        return []
    if block.type in _ATOMIC_BLOCK_TYPES or block.type == BlockType.HEADING:
        return [_Unit(text=text, block_id=block.id, page=block.page)]

    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    if not sentences:
        return [_Unit(text=text, block_id=block.id, page=block.page)]
    return [_Unit(text=sentence, block_id=block.id, page=block.page) for sentence in sentences]


def _token_length(text: str) -> int:
    return len(text.split())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _embed_pair(model: _Encoder, left: str, right: str) -> tuple[np.ndarray, np.ndarray]:
    vectors = model.encode([left, right], normalize_embeddings=True)
    arr = np.asarray(vectors, dtype=np.float32)
    return arr[0], arr[1]


def _group_units(
    units: list[_Unit],
    *,
    model: _Encoder,
    threshold: float,
    min_tokens: int,
    max_tokens: int,
) -> list[list[_Unit]]:
    if not units:
        return []

    groups: list[list[_Unit]] = []
    current: list[_Unit] = [units[0]]

    for unit in units[1:]:
        current_text = " ".join(item.text for item in current)
        combined_text = f"{current_text} {unit.text}"
        combined_tokens = _token_length(combined_text)

        if combined_tokens < min_tokens:
            if combined_tokens <= max_tokens:
                current.append(unit)
            else:
                groups.append(current)
                current = [unit]
            continue

        if combined_tokens <= max_tokens:
            left, right = _embed_pair(model, current_text, unit.text)
            if _cosine(left, right) < threshold:
                groups.append(current)
                current = [unit]
            else:
                current.append(unit)
            continue

        groups.append(current)
        current = [unit]

    if current:
        groups.append(current)
    return groups
