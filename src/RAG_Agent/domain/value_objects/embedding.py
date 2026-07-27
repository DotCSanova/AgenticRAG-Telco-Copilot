from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SparseEmbedding:
    """Embedding disperso (índices + pesos), p. ej. BM25 / SPLADE."""

    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            msg = "sparse indices and values must have the same length"
            raise ValueError(msg)


@dataclass(frozen=True)
class TextEmbedding:
    """Embedding de un texto (documento o query): dense obligatorio, sparse opcional."""

    dense: tuple[float, ...]
    sparse: SparseEmbedding | None = None

    def __post_init__(self) -> None:
        if not self.dense:
            msg = "dense embedding must be non-empty"
            raise ValueError(msg)
