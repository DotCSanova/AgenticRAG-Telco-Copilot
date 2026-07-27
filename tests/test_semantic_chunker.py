from pathlib import Path

import numpy as np

from RAG_Agent.domain.value_objects.block import Block, BlockType, TableData
from RAG_Agent.domain.value_objects.canonical_document import CanonicalDocument, DocumentMetadata
from RAG_Agent.infrastructure.indexing.semantic_chunker import SemanticChunker


class _FakeEncoder:
    """Embeddings fake: mismas palabras → vectores cercanos."""

    def encode(self, texts: list[str], **kwargs):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "alpha" in lowered else 0.0,
                    1.0 if "beta" in lowered else 0.0,
                    1.0 if "gamma" in lowered else 0.0,
                    0.1,
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


def _doc(blocks: dict[str, Block]) -> CanonicalDocument:
    return CanonicalDocument(
        metadata=DocumentMetadata(source_path=Path("data/demo-doc.pdf")),
        blocks=blocks,
    )


def test_semantic_chunker_splits_on_topic_shift():
    # Many short alpha sentences (fill min_tokens) then beta shift.
    alpha = " ".join(["Alpha topic sentence number %d." % i for i in range(20)])
    beta = " ".join(["Beta topic sentence number %d." % i for i in range(5)])
    blocks = {
        "b0": Block(id="b0", type=BlockType.PARAGRAPH, order=0, page=1, text=alpha),
        "b1": Block(id="b1", type=BlockType.PARAGRAPH, order=1, page=2, text=beta),
    }
    chunker = SemanticChunker(
        model=_FakeEncoder(),
        threshold=0.5,
        min_tokens=30,
        max_tokens=80,
    )
    chunks = chunker.chunk(_doc(blocks))
    assert len(chunks) >= 2
    assert chunks[0].metadata["chunker"] == "semantic"
    assert "Alpha" in chunks[0].text
    assert any("Beta" in chunk.text for chunk in chunks[1:])
    assert chunks[0].block_ids[0] == "b0"


def test_semantic_chunker_keeps_tables_atomic():
    table = TableData(headers=["A", "B"], rows=[["1", "2"], ["3", "4"]])
    blocks = {
        "t0": Block(id="t0", type=BlockType.TABLE, order=0, page=1, table=table),
        "p0": Block(
            id="p0",
            type=BlockType.PARAGRAPH,
            order=1,
            page=1,
            text="Alpha follows the table with more alpha words " * 8,
        ),
    }
    chunks = SemanticChunker(
        model=_FakeEncoder(),
        threshold=0.9,
        min_tokens=10,
        max_tokens=40,
    ).chunk(_doc(blocks))
    assert chunks
    assert any("A | B" in chunk.text for chunk in chunks)


def test_semantic_chunker_empty_document():
    assert SemanticChunker(model=_FakeEncoder()).chunk(_doc({})) == []
