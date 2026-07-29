from __future__ import annotations

from dataclasses import dataclass

from RAG_Agent.domain.ports.embedder import Embedder
from RAG_Agent.domain.ports.reranker import Reranker
from RAG_Agent.domain.ports.vector_store import VectorStore
from RAG_Agent.domain.value_objects.search_hit import SearchHit


@dataclass(frozen=True)
class SearchDocumentsService:
    """Caso de uso: embed query → hybrid/dense search → rerank → hits."""

    embedder: Embedder
    vector_store: VectorStore
    reranker: Reranker
    candidate_limit: int
    rerank_top_n: int

    def __post_init__(self) -> None:
        if self.candidate_limit < 1:
            raise ValueError(f"candidate_limit must be >= 1, got {self.candidate_limit}")
        if self.rerank_top_n < 1:
            raise ValueError(f"rerank_top_n must be >= 1, got {self.rerank_top_n}")

    def execute(self, query: str) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("query must be non-empty")

        embedding = self.embedder.embed_query(query)
        candidates = self.vector_store.search(embedding, limit=self.candidate_limit)
        if not candidates:
            return []

        ranked = self.reranker.rerank(
            query,
            [chunk.text for chunk in candidates],
            top_n=min(self.rerank_top_n, len(candidates)),
        )
        hits: list[SearchHit] = []
        for item in ranked:
            if item.index < 0 or item.index >= len(candidates):
                continue
            chunk = candidates[item.index]
            hits.append(
                SearchHit(
                    text=chunk.text,
                    score=item.score,
                    doc_id=chunk.doc_id,
                    section_path=chunk.section_path,
                )
            )
        return hits
