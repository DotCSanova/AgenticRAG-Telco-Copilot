from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Modifier,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from RAG_Agent.config import settings
from RAG_Agent.domain.value_objects.chunk import Chunk
from RAG_Agent.domain.value_objects.embedding import TextEmbedding
from RAG_Agent.domain.value_objects.search_hit import RetrievedChunk

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantVectorStore:
    """Adapter Qdrant: dense (+ sparse opcional) con vectores nombrados."""

    def __init__(
        self,
        *,
        collection_name: str | None = None,
        client: QdrantClient | None = None,
        enable_sparse: bool | None = None,
        prefetch_limit: int | None = None,
    ) -> None:
        self._collection = collection_name or settings.qdrant_collection
        self._enable_sparse = (
            settings.qdrant_enable_sparse if enable_sparse is None else enable_sparse
        )
        self._prefetch_limit = (
            settings.retrieval_prefetch_limit if prefetch_limit is None else prefetch_limit
        )
        if self._prefetch_limit < 1:
            raise ValueError(f"prefetch_limit must be >= 1, got {self._prefetch_limit}")
        self._client = client or self._build_client()
        self._ready_for_dim: int | None = None

    @staticmethod
    def _build_client() -> QdrantClient:
        if settings.qdrant_in_memory:
            return QdrantClient(location=":memory:")

        url = (settings.qdrant_url or "http://localhost:6333").strip()
        api_key = (settings.qdrant_api_key or "").strip() or None
        return QdrantClient(url=url, api_key=api_key)

    def _ensure_collection(self, dense_dim: int) -> None:
        if self._ready_for_dim == dense_dim:
            return

        if not self._client.collection_exists(self._collection):
            sparse_config = None
            if self._enable_sparse:
                sparse_config = {
                    SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
                }
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    DENSE_VECTOR_NAME: VectorParams(size=dense_dim, distance=Distance.COSINE),
                },
                sparse_vectors_config=sparse_config,
            )
            logger.info(
                "Created Qdrant collection %s (dense_dim=%d, sparse=%s)",
                self._collection,
                dense_dim,
                self._enable_sparse,
            )

        self._ready_for_dim = dense_dim

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    @staticmethod
    def _payload(chunk: Chunk) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chunk_id": chunk.id,
            "doc_id": chunk.doc_id,
            "text": chunk.text,
            "block_ids": list(chunk.block_ids),
            **chunk.metadata,
        }
        if chunk.page_start is not None:
            payload["page_start"] = chunk.page_start
        if chunk.page_end is not None:
            payload["page_end"] = chunk.page_end
        if chunk.section_id is not None:
            payload["section_id"] = chunk.section_id
        return payload

    def _point_vector(self, embedding: TextEmbedding) -> dict[str, Any]:
        vector: dict[str, Any] = {DENSE_VECTOR_NAME: list(embedding.dense)}
        if embedding.sparse is not None:
            if not self._enable_sparse:
                raise ValueError(
                    "Received sparse embedding but qdrant_enable_sparse is False"
                )
            vector[SPARSE_VECTOR_NAME] = SparseVector(
                indices=list(embedding.sparse.indices),
                values=list(embedding.sparse.values),
            )
        return vector

    def upsert(self, chunks: list[Chunk], embeddings: list[TextEmbedding]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
            )
        if not chunks:
            return 0

        dense_dim = len(embeddings[0].dense)
        if any(len(item.dense) != dense_dim for item in embeddings):
            raise ValueError("all dense embeddings must share the same dimension")

        self._ensure_collection(dense_dim)

        points = [
            PointStruct(
                id=self._point_id(chunk.id),
                vector=self._point_vector(embedding),
                payload=self._payload(chunk),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)
        logger.info("Upserted %d points into %s", len(points), self._collection)
        return len(points)

    def search(self, embedding: TextEmbedding, *, limit: int) -> list[RetrievedChunk]:
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")

        dense = list(embedding.dense)
        prefetch_limit = max(limit, self._prefetch_limit)

        if self._enable_sparse and embedding.sparse is not None:
            sparse = SparseVector(
                indices=list(embedding.sparse.indices),
                values=list(embedding.sparse.values),
            )
            response = self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    Prefetch(query=dense, using=DENSE_VECTOR_NAME, limit=prefetch_limit),
                    Prefetch(query=sparse, using=SPARSE_VECTOR_NAME, limit=prefetch_limit),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        else:
            response = self._client.query_points(
                collection_name=self._collection,
                query=dense,
                using=DENSE_VECTOR_NAME,
                limit=limit,
                with_payload=True,
            )

        return [_point_to_retrieved(point) for point in response.points]

    def delete_by_doc_id(self, doc_id: str) -> int:
        if not doc_id:
            raise ValueError("doc_id must be a non-empty string")
        if not self._client.collection_exists(self._collection):
            return 0

        doc_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )
        counted = self._client.count(
            collection_name=self._collection,
            count_filter=doc_filter,
            exact=True,
        )
        deleted = int(counted.count)
        if deleted == 0:
            return 0

        self._client.delete(
            collection_name=self._collection,
            points_selector=doc_filter,
            wait=True,
        )
        logger.info(
            "Deleted %d points with doc_id=%r from %s",
            deleted,
            doc_id,
            self._collection,
        )
        return deleted


def _point_to_retrieved(point: Any) -> RetrievedChunk:
    payload = point.payload or {}
    return RetrievedChunk(
        text=str(payload.get("text") or ""),
        doc_id=str(payload.get("doc_id") or ""),
        section_path=str(payload.get("section_path") or ""),
        score=float(point.score) if point.score is not None else None,
    )
