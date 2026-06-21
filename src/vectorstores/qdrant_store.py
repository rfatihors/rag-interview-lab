from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from src.core.interfaces import BaseVectorStore


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant tabanlı vector store implementasyonu.
    """

    def __init__(
        self,
        url: str,
        collection_name: str,
        api_key: str | None = None,
    ):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name

    def create_collection(self, vector_size: int, recreate: bool = False) -> None:
        if self.client.collection_exists(self.collection_name):
            if not recreate:
                return

            self.client.delete_collection(collection_name=self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def _stable_point_id(self, document: dict[str, Any]) -> str:
        raw_id = str(document.get("id") or f"{document.get('source', '')}:{document.get('text', '')[:200]}")
        return str(uuid5(NAMESPACE_URL, raw_id))

    def _build_payload(self, document: dict[str, Any]) -> dict[str, Any]:
        metadata = document.get("metadata", {}) or {}

        payload = {
            "text": document["text"],
            "source": document.get("source", "unknown"),
            "metadata": metadata,
            "original_id": document.get("id"),
        }

        # Basit payload filtering için scalar metadata alanlarını flatten ediyoruz.
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                payload[f"metadata_{key}"] = value

        return payload

    def _build_filter(self, filters: dict[str, Any] | None) -> Filter | None:
        if not filters:
            return None

        conditions = []

        for key, value in filters.items():
            conditions.append(
                FieldCondition(
                    key=f"metadata_{key}",
                    match=MatchValue(value=value),
                )
            )

        return Filter(must=conditions)

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("documents ve embeddings uzunlukları eşit olmalı.")

        points = []

        for document, embedding in zip(documents, embeddings):
            points.append(
                PointStruct(
                    id=self._stable_point_id(document),
                    vector=embedding,
                    payload=self._build_payload(document),
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_filter = self._build_filter(filters)

        # qdrant-client sürüm farklarına karşı iki yolu da destekliyoruz.
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            hits = response.points
        except AttributeError:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )

        results = []

        for hit in hits:
            payload = hit.payload or {}

            results.append(
                {
                    "id": str(hit.id),
                    "score": float(hit.score),
                    "text": payload.get("text", ""),
                    "source": payload.get("source", "unknown"),
                    "metadata": payload.get("metadata", {}),
                }
            )

        return results
