from src.config import settings
from src.vectorstores.qdrant_store import QdrantVectorStore


def build_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
    )
