from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    """
    LLM ve embedding sağlayıcıları için ortak arayüz.

    OpenAI, Ollama, Azure OpenAI, Bedrock, Vertex AI veya mock provider
    bu interface'i implemente edebilir.
    """

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Verilen prompt ile LLM cevabı üretir.
        """
        raise NotImplementedError

    @abstractmethod
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Metin listesi için embedding listesi döndürür.
        """
        raise NotImplementedError


class BaseVectorStore(ABC):
    """
    Vector database işlemleri için ortak arayüz.

    Qdrant, Supabase/pgvector, Pinecone, Weaviate veya FAISS implementasyonu
    bu interface'i kullanabilir.
    """

    @abstractmethod
    def create_collection(self, vector_size: int, recreate: bool = False) -> None:
        """
        Collection/index oluşturur.
        """
        raise NotImplementedError

    @abstractmethod
    def add_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """
        Doküman chunk'larını embedding'leriyle birlikte vector store'a ekler.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query embedding'e en yakın chunk'ları döndürür.
        """
        raise NotImplementedError
