import hashlib
import numpy as np

from src.core.interfaces import BaseLLMProvider


class MockProvider(BaseLLMProvider):
    """
    API key veya local model yoksa pipeline'ı göstermek için kullanılan provider.

    Bu provider gerçek semantik kalite vermez.
    Ama loader → splitter → vector store → retrieval → answer akışını test etmeyi sağlar.
    """

    def __init__(self, vector_size: int = 384):
        self.vector_size = vector_size

    def _hash_embedding(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], byteorder="little")
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=self.vector_size)
        norm = np.linalg.norm(vector)

        if norm == 0:
            return vector.astype(float).tolist()

        vector = vector / norm
        return vector.astype(float).tolist()

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embedding(text) for text in texts]

    def generate_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        return (
            "Mock cevap: Gerçek LLM provider bağlı değil. "
            "Ancak RAG pipeline, Qdrant entegrasyonu ve prompt üretimi çalışıyor.\n\n"
            "Not: Semantik cevap kalitesi için OpenAI/Ollama/Azure gibi gerçek provider bağlanmalıdır."
        )
