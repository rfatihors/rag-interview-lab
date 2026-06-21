import requests

from src.core.interfaces import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """
    Local Ollama tabanlı chat + embedding provider.
    """

    def __init__(
        self,
        base_url: str,
        chat_model: str,
        embed_model: str,
    ):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = requests.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.embed_model,
                "input": texts,
            },
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()
        embeddings = data.get("embeddings")

        if embeddings is None:
            raise ValueError("Ollama embed response içinde 'embeddings' bulunamadı.")

        return embeddings

    def generate_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        payload = {
            "model": self.chat_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=180,
        )
        response.raise_for_status()

        data = response.json()
        answer = data.get("response")

        if answer is None:
            raise ValueError("Ollama generate response içinde 'response' bulunamadı.")

        return answer.strip()
