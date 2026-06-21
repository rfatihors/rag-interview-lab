from openai import OpenAI

from src.core.interfaces import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI tabanlı chat + embedding provider.

    API key OPENAI_API_KEY environment variable ile veya config üzerinden verilebilir.
    """

    def __init__(
        self,
        api_key: str | None,
        chat_model: str,
        embed_model: str,
    ):
        self.client = OpenAI(api_key=api_key)
        self.chat_model = chat_model
        self.embed_model = embed_model

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.embed_model,
            input=texts,
        )

        return [item.embedding for item in response.data]

    def generate_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        full_prompt = prompt

        if system_prompt:
            full_prompt = f"{system_prompt.strip()}\\n\\n{prompt.strip()}"

        response = self.client.responses.create(
            model=self.chat_model,
            input=full_prompt,
        )

        return response.output_text.strip()
