from src.config import settings
from src.core.interfaces import BaseLLMProvider
from src.providers.mock_provider import MockProvider
from src.providers.openai_provider import OpenAIProvider
from src.providers.ollama_provider import OllamaProvider


def build_provider() -> BaseLLMProvider:
    provider_name = settings.llm_provider.lower().strip()

    if provider_name == "mock":
        return MockProvider()

    if provider_name == "openai":
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            chat_model=settings.openai_chat_model,
            embed_model=settings.openai_embed_model,
        )

    if provider_name == "ollama":
        return OllamaProvider(
            base_url=settings.ollama_base_url,
            chat_model=settings.ollama_chat_model,
            embed_model=settings.ollama_embed_model,
        )

    raise ValueError(f"Desteklenmeyen LLM_PROVIDER: {settings.llm_provider}")
