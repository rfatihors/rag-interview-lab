from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "local"
    log_level: str = "INFO"

    llm_provider: str = "mock"

    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embed_model: str = "text-embedding-3-small"

    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3.5:2b"
    ollama_embed_model: str = "embeddinggemma:latest"
    ollama_timeout: int = 600
    ollama_num_predict: int = 256

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "rag_interview_docs"

    deepeval_judge_provider: str = "ollama"
    deepeval_judge_model: str = "qwen3.5:4b"
    deepeval_judge_base_url: str = "http://localhost:11434"
    deepeval_judge_num_predict: int = 512

    top_k: int = 5
    min_score: float = 0.35

    docs_dir: str = "data/sample_docs"



settings = Settings()
