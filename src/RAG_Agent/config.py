from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    # Perfil de hardware para ingest (hoy: local). Override fino opcional.
    ingest_profile: str = "local"
    ingest_pages_per_shard: int | None = None
    chunker: str = "semantic"  # section | semantic

    cohere_api_key: str | None = None

    # Semantic chunker
    semantic_chunk_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_chunk_threshold: float = 0.7
    semantic_chunk_min_tokens: int = 64
    semantic_chunk_max_tokens: int = 128

    # Qdrant: mode=local|cloud|memory
    qdrant_mode: str = "local"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "tech_docs"
    qdrant_enable_sparse: bool = True

    # Retrieval + rerank
    retrieval_prefetch_limit: int = 30
    retrieval_candidate_limit: int = 30
    rerank_top_n: int = 10
    rerank_model: str = "rerank-v3.5"

    # Agent (ADK + LiteLLM)
    agent_model: str = "cohere_chat/command-a-03-2025"
    agent_app_name: str = "Agentic RAG Engineering Copilot for Telco"


settings = Settings()
