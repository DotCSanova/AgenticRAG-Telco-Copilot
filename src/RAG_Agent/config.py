from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    # When true (Cloud Run ingest), load Cohere/Qdrant secrets from Secret Manager.
    use_secret_manager: bool = False
    google_cloud_project: str | None = None

    # Docling hardware knobs (same Compose / Cloud Run). Env: INGEST_*.
    ingest_pages_per_shard: int = Field(default=50, ge=1)
    ingest_layout_batch_size: int = Field(default=4, ge=1)
    ingest_table_batch_size: int = Field(default=2, ge=1)
    chunker: str = "section"  # section | semantic

    # LLM Providers API Keys
    cohere_api_key: str | None = None

    # Semantic chunker settings
    semantic_chunk_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_chunk_threshold: float = 0.7
    semantic_chunk_min_tokens: int = 64
    semantic_chunk_max_tokens: int = 128

    # Qdrant: URL (+ optional API key for Cloud). Empty URL → localhost.
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "tech_docs"
    qdrant_enable_sparse: bool = True
    qdrant_in_memory: bool = False  # tests only

    # Retrieval + rerank
    retrieval_prefetch_limit: int = 30
    retrieval_candidate_limit: int = 30
    rerank_top_n: int = 10
    rerank_model: str = "rerank-v3.5"

    # Provider switches (composition root). Hoy solo una opción por slot.
    dense_embedder: str = "cohere"  # cohere
    reranker_provider: str = "cohere"  # cohere
    vector_store_provider: str = "qdrant"  # qdrant

    # Agent (ADK + LiteLLM)
    agent_model: str = "cohere_chat/command-a-03-2025"
    agent_app_name: str = "rag_agent"

    # Sessions (ADK). Compose for local development sets SESSIONS_DB_URL.
    # Cloud Run serving builds /cloudsql/PROJECT:REGION:INSTANCE from the fields below.
    sessions_db_url: str | None = None
    region: str | None = None
    instance_name: str | None = None
    db_user: str = "app"
    db_pass: str | None = None


settings = Settings()
