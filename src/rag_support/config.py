from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql://rag:rag@localhost:5433/rag"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    generation_provider: str = "openai"
    generation_model: str = "gpt-4o-mini"
    jwt_issuer: str = ""
    jwt_audience: str = ""
    jwt_jwks_url: str = ""
    admin_role: str = "knowledge_admin"
    cors_origins: str = ""
    max_upload_bytes: int = 20_000_000
    max_history_turns: int = 6
    min_retrieval_score: float = 0.24


settings = Settings()
