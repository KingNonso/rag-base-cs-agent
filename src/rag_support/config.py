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
    rerank_provider: str = "lexical"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    jwt_issuer: str = ""
    jwt_audience: str = ""
    jwt_jwks_url: str = ""
    jwt_public_key: str = ""
    jwt_algorithms: str = "RS256,ES256"
    account_claim: str = "account_id"
    roles_claim: str = "roles"
    admin_role: str = "knowledge_admin"
    cors_origins: str = ""
    max_upload_bytes: int = 20_000_000
    max_history_turns: int = 6
    min_retrieval_score: float = 0.24
    retrieval_candidates: int = 12
    context_passages: int = 6
    chunk_max_chars: int = 1300
    chunk_overlap: int = 180
    ocr_min_chars: int = 40
    ocr_min_confidence: float = 60.0
    session_retention_days: int = 90

    @property
    def embedding_id(self) -> str:
        return f"{self.embedding_provider}:{self.embedding_model}"

    @property
    def algorithms(self) -> list[str]:
        return [a.strip() for a in self.jwt_algorithms.split(",") if a.strip()]


settings = Settings()
