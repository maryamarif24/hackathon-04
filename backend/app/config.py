"""
Configuration management using pydantic-settings.
Loads and validates environment variables from .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI API Configuration (using OpenRouter)
    openai_api_key: Optional[str] = None  # Optional, defaults to OPENROUTER_API_KEY environment variable
    openai_model: str = "google/gemini-pro"  # Updated default for OpenRouter

    # Qdrant Cloud Configuration
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str = "physical-ai-textbook"

    # Neon Postgres Configuration
    neon_connection_string: str
    neon_database_name: str = "neondb"

    # Application Settings
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000,https://your-website.vercel.app"
    rate_limit_requests_per_minute: int = 10

    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Retrieval Configuration
    top_k_default: int = 5
    min_relevance_score: float = 0.7

    # OpenAI Generation Configuration
    openai_temperature: float = 0.3
    openai_max_tokens: int = 800

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse comma-separated CORS_ORIGINS into list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]


# Global settings instance
settings = Settings()
