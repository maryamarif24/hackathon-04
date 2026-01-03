"""
Configuration management using pydantic-settings.
Loads and validates environment variables from .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
<<<<<<< HEAD
<<<<<<< HEAD
from typing import List, Optional
=======
from typing import List
>>>>>>> master
=======
from typing import List, Optional
>>>>>>> 001-rag-chatbot


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

<<<<<<< HEAD
<<<<<<< HEAD
    # OpenAI API Configuration (using OpenRouter)
    openai_api_key: Optional[str] = None  # Optional, defaults to OPENROUTER_API_KEY environment variable
    openai_model: str = "google/gemini-flash-1.5"  # Updated default for OpenRouter - more reliable

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
=======
    # Qdrant Configuration
=======
    # OpenAI API Configuration (using OpenRouter)
    openai_api_key: Optional[str] = None  # Optional, defaults to OPENROUTER_API_KEY environment variable
    openai_model: str = "google/gemini-pro"  # Updated default for OpenRouter

    # Qdrant Cloud Configuration
>>>>>>> 001-rag-chatbot
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str = "physical-ai-textbook"

    # Neon Postgres Configuration
    neon_connection_string: str
    neon_database_name: str = "neondb"

    # Application Settings
    log_level: str = "INFO"
<<<<<<< HEAD
>>>>>>> master
=======
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
>>>>>>> 001-rag-chatbot

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @property
<<<<<<< HEAD
<<<<<<< HEAD
    def cors_origins_list(self) -> List[str]:
        """Parse comma-separated CORS_ORIGINS into list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]
=======
    def cors_origins(self) -> List[str]:
        """Parse comma-separated ALLOWED_ORIGINS into list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]
>>>>>>> master
=======
    def cors_origins_list(self) -> List[str]:
        """Parse comma-separated CORS_ORIGINS into list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]
>>>>>>> 001-rag-chatbot


# Global settings instance
settings = Settings()
