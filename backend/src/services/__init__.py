"""
Service layer initialization.

Creates singleton instances of all services with proper dependency injection.
"""
from app.config import settings
from src.services.embedding_service import EmbeddingService
from src.services.qdrant_service import QdrantService
from src.services.retrieval_service import RetrievalService
from src.services.openai_service import OpenRouterService
from src.services.rag_service import RAGService

# Initialize services in dependency order

# 1. Embedding service (no dependencies)
embedding_service = EmbeddingService()

# 2. Qdrant service (no dependencies)
qdrant_service = QdrantService(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
    collection_name=settings.qdrant_collection_name,
    embedding_dim=settings.embedding_dim,
)

# 3. Retrieval service (depends on embedding + qdrant)
retrieval_service = RetrievalService(
    embedding_service=embedding_service, qdrant_service=qdrant_service, config=settings
)

# 4. OpenAI service (no dependencies) - Using OpenRouter
openai_service = OpenRouterService(
    model=settings.openai_model,
    temperature=settings.openai_temperature,
    max_tokens=settings.openai_max_tokens,
)

# 5. RAG service (depends on retrieval + openai)
rag_service = RAGService(retrieval_service=retrieval_service, openai_service=openai_service)

__all__ = [
    "embedding_service",
    "qdrant_service",
    "retrieval_service",
    "openai_service",
    "rag_service",
]
