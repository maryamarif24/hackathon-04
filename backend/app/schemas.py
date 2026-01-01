"""
Pydantic schemas for request/response validation.

Re-exports models from src.models.rag_models for backwards compatibility.
"""
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

# Import RAG models
from src.models.rag_models import ChatQuery, ChatResponse, RetrievedChunk

# Chunk Metadata model for Neon database
from uuid import UUID
from datetime import datetime
from typing import Optional


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk stored in Neon PostgreSQL database."""

    chunk_id: UUID = Field(..., description="Unique identifier for the chunk")
    chapter_id: int = Field(..., ge=1, le=6, description="Chapter number (1-6)")
    section_id: str = Field(..., description="Section identifier within chapter")
    section_title: str = Field(..., description="Human-readable section title")
    chunk_index: int = Field(..., ge=0, description="Sequential index of chunk in section")
    token_count: int = Field(..., ge=0, description="Number of tokens in chunk")
    char_count: int = Field(..., ge=0, description="Number of characters in chunk")
    preview_text: str = Field(..., max_length=200, description="First 200 characters for display")
    indexed_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when chunk was indexed")


# Re-export for backwards compatibility
__all__ = ["ChatQuery", "ChatResponse", "RetrievedChunk", "ChunkMetadata", "ErrorResponse", "HealthResponse"]


# ============================================================================
# Additional Schemas
# ============================================================================


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    error: str = Field(..., description="Error type (e.g., 'validation_error', 'rate_limit')")
    message: str = Field(..., description="Human-readable error message")
    details: dict = Field(default_factory=dict, description="Additional error context")


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Overall health status: 'healthy' or 'degraded'")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
