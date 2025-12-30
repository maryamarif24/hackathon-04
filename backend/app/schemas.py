"""
Pydantic schemas for request/response validation.

Re-exports models from src.models.rag_models for backwards compatibility.
"""
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

# Import RAG models
from src.models.rag_models import ChatQuery, ChatResponse, RetrievedChunk

# Re-export for backwards compatibility
__all__ = ["ChatQuery", "ChatResponse", "RetrievedChunk", "ErrorResponse", "HealthResponse"]


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
