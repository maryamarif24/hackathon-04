"""
Pydantic models for RAG chatbot system.

Defines request/response models for query processing.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal


class ChatQuery(BaseModel):
    """
    User query request model.

    Represents a question with optional context constraints.
    """

    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="User's natural language question"
    )
    context: Optional[str] = Field(
        None,
        description="Selected text passage (for selected-text-only mode)"
    )
    use_context_only: bool = Field(
        False,
        description="When true, answers use ONLY the provided context"
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve"
    )
    chapter_id: Optional[int] = Field(
        None,
        ge=1,
        le=6,
        description="Current chapter context (for chapter-aware mode)"
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Validate question is non-empty after stripping."""
        if not v or not v.strip():
            raise ValueError("Question must be non-empty")
        return v.strip()

    @field_validator("context")
    @classmethod
    def validate_context(cls, v: Optional[str]) -> Optional[str]:
        """Validate context if provided."""
        if v is not None:
            return v.strip() if v.strip() else None
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the key components of a humanoid robot?",
                "context": None,
                "use_context_only": False,
                "top_k": 5,
                "chapter_id": None
            }
        }


class RetrievedChunk(BaseModel):
    """
    Retrieved textbook content chunk.

    Represents a piece of content from vector search.
    """

    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    chapter_id: int = Field(..., description="Chapter number (1-6)")
    section_id: str = Field(..., description="Section identifier within chapter")
    section_title: str = Field(..., description="Human-readable section title")
    preview_text: str = Field(
        ...,
        max_length=200,
        description="First 200 characters for display"
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Semantic similarity score (0.0-1.0)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "ch1_sec2_components_001",
                "chapter_id": 1,
                "section_id": "2",
                "section_title": "Introduction to Humanoid Robotics",
                "preview_text": "Humanoid robots typically consist of...",
                "relevance_score": 0.87
            }
        }


class ChatResponse(BaseModel):
    """
    Chatbot response model.

    Contains generated answer with source citations.
    """

    answer: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Generated explanation text"
    )
    sources: List[RetrievedChunk] = Field(
        default_factory=list,
        max_length=10,
        description="Relevant chunks used to generate answer"
    )
    query_time_ms: int = Field(
        ...,
        ge=0,
        description="Time taken to process query (milliseconds)"
    )
    mode: Literal["book-wide", "selected-text-only", "chapter-aware"] = Field(
        ...,
        description="Answering mode used"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Humanoid robots typically use electric or hydraulic actuators...",
                "sources": [
                    {
                        "chunk_id": "ch1_sec2_components_001",
                        "chapter_id": 1,
                        "section_id": "2",
                        "section_title": "Introduction to Humanoid Robotics",
                        "preview_text": "Humanoid robots typically consist of...",
                        "relevance_score": 0.87
                    }
                ],
                "query_time_ms": 1250,
                "mode": "book-wide"
            }
        }
