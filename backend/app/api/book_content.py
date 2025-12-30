"""
Real-time book content API endpoints.
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.qdrant_service import qdrant_service
from app.services.neon_service import neon_service
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class ContentQueryRequest(BaseModel):
    """Request model for content queries."""
    query: str
    chapter_id: Optional[int] = None
    section_id: Optional[str] = None
    top_k: int = 5


class ContentChunk(BaseModel):
    """Model for content chunks."""
    chunk_id: str
    chapter_id: int
    section_id: str
    section_title: str
    content: str
    relevance_score: Optional[float] = None


class ContentQueryResponse(BaseModel):
    """Response model for content queries."""
    chunks: List[ContentChunk]
    total_chunks: int
    query_time_ms: int


class ChapterInfo(BaseModel):
    """Model for chapter information."""
    chapter_id: int
    title: str
    description: str
    sections_count: int


class BookContentResponse(BaseModel):
    """Response model for book content."""
    chapters: List[ChapterInfo]
    total_chapters: int


@router.get("/api/book-content", response_model=BookContentResponse, tags=["Book Content"])
async def get_book_content():
    """
    Get structured information about the book content.

    Returns a list of chapters with their metadata.
    """
    try:
        # In a real implementation, this would fetch from the database
        # For now, we'll return a sample structure based on the current book content
        chapters = [
            ChapterInfo(
                chapter_id=1,
                title="Introduction to Physical AI",
                description="Foundations of Physical AI and its applications",
                sections_count=5
            ),
            ChapterInfo(
                chapter_id=2,
                title="Humanoid Robotics Fundamentals",
                description="Core concepts in humanoid robot design and control",
                sections_count=7
            ),
            ChapterInfo(
                chapter_id=3,
                title="ROS 2 for Robotics Development",
                description="Robot Operating System 2 for building robotic applications",
                sections_count=6
            ),
            ChapterInfo(
                chapter_id=4,
                title="Simulation and Development",
                description="Simulation environments and development practices",
                sections_count=4
            ),
            ChapterInfo(
                chapter_id=5,
                title="Advanced Control Systems",
                description="Advanced control algorithms and implementation",
                sections_count=5
            ),
            ChapterInfo(
                chapter_id=6,
                title="Real-World Applications",
                description="Practical applications and case studies",
                sections_count=3
            )
        ]

        return BookContentResponse(
            chapters=chapters,
            total_chapters=len(chapters)
        )
    except Exception as e:
        logger.error(f"Failed to fetch book content: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch book content")


@router.post("/api/book-content/search", response_model=ContentQueryResponse, tags=["Book Content"])
async def search_book_content(request: ContentQueryRequest):
    """
    Search for specific content within the book using semantic search.

    Performs vector search in Qdrant and retrieves relevant content chunks.
    """
    try:
        from app.services.embedding_service import embedding_service
        import time

        start_time = time.time()

        # Embed the query
        query_embedding = embedding_service.encode_text(request.query)

        # Search in Qdrant
        search_results = qdrant_service.search(
            query_vector=query_embedding,
            top_k=request.top_k
        )

        if not search_results:
            return ContentQueryResponse(
                chunks=[],
                total_chunks=0,
                query_time_ms=int((time.time() - start_time) * 1000)
            )

        # Get chunk IDs
        chunk_ids = [result.id for result in search_results]

        # Retrieve metadata from Neon
        metadata_records = neon_service.get_chunks_by_ids([result.id for result in search_results])

        # Create content chunks
        chunks = []
        for result in search_results:
            # Find corresponding metadata
            metadata = next(
                (rec for rec in metadata_records if str(rec['chunk_id']) == result.id),
                None
            )

            if metadata:
                chunks.append(ContentChunk(
                    chunk_id=result.id,
                    chapter_id=metadata['chapter_id'],
                    section_id=metadata['section_id'],
                    section_title=metadata['section_title'],
                    content=metadata['preview_text'],
                    relevance_score=round(result.score, 3)
                ))

        query_time_ms = int((time.time() - start_time) * 1000)

        return ContentQueryResponse(
            chunks=chunks,
            total_chunks=len(chunks),
            query_time_ms=query_time_ms
        )

    except Exception as e:
        logger.error(f"Content search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Content search failed")


@router.get("/api/book-content/chapter/{chapter_id}", response_model=List[ContentChunk], tags=["Book Content"])
async def get_chapter_content(chapter_id: int):
    """
    Get all content chunks for a specific chapter.
    """
    try:
        # Retrieve chunks for the specified chapter from Neon
        chunks = neon_service.get_chunks_by_chapter(chapter_id)

        content_chunks = [
            ContentChunk(
                chunk_id=str(chunk['chunk_id']),
                chapter_id=chunk['chapter_id'],
                section_id=chunk['section_id'],
                section_title=chunk['section_title'],
                content=chunk['preview_text'],
            )
            for chunk in chunks
        ]

        return content_chunks
    except Exception as e:
        logger.error(f"Failed to fetch chapter {chapter_id} content: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch chapter {chapter_id} content")


@router.get("/api/book-content/health", tags=["Book Content"])
async def content_health():
    """
    Health check for book content services.
    """
    try:
        # Check if Qdrant is accessible
        qdrant_healthy = qdrant_service.health_check()

        # Check if Neon is accessible
        neon_healthy = neon_service.health_check()

        return {
            "status": "healthy",
            "services": {
                "qdrant": "healthy" if qdrant_healthy else "unhealthy",
                "neon": "healthy" if neon_healthy else "unhealthy"
            }
        }
    except Exception as e:
        logger.error(f"Content health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Content service health check failed")