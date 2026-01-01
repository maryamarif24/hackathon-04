"""
Query API endpoint for RAG chatbot.

Handles POST /query requests with rate limiting.
"""
import logging
from fastapi import APIRouter, Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.models.rag_models import ChatQuery, ChatResponse, RetrievedChunk
from src.services import rag_service

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/query", response_model=ChatResponse)
@limiter.limit("10/minute")  # Rate limit: 10 requests per minute per IP
async def query_textbook(request: Request, query: ChatQuery):
    """
    Query the Physical AI textbook using RAG.

    This endpoint processes user questions through the RAG pipeline:
    1. Retrieval: Search for relevant textbook chunks (vector search or context-only)
    2. Generation: Generate grounded answer using OpenAI Agents SDK
    3. Citation: Return answer with source references

    Args:
        request: FastAPI request (for rate limiting)
        query: ChatQuery with question, context, use_context_only, top_k, chapter_id

    Returns:
        ChatResponse with answer, sources, query_time_ms, mode

    Raises:
        400: Invalid query parameters
        429: Rate limit exceeded
        503: Service unavailable (database or AI service down)

    Example:
        Request:
        ```json
        {
            "question": "What are the key components of a humanoid robot?",
            "top_k": 5
        }
        ```

        Response:
        ```json
        {
            "answer": "The key components of a humanoid robot include...",
            "sources": [
                {
                    "chunk_id": "ch1_sec2_001",
                    "chapter_id": 1,
                    "section_id": "2",
                    "section_title": "Introduction to Humanoid Robotics",
                    "preview_text": "Humanoid robots consist of...",
                    "relevance_score": 0.87
                }
            ],
            "query_time_ms": 1250,
            "mode": "book-wide"
        }
        ```
    """
    try:
        # Validate query
        if not query.question or len(query.question.strip()) < 3:
            raise HTTPException(status_code=400, detail="Question must be at least 3 characters")

        if query.use_context_only and not query.context:
            raise HTTPException(
                status_code=400, detail="Context is required when use_context_only is true"
            )

        # Process query through RAG pipeline
        answer, sources, query_time_ms = await rag_service.process_query(
            question=query.question,
            context=query.context,
            use_context_only=query.use_context_only,
            top_k=query.top_k,
            chapter_id=query.chapter_id,
        )

        # Determine mode
        if query.use_context_only:
            mode = "selected-text-only"
        elif query.chapter_id is not None:
            mode = "chapter-aware"
        else:
            mode = "book-wide"

        # Convert sources to RetrievedChunk models
        source_chunks = [
            RetrievedChunk(
                chunk_id=s.get("chunk_id", ""),
                chapter_id=s.get("chapter_id", 0),
                section_id=s.get("section_id", ""),
                section_title=s.get("section_title", ""),
                preview_text=s.get("preview_text", ""),
                relevance_score=s.get("relevance_score", 0.0),
            )
            for s in sources
        ]

        return ChatResponse(
            answer=answer, sources=source_chunks, query_time_ms=query_time_ms, mode=mode
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503, detail="Failed to process your query. Please try again later."
        )
