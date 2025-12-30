"""
Query API endpoint for RAG system.
Handles POST /api/query requests with rate limiting.
"""
import logging
from fastapi import APIRouter, Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.schemas import ChatQueryRequest, ChatQueryResponse, ErrorResponse
from app.services.rag_service import rag_service
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post(
    "/api/query",
    response_model=ChatQueryResponse,
    responses={
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Query"]
)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def query_textbook(request: Request, query: ChatQueryRequest):
    """
    Query the textbook using RAG.

    Processes the user's question through:
    1. Semantic search in vector database
    2. Metadata retrieval from PostgreSQL
    3. Re-ranking and citation generation
    4. Answer synthesis

    **Rate Limit**: 10 requests per minute per IP

    **Example Request**:
    ```json
    {
      "question": "What is Physical AI?",
      "top_k": 5
    }
    ```

    **Example Response**:
    ```json
    {
      "answer": "Based on the textbook content...",
      "sources": [
        {
          "chunk_id": "uuid",
          "chapter_id": 1,
          "section_id": "1.1",
          "section_title": "Introduction",
          "preview_text": "Physical AI is...",
          "relevance_score": 0.89
        }
      ],
      "query_time_ms": 145
    }
    ```
    """
    try:
        logger.info(f"Received query: {query.question[:50]}...")

        # Process query through RAG pipeline
        answer, sources, query_time_ms = rag_service.process_query(
            question=query.question,
            context=query.context,
            use_context_only=query.use_context_only,
            top_k=query.top_k
        )

        # Return response
        return ChatQueryResponse(
            answer=answer,
            sources=sources,
            query_time_ms=query_time_ms
        )

    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "query_processing_error",
                "message": "Failed to process your query. Please try again.",
                "details": {}
            }
        )
