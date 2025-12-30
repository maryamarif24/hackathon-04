"""
Retrieval service for orchestrating vector search.

Coordinates embedding generation and Qdrant search.
"""
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Service for retrieving relevant textbook chunks.

    Supports three modes:
    - Book-wide: Search across entire textbook
    - Selected-text-only: Use provided context directly (no vector search)
    - Chapter-aware: Prioritize chunks from specific chapter
    """

    def __init__(self, embedding_service, qdrant_service, config):
        """
        Initialize retrieval service.

        Args:
            embedding_service: Service for generating embeddings
            qdrant_service: Service for vector search
            config: Application configuration
        """
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        self.default_top_k = config.top_k_default
        self.min_relevance_score = config.min_relevance_score
        logger.info("RetrievalService initialized")

    def retrieve_book_wide(
        self, question: str, top_k: Optional[int] = None, chapter_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a book-wide query.

        Args:
            question: User's natural language question
            top_k: Number of chunks to retrieve (default from config)
            chapter_id: Optional chapter context for chapter-aware mode

        Returns:
            List of RetrievedChunk dicts with relevance scores
        """
        if top_k is None:
            top_k = self.default_top_k

        logger.info(
            f"Book-wide retrieval: question='{question[:50]}...', top_k={top_k}, chapter={chapter_id}"
        )

        # Step 1: Generate embedding for the question
        query_embedding = self.embedding_service.generate_embedding(question)

        # Step 2: Search Qdrant for similar chunks
        chunks = self.qdrant_service.search(
            query_embedding=query_embedding,
            top_k=top_k,
            chapter_filter=chapter_id,  # None for book-wide, int for chapter-aware
            min_score=self.min_relevance_score,
        )

        # Step 3: Return chunks (empty list if no relevant content found)
        logger.info(f"Retrieved {len(chunks)} chunks (min_score={self.min_relevance_score})")
        return chunks

    def retrieve_context_only(self, context: str, question: str) -> List[Dict]:
        """
        Create a synthetic chunk from user-selected text.

        No vector search is performed. The selected text becomes the only "chunk".

        Args:
            context: User-selected text passage
            question: User's question (for logging)

        Returns:
            Single-item list with synthetic chunk

        Raises:
            ValueError: If context is empty or too short
        """
        if not context or len(context.strip()) < 10:
            raise ValueError(
                "Context must be non-empty and at least 10 characters for selected-text-only mode"
            )

        logger.info(
            f"Context-only retrieval: question='{question[:50]}...', context_length={len(context)}"
        )

        # Create synthetic chunk (no vector search needed)
        synthetic_chunk = {
            "chunk_id": "selected-text",
            "chapter_id": 0,  # Unknown chapter
            "section_id": "selected",
            "section_title": "Selected Text",
            "preview_text": context[:200],  # First 200 chars
            "full_text": context,
            "relevance_score": 1.0,  # Perfect "relevance" since user provided it
        }

        return [synthetic_chunk]


# Note: Singleton instance created with dependency injection in app initialization
