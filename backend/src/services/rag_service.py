"""
RAG service for coordinating the full retrieval-augmented generation pipeline.

Main orchestrator that combines retrieval and generation services.
"""
import logging
import time
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class RAGService:
    """
    Main RAG pipeline coordinator.

    Orchestrates retrieval service and OpenAI service to process queries.
    """

    def __init__(self, retrieval_service, openai_service):
        """
        Initialize RAG service.

        Args:
            retrieval_service: Service for retrieving textbook chunks
            openai_service: Service for generating answers
        """
        self.retrieval_service = retrieval_service
        self.openai_service = openai_service
        logger.info("RAGService initialized")

    async def process_query(
        self,
        question: str,
        context: Optional[str] = None,
        use_context_only: bool = False,
        top_k: int = 5,
        chapter_id: Optional[int] = None,
    ) -> Tuple[str, List[Dict], int]:
        """
        Main RAG pipeline processing.

        Args:
            question: User's natural language question
            context: Selected text passage (for selected-text-only mode)
            use_context_only: Flag to restrict to selected text only
            top_k: Number of chunks to retrieve (for book-wide mode)
            chapter_id: Optional chapter context (for chapter-aware mode)

        Returns:
            Tuple of (answer, sources, query_time_ms)

        Raises:
            ValueError: If validation fails
            Exception: If pipeline processing fails
        """
        start_time = time.time()

        logger.info(
            f"Processing query: use_context_only={use_context_only}, "
            f"chapter_id={chapter_id}, top_k={top_k}"
        )

        try:
            # Step 1: Determine answering mode and retrieve sources
            if use_context_only:
                mode = "selected-text-only"
                sources = self.retrieval_service.retrieve_context_only(context, question)
                logger.info("Using selected-text-only mode")
            elif chapter_id is not None:
                mode = "chapter-aware"
                sources = self.retrieval_service.retrieve_book_wide(question, top_k, chapter_id)
                logger.info(f"Using chapter-aware mode (chapter {chapter_id})")
            else:
                mode = "book-wide"
                sources = self.retrieval_service.retrieve_book_wide(question, top_k)
                logger.info("Using book-wide mode")

            # Step 2: Generate answer using OpenAI Agents SDK
            answer = await self.openai_service.generate_answer(
                question=question, retrieved_chunks=sources, mode=mode
            )

            # Step 3: Calculate query time
            query_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"Query processed successfully: mode={mode}, "
                f"sources={len(sources)}, time={query_time_ms}ms"
            )

            return answer, sources, query_time_ms

        except ValueError as e:
            # Validation errors (e.g., empty context in selected-text-only mode)
            logger.warning(f"Validation error: {e}")
            raise

        except Exception as e:
            # Pipeline processing errors
            logger.error(f"RAG pipeline failed: {e}", exc_info=True)
            raise


# Note: Singleton instance created with dependency injection in app initialization
