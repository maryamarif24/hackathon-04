"""
RAG (Retrieval-Augmented Generation) service.
Handles query processing, semantic search, re-ranking, and response generation.
"""
import logging
import time
from typing import List
from uuid import UUID

import google.generativeai as genai
from app.config import settings
from app.schemas import SourceCitation
from app.services.embedding_service import embedding_service
from app.services.neon_service import neon_service
from app.services.qdrant_service import qdrant_service

logger = logging.getLogger(__name__)

# --- Prompt Templates ---

CONTEXT_ONLY_PROMPT = """
You are a helpful AI assistant answering questions about a textbook.
Your task is to answer the user's question based *only* on the provided text context.
Do not use any external knowledge. If the answer is not found in the context, state that clearly.

Context:
---
{context}
---

Question: {question}

Answer:
"""

RAG_PROMPT = """
You are a helpful AI assistant answering questions about a textbook on Physical AI and Robotics.
Your task is to synthesize an answer to the user's question based *only* on the provided context chunks from the textbook.
Cite the relevant sources by referring to the chapter and section.

Context Chunks:
---
{context}
---

Question: {question}

Answer:
"""


class RAGService:
    """Service for RAG query processing."""

    def __init__(self):
        """Initialize RAG service and Gemini client."""
        try:
            logger.info("Initializing RAGService and Gemini client...")
            if not settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is not set in environment.")
            
            genai.configure(api_key=settings.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini client initialized successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}", exc_info=True)
            # Depending on desired behavior, you might want to re-raise the exception
            # or handle it gracefully, so the app can start but query processing fails.
            self.model = None

    def process_query(
        self,
        question: str,
        context: str = "",
        use_context_only: bool = False,
        top_k: int = 5
    ) -> tuple[str, List[SourceCitation], int]:
        """
        Process a user query end-to-end.
        """
        start_time = time.time()

        if not self.model:
            return "Error: The AI model is not available. Please check the server configuration.", [], 0

        try:
            if use_context_only and context:
                logger.info(f"Processing context-only query: '{question[:50]}...'")
                answer = self._generate_context_only_answer(question, context)
                sources = [SourceCitation(
                    chunk_id=UUID('11111111-1111-1111-1111-111111111111'),
                    chapter_id=0, section_id='context', section_title='User Provided Context',
                    preview_text=context[:150] + '...' if len(context) > 150 else context,
                    relevance_score=1.0
                )]
                query_time_ms = int((time.time() - start_time) * 1000)
                logger.info(f"Context-only query processed in {query_time_ms}ms")
                return answer, sources, query_time_ms

            logger.info(f"Processing RAG query: '{question[:50]}...'")
            query_embedding = embedding_service.encode_text(question)

            logger.info(f"Searching Qdrant for top {top_k} results.")
            search_results = qdrant_service.search(query_vector=query_embedding, top_k=top_k * 2)

            if not search_results:
                logger.warning("No relevant chunks found in Qdrant.")
                return self._generate_no_results_response(question)

            chunk_ids = [UUID(result.id) for result in search_results]
            logger.info(f"Retrieving metadata for {len(chunk_ids)} chunks from database.")
            metadata_records = neon_service.get_chunks_by_ids(chunk_ids)

            metadata_map = {rec['chunk_id']: rec for rec in metadata_records}
            
            sources = []
            for result in search_results[:top_k]:
                chunk_id = UUID(result.id)
                if metadata := metadata_map.get(chunk_id):
                    sources.append(SourceCitation(
                        chunk_id=chunk_id,
                        chapter_id=metadata['chapter_id'],
                        section_id=metadata['section_id'],
                        section_title=metadata['section_title'],
                        preview_text=metadata['preview_text'],
                        relevance_score=round(result.score, 3)
                    ))
                else:
                    logger.warning(f"Metadata not found for chunk {chunk_id}")

            answer = self._generate_answer(question, sources)
            query_time_ms = int((time.time() - start_time) * 1000)
            logger.info(f"RAG query processed in {query_time_ms}ms, found {len(sources)} sources.")

            return answer, sources, query_time_ms

        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
            raise

    def _generate_answer(self, question: str, sources: List[SourceCitation]) -> str:
        """Generate answer from retrieved sources using Gemini."""
        if not sources:
            return "I couldn't find relevant information to answer your question. Please try rephrasing."

        context_str = "\n\n".join([
            f"Source (Chapter {s.chapter_id}, Section {s.section_title}):\n{s.preview_text}"
            for s in sources
        ])
        
        prompt = RAG_PROMPT.format(context=context_str, question=question)

        try:
            logger.info("Calling Gemini for RAG-based answer synthesis.")
            response = self.model.generate_content(prompt)
            logger.info("Successfully received response from Gemini.")
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}", exc_info=True)
            return "Error: Failed to generate an answer from the AI model."

    def _generate_context_only_answer(self, question: str, context: str) -> str:
        """Generate answer based only on the provided context text using Gemini."""
        prompt = CONTEXT_ONLY_PROMPT.format(context=context, question=question)

        try:
            logger.info("Calling Gemini for context-only answer.")
            response = self.model.generate_content(prompt)
            logger.info("Successfully received response from Gemini for context-only query.")
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed for context-only query: {e}", exc_info=True)
            return "Error: Failed to generate an answer from the AI model using the provided context."

    def _generate_no_results_response(self, question: str) -> tuple[str, List[SourceCitation], int]:
        """Generate response when no results are found."""
        answer = (
            "I couldn't find relevant information in the textbook for that question. "
            "Try asking about topics like Physical AI, Humanoid Robotics, ROS 2, or Simulation."
        )
        return answer, [], 0


# Global service instance
rag_service = RAGService()
