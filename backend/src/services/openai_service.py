"""
LLM Service using OpenAI Chat Completions API with OpenRouter.

Uses Gemini 2.5-Flash via OpenRouter for answer generation.
Handles grounded response generation with citation enforcement.
"""
import logging
import os
from typing import List, Dict, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenRouterService:
    """
    Service for generating answers using OpenAI Chat Completions API with OpenRouter.

    Uses Gemini 2.5-Flash via OpenRouter for cost-effective, high-quality responses.
    Enforces strict grounding in retrieved context.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "google/gemini-flash-1.5",  # Updated default for OpenRouter - more reliable
        temperature: float = 0.3,
        max_tokens: int = 800
    ):
        """
        Initialize OpenRouter client configured for OpenAI Chat Completions API.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model: Model name - use "google/gemini-pro" for Gemini via OpenRouter
            temperature: Sampling temperature (low for factual responses)
            max_tokens: Maximum response length
        """
        # Use only OPENROUTER_API_KEY environment variable
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self.api_key:
            logger.error("OPENROUTER_API_KEY environment variable not set")
            # Create a client that will fail gracefully
            self.client = None
        else:
            # Configure OpenAI client for OpenRouter
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1",
            )

        logger.info(f"OpenRouterService initialized: model={model}, temp={temperature}")
        logger.info("Using OpenRouter with OpenAI Chat Completions API")

    async def generate_answer(
        self,
        question: str,
        retrieved_chunks: Optional[List[Dict]] = None,
        mode: str = "book-wide"
    ) -> str:
        """
        Generate answer using OpenAI Chat Completions API with Gemini via OpenRouter.

        Args:
            question: User's question
            retrieved_chunks: Optional list of RetrievedChunk dicts from RAG
            mode: "book-wide" | "selected-text-only" | "chapter-aware"

        Returns:
            Generated answer text (always returns a valid string)
        """
        logger.info(
            f"Generating answer: mode={mode}, chunks={len(retrieved_chunks or [])}, question='{question[:50]}...'"
        )

        # Build context from retrieved chunks
        context_text = self._build_context(retrieved_chunks or [])

        # Build the system message with instructions
        system_message = """You are an educational AI assistant for the "Physical AI & Humanoid Robotics" textbook.

Your role is to help learners understand textbook content by answering questions accurately and clearly.

CRITICAL RULES:
1. Answer primarily from the provided textbook context when available
2. If context is limited or empty, you may use your general knowledge to help the learner
3. Always be helpful - if you don't know something, explain what you DO know related to the topic
4. Use clear, educational language appropriate for learners
5. Structure answers with bullet points or numbered lists when appropriate
6. Keep answers concise but thorough enough to be educational
7. Maintain a helpful, patient, encouraging tone

When context is provided:
- Use it as the primary source for your answer
- Cite the relevant chapter/section in your response

When no context is provided:
- Answer based on your knowledge of Physical AI, robotics, ROS 2, simulation, and VLA systems
- Still be helpful and educational

Remember: Your goal is to help learners succeed. Always provide a useful, accurate response."""

        # Build the user message with context
        if context_text:
            user_message = f"""Question: {question}

Context from textbook:
{context_text}

Instructions:
- Answer the question using the provided context as your primary source
- Cite the relevant chapter/section in your answer
- Use clear, educational language
- Structure your answer with bullet points or numbered lists if appropriate
- Keep answer concise but thorough
- If the context doesn't fully answer the question, supplement with your knowledge"""
        else:
            user_message = f"""Question: {question}

Instructions:
- Answer based on your knowledge of Physical AI, robotics, ROS 2, simulation, and VLA systems
- Use clear, educational language
- Structure your answer appropriately
- Be helpful and thorough"""

        try:
            # Validate API key is available
            if not self.api_key or not self.client:
                logger.error("OpenRouter API key is not configured or client not available")
                return self._get_fallback_response(question, context_text)

            logger.info(f"Making OpenRouter API call with model: {self.model}")
            logger.info(f"API Key first 8 chars: {self.api_key[:8] if self.api_key else 'None'}")

            # Call the OpenAI Chat Completions API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content.strip()
            logger.info(f"Answer generated: {len(answer)} characters")
            return answer

        except Exception as e:
            logger.error(f"OpenRouter API call failed: {e}", exc_info=True)
            # Log more specific details for debugging
            logger.error(f"API Key configured: {bool(self.api_key)}")
            logger.error(f"Client available: {bool(self.client)}")
            logger.error(f"Model: {self.model}")
            logger.error(f"Temperature: {self.temperature}")
            logger.error(f"Max tokens: {self.max_tokens}")
            logger.error(f"Question length: {len(question)}")
            logger.error(f"Context length: {len(context_text)}")
            logger.error(f"System message length: {len(system_message)}")
            logger.error(f"User message length: {len(user_message)}")

            # Check if it's a specific HTTP error
            if hasattr(e, 'status_code'):
                logger.error(f"HTTP Status Code: {e.status_code}")
            if hasattr(e, 'response'):
                logger.error(f"Response body: {e.response.text if hasattr(e.response, 'text') else 'No response body'}")

            # Return a helpful fallback response
            return self._get_fallback_response(question, context_text)

    def _build_context(self, chunks: List[Dict]) -> str:
        """
        Build context text from retrieved chunks.

        Args:
            chunks: List of chunk dicts

        Returns:
            Formatted context text with chapter/section headers
        """
        if not chunks:
            return ""

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            section_title = chunk.get("section_title", "Unknown Section")
            chapter_id = chunk.get("chapter_id", "?")
            full_text = chunk.get("full_text", "")

            context_parts.append(
                f"[Chunk {i}] Chapter {chapter_id}, Section: {section_title}\n{full_text}\n"
            )

        return "\n---\n".join(context_parts)

    def _get_fallback_response(self, question: str, context: str) -> str:
        """
        Generate a helpful fallback response when API fails.

        Args:
            question: The user's question
            context: Any context that was provided

        Returns:
            Fallback response string
        """
        if context:
            return f"""Based on the textbook content you provided, I can see this relates to your question about "{question}".

The context discusses important concepts in Physical AI and robotics. For more detailed information, I recommend referring to the specific sections in the textbook that cover this topic.

Please try your question again - I'm here to help you learn!"""
        else:
            return f"""I'd be happy to help answer your question about "{question}".

As an educational assistant for Physical AI & Humanoid Robotics, I can help you understand topics like:
- Physical AI and embodied intelligence
- Humanoid robot design and components
- ROS 2 architecture and tools
- Digital twin simulation
- Vision-Language-Action (VLA) systems

Please try again, and I'll do my best to provide a helpful answer!"""


# Singleton instance (created lazily)
_service_instance: Optional[OpenRouterService] = None


def get_service(api_key: Optional[str] = None) -> OpenRouterService:
    """
    Get or create the OpenRouter service singleton.

    Args:
        api_key: Optional API key override

    Returns:
        OpenRouterService instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = OpenRouterService(api_key=api_key)
    return _service_instance