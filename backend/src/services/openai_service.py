"""
OpenAI service for answer generation using Agents SDK.

Handles grounded response generation with citation enforcement.
"""
import logging
from typing import List, Dict
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIService:
    """
    Service for generating answers using OpenAI Agents SDK.

    Enforces strict grounding in retrieved context.
    """

    def __init__(self, api_key: str, model: str, temperature: float = 0.3, max_tokens: int = 800):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model name (e.g., "gpt-4o-mini")
            temperature: Sampling temperature (low for factual responses)
            max_tokens: Maximum response length
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"OpenAIService initialized: model={model}, temp={temperature}")

    def generate_answer(
        self, question: str, retrieved_chunks: List[Dict], mode: str = "book-wide"
    ) -> str:
        """
        Generate grounded answer using OpenAI Agents SDK.

        Args:
            question: User's question
            retrieved_chunks: List of RetrievedChunk dicts
            mode: "book-wide" | "selected-text-only" | "chapter-aware"

        Returns:
            Generated answer text

        Raises:
            Exception: If OpenAI API call fails
        """
        logger.info(
            f"Generating answer: mode={mode}, chunks={len(retrieved_chunks)}, question='{question[:50]}...'"
        )

        # Build context from retrieved chunks
        context_text = self._build_context(retrieved_chunks)

        # Build system prompt based on mode
        system_prompt = self._build_system_prompt(mode, retrieved_chunks)

        # Build user prompt
        user_prompt = f"""Question: {question}

Context from textbook:
{context_text}

Instructions:
- Answer the question using ONLY the provided context
- If the context does not contain enough information, respond: "This is not covered in the book"
- Use clear, educational language
- Structure your answer with bullet points or numbered lists if appropriate
- Keep answer concise (under 300 words for simple questions)
- Do NOT add external knowledge or information not in the context
"""

        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content.strip()
            logger.info(f"Answer generated: {len(answer)} characters")
            return answer

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}", exc_info=True)
            raise

    def _build_context(self, chunks: List[Dict]) -> str:
        """
        Build context text from retrieved chunks.

        Args:
            chunks: List of chunk dicts

        Returns:
            Formatted context text with chapter/section headers
        """
        if not chunks:
            return "[No relevant content found in textbook]"

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            section_title = chunk.get("section_title", "Unknown Section")
            chapter_id = chunk.get("chapter_id", "?")
            full_text = chunk.get("full_text", "")

            context_parts.append(
                f"[Chunk {i}] Chapter {chapter_id}, Section: {section_title}\n{full_text}\n"
            )

        return "\n---\n".join(context_parts)

    def _build_system_prompt(self, mode: str, chunks: List[Dict]) -> str:
        """
        Build system prompt based on answering mode.

        Args:
            mode: Answering mode
            chunks: Retrieved chunks

        Returns:
            System prompt with mode-specific instructions
        """
        base_prompt = """You are an educational AI assistant for the "Physical AI & Humanoid Robotics" textbook.

Your role is to help learners understand textbook content by answering questions accurately and clearly.

CRITICAL RULES:
1. Answer ONLY from the provided textbook context
2. NEVER add external knowledge or information not in the context
3. If information is missing, explicitly state: "This is not covered in the book"
4. Use clear, educational language appropriate for learners
5. Structure answers with bullet points or numbered lists when appropriate
6. Keep answers concise (under 300 words for simple questions)
7. Maintain a helpful, patient tone
"""

        if mode == "selected-text-only":
            base_prompt += (
                "\nMODE: Selected-Text-Only - Answer using ONLY the selected text passage provided."
            )
        elif mode == "chapter-aware":
            chapter_id = chunks[0].get("chapter_id") if chunks else None
            base_prompt += f"\nMODE: Chapter-Aware - Prioritize content from Chapter {chapter_id} when answering."
        else:
            base_prompt += "\nMODE: Book-wide - Answer using relevant content from across the entire textbook."

        return base_prompt


# Note: Singleton instance created in config using environment variables
