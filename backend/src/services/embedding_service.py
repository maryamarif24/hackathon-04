"""
Embedding service for converting text to vectors.

Uses sentence-transformers to generate 384-dimensional embeddings.
"""
import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using sentence-transformers.

    Loads all-MiniLM-L6-v2 model (384-dim) on initialization.
    """

    def __init__(self):
        self.model = None
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_dim = 384
        logger.info(f"EmbeddingService initialized with model: {self.model_name}")

    def load_model(self) -> bool:
        """
        Load embedding model on application startup.

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.model is None:
                logger.info(f"Loading embedding model: {self.model_name}...")
                self.model = SentenceTransformer(self.model_name)
                logger.info(
                    f"✅ Embedding model loaded: {self.model_name} ({self.embedding_dim}-dim)"
                )
            return True
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            return False

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate 384-dimensional embedding for input text.

        Args:
            text: Input text to embed

        Returns:
            384-dim vector as list of floats (normalized for cosine similarity)

        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError("Embedding model not loaded. Call load_model() first.")

        if not text or not text.strip():
            raise ValueError("Text must be non-empty")

        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)

        # Normalize vector for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.tolist()

    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts: List of input texts

        Returns:
            List of 384-dim vectors (normalized)

        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError("Embedding model not loaded. Call load_model() first.")

        if not texts:
            raise ValueError("Texts list must be non-empty")

        # Generate embeddings in batch
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, batch_size=32, show_progress_bar=True
        )

        # Normalize each vector
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms

        return normalized.tolist()


# Singleton instance
embedding_service = EmbeddingService()
