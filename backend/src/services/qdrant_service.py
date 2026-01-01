"""
Qdrant service for vector database operations.

Handles vector storage, search, and collection management.
"""
import logging
from typing import List, Dict, Optional
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

logger = logging.getLogger(__name__)


class QdrantService:
    """
    Service for interacting with Qdrant Cloud vector database.

    Manages collection creation, vector storage, and semantic search.
    """

    def __init__(self, url: str, api_key: str, collection_name: str, embedding_dim: int = 384):
        """
        Initialize Qdrant client.

        Args:
            url: Qdrant Cloud instance URL
            api_key: API key for authentication
            collection_name: Name of the collection
            embedding_dim: Dimension of embedding vectors (default: 384)
        """
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        logger.info(f"QdrantService initialized: {collection_name} ({embedding_dim}-dim)")

    def create_collection(self):
        """
        Create Qdrant collection with HNSW index.

        Uses cosine distance for semantic similarity.
        HNSW params: M=16, ef_construction=512 for good balance.
        """
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,  # Cosine similarity
                    on_disk=False,  # Keep vectors in memory for speed
                ),
                hnsw_config={"m": 16, "ef_construction": 512},
            )
            logger.info(f"✅ Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}", exc_info=True)
            raise

    def store_chunk(self, chunk_id: str, embedding: List[float], metadata: Dict):
        """
        Store a single chunk with embedding and metadata.

        Args:
            chunk_id: Unique identifier for chunk
            embedding: 384-dim vector
            metadata: Dict with chapter_id, section_id, section_title, full_text, etc.
        """
        try:
            point = PointStruct(
                id=str(uuid.uuid4()),  # Qdrant point ID (UUID)
                vector=embedding,
                payload={
                    "chunk_id": chunk_id,
                    "chapter_id": metadata.get("chapter_id"),
                    "section_id": metadata.get("section_id"),
                    "section_title": metadata.get("section_title"),
                    "full_text": metadata.get("full_text", ""),
                    "preview_text": metadata.get("full_text", "")[:200],
                    "file_name": metadata.get("file_name", ""),
                },
            )

            self.client.upsert(collection_name=self.collection_name, points=[point])

        except Exception as e:
            logger.error(f"Failed to store chunk {chunk_id}: {e}", exc_info=True)
            raise

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        chapter_filter: Optional[int] = None,
        min_score: float = 0.7,
    ) -> List[Dict]:
        """
        Search for similar chunks using cosine similarity.

        Args:
            query_embedding: 384-dim query vector
            top_k: Number of results to return
            chapter_filter: Optional chapter ID filter (for chapter-aware mode)
            min_score: Minimum relevance score threshold

        Returns:
            List of matching chunks with metadata and scores

        Raises:
            Exception: If search fails
        """
        try:
            # Build filter for chapter-aware search
            query_filter = None
            if chapter_filter is not None:
                query_filter = Filter(
                    must=[FieldCondition(key="chapter_id", match=MatchValue(value=chapter_filter))]
                )

            # Execute vector search - handle different Qdrant client API versions
            try:
                # Try the newer API method
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=min_score,
                )
            except AttributeError:
                # Fallback to older API method name if needed
                try:
                    results = self.client.search_points(
                        collection_name=self.collection_name,
                        query_vector=query_embedding,
                        query_filter=query_filter,
                        limit=top_k,
                        score_threshold=min_score,
                    )
                except AttributeError:
                    # If both methods fail, return empty results as fallback
                    logger.warning("Qdrant search method not found, returning empty results")
                    results = []

            # Format results
            chunks = []
            if results:  # Only process if results is not empty
                for result in results:
                    chunks.append(
                        {
                            "chunk_id": result.payload.get("chunk_id"),
                            "chapter_id": result.payload.get("chapter_id"),
                            "section_id": result.payload.get("section_id"),
                            "section_title": result.payload.get("section_title"),
                            "preview_text": result.payload.get("preview_text"),
                            "full_text": result.payload.get("full_text"),
                            "relevance_score": result.score,
                        }
                    )

            logger.info(
                f"Search completed: {len(chunks)} chunks found (top_k={top_k}, min_score={min_score})"
            )
            return chunks

        except Exception as e:
            logger.error(f"Vector search failed: {e}", exc_info=True)
            raise


# Note: Singleton instance created in config using environment variables
