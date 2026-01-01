"""
Neon PostgreSQL service client for chunk metadata storage.
Handles connection pooling and CRUD operations for chunk_metadata table.
"""
import logging
from typing import List, Optional
from uuid import UUID
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

from app.config import settings
from app.schemas import ChunkMetadata

logger = logging.getLogger(__name__)


class NeonService:
    """Service for interacting with Neon PostgreSQL database."""

    def __init__(self):
        """Initialize connection pool to Neon database."""
        try:
            self.pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=settings.neon_connection_string
            )
            logger.info("✅ Neon connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Neon pool: {e}")
            self.pool = None

    def get_connection(self):
        """Get a connection from the pool."""
        if not self.pool:
            raise RuntimeError("Database connection pool not initialized")
        return self.pool.getconn()

    def release_connection(self, conn):
        """Return a connection to the pool."""
        if self.pool:
            self.pool.putconn(conn)

    def insert_chunk_metadata(self, chunk: ChunkMetadata) -> bool:
        """
        Insert chunk metadata into database.

        Args:
            chunk: ChunkMetadata object

        Returns:
            True if insertion succeeded
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO chunk_metadata (
                    chunk_id, chapter_id, section_id, section_title,
                    chunk_index, token_count, char_count, preview_text, indexed_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (chapter_id, section_id, chunk_index)
                DO UPDATE SET
                    token_count = EXCLUDED.token_count,
                    char_count = EXCLUDED.char_count,
                    preview_text = EXCLUDED.preview_text,
                    indexed_at = EXCLUDED.indexed_at;
                """,
                (
                    chunk.chunk_id,
                    chunk.chapter_id,
                    chunk.section_id,
                    chunk.section_title,
                    chunk.chunk_index,
                    chunk.token_count,
                    chunk.char_count,
                    chunk.preview_text,
                    chunk.indexed_at
                )
            )

            conn.commit()
            cursor.close()
            return True

        except Exception as e:
            logger.error(f"Failed to insert chunk metadata: {e}")
            if conn:
                conn.rollback()
            return False

        finally:
            if conn:
                self.release_connection(conn)

    def get_chunk_by_id(self, chunk_id: UUID) -> Optional[dict]:
        """
        Retrieve chunk metadata by UUID.

        Args:
            chunk_id: Chunk UUID

        Returns:
            Dict with chunk metadata or None if not found
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute(
                """
                SELECT chunk_id, chapter_id, section_id, section_title,
                       chunk_index, token_count, char_count, preview_text, indexed_at
                FROM chunk_metadata
                WHERE chunk_id = %s;
                """,
                (chunk_id,)
            )

            result = cursor.fetchone()
            cursor.close()

            return dict(result) if result else None

        except Exception as e:
            logger.error(f"Failed to get chunk by ID: {e}")
            return None

        finally:
            if conn:
                self.release_connection(conn)

    def get_chunks_by_ids(self, chunk_ids: List[UUID]) -> List[dict]:
        """
        Retrieve multiple chunks by UUIDs (for source citations).

        Args:
            chunk_ids: List of chunk UUIDs

        Returns:
            List of dicts with chunk metadata
        """
        if not chunk_ids:
            return []

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute(
                """
                SELECT chunk_id, chapter_id, section_id, section_title,
                       chunk_index, token_count, char_count, preview_text, indexed_at
                FROM chunk_metadata
                WHERE chunk_id = ANY(%s);
                """,
                (chunk_ids,)
            )

            results = cursor.fetchall()
            cursor.close()

            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get chunks by IDs: {e}")
            return []

        finally:
            if conn:
                self.release_connection(conn)

    def get_chapter_stats(self, chapter_id: int) -> dict:
        """
        Get statistics for a specific chapter.

        Args:
            chapter_id: Chapter number (1-6)

        Returns:
            Dict with chapter statistics
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute(
                """
                SELECT
                    COUNT(*) as chunk_count,
                    SUM(token_count) as total_tokens,
                    AVG(token_count) as avg_tokens_per_chunk
                FROM chunk_metadata
                WHERE chapter_id = %s;
                """,
                (chapter_id,)
            )

            result = cursor.fetchone()
            cursor.close()

            return dict(result) if result else {}

        except Exception as e:
            logger.error(f"Failed to get chapter stats: {e}")
            return {}

        finally:
            if conn:
                self.release_connection(conn)

    def get_chunks_by_chapter(self, chapter_id: int) -> List[dict]:
        """
        Retrieve all chunks for a specific chapter.

        Args:
            chapter_id: Chapter number (1-6)

        Returns:
            List of dicts with chunk metadata
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute(
                """
                SELECT chunk_id, chapter_id, section_id, section_title,
                       chunk_index, token_count, char_count, preview_text, indexed_at
                FROM chunk_metadata
                WHERE chapter_id = %s
                ORDER BY chunk_index;
                """,
                (chapter_id,)
            )

            results = cursor.fetchall()
            cursor.close()

            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get chunks by chapter {chapter_id}: {e}")
            return []

        finally:
            if conn:
                self.release_connection(conn)

    def health_check(self) -> bool:
        """
        Check if database is reachable.

        Returns:
            True if healthy
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            cursor.close()
            return True

        except Exception as e:
            logger.error(f"Neon health check failed: {e}")
            return False

        finally:
            if conn:
                self.release_connection(conn)


# Global service instance
neon_service = NeonService()
