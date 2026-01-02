# RAG Chatbot Implementation Guide

**Feature**: 001-rag-chatbot
**Date**: 2025-12-30
**Status**: Implementation guide for RAG chatbot system
**Purpose**: Detailed explanation of folder structure, data flows, and implementation details

---

## Table of Contents

1. [Folder and File Structure](#folder-and-file-structure)
2. [Markdown Content Ingestion](#markdown-content-ingestion)
3. [Chunking Strategy and Metadata Schema](#chunking-strategy-and-metadata-schema)
4. [Embedding and Storage Flow](#embedding-and-storage-flow)
5. [Retrieval Process for Normal Queries](#retrieval-process-for-normal-queries)
6. [Retrieval Process for Selected-Text-Only Queries](#retrieval-process-for-selected-text-only-queries)
7. [Agent Collaboration for Answer Generation](#agent-collaboration-for-answer-generation)
8. [FastAPI Endpoint Definitions](#fastapi-endpoint-definitions)
9. [Safety and Hallucination Prevention](#safety-and-hallucination-prevention)
10. [Implementation Checklist](#implementation-checklist)

---

## Folder and File Structure

### Project Root Structure

```
hackathon-04/
├── backend/                  # FastAPI RAG backend
│   ├── src/
│   │   ├── models/            # Pydantic models
│   │   │   └── rag_models.py  # ChatQuery, RetrievedChunk, ChatResponse
│   │   ├── services/          # Business logic
│   │   │   ├── embedding_service.py     # Text → vectors
│   │   │   ├── qdrant_service.py        # Vector database operations
│   │   │   ├── neon_service.py          # Postgres metadata
│   │   │   ├── retrieval_service.py     # Vector search orchestration
│   │   │   ├── openai_service.py        # OpenAI Agents SDK
│   │   │   └── rag_service.py           # RAG pipeline coordinator
│   │   ├── api/               # API endpoints
│   │   │   ├── query.py       # POST /api/query
│   │   │   ├── health.py      # GET /health
│   │   │   └── book_content.py  # Future extension
│   │   ├── config.py          # Environment configuration
│   │   └── schemas.py         # Additional Pydantic schemas
│   ├── app/
│   │   └── main.py            # FastAPI application entry point
│   ├── tests/                 # Test suite (future)
│   │   ├── integration/
│   │   └── unit/
│   ├── .env                   # Environment variables (not in git)
│   ├── .env.example           # Example environment variables
│   └── requirements.txt       # Python dependencies
├── frontend/                  # Docusaurus site (existing)
│   └── src/
│       └── hooks/
│           └── useRAGQuery.ts # Future: Backend query hook
├── website/                   # Docusaurus textbook content
│   └── docs/                  # Markdown chapters
│       ├── chapter-1-introduction-to-physical-ai.md
│       ├── chapter-2-basics-of-humanoid-robotics.md
│       ├── chapter-3-ros-2-fundamentals.md
│       ├── chapter-4-digital-twin-simulation.md
│       ├── chapter-5-vision-language-action-systems.md
│       └── chapter-6-capstone-simple-ai-robot-pipeline.md
├── specs/
│   └── 001-rag-chatbot/
│       ├── spec.md             # Feature specification
│       ├── plan.md             # Implementation plan
│       ├── tasks.md            # Task breakdown
│       ├── data-model.md       # Entity definitions
│       ├── research.md         # Research decisions
│       ├── quickstart.md       # Developer guide
│       └── IMPLEMENTATION_GUIDE.md  # This file
└── .gitignore                 # Git ignore patterns
```

### Key Files Explained

- **`backend/src/models/rag_models.py`**: Pydantic models for request/response validation
- **`backend/src/services/rag_service.py`**: Main orchestrator - coordinates retrieval and generation
- **`backend/src/services/embedding_service.py`**: Converts text to 384-dim vectors using sentence-transformers
- **`backend/src/services/qdrant_service.py`**: Interfaces with Qdrant Cloud for vector storage/search
- **`backend/src/services/openai_service.py`**: Integrates with OpenAI Agents SDK for answer generation
- **`backend/app/main.py`**: FastAPI application with CORS, rate limiting, and endpoint registration

---

## Markdown Content Ingestion

### Source Content Format

Textbook chapters are stored as markdown files with frontmatter:

```markdown
---
title: "Chapter 1: Introduction to Physical AI"
chapter: 1
section_id: "intro"
---

# Introduction to Physical AI

Physical AI refers to artificial intelligence systems that interact with the physical world...

## Key Components

1. **Sensors**: Collect data from the environment...
2. **Actuators**: Execute physical actions...
3. **Control Systems**: Coordinate sensor-actuator loops...
```

### Ingestion Process

```python
# scripts/ingest_textbook.py (to be created)

import os
import frontmatter
from pathlib import Path
from src.services.embedding_service import embedding_service
from src.services.qdrant_service import qdrant_service

def ingest_textbook():
    """
    Ingest markdown chapters from website/docs/ directory
    """
    docs_dir = Path("website/docs")
    chunks = []

    for md_file in docs_dir.glob("chapter-*.md"):
        # Parse markdown with frontmatter
        post = frontmatter.load(md_file)
        content = post.content
        metadata = post.metadata

        # Extract chapter and section info
        chapter_id = metadata.get('chapter', 0)

        # Chunk the content (see next section)
        chapter_chunks = chunk_by_headings(
            content=content,
            chapter_id=chapter_id,
            file_name=md_file.name
        )
        chunks.extend(chapter_chunks)

    # Generate embeddings and store in Qdrant
    for chunk in chunks:
        embedding = embedding_service.generate_embedding(chunk['text'])
        qdrant_service.store_chunk(
            chunk_id=chunk['id'],
            embedding=embedding,
            metadata=chunk['metadata']
        )

    print(f"Ingested {len(chunks)} chunks from {len(list(docs_dir.glob('chapter-*.md')))} chapters")
```

---

## Chunking Strategy and Metadata Schema

### Chunking Strategy

**Decision**: Semantic chunking by markdown headings (500-1000 tokens per chunk with 100-token overlap)

**Rationale**:
- Preserves semantic boundaries (headings indicate topic changes)
- Overlap ensures no content loss at boundaries
- Chunk size (500-1000 tokens) fits LLM context windows and preserves readability

### Implementation

```python
# src/services/chunking_service.py

import re
from typing import List, Dict

def chunk_by_headings(content: str, chapter_id: int, file_name: str) -> List[Dict]:
    """
    Chunk markdown content by headings with overlap

    Args:
        content: Markdown text
        chapter_id: Chapter number
        file_name: Source file name

    Returns:
        List of chunk dictionaries with metadata
    """
    chunks = []

    # Split by markdown headings (## or ###)
    sections = re.split(r'\n(#{2,3})\s+(.+?)\n', content)

    current_heading = ""
    current_text = ""
    chunk_counter = 0

    for i in range(0, len(sections), 3):
        if i + 2 < len(sections):
            heading_level = sections[i + 1]
            heading_text = sections[i + 2]
            section_content = sections[i + 3] if i + 3 < len(sections) else ""

            # Combine heading + content
            full_text = f"{heading_text}\n\n{section_content}".strip()

            # If chunk is too large (>1000 tokens ≈ 4000 chars), split further
            if len(full_text) > 4000:
                sub_chunks = split_large_section(full_text, max_chars=4000, overlap=400)
                for sub_chunk in sub_chunks:
                    chunks.append({
                        'id': f"ch{chapter_id}_sec{chunk_counter}",
                        'text': sub_chunk,
                        'metadata': {
                            'chapter_id': chapter_id,
                            'section_id': str(chunk_counter),
                            'section_title': heading_text,
                            'file_name': file_name
                        }
                    })
                    chunk_counter += 1
            else:
                chunks.append({
                    'id': f"ch{chapter_id}_sec{chunk_counter}",
                    'text': full_text,
                    'metadata': {
                        'chapter_id': chapter_id,
                        'section_id': str(chunk_counter),
                        'section_title': heading_text,
                        'file_name': file_name
                    }
                })
                chunk_counter += 1

    return chunks

def split_large_section(text: str, max_chars: int = 4000, overlap: int = 400) -> List[str]:
    """
    Split large text into overlapping chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Overlap for context preservation

    return chunks
```

### Metadata Schema

**Qdrant Payload Schema**:

```python
{
    "chunk_id": "ch1_sec3_actuators_001",        # Unique identifier
    "chapter_id": 1,                            # Chapter number (1-6)
    "section_id": "3",                          # Section within chapter
    "section_title": "Actuators and Motion",    # Human-readable section title
    "full_text": "Actuators are the muscles...", # Complete chunk content
    "preview_text": "Actuators are the musc...", # First 200 chars for display
    "file_name": "chapter-1-introduction-to-physical-ai.md",
    "indexed_at": "2025-12-30T12:00:00Z"        # Timestamp for debugging
}
```

---

## Embedding and Storage Flow

### Embedding Generation

```python
# backend/src/services/embedding_service.py

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingService:
    def __init__(self):
        self.model = None
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_dim = 384

    def load_model(self):
        """Load embedding model on application startup"""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            print(f"Loaded embedding model: {self.model_name} ({self.embedding_dim}-dim)")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate 384-dimensional embedding for input text

        Args:
            text: Input text to embed

        Returns:
            384-dim vector as list of floats
        """
        if self.model is None:
            self.load_model()

        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)

        # Normalize vector (for cosine similarity)
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.tolist()

    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch

        Args:
            texts: List of input texts

        Returns:
            List of 384-dim vectors
        """
        if self.model is None:
            self.load_model()

        embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)

        # Normalize each vector
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return normalized.tolist()

# Singleton instance
embedding_service = EmbeddingService()
```

### Storage in Qdrant Cloud

```python
# backend/src/services/qdrant_service.py

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Optional
import uuid
from src.config import settings

class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.embedding_dim = 384

    def create_collection(self):
        """
        Create Qdrant collection with HNSW index
        """
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE,  # Cosine similarity
                on_disk=False  # Keep vectors in memory for speed
            ),
            hnsw_config={
                "m": 16,  # Number of connections per node
                "ef_construction": 512  # Size of dynamic candidate list for construction
            }
        )
        print(f"Created collection: {self.collection_name}")

    def store_chunk(self, chunk_id: str, embedding: List[float], metadata: Dict):
        """
        Store a single chunk with embedding and metadata

        Args:
            chunk_id: Unique identifier for chunk
            embedding: 384-dim vector
            metadata: Metadata dict (chapter_id, section_id, section_title, full_text, etc.)
        """
        point = PointStruct(
            id=str(uuid.uuid4()),  # Qdrant point ID
            vector=embedding,
            payload={
                "chunk_id": chunk_id,
                "chapter_id": metadata.get("chapter_id"),
                "section_id": metadata.get("section_id"),
                "section_title": metadata.get("section_title"),
                "full_text": metadata.get("full_text", ""),
                "preview_text": metadata.get("full_text", "")[:200],  # First 200 chars
                "file_name": metadata.get("file_name", "")
            }
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        chapter_filter: Optional[int] = None,
        min_score: float = 0.7
    ) -> List[Dict]:
        """
        Search for similar chunks using cosine similarity

        Args:
            query_embedding: 384-dim query vector
            top_k: Number of results to return
            chapter_filter: Optional chapter ID filter
            min_score: Minimum relevance score threshold

        Returns:
            List of matching chunks with metadata
        """
        # Build filter for chapter-aware search
        query_filter = None
        if chapter_filter is not None:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="chapter_id",
                        match=MatchValue(value=chapter_filter)
                    )
                ]
            )

        # Execute vector search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=min_score  # Filter by relevance score
        )

        # Format results
        chunks = []
        for result in results:
            chunks.append({
                "chunk_id": result.payload.get("chunk_id"),
                "chapter_id": result.payload.get("chapter_id"),
                "section_id": result.payload.get("section_id"),
                "section_title": result.payload.get("section_title"),
                "preview_text": result.payload.get("preview_text"),
                "full_text": result.payload.get("full_text"),
                "relevance_score": result.score
            })

        return chunks

# Singleton instance
qdrant_service = QdrantService()
```

---

## Retrieval Process for Normal Queries

**Normal query**: User asks a general question across the entire textbook (book-wide mode).

### Flow Diagram

```
User Question
    ↓
[Embedding Service] Convert question to 384-dim vector
    ↓
[Qdrant Service] Vector similarity search (cosine distance)
    ↓
Filter results by relevance score (>0.7)
    ↓
Return top-k chunks with metadata
    ↓
[RAG Service] Pass chunks to OpenAI Agents SDK
    ↓
Generate grounded answer with citations
    ↓
ChatResponse returned to user
```

### Implementation

```python
# backend/src/services/retrieval_service.py

from typing import List, Dict, Optional
from src.services.embedding_service import embedding_service
from src.services.qdrant_service import qdrant_service

class RetrievalService:
    def __init__(self):
        self.default_top_k = 5
        self.min_relevance_score = 0.7

    def retrieve_book_wide(
        self,
        question: str,
        top_k: int = None,
        chapter_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a book-wide query

        Args:
            question: User's natural language question
            top_k: Number of chunks to retrieve (default: 5)
            chapter_id: Optional chapter context for chapter-aware mode

        Returns:
            List of RetrievedChunk dicts with relevance scores
        """
        if top_k is None:
            top_k = self.default_top_k

        # Step 1: Generate embedding for the question
        query_embedding = embedding_service.generate_embedding(question)

        # Step 2: Search Qdrant for similar chunks
        chunks = qdrant_service.search(
            query_embedding=query_embedding,
            top_k=top_k,
            chapter_filter=chapter_id,  # None for book-wide, int for chapter-aware
            min_score=self.min_relevance_score
        )

        # Step 3: Return chunks (empty list if no relevant content found)
        return chunks

# Singleton instance
retrieval_service = RetrievalService()
```

---

## Retrieval Process for Selected-Text-Only Queries

**Selected-text-only query**: User selects specific text from the page and asks a question constrained to ONLY that text.

### Key Differences from Normal Queries

1. **No vector search**: Skip Qdrant entirely
2. **Direct text usage**: Use the provided `context` field directly
3. **Strict grounding**: OpenAI prompt enforces answering ONLY from selected text

### Flow Diagram

```
User Question + Selected Text
    ↓
[Validation] Check if use_context_only=true and context is non-empty
    ↓
[Retrieval Service] Create synthetic "chunk" from selected text
    ↓
[RAG Service] Pass single chunk to OpenAI Agents SDK
    ↓
Generate answer using ONLY selected text
    ↓
ChatResponse returned with single source (selected text)
```

### Implementation

```python
# backend/src/services/retrieval_service.py (continued)

class RetrievalService:
    # ... (previous methods)

    def retrieve_context_only(
        self,
        context: str,
        question: str
    ) -> List[Dict]:
        """
        Create a synthetic chunk from user-selected text

        Args:
            context: User-selected text passage
            question: User's question (for validation)

        Returns:
            Single-item list with synthetic chunk
        """
        if not context or len(context.strip()) < 10:
            raise ValueError("Context must be non-empty and at least 10 characters for selected-text-only mode")

        # Create synthetic chunk (no vector search needed)
        synthetic_chunk = {
            "chunk_id": "selected-text",
            "chapter_id": None,  # Unknown chapter
            "section_id": "selected",
            "section_title": "Selected Text",
            "preview_text": context[:200],  # First 200 chars
            "full_text": context,
            "relevance_score": 1.0  # Perfect "relevance" since user provided it
        }

        return [synthetic_chunk]
```

---

## Agent Collaboration for Answer Generation

### OpenAI Agents SDK Integration

The OpenAI Agents SDK enables **function calling** and **structured response generation** for controlled, grounded answers.

### Agent Responsibilities

1. **Retrieval Agent** (implicit via retrieval_service):
   - Search Qdrant for relevant chunks
   - Filter by relevance score
   - Apply chapter context if needed

2. **Generation Agent** (via openai_service):
   - Receive retrieved chunks as context
   - Generate educational, grounded answers
   - Enforce citation requirements
   - Refuse to answer if information not present

### Implementation

```python
# backend/src/services/openai_service.py

from openai import OpenAI
from typing import List, Dict
from src.config import settings

class OpenAIService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL  # e.g., "gpt-4o-mini"

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        mode: str = "book-wide"
    ) -> str:
        """
        Generate grounded answer using OpenAI Agents SDK

        Args:
            question: User's question
            retrieved_chunks: List of RetrievedChunk dicts
            mode: "book-wide" | "selected-text-only" | "chapter-aware"

        Returns:
            Generated answer text
        """
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

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Low temperature for factual responses
            max_tokens=800    # Limit response length
        )

        answer = response.choices[0].message.content.strip()

        return answer

    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context text from retrieved chunks"""
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
        """Build system prompt based on answering mode"""
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
            base_prompt += "\nMODE: Selected-Text-Only - Answer using ONLY the selected text passage provided."
        elif mode == "chapter-aware":
            chapter_id = chunks[0].get("chapter_id") if chunks else None
            base_prompt += f"\nMODE: Chapter-Aware - Prioritize content from Chapter {chapter_id} when answering."
        else:
            base_prompt += "\nMODE: Book-wide - Answer using relevant content from across the entire textbook."

        return base_prompt

# Singleton instance
openai_service = OpenAIService()
```

### RAG Service Coordinator

```python
# backend/src/services/rag_service.py

from typing import Dict, List, Tuple
import time
from src.services.retrieval_service import retrieval_service
from src.services.openai_service import openai_service

class RAGService:
    def process_query(
        self,
        question: str,
        context: str = None,
        use_context_only: bool = False,
        top_k: int = 5,
        chapter_id: int = None
    ) -> Tuple[str, List[Dict], int]:
        """
        Main RAG pipeline coordinator

        Args:
            question: User's natural language question
            context: Selected text passage (for selected-text-only mode)
            use_context_only: Flag to restrict to selected text only
            top_k: Number of chunks to retrieve (for book-wide mode)
            chapter_id: Optional chapter context (for chapter-aware mode)

        Returns:
            Tuple of (answer, sources, query_time_ms)
        """
        start_time = time.time()

        # Step 1: Determine answering mode
        if use_context_only:
            mode = "selected-text-only"
            sources = retrieval_service.retrieve_context_only(context, question)
        elif chapter_id is not None:
            mode = "chapter-aware"
            sources = retrieval_service.retrieve_book_wide(question, top_k, chapter_id)
        else:
            mode = "book-wide"
            sources = retrieval_service.retrieve_book_wide(question, top_k)

        # Step 2: Generate answer using OpenAI Agents SDK
        answer = openai_service.generate_answer(
            question=question,
            retrieved_chunks=sources,
            mode=mode
        )

        # Step 3: Calculate query time
        query_time_ms = int((time.time() - start_time) * 1000)

        return answer, sources, query_time_ms

# Singleton instance
rag_service = RAGService()
```

---

## FastAPI Endpoint Definitions

### Main Application Entry Point

```python
# backend/app/main.py

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.config import settings
from src.api.health import router as health_router
from src.api.query import router as query_router
from src.services.embedding_service import embedding_service

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle manager"""
    # Startup
    logger.info("🚀 Starting Physical AI Textbook RAG API...")
    logger.info(f"Environment: {settings.LOG_LEVEL}")
    logger.info(f"CORS Origins: {settings.CORS_ORIGINS}")

    # Load embedding model
    logger.info("Loading embedding model...")
    embedding_service.load_model()
    logger.info("✅ Embedding model loaded successfully")

    yield

    # Shutdown
    logger.info("Shutting down API...")

# Create FastAPI application
app = FastAPI(
    title="Physical AI & Humanoid Robotics Textbook RAG API",
    description="RAG-powered API for querying textbook content",
    version="1.0.0",
    lifespan=lifespan
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health_router, prefix="", tags=["Health"])
app.include_router(query_router, prefix="/api", tags=["Query"])

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Physical AI Textbook RAG API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
```

### Query Endpoint

```python
# backend/src/api/query.py

import logging
from fastapi import APIRouter, Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.models.rag_models import ChatQuery, ChatResponse, RetrievedChunk
from src.services.rag_service import rag_service

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

@router.post("/query", response_model=ChatResponse)
@limiter.limit("10/minute")  # Rate limit: 10 requests per minute per IP
async def query_textbook(request: Request, query: ChatQuery):
    """
    Query the Physical AI textbook using RAG

    Args:
        query: ChatQuery with question, context, use_context_only, top_k, chapter_id

    Returns:
        ChatResponse with answer, sources, query_time_ms, mode

    Raises:
        400: Invalid query parameters
        503: Service unavailable (database or AI service down)
    """
    try:
        # Validate query
        if not query.question or len(query.question.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Question must be at least 3 characters"
            )

        if query.use_context_only and not query.context:
            raise HTTPException(
                status_code=400,
                detail="Context is required when use_context_only is true"
            )

        # Process query through RAG pipeline
        answer, sources, query_time_ms = rag_service.process_query(
            question=query.question,
            context=query.context,
            use_context_only=query.use_context_only,
            top_k=query.top_k,
            chapter_id=query.chapter_id
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
                relevance_score=s.get("relevance_score", 0.0)
            )
            for s in sources
        ]

        return ChatResponse(
            answer=answer,
            sources=source_chunks,
            query_time_ms=query_time_ms,
            mode=mode
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Failed to process your query. Please try again later."
        )
```

### Health Check Endpoint

```python
# backend/src/api/health.py

import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Health check endpoint

    Returns:
        Status and version information
    """
    return {
        "status": "healthy",
        "version": "1.0.0"
    }
```

---

## Safety and Hallucination Prevention

### 1. Constitution-Level Enforcement

**Constitution Principle**: "The chatbot MUST answer strictly from retrieved book content only"

**Implementation**:
- System prompt explicitly instructs: "Answer ONLY from provided context"
- System prompt includes refusal instruction: "If information missing, say 'This is not covered in the book'"

### 2. Retrieval-Before-Generation (RAG Pattern)

**Constitution Principle**: "The chatbot MUST always retrieve relevant content before generating an answer"

**Implementation**:
- `rag_service.process_query()` **always** calls `retrieval_service` before `openai_service`
- No code path exists that skips retrieval

### 3. Relevance Score Filtering

**Implementation**:
```python
# In qdrant_service.search()
min_score = 0.7  # Minimum cosine similarity threshold
```

**Effect**: Excludes low-quality matches to prevent answering from irrelevant content

### 4. Low Temperature Setting

**Implementation**:
```python
# In openai_service.generate_answer()
temperature = 0.3  # Low temperature for factual, deterministic responses
```

**Effect**: Reduces hallucination by favoring high-probability tokens

### 5. Max Token Limit

**Implementation**:
```python
# In openai_service.generate_answer()
max_tokens = 800  # Limit response length
```

**Effect**: Enforces conciseness constraint (<300 words for simple queries ≈ 400-600 tokens)

### 6. Empty Context Handling

**Implementation**:
```python
# In openai_service._build_context()
if not chunks:
    return "[No relevant content found in textbook]"
```

**Effect**: If vector search returns no results, OpenAI prompt receives explicit "no content" signal

### 7. Selected-Text-Only Enforcement

**Implementation**:
```python
# In retrieval_service.retrieve_context_only()
if not context or len(context.strip()) < 10:
    raise ValueError("Context must be non-empty...")
```

**Effect**: Prevents selected-text-only mode when no context provided

### 8. System Prompt Mode-Specific Instructions

**Implementation**:
```python
# In openai_service._build_system_prompt()
if mode == "selected-text-only":
    base_prompt += "\nMODE: Selected-Text-Only - Answer using ONLY the selected text passage provided."
```

**Effect**: Reinforces grounding constraint for selected-text-only queries

### 9. Logging for Auditing

**Implementation**:
```python
# In rag_service.process_query()
logger.info(f"Query: {question}, Mode: {mode}, Sources: {len(sources)}, Time: {query_time_ms}ms")
```

**Effect**: Enables post-hoc auditing of query behavior and hallucination detection

### 10. Rate Limiting

**Implementation**:
```python
# In query.py
@limiter.limit("10/minute")  # 10 requests per minute per IP
```

**Effect**: Prevents abuse and ensures fair resource allocation

---

## Implementation Checklist

### Phase 1: Setup (T001-T010)

- [ ] T001 Create project directory structure
- [ ] T002 Initialize Python project with FastAPI dependencies
- [ ] T003 Configure linting (ruff)
- [ ] T004 Create backend/src/models/ directory
- [ ] T005 Create backend/src/services/ directory
- [ ] T006 Create backend/src/api/ directory
- [ ] T007 Create backend/tests/ directory
- [ ] T008 Create requirements.txt
- [ ] T009 Create .env.example
- [ ] T010 Create backend/src/config.py

### Phase 2: US1 - Book-wide QA (T011-T025)

- [ ] T011 Create ChatQuery model
- [ ] T012 Create RetrievedChunk model
- [ ] T013 Create ChatResponse model
- [ ] T014 Create config.py
- [ ] T015 Create embedding_service.py
- [ ] T016 Create qdrant_service.py
- [ ] T017 Create retrieval_service.py
- [ ] T018 Create openai_service.py
- [ ] T019 Create rag_service.py
- [ ] T020 Create /api/query endpoint
- [ ] T021 Create /health endpoint
- [ ] T022 Add CORS middleware
- [ ] T023 Configure rate limiting
- [ ] T024 Update schemas.py
- [ ] T025 Update requirements.txt

### Phase 3: US2 - Selected-Text-Only QA (T026-T029)

- [ ] T026 Update ChatQuery model (add use_context_only)
- [ ] T027 Add context-only logic to retrieval_service
- [ ] T028 Update /api/query endpoint
- [ ] T029 Add error handling for empty context

---

## Next Steps

1. **Execute Setup Tasks** (Phase 1): Create project structure and install dependencies
2. **Implement Core Services** (Phase 2): Embedding, Qdrant, Retrieval, OpenAI, RAG services
3. **Build API Endpoints** (Phase 2): Query and health check endpoints
4. **Add Selected-Text Support** (Phase 3): Context-only retrieval logic
5. **Test End-to-End**: Verify both book-wide and selected-text-only modes work correctly
6. **Deploy** (Future): Deploy backend to Railway/Render, frontend to Vercel

---

**Document Version**: 1.0.0
**Last Updated**: 2025-12-30
