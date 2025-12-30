---
id: 004
title: rag-chatbot-implementation
stage: green
date: 2025-12-30
surface: agent
model: claude-sonnet-4.5
feature: rag-chatbot
branch: 001-rag-chatbot
user: unknown
command: /sp.implement
labels: [implementation, architecture, rag-pipeline]
links:
  spec: D:\Spec-driven-dev\Hackathon04\hackathon-04\specs\001-rag-chatbot\spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/001-rag-chatbot/IMPLEMENTATION_GUIDE.md (created)
tests:
 - none
---

## Prompt

Implement the RAG chatbot system according to:
- The RAG constitution
- The RAG specification
- The system plan
- The task breakdown

Describe in detail:
- Folder and file structure
- How markdown content is ingested
- Chunking strategy and metadata schema
- Embedding and storage flow
- Retrieval process for normal queries
- Retrieval process for selected-text-only queries
- How agents collaborate to generate answers
- FastAPI endpoint definitions
- Safety and hallucination prevention mechanisms

Do not include API keys.
Do not deploy the system.
Focus on clarity, correctness, and explainability.

## Response snapshot

Created comprehensive implementation guide for RAG chatbot system.

**Implementation Guide Sections**:

1. **Folder and File Structure**: Complete project layout (backend services, models, API endpoints, tests, frontend hooks)

2. **Markdown Content Ingestion**:
   - Frontmatter parsing for chapter/section metadata
   - Ingestion script reads from website/docs/ directory
   - Batch processing of all chapters

3. **Chunking Strategy**:
   - Semantic chunking by markdown headings
   - 500-1000 tokens per chunk with 100-token overlap
   - Metadata schema includes chunk_id, chapter_id, section_id, section_title, full_text

4. **Embedding and Storage Flow**:
   - sentence-transformers/all-MiniLM-L6-v2 (384-dim)
   - Qdrant Cloud with HNSW index (M=16, ef_construction=512)
   - Cosine similarity for semantic search

5. **Retrieval Process (Normal Queries)**:
   - Question → Embedding → Vector search → Filter by relevance (>0.7) → Return top-k
   - Chapter-aware mode: Apply chapter_id filter in Qdrant

6. **Retrieval Process (Selected-Text-Only)**:
   - Skip vector search entirely
   - Create synthetic chunk from selected text
   - Pass directly to generation agent

7. **Agent Collaboration**:
   - Retrieval Agent: Vector search orchestration
   - Generation Agent: OpenAI Agents SDK with function calling
   - System prompt enforces grounding and refusal instructions

8. **FastAPI Endpoints**:
   - POST /api/query: Main RAG endpoint with rate limiting (10/min)
   - GET /health: Service health check
   - CORS middleware, error handling, logging

9. **Safety and Hallucination Prevention**:
   - 10 mechanisms: Constitution enforcement, retrieval-before-generation, relevance filtering, low temperature (0.3), max tokens (800), empty context handling, selected-text enforcement, mode-specific prompts, audit logging, rate limiting

**Key Implementation Details**:
- All services use singleton pattern
- Error handling with user-friendly messages (503 for service unavailable)
- Metadata schema includes chapter/section for citations
- Three answering modes: book-wide, selected-text-only, chapter-aware

## Outcome

- ✅ Impact: Comprehensive RAG chatbot implementation guide created with detailed code examples and architecture explanations
- 🧪 Tests: none (not requested in spec)
- 📁 Files: 1 created (IMPLEMENTATION_GUIDE.md)
- 🔁 Next prompts: Begin actual implementation following the guide, or use /sp.phr to record further prompts
- 🧠 Reflection: Implementation guide provides complete architecture, data flows, and safety mechanisms. Ready for code execution.

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): Requirements checklist - all items pass
- Prompt variant (if applicable): none
- Next experiment (smallest change to try): none
