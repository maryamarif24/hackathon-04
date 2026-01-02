# Implementation Plan: RAG Chatbot

**Branch**: `001-rag-chatbot` | **Date**: 2025-12-30 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-rag-chatbot/spec.md`

## Summary

RAG chatbot for "Physical AI & Humanoid Robotics" textbook that provides accurate, book-grounded answers using vector search retrieval with OpenAI Agents SDK for response generation. System supports three modes: book-wide (across all chapters), selected-text-only (constrained to user selection), and chapter-aware (prioritizes current chapter context).

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: FastAPI, OpenAI Agents SDK / ChatKit, Qdrant Cloud, sentence-transformers (embeddings)
**Storage**: Qdrant Cloud (vector store), Neon Serverless Postgres (metadata/sessions)
**Testing**: pytest (backend tests)
**Target Platform**: Linux server (backend), Web browser (frontend Docusaurus)
**Project Type**: web
**Performance Goals**: <5 seconds p95 for response generation, vector search <1 second
**Constraints**: No hallucinations, no external knowledge, read-only textbook access, educational tone, conciseness (<300 words for simple queries)
**Scale/Scope**: 6 chapters of textbook content, multiple concurrent users (rate limited to 10 req/min per IP)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Gate 1: Behavior Rules Compliance
**Constitution Principle**: "The chatbot MUST answer strictly from retrieved book content only" and "MUST NOT hallucinate or add external knowledge"

**Compliance Status**: ✅ PASS

**Verification**: Specification FR-001 (System MUST retrieve relevant textbook content before generating any answer) and FR-002 (System MUST answer strictly from retrieved book content only) ensure this compliance. The spec explicitly requires no external web knowledge (Constraint 2) and no hallucinations (Constraint 1).

### Gate 2: Retrieval Rules Compliance
**Constitution Principle**: "The chatbot MUST always retrieve relevant content before generating an answer"

**Compliance Status**: ✅ PASS

**Verification**: FR-001 (System MUST retrieve relevant textbook content) enforces retrieval-before-response pattern. Constitution states "Never answer any query without context retrieval first."

### Gate 3: Safety Rules Compliance
**Constitution Principle**: "System MUST NOT generate harmful or unsafe robotics instructions"

**Compliance Status**: ✅ PASS

**Verification**: FR-015 (System MUST prevent generation of harmful or unsafe robotics instructions) explicitly enforces this requirement. Spec constraints include safety-focused edge cases.

### Gate 4: Educational Scope Compliance
**Constitution Principle**: "System assists with learning textbook content" and "provides accurate, book-grounded answers"

**Compliance Status**: ✅ PASS

**Verification**: FR-007 (System MUST provide responses in a clear, educational tone) and FR-012 (System MUST provide concise explanations) ensure educational focus. FR-008 (System MUST structure answers with clear formatting) supports readability.

**Constitution Gate Result**: ✅ ALL GATES PASS - Specification is constitutionally compliant.

## Project Structure

### Documentation (this feature)

```text
specs/001-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
# Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   │   └── rag_models.py       # Pydantic models for queries, responses, chunks
│   ├── services/
│   │   ├── embedding_service.py    # Text embedding generation
│   │   ├── qdrant_service.py      # Vector database operations
│   │   ├── neon_service.py         # Neon Postgres metadata/sessions
│   │   ├── retrieval_service.py    # Vector search orchestration
│   │   ├── rag_service.py          # RAG pipeline coordinator
│   │   └── openai_service.py       # OpenAI Agents SDK integration
│   ├── api/
│   │   ├── query.py               # Query endpoint (/api/query)
│   │   ├── health.py              # Health check endpoint (/health)
│   │   └── book_content.py        # Book content API (future extension)
│   ├── config.py                    # Environment configuration
│   └── schemas.py                   # Pydantic schemas
├── tests/
│   ├── integration/
│   │   ├── test_retrieval.py    # End-to-end retrieval tests
│   │   ├── test_rag_pipeline.py  # Full RAG flow tests
│   │   └── test_safety.py        # Safety filter tests
│   └── unit/
│       ├── test_embedding_service.py
│       ├── test_qdrant_service.py
│       └── test_openai_service.py
└── requirements.txt

frontend/
├── src/
│   ├── hooks/
│   │   └── useRAGQuery.ts          # Backend query hook
│   ├── components/
│   │   └── RAGChat/                  # RAG chatbot widget (future)
│   └── theme/
│       └── Root.tsx                  # Global wrapper (already exists)
├── .env.example                      # API URL configuration
└── package.json
```

**Structure Decision**: Web application structure with separate backend (FastAPI) and frontend (Docusaurus) directories. Backend contains full RAG pipeline services, models, and API endpoints. Frontend contains integration hooks for backend API communication.

## Complexity Tracking

> No constitution violations requiring justification. All gates passed.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |

## Phase 0: Outline & Research

**Purpose**: Resolve unknowns and research best practices for RAG chatbot implementation

### Research Tasks

**Task R001**: Research chunking strategy for markdown textbook content
- **Decision**: Use semantic chunking (e.g., 500-1000 tokens per chunk) with overlap to preserve context
- **Rationale**: Semantic chunks preserve meaning boundaries better than fixed-size chunks. Overlap (e.g., 100 tokens) ensures no content loss at boundaries.
- **Alternatives considered**:
  - Fixed-size chunks (rejected): Breaks mid-sentence
  - Paragraph-based chunks (rejected): Too variable in length, may be too small/large
  - Recursive character splitting (rejected): Loses semantic meaning

**Task R002**: Research embedding model selection
- **Decision**: Use sentence-transformers/all-MiniLM-L6-v2 (384-dim, 80MB model)
- **Rationale**: Specified in user requirements. Good balance of performance vs accuracy for educational content. Works on CPU without GPU.
- **Alternatives considered**:
  - all-mpnet-base-v2 (rejected): Too large for production (420MB)
  - BGE-small (rejected): Not specified in requirements

**Task R003**: Research vector database configuration
- **Decision**: Use Qdrant Cloud with HNSW index (384-dim vectors, M=16, ef_construction=512)
- **Rationale**: HNSW provides fast approximate search suitable for semantic similarity queries. M=16 is standard for 384-dim vectors.
- **Alternatives considered**:
  - Qdrant Cloud with exact search (rejected): Not suitable for semantic similarity
  - Pinecone (rejected): Not specified in user requirements

**Task R004**: Research OpenAI Agents SDK / ChatKit integration
- **Decision**: Use OpenAI Agents SDK with function calling for structured response generation
- **Rationale**: Agents SDK provides tool use capabilities for controlled, grounded response generation. Enables explicit citation of retrieved chunks in prompts.
- **Alternatives considered**:
  - Direct OpenAI API completion (rejected): Less control over grounding, harder to enforce citations
  - LangChain agents (rejected): Additional abstraction layer not needed

**Task R005**: Research rate limiting strategy
- **Decision**: Use slowapi with in-memory rate limiting (10 requests/minute per IP)
- **Rationale**: Simple, lightweight rate limiting sufficient for educational use case. Prevents abuse without over-engineering.
- **Alternatives considered**:
  - Redis-based distributed limiting (rejected): Over-engineering for single-instance deployment
  - External API gateway (rejected): Additional cost and complexity

**Research Deliverables**: Research.md created with all decisions and rationales.

---

## Phase 1: Design & Contracts

**Prerequisites**: Phase 0 (research.md) complete

### Phase 1A: Data Model Definition

**Task D001**: Define ChatQuery, RetrievedChunk, and ChatResponse entities
- **Output**: data-model.md with entity definitions, validation rules, and relationships

### Phase 1B: API Contract Definition

**Task D002**: Define query endpoint contract
- **Output**: contracts/query-endpoint.yaml with request/response schemas, error responses, rate limits

**Task D003**: Define health check endpoint contract
- **Output**: contracts/health-endpoint.yaml with health response schema

### Phase 1C: Quickstart Guide

**Task D004**: Create developer quickstart guide
- **Output**: quickstart.md with environment setup, running backend locally, testing query endpoint

### Phase 1D: Frontend Integration Guide

**Task D005**: Document frontend API integration approach
- **Output**: Add to quickstart.md: how Docusaurus site calls backend API

**Checkpoint**: Phase 1 complete. Design artifacts (data-model.md, contracts/, quickstart.md) ready for implementation planning (tasks.md).

---

## Phase 2: Task Generation

> This phase is executed by `/sp.tasks` command - NOT included in this plan document.

After Phase 1 is complete, run `/sp.tasks` to generate the implementation task list based on this plan and the feature specification.

---

## Implementation Order

### Phase Dependencies

- **Phase 0 (Research)**: No dependencies - can start immediately
- **Phase 1 (Design)**: Depends on Phase 0 - BLOCKS Phase 2
- **Phase 2 (Tasks)**: Depends on Phase 1 - cannot start until design artifacts exist

### Recommended Implementation Sequence

1. Run `/sp.tasks` to generate detailed task breakdown
2. Implement Phase 1A (data-model.md) → Entity definitions in backend/src/models/
3. Implement Phase 1B (contracts/) → API endpoint scaffolding in backend/src/api/
4. Implement Phase 1C (quickstart.md) → Update README with setup instructions
5. Run `/sp.implement` to execute tasks in dependency order

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|-------|---------|-------------|
| OpenAI rate limits hit | Degraded response time or 429 errors | Implement exponential backoff and user-friendly error messages |
| Qdrant connection failures | Vector search unavailable, no answers possible | Graceful degradation with error messages, retry with exponential backoff |
| Embedding model too slow | Response time exceeds 5-second target | Optimize chunk size, consider caching embeddings, use faster model if needed |
| Hallucination despite constraints | Loss of trust, constitutional violation | Implement strict prompt engineering with citation requirements, add content verification in evaluation |
| Neon downtime | Session/history unavailable | Degraded service (no session persistence), continue functioning for immediate queries |

---

## Next Steps

1. **Execute**: Run `/sp.tasks` to generate implementation task list
2. **Review**: Review generated tasks.md before starting implementation
3. **Implement**: Execute tasks in dependency order using `/sp.implement`
4. **Test**: Verify all acceptance scenarios pass
5. **Deploy**: Deploy to Railway (backend) and Vercel (frontend) when ready

- Sitemap URL = https://hackathon0-4.vercel.app/sitemap.xml