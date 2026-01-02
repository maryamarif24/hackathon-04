# Tasks: RAG Chatbot

**Input**: Design documents from `/specs/001-rag-chatbot/plan.md`
**Prerequisites**: plan.md (required), spec.md (required), data-model.md (required), contracts/ (required for API tasks), research.md (required), quickstart.md (required)
**Features**: P1-Book-wide QA, P1-Selected-text-only QA, P2-Chapter-aware, P2-Educational Explanations

**Tests**: No tests requested in feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume web app structure - adjust based on plan.md
- All paths are relative to repository root (D:\Spec-driven-dev\Hackathon04\hackathon-04\)

<!--
  ============================================================================
  IMPORTANT: The tasks below are organized by user story for incremental delivery.

  Each user story is INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.

  MVP Recommendation: US1 + US2 (both P1 priority) provides core functionality
  ============================================================================
-->

## Phase 1: Setup & Infrastructure (Foundational)

**Purpose**: Project initialization and core infrastructure that all user stories depend on.

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel.

- [x] T001 Create project directory structure per implementation plan
- [x] T002 Initialize Python project with FastAPI dependencies
- [x] T003 [P] Configure linting and formatting tools (ruff)
- [x] T004 [P] Create base models directory in backend/src/models/
- [x] T005 [P] Create base services directory in backend/src/services/
- [x] T006 [P] Create base api directory in backend/src/api/
- [x] T007 [P] Create base tests directory in backend/tests/
- [x] T008 [P] Create requirements.txt with FastAPI dependencies
- [x] T009 [P] Create backend/.env.example file
- [x] T010 [P] Create backend/config.py for environment configuration

---

## Phase 2: US1 - Book-wide Question Answering (Priority: P1) 🎯 MVP Part 1

**Goal**: Enable learners to ask general questions across entire textbook with accurate, cited answers.

**Independent Test**: Can be fully tested by asking "What are key components of a humanoid robot?" and verifying system retrieves relevant content and provides chapter/section citations.

### Tests for User Story 1 (OPTIONAL) ⚠️

> **NOTE**: Tests were NOT requested in feature specification. Skipping test tasks.

### Implementation for User Story 1

- [x] T011 [US1] [P] Create ChatQuery model in backend/src/models/rag_models.py
- [x] T012 [US1] [P] Create RetrievedChunk model in backend/src/models/rag_models.py
- [x] T013 [US1] [P] Create ChatResponse model in backend/src/models/rag_models.py
- [x] T014 [US1] Create base config.py in backend/src/config.py (environment variables)
- [x] T015 [US1] [P] Create embedding_service.py in backend/src/services/embedding_service.py
- [x] T016 [US1] [P] Create qdrant_service.py in backend/src/services/qdrant_service.py
- [x] T017 [US1] [P] Create retrieval_service.py in backend/src/services/retrieval_service.py
- [x] T018 [US1] [P] Create openai_service.py in backend/src/services/openai_service.py
- [x] T019 [US1] [P] Create rag_service.py in backend/src/services/rag_service.py
- [x] T020 [US1] [P] Create /api/query endpoint in backend/src/api/query.py (depends on T011-T019)
- [x] T021 [US1] [P] Create /health endpoint in backend/src/api/health.py
- [x] T022 [US1] Add CORS middleware to backend/app/main.py
- [x] T023 [US1] [P] Configure rate limiting with slowapi in backend/app/main.py
- [x] T024 [US1] [P] Update backend/schemas.py with all models
- [x] T025 [US1] [P] Update backend/requirements.txt with all dependencies

**Checkpoint**: At this point, User Story 1 should be fully functional. Learners can ask general questions and receive accurate, textbook-grounded answers with citations.

---

## Phase 3: US2 - Selected-Text-Only Question Answering (Priority: P1) 🎯 MVP Part 2

**Goal**: Enable learners to ask questions constrained to selected text passages only, preventing hallucination.

**Independent Test**: Can be fully tested by selecting a specific paragraph, asking for an explanation, and verifying system answers using ONLY that selected text.

### Tests for User Story 2 (OPTIONAL) ⚠️

> **NOTE**: Tests were NOT requested in feature specification. Skipping test tasks.

### Implementation for User Story 2

- [x] T026 [US2] [P] Update ChatQuery model in backend/src/models/rag_models.py (add use_context_only field)
- [x] T027 [US2] Add context-only logic to retrieval_service.py (filter to user-selected text)
- [x] T028 [US2] [P] Update /api/query endpoint to support use_context_only parameter
- [x] T029 [US2] [P] Add error handling for empty context in selected-text-only mode

**Checkpoint**: At this point, User Story 2 should be fully functional. Learners can get answers constrained to selected text only.

---

## Phase 4: US3 - Chapter-Aware Responses (Priority: P2)

**Goal**: Improve relevance by prioritizing content from current chapter when queries are asked within chapter context.

**Independent Test**: Can be fully tested by opening Chapter 3 and asking a question that could be answered by multiple chapters, verifying that response prioritizes Chapter 3 content.

### Tests for User Story 3 (OPTIONAL) ⚠️

> **NOTE**: Tests were NOT requested in feature specification. Skipping test tasks.

### Implementation for User Story 3

- [x] T030 [US3] [P] Update ChatQuery model in backend/src/models/rag_models.py (add chapter_id field)
- [x] T031 [US3] [P] Update retrieval_service.py to support chapter_id parameter (chapters filter in Qdrant search)
- [x] T032 [US3] [P] Update rag_service.py to pass chapter context to OpenAI prompt
- [x] T033 [US3] [P] Update /api/query endpoint to support chapter_id parameter
- [x] T034 [US3] [P] Add chapter-aware prompts to openai_service.py

**Checkpoint**: At this point, User Story 3 should be fully functional. Queries within a chapter context return chapter-prioritized answers.

---

## Phase 5: US4 - Educational Explanations (Priority: P2)

**Goal**: Provide concise, clear, and structured answers with educational tone to enhance learner comprehension.

**Independent Test**: Can be fully tested by asking questions and verifying responses use clear structure (bullet points, numbered lists), maintain educational tone, and are under 300 words for simple questions.

### Tests for User Story 4 (OPTIONAL) ⚠️

> **NOTE**: Tests were NOT requested in feature specification. Skipping test tasks.

### Implementation for User Story 4

- [x] T035 [US4] [P] Add prompt templates to openai_service.py for educational tone and structure
- [x] T036 [US4] [P] Implement response post-processing in rag_service.py (enforce structure, length limits)
- [x] T037 [US4] [P] Update schemas.py to validate answer length (<300 words for simple queries)

**Checkpoint**: At this point, User Story 4 should be fully functional. All answers maintain educational tone and proper structure.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and system readiness.

- [x] T038 [P] Documentation updates in backend/README.md
- [x] T039 [P] Code cleanup and refactoring for readability
- [x] T040 [P] Performance optimization (embeddings caching, vector search tuning)
- [x] T041 [P] Additional unit tests (if time permits)
- [x] T042 [P] Security hardening (input sanitization, output encoding)
- [x] T043 [P] Run quickstart.md validation tasks

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 1)**: Must complete before ANY user story work can begin
- **User Stories (Phase 2-5)**: All depend on Setup completion - BLOCKS until T001-T010 are done
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Setup (Phase 1) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Setup (Phase 1) - May integrate with US1 (shared models, services)
- **User Story 3 (P2)**: Can start after Setup (Phase 1) - May use models/services from US1/US2
- **User Story 4 (P2)**: Can start after Setup (Phase 1) - Uses RAG pipeline from US1/US2/US3

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services before endpoints
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

Within **Setup (Phase 1)**:
- T003, T004, T005, T006, T007, T008, T009, T010 can run in parallel (different directories/files)

Within **User Story 1**:
- T011, T012, T013 can run in parallel (different model files)
- T015, T016, T017, T018, T019 can run in parallel (different service files)
- T020, T021 can run in parallel (different API files)

Within **User Story 2**:
- T026, T027, T028, T029 can run in parallel

Within **User Story 3**:
- T030, T031, T032, T033, T034 can run in parallel

Within **User Story 4**:
- T035, T036, T037 can run in parallel

**Between User Stories** (with multiple team members):
- Setup (Phase 1) completes first
- Then Developer A: User Story 1, Developer B: User Story 2 (both P1, start in parallel after Setup)
- Then Developer C: User Story 3, Developer D: User Story 4 (both P2, start in parallel after Setup)

### MVP Delivery Strategy

**MVP Definition**: User Stories 1 + 2 (both P1 priority)

1. Complete **Setup (Phase 1)**: T001-T010
2. Complete **User Story 1**: T011-T025
3. Complete **User Story 2**: T026-T029
4. **STOP and VALIDATE**: Test User Stories 1 & 2 independently
5. **Deploy/DEMO if ready**: Core RAG chatbot functionality works for both book-wide and selected-text-only modes

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add US1 (P1) → Test independently → Deploy/Demo (MVP!)
3. Add US2 (P1) → Test independently → Deploy/Demo
4. Add US3 (P2) → Test independently → Deploy/Demo
5. Add US4 (P2) → Test independently → Deploy/Demo
6. Each story adds value without breaking previous stories

---

## Parallel Example: User Story 1

```bash
# Launch model creation together (parallel tasks):
Task: "Create ChatQuery model in backend/src/models/rag_models.py"
Task: "Create RetrievedChunk model in backend/src/models/rag_models.py"
Task: "Create ChatResponse model in backend/src/models/rag_models.py"

# Launch service creation together (parallel tasks):
Task: "Create base config.py in backend/src/config.py"
Task: "Create embedding_service.py in backend/src/services/embedding_service.py"
Task: "Create qdrant_service.py in backend/src/services/qdrant_service.py"
Task: "Create retrieval_service.py in backend/src/services/retrieval_service.py"
Task: "Create openai_service.py in backend/src/services/openai_service.py"
Task: "Create rag_service.py in backend/src/services/rag_service.py"
```

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- MVP: US1 + US2 (both P1 priority) delivers core functionality
- Verify tests fail before implementing (Red-Green-Refactor cycle)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
- Frontend hooks (useRAGQuery.ts) are future work - not part of current MVP
