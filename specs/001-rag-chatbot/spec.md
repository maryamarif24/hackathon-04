# Feature Specification: RAG Chatbot

**Feature Branch**: `001-rag-chatbot`
**Created**: 2025-12-30
**Status**: Draft
**Input**: User description: "Specify a Retrieval-Augmented Generation (RAG) chatbot for 'Physical AI & Humanoid Robotics' online textbook."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Book-wide Question Answering (Priority: P1)

A learner reading the "Physical AI & Humanoid Robotics" textbook has questions about concepts across multiple chapters and wants accurate, textbook-grounded answers with chapter and section references.

**Why this priority**: This is the primary use case - learners need to ask general questions across the entire textbook to understand complex topics that may span multiple chapters.

**Independent Test**: This can be fully tested by asking questions like "What are the key components of a humanoid robot?" and verifying that the response is accurate, cites correct sources, and provides chapter/section context.

**Acceptance Scenarios**:

1. **Given** the user is on any textbook page, **When** the user asks a general question about Physical AI topics, **Then** the system retrieves relevant content from across the textbook and provides an accurate, cited answer
2. **Given** the textbook content exists in the vector database, **When** the user asks a question about a specific topic covered in the book, **Then** the system returns an answer with relevant excerpts and source citations
3. **Given** the user asks a question about a topic not covered in the textbook, **Then** the system explicitly states "This is not covered in the book" without attempting to answer from external knowledge

---

### User Story 2 - Selected-Text Question Answering (Priority: P1)

A learner has selected specific text from the textbook (e.g., a paragraph about actuators) and wants an explanation or clarification limited strictly to that selected passage.

**Why this priority**: This is critical for focused learning - learners often want explanations of specific passages rather than general knowledge, and constraining to selected text prevents hallucination.

**Independent Test**: This can be fully tested by selecting a specific paragraph and asking for an explanation, verifying that the response only references the selected text.

**Acceptance Scenarios**:

1. **Given** the user has selected a specific text passage, **When** the user asks a question with "use selected text only" mode enabled, **Then** the system answers using ONLY the provided text and cites that text as the sole source
2. **Given** the user provides selected text and a question, **When** the selected text does not contain sufficient information to answer, **Then** the system states "The selected text does not contain enough information to answer this question"
3. **Given** the user selects text from Chapter 2 about sensors, **When** the user asks for clarification with context-only mode, **Then** the response is grounded only in that Chapter 2 sensor content

---

### User Story 3 - Chapter-Aware Responses (Priority: P2)

A learner is reading a specific chapter (e.g., ROS 2 Fundamentals) and asks questions that should be answered with awareness of that chapter's context and terminology.

**Why this priority**: This improves relevance - questions asked within a chapter context should preferentially use content from that chapter and use chapter-specific terminology.

**Independent Test**: This can be fully tested by opening Chapter 3 and asking a question that could be answered by multiple chapters, verifying that the response prioritizes Chapter 3 content.

**Acceptance Scenarios**:

1. **Given** the user is viewing Chapter 4 "Digital Twin Simulation", **When** the user asks a question about simulation tools, **Then** the system retrieves and prioritizes Chapter 4 content over other chapters
2. **Given** the user has chapter context explicitly provided, **When** generating a response, **Then** the system uses chapter-aware terminology and references chapter-specific examples
3. **Given** the user is in a chapter without relevant content for their question, **When** the answer is generated, **Then** the system still provides an accurate answer from other chapters if available, but acknowledges it's outside the current chapter scope

---

### User Story 4 - Educational Explanations (Priority: P2)

A learner receives answers from the chatbot that are concise, clear, and structured to facilitate understanding rather than overwhelming them with lengthy responses.

**Why this priority**: Educational effectiveness - concise, structured answers help learners grasp concepts quickly without cognitive overload.

**Independent Test**: This can be fully tested by asking questions and verifying responses use clear structure (e.g., bullet points for multiple items, step-by-step for processes), maintain an educational tone, and are under 300 words for simple questions.

**Acceptance Scenarios**:

1. **Given** the user asks "How do sensors work in humanoid robots?", **When** the system generates an answer, **Then** the response is structured with clear sections (e.g., types, function, examples) and uses simple language
2. **Given** the user asks a multi-part question, **When** the system responds, **Then** the answer uses structured formatting (numbered lists or bullet points) to separate distinct parts
3. **Given** a technical concept is complex, **When** an explanation is provided, **Then** the system uses analogies or examples from the textbook to clarify without adding external knowledge

---

### Edge Cases

- What happens when the vector database is empty (no textbook content indexed)?
  - The system should respond "Textbook content is not available. Please contact the administrator." without attempting to generate answers from pre-training knowledge.
- How does the system handle when no relevant content is found in the vector search?
  - The system should respond "I couldn't find relevant information in the textbook. Could you rephrase your question?" instead of hallucinating or providing external knowledge.
- What happens when the user provides both selected text and asks a general question?
  - The system should default to general book-wide search unless the user explicitly enables "use selected text only" mode via a UI toggle or parameter.
- How does the system handle ambiguous questions that match multiple textbook sections?
  - The system should retrieve multiple relevant sections and synthesize a response that references all sources, or ask a clarifying question "Are you asking about [topic A] or [topic B]?" if ambiguity is high.
- What happens when the embedding model fails or the vector database is unreachable?
  - The system should return a user-friendly error message "I'm having trouble accessing the textbook database. Please try again later." with an appropriate HTTP status code (503).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST retrieve relevant textbook content before generating any answer
- **FR-002**: System MUST answer strictly from retrieved book content only, never from external web search or pre-training knowledge
- **FR-003**: System MUST explicitly state "This is not covered in the book" when the question topic is absent from textbook content
- **FR-004**: System MUST support question answering across the entire textbook content (book-wide mode)
- **FR-005**: System MUST support question answering constrained to user-selected text only (selected-text-only mode)
- **FR-006**: System MUST incorporate chapter context when available (chapter-aware mode)
- **FR-007**: System MUST provide responses in a clear, educational tone appropriate for learners
- **FR-008**: System MUST structure answers with clear formatting (bullet points, numbered lists, sections) for readability
- **FR-009**: System MUST cite source references internally (chapter ID, section ID) for each retrieved chunk
- **FR-010**: System MUST NOT hallucinate or add external knowledge beyond the textbook content
- **FR-011**: System MUST respect user intent to limit answers to selected text when explicitly indicated
- **FR-012**: System MUST provide concise explanations (target under 300 words for simple questions)
- **FR-013**: System MUST handle edge cases gracefully (empty database, no matches, service unavailability)
- **FR-014**: System MUST use read-only access to textbook content (no modifications allowed)
- **FR-015**: System MUST prevent generation of harmful or unsafe robotics instructions

### Key Entities

- **ChatQuery**: Represents a user question with context constraints
  - `question`: User's natural language question (required)
  - `context`: Selected text passage (optional)
  - `use_context_only`: Boolean flag for selected-text-only mode (default: false)
  - `top_k`: Number of relevant chunks to retrieve (default: 5)

- **RetrievedChunk**: A single piece of textbook content from vector search
  - `chunk_id`: Unique identifier for the content chunk
  - `chapter_id`: Chapter number the chunk belongs to
  - `section_id`: Section identifier within the chapter
  - `section_title`: Human-readable title of the section
  - `preview_text`: First 200 characters of the chunk for display
  - `relevance_score`: Semantic similarity score from vector search (0-1)
  - `full_text`: Complete content of the chunk (not exposed to user, used internally)

- **ChatResponse**: Answer returned to the user
  - `answer`: Generated explanation text
  - `sources`: Array of RetrievedChunk references (chapter, section, preview)
  - `query_time_ms`: Time taken to process the query (for monitoring)
  - `mode`: Answering mode used (book-wide | selected-text-only | chapter-aware)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 95% of responses are grounded in textbook content with correct citations
- **SC-002**: 0% of responses contain external knowledge or information not present in the textbook
- **SC-003**: 90% of users report that answers are helpful and relevant to their questions
- **SC-004**: Response generation completes within 5 seconds for typical queries (p95)
- **SC-005**: Vector search retrieves relevant content (relevance score > 0.7) for 90% of textbook-covered questions
- **SC-006**: All selected-text-only queries answer using ONLY the provided text (100% compliance)
- **SC-007**: 95% of responses use clear structure (bullet points, numbered lists, or section headers)

## Assumptions

- Textbook content is available as markdown files with frontmatter containing chapter and section metadata
- Vector database (Qdrant Cloud) is pre-populated with embeddings of textbook content
- OpenAI Agents SDK / ChatKit API is accessible and has appropriate rate limits
- Frontend integration provides context (current chapter, selected text) to backend via API
- Neon Serverless Postgres stores metadata and minimal session history (no PII required)
- No training or fine-tuning of AI models is required (pre-trained models sufficient)
- Read-only access means the system can query content but cannot modify the textbook markdown files

## Constraints

- **No Hallucinations**: Strict grounding requirement - answer only what is in the retrieved content
- **No External Knowledge**: Cannot supplement answers with information from web search or general knowledge
- **No Model Training**: Use pre-trained models only, no fine-tuning on this textbook
- **Read-Only Access**: System can read textbook content but cannot write or modify it
- **Educational Scope**: Only assist with learning textbook content, not provide instructions outside book scope
- **Safety**: Must not generate harmful, unsafe, or dangerous robotics instructions
- **Performance**: Response time target is under 5 seconds for typical queries (p95)
- **Conciseness**: Responses should be concise (under 300 words for simple questions, structured for complex ones)
