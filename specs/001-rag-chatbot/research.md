# Phase 0 Research: RAG Chatbot

**Purpose**: Resolve unknowns and research best practices for RAG chatbot implementation

## Research Decisions

### Chunking Strategy for Markdown Textbook Content

**Decision**: Use semantic chunking (500-1000 tokens per chunk) with overlap to preserve context

**Rationale**: Semantic chunks preserve meaning boundaries better than fixed-size chunks. Overlap (e.g., 100 tokens) ensures no content loss at boundaries. This approach maintains readability of retrieved passages while providing sufficient context for answering questions.

**Alternatives Considered**:

| Alternative | Rejected | Reason |
|------------|----------|---------|
| Fixed-size chunks | Rejected | Breaks mid-sentence, loses context meaning |
| Paragraph-based chunks | Rejected | Too variable in length, may be too small or too large |
| Recursive character splitting | Rejected | Loses semantic meaning, harder to retrieve relevant content |

### Embedding Model Selection

**Decision**: Use sentence-transformers/all-MiniLM-L6-v2 (384-dim, 80MB model)

**Rationale**: Specified in user requirements. Good balance of performance vs accuracy for educational content. Works on CPU without GPU (important for serverless deployment).

**Alternatives Considered**:

| Alternative | Rejected | Reason |
|------------|----------|---------|
| all-mpnet-base-v2 | Rejected | Too large for production (420MB), slower inference |
| BGE-small | Rejected | Not specified in user requirements |

### Vector Database Configuration

**Decision**: Use Qdrant Cloud with HNSW index (384-dim vectors, M=16, ef_construction=512)

**Rationale**: HNSW provides fast approximate search suitable for semantic similarity queries. M=16 is standard for 384-dim vectors. ef_construction=512 provides good balance between index size and search accuracy.

**Alternatives Considered**:

| Alternative | Rejected | Reason |
|------------|----------|---------|
| Qdrant Cloud with exact search | Rejected | Not suitable for semantic similarity queries |
| Pinecone | Rejected | Not specified in user requirements |

### OpenAI Agents SDK / ChatKit Integration

**Decision**: Use OpenAI Agents SDK with function calling for structured response generation

**Rationale**: Agents SDK provides tool use capabilities for controlled, grounded response generation. Enables explicit citation of retrieved chunks in prompts. Function calling ensures system follows defined tools rather than generating free-form text.

**Alternatives Considered**:

| Alternative | Rejected | Reason |
|------------|----------|---------|
| Direct OpenAI API completion | Rejected | Less control over grounding, harder to enforce citations |
| LangChain agents | Rejected | Additional abstraction layer not needed |

### Rate Limiting Strategy

**Decision**: Use slowapi with in-memory rate limiting (10 requests/minute per IP)

**Rationale**: Simple, lightweight rate limiting sufficient for educational use case. Prevents abuse without over-engineering. In-memory approach avoids external service dependencies.

**Alternatives Considered**:

| Alternative | Rejected | Reason |
|------------|----------|---------|
| Redis-based distributed limiting | Rejected | Over-engineering for single-instance deployment |
| External API gateway | Rejected | Additional cost and complexity |

## Best Practices Researched

### RAG Pipeline Architecture
- Standard pattern: Retrieve → Rerank → Context Assemble → Generate Response
- Context window management: Keep top-k chunks within model context limit
- Citation metadata: Store chapter/section ID with each chunk for accurate references

### Semantic Search Optimization
- Use HNSW (Hierarchical Navigable Small World) index for approximate search
- Cosine similarity for embedding vectors
- Relevance filtering: Set threshold (e.g., >0.7) to exclude low-quality results

### Prompt Engineering for Grounding
- Provide retrieved chunks as context in system message
- Instruct model to answer ONLY from provided context
- Include citation instructions: "Use chapters X and Y to answer"
- Include refusal instruction: "If information not in context, say 'This is not covered in book'"

### Error Handling Patterns
- Vector database errors: Return 503 with user-friendly message
- OpenAI rate limits: Return 429, implement exponential backoff
- Empty database state: Explicit error rather than hallucination
- Embedding failures: Fallback to cached embeddings if available

### Frontend Integration Patterns
- Debounce API calls on text selection
- Show loading states during retrieval
- Display source citations (chapter/section) for transparency
- Handle selected-text vs book-wide mode toggling

## Dependencies & Risks

| Dependency | Risk | Mitigation |
|------------|-------|-------------|
| OpenAI Agents SDK | New API, potential instability | Implement fallback, monitor API status |
| Qdrant Cloud | Service downtime | Local error messages, retry logic |
| Neon Serverless | Connection limits | Simple retry with exponential backoff |
| sentence-transformers | Model loading time | Pre-load model on application startup |
