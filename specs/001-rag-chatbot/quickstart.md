# Quickstart Guide: RAG Chatbot Development

**Feature**: 001-rag-chatbot
**Updated**: 2025-12-30
**Purpose**: Get started quickly with backend RAG chatbot development

## Prerequisites

### Required Tools

- **Python 3.11+** - Backend development language
- **Git** - Version control (already installed)
- **Qdrant Cloud Account** - Vector database
- **Neon Postgres Account** - Metadata and session storage
- **OpenAI API Key** - Agents SDK / ChatKit

### Optional Tools

- **VS Code** or **PyCharm** - Python IDE
- **Postman** or **Insomnia** - API testing

## Environment Setup

### 1. Clone and Navigate

```bash
git clone <your-repo-url>
cd hackathon-04
git checkout 001-rag-chatbot
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables

Create `.env` file in `backend/` directory:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini

# Qdrant Cloud Configuration
QDRANT_URL=https://<your-cloud-instance>.qdrant.io:6333
QDRANT_API_KEY=<your-qdrant-api-key>
QDRANT_COLLECTION_NAME=physical-ai-textbook

# Neon Postgres Configuration
NEON_CONNECTION_STRING=postgresql://<user>:<password>@<host>.neon.tech/neondb?sslmode=require
NEON_DATABASE_NAME=neondb

# Application Settings
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000,https://your-website.vercel.app
RATE_LIMIT_REQUESTS_PER_MINUTE=10
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K_DEFAULT=5
```

### 4. Qdrant Setup (One-Time)

Create Qdrant Cloud collection and configure HNSW index:

```bash
# Using Qdrant Cloud REST API
curl -X PUT https://<your-instance>.qdrant.io:6333/collections/physical-ai-textbook \
  -H "Content-Type: application/json" \
  -H "api-key: <your-qdrant-api-key>" \
  -d '{
    "vectors": {
      "size": 384,
      "distance": "Cosine",
      "index_type": "HNSW",
      "m": 16,
      "ef_construction": 512
    }
  }'
```

## Running Locally

### Start Backend Server

```bash
cd backend

# Ensure environment variables are loaded
source venv/bin/activate  # or venv\Scripts\activate

# Run FastAPI with auto-reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Verify Backend Health

```bash
# Check health endpoint
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "version": "1.0.0"
# }
```

## Testing the API

### Test Query Endpoint

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key components of a humanoid robot?",
    "top_k": 5
  }'
```

### Expected Success Response

```json
{
  "answer": "The key components of a humanoid robot include...",
  "sources": [
    {
      "chunk_id": "ch1_sec2_components_001",
      "chapter_id": 1,
      "section_id": "2",
      "section_title": "Introduction to Humanoid Robotics",
      "preview_text": "Humanoid robots typically consist of...",
      "relevance_score": 0.87
    }
  ],
  "query_time_ms": 1250,
  "mode": "book-wide"
}
```

### Test Selected-Text Mode

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do these actuators work?",
    "context": "Actuators convert electrical energy into mechanical motion. They are the muscles of a humanoid robot.",
    "use_context_only": true,
    "top_k": 3
  }'
```

### Expected "Not Covered" Response

```json
{
  "answer": "This is not covered in the book",
  "sources": [],
  "query_time_ms": 450
}
```

## Troubleshooting

### Backend Won't Start

**Issue**: Address already in use
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill process using that port
kill -9 <PID>

# Try starting again
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Issue**: Qdrant Connection Failed
```bash
# Verify Qdrant URL and API key in .env file
cat backend/.env | grep QDRANT

# Test Qdrant connection
curl https://<your-instance>.qdrant.io:6333/collections

# Check for authentication errors
```

**Issue**: OpenAI API Key Invalid
```bash
# Verify API key format
echo $OPENAI_API_KEY | cut -c1-3

# API keys should start with "sk-proj-"
```

**Issue**: Embedding Model Not Found
```bash
# Install models directory
mkdir -p backend/models

# Download embedding model (manual one-time)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('models/embedding_model/')"
```

## Frontend Integration (Docusaurus)

The backend API is ready for frontend integration. For Docusaurus:

```typescript
// frontend/src/hooks/useRAGQuery.ts (future)
import { useState } from 'react';

export function useRAGQuery() {
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState([]);

  const query = async (question: string, context?: string, useContextOnly?: boolean) => {
    setLoading(true);
    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, context, use_context_only: useContextOnly, top_k: 5 })
      });

      const data = await response.json();
      setAnswer(data.answer);
      setSources(data.sources);
    } catch (error) {
      console.error('Query failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return { loading, answer, sources, query };
}
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                                                     │
│  User Browser                                         │
│                                                     │
│              ┌───────────────────────────────────────────┐    │
│              │  Docusaurus Frontend               │    │
│              │  (Static Site)                    │    │
│              └─────────────┬───────────────────────┘    │
│                          │                     │         │
│                          │   FastAPI Backend      │         │
│                          │   (REST API)            │         │
│                          │         │                │         │
│              ┌──────────────┴──────────────┐    │         │
│              │  RAG Pipeline            │         │
│              │  ├─ Embedding Service      │         │
│              │  ├─ Qdrant Service         │         │
│              │  ├─ Retrieval Service      │         │
│              │  ├─ OpenAI Agents SDK     │         │
│              │  └─ Neon Service           │         │
│              └─────────────────────────────┘         │
│                          │                     │         │
│                          └────────────┴───────────────┘         │
│                                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Next Steps

1. **Textbook Content Preparation** - Organize markdown files with frontmatter (chapter/section metadata)
2. **Content Ingestion** - Implement embedding service to process textbook and store vectors in Qdrant
3. **Query API Implementation** - Build `/api/query` endpoint with RAG pipeline
4. **Testing** - Write integration tests for retrieval and response generation
5. **Frontend Integration** - Create React hooks for Docusaurus to call backend API
6. **Deployment** - Deploy backend to Railway and frontend to Vercel

## Additional Resources

- **Constitution**: `.specify/memory/constitution.md` - RAG chatbot principles and constraints
- **OpenAPI Documentation**: https://platform.openai.com/docs/
- **Qdrant Cloud Docs**: https://qdrant.tech/documentation/
- **FastAPI Docs**: https://fastapi.tiangolo.com/

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review architecture decisions in `plan.md`
3. Check data models in `data-model.md`
