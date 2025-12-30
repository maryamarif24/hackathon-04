---
title: RAG Chatbot Backend
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# RAG Chatbot Backend

FastAPI-based backend for the Physical AI Textbook Chatbot.

## Features

- **Book-wide QA**: Ask questions about Physical AI, humanoid robotics, ROS 2, simulation, and VLA systems
- **Context-aware**: Answer questions based on selected textbook text
- **Chapter prioritization**: Prioritize content from specific chapters
- **Educational formatting**: Responses with proper structure and tone

## API Endpoints

### GET /
Root endpoint with API status.

### GET /health
Health check endpoint.

### POST /api/query
Query the chatbot.

**Request Body:**
```json
{
  "question": "What is Physical AI?",
  "context": "",           // Optional: Selected text from textbook
  "use_context_only": false, // Optional: Only answer from context
  "chapter_id": null       // Optional: Prioritize specific chapter
}
```

**Response:**
```json
{
  "answer": "**Physical AI** represents...",
  "sources": [...],
  "chapter_id": null,
  "query_time_ms": 42.5,
  "educational_metadata": {
    "questionType": "definition",
    "complexity": "simple",
    "estimatedWordCount": "< 300",
    "needsStructure": false
  }
}
```

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python simple_server.py

# API available at http://localhost:8000
```

## Deployment

This Space uses Docker. The application runs on port 7860 by default.
