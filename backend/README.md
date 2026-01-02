# Physical AI Textbook API (Hugging Face Deployment)

This is a FastAPI application for the Physical AI Textbook project, designed for deployment on Hugging Face Spaces using Docker.

## Docker Deployment

The application is containerized with the following configuration:

- **Base Image**: python:3.11-slim
- **Port**: 8000
- **Command**: uvicorn simple_server:app --host 0.0.0.0 --port 8000

## Requirements

The application uses the following key dependencies:
- FastAPI for the web framework
- Uvicorn as the ASGI server
- Sentence-transformers for embeddings
- Qdrant-client for vector database operations
- Various other packages for AI and web functionality

## Environment Variables

If needed, you can configure the application using environment variables. Create a `.env` file in the root directory with the following variables:

```
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Database Configuration (if using Qdrant)
QDRANT_URL=
QDRANT_API_KEY=

# Database Configuration (if using PostgreSQL)
DATABASE_URL=

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Deployment on Hugging Face Spaces

To deploy this application on Hugging Face Spaces with Docker:

1. Fork this repository to your Hugging Face account
2. Create a new Space with Docker as the SDK
3. The Dockerfile will automatically build and run the application
4. The application will be accessible on port 8000

## API Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint
- `POST /api/query` - Query endpoint for the chatbot

## Notes

- The application runs in mock mode by default, providing sample responses for testing
- For full functionality with Qdrant and Neon database, additional configuration is required
- The Docker image includes all necessary dependencies for the application to run