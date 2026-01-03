"""
FastAPI application entry point for Physical AI textbook RAG system.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import settings
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 001-rag-chatbot
from src.api.health import router as health_router
from src.api.query import router as query_router
from app.api.book_content import router as book_content_router
from src.services import embedding_service
<<<<<<< HEAD
=======
from app.api.health import router as health_router
from app.api.query import router as query_router
from app.services.embedding_service import embedding_service
>>>>>>> master
=======
>>>>>>> 001-rag-chatbot

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown lifecycle manager.

    Loads embedding model on startup.
    """
    # Startup
<<<<<<< HEAD
<<<<<<< HEAD
    logger.info("🚀 Starting Physical AI Textbook RAG API...")
    logger.info(f"Environment: {settings.log_level}")
    logger.info(f"CORS Origins: {settings.cors_origins_list}")
=======
    logger.info("🚀 Starting Physical AI Textbook API...")
    logger.info(f"Environment: {settings.log_level}")
    logger.info(f"CORS Origins: {settings.cors_origins}")
>>>>>>> master
=======
    logger.info("🚀 Starting Physical AI Textbook RAG API...")
    logger.info(f"Environment: {settings.log_level}")
    logger.info(f"CORS Origins: {settings.cors_origins_list}")
>>>>>>> 001-rag-chatbot

    # Load embedding model
    logger.info("Loading embedding model...")
    success = embedding_service.load_model()
    if success:
        logger.info("✅ Embedding model loaded successfully")
    else:
        logger.error("❌ Failed to load embedding model")

    yield

    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI application
app = FastAPI(
    title="Physical AI & Humanoid Robotics Textbook API",
<<<<<<< HEAD
<<<<<<< HEAD
    description="API for querying textbook content using RAG",
    version="2.0.0",
=======
    description="RAG-powered chatbot API for querying textbook content",
=======
    description="API for querying textbook content using RAG",
>>>>>>> 001-rag-chatbot
    version="1.0.0",
>>>>>>> master
    lifespan=lifespan
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
<<<<<<< HEAD
<<<<<<< HEAD
    allow_origins=["*"],
=======
    allow_origins=settings.cors_origins,
>>>>>>> master
=======
    allow_origins=["*"],
>>>>>>> 001-rag-chatbot
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health_router, prefix="", tags=["Health"])
app.include_router(query_router, prefix="", tags=["Query"])
<<<<<<< HEAD
<<<<<<< HEAD
app.include_router(book_content_router, prefix="", tags=["Book Content"])
=======
>>>>>>> master
=======
app.include_router(book_content_router, prefix="", tags=["Book Content"])
>>>>>>> 001-rag-chatbot

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Physical AI Textbook API",
<<<<<<< HEAD
        "version": "2.0.0",
=======
        "version": "1.0.0",
>>>>>>> master
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
        log_level=settings.log_level.lower()
    )
