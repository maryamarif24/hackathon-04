"""
Health check API endpoint.

Provides service health status and version information.
"""
import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns service status and version information.

    Returns:
        Dict with status and version

    Example:
        Response:
        ```json
        {
            "status": "healthy",
            "version": "2.0.0"
        }
        ```
    """
    return {"status": "healthy", "version": "2.0.0"}
