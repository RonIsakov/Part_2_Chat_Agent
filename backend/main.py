"""
FastAPI main application for Medical Services Chatbot.

RESTful API with two endpoints:
- POST /api/v1/chat - Main chat endpoint (handles both collection and Q&A phases)
- GET /api/v1/health - Health check endpoint

Architecture:
- Stateless backend (all context in request body)
- Rate limiting with semaphore (max 10 concurrent OpenAI calls)
- Lifespan management for vector store initialization
"""

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
from asyncio import Semaphore

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use relative imports that work both locally and in Docker
from models import ChatRequest, ChatResponse, HealthResponse
from services.vector_store import get_vector_store
from services.collection_handler import handle_collection_phase
from services.qa_handler import handle_qa_phase
from config import get_backend_settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiting: Max 10 concurrent OpenAI API calls
# This prevents overwhelming Azure OpenAI and hitting rate limits
openai_semaphore = Semaphore(10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.

    Handles startup and shutdown events:
    - Startup: Initialize vector store (load ChromaDB collection)
    - Shutdown: Cleanup and logging
    """
    # Startup
    logger.info("Starting up Medical Services Chatbot API...")

    try:
        # Initialize vector store for RAG retrieval
        vector_store = get_vector_store()
        logger.info("Vector store initialized successfully")

        # Log settings
        settings = get_backend_settings()
        logger.info(f"Settings loaded: RAG_TOP_K={settings.RAG_TOP_K}")
        logger.info("Startup complete - API ready to serve requests")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Medical Services Chatbot API...")
    logger.info("Shutdown complete")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Medical Services Chatbot API",
    description="RESTful API for Israeli health fund services chatbot with RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware (allow all origins for development)
# In production, replace "*" with specific frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.

    Logs the error and returns a clean JSON response to the client.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc)
        }
    )


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint - handles both collection and Q&A phases.

    Flow:
    1. Check if user_data is complete
    2. If incomplete → Collection phase (gather user info)
    3. If complete → Q&A phase (answer questions using RAG)

    Rate Limiting:
    - Uses semaphore to limit concurrent OpenAI API calls to 10
    - Prevents overwhelming Azure OpenAI service

    Args:
        request: ChatRequest with message, user_data, conversation_history, language

    Returns:
        ChatResponse with response, phase, user_data, sources, metadata

    Raises:
        HTTPException: If processing fails
    """
    try:
        # Use semaphore for rate limiting
        async with openai_semaphore:
            logger.info(f"Chat request: phase check, message length={len(request.message)}")

            # Check if user data is complete AND confirmed
            # Stay in collection phase until user explicitly confirms
            if not request.user_data.is_complete() or not request.user_data.confirmed:
                # Collection phase: gather user information
                logger.info(f"→ Collection phase (complete={request.user_data.is_complete()}, confirmed={request.user_data.confirmed})")
                response = await handle_collection_phase(request)
                logger.info(f"← Collection phase complete: missing_fields={response.missing_fields}")
                return response

            else:
                # Q&A phase: answer questions using RAG
                logger.info(f"→ Q&A phase: hmo={request.user_data.hmo}, tier={request.user_data.tier}")
                response = await handle_qa_phase(request)
                logger.info(
                    f"← Q&A phase complete: chunks={response.metadata.get('chunks_retrieved')}, "
                    f"tokens={response.metadata.get('tokens_used')}"
                )
                return response

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat request: {str(e)}"
        )


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring and orchestration.

    Checks:
    - Vector store connectivity (ChromaDB collection accessible)
    - Azure OpenAI availability (implicitly checked during requests)

    Returns:
        HealthResponse with status, timestamp, and component health

    Raises:
        HTTPException: If health check fails
    """
    try:
        # Check vector store
        vector_store = get_vector_store()
        vector_store_status = "connected" if vector_store.health_check() else "disconnected"

        # Overall status
        overall_status = "healthy" if vector_store_status == "connected" else "degraded"

        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            components={
                "vector_store": vector_store_status,
                "azure_openai": "available"  # Checked implicitly during requests
            }
        )

    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}"
        )


@app.get("/")
async def root():
    """
    Root endpoint - returns API information.
    """
    return {
        "name": "Medical Services Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn

    # Run with: python backend/main.py
    # Or: uvicorn backend.main:app --reload
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
