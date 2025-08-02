"""
Main application entry point for HackRx6 RAG API.

This module starts the FastAPI server with proper configuration,
middleware, and error handling for the intelligent query-retrieval system.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the API endpoints
from api.endpoints import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_api.log')
    ]
)
logger = logging.getLogger(__name__)

# Global variables for application state
app_start_time = None
request_count = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global app_start_time
    
    # Startup
    app_start_time = datetime.utcnow()
    logger.info("🚀 Starting HackRx6 RAG API...")
    logger.info(f"📅 Start time: {app_start_time}")
    
    # Log configuration
    logger.info("📋 Configuration:")
    logger.info(f"   - API Token: {'*' * 20}...")
    logger.info(f"   - Max Questions per Request: 10")
    logger.info(f"   - Max Document Size: 100MB")
    logger.info(f"   - Default Top-K: 5")
    logger.info(f"   - Default Similarity Threshold: 0.7")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down HackRx6 RAG API...")
    logger.info(f"📊 Total requests processed: {request_count}")


# Create the main FastAPI application
main_app = FastAPI(
    title="HackRx6 RAG API",
    description="""
    Intelligent Query-Retrieval System for Insurance, Legal, HR, and Compliance
    
    ## Features
    - Process large documents (PDF, DOCX, TXT, Emails)
    - Answer questions with high accuracy (>95%)
    - Provide structured JSON responses with source citations
    - Support batch question processing
    - Token-efficient context handling
    
    ## Endpoints
    - `POST /api/v1/hackrx/run` - Process documents and answer questions
    - `GET /api/v1/health` - Health check
    - `GET /api/v1/info` - API information
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
main_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@main_app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Middleware for request logging and monitoring."""
    global request_count
    
    start_time = datetime.utcnow()
    request_count += 1
    
    # Log request
    logger.info(f"📥 Request #{request_count}: {request.method} {request.url}")
    
    try:
        response = await call_next(request)
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Log response
        logger.info(f"📤 Response #{request_count}: {response.status_code} ({processing_time:.3f}s)")
        
        return response
        
    except Exception as e:
        # Log error
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        logger.error(f"❌ Error #{request_count}: {str(e)} ({processing_time:.3f}s)")
        raise


@main_app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@main_app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Include the API routes
main_app.include_router(app, prefix="")


@main_app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "HackRx6 RAG API",
        "description": "Intelligent Query-Retrieval System",
        "version": "1.0.0",
        "status": "running",
        "uptime": str(datetime.utcnow() - app_start_time) if app_start_time else "unknown",
        "requests_processed": request_count,
        "endpoints": {
            "POST /api/v1/hackrx/run": "Process documents and answer questions",
            "GET /api/v1/health": "Health check",
            "GET /api/v1/info": "API information",
            "GET /docs": "Interactive API documentation",
            "GET /redoc": "ReDoc API documentation"
        }
    }


@main_app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "uptime": str(datetime.utcnow() - app_start_time) if app_start_time else "unknown",
        "requests_processed": request_count
    }


def run_server():
    """Run the FastAPI server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="HackRx6 RAG API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    logger.info(f"🌐 Starting server on {args.host}:{args.port}")
    logger.info(f"🔄 Auto-reload: {'enabled' if args.reload else 'disabled'}")
    logger.info(f"👥 Workers: {args.workers}")
    
    uvicorn.run(
        "main:main_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info"
    )


if __name__ == "__main__":
    run_server() 