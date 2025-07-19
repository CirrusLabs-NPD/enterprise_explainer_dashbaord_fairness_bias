"""
ML Explainer Dashboard Backend
Enterprise-grade ML model explainability and monitoring system
"""

import asyncio
import logging
import multiprocessing as mp
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app
import structlog

from app.config import settings
from app.core.mock_database import init_db, close_db
from app.core.mock_services import mock_websocket_manager as websocket_manager
from app.core.mock_services import cleanup_dependencies
from app.api.v1.mock_router import api_router
from app.core.exceptions import (
    mlexplainer_exception_handler, http_exception_handler,
    validation_exception_handler, general_exception_handler,
    MLExplainerException
)
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management"""
    logger.info("Starting ML Explainer Dashboard Backend")
    
    try:
        # Initialize database
        await init_db()
        logger.info("Database initialized")
        
        # Start WebSocket manager
        await websocket_manager.start()
        logger.info("WebSocket manager started")
        
        logger.info("All services initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down services")
        
        # Cleanup all dependencies
        await cleanup_dependencies()
        
        # Close database
        await close_db()
        
        logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="ML Explainer Dashboard API",
    description="Enterprise-grade ML model explainability and monitoring system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Add exception handlers
app.add_exception_handler(MLExplainerException, mlexplainer_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Serve static files (disabled for development)
# app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ML Explainer Dashboard API",
        "version": "1.0.0",
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connectivity
        from app.core.mock_database import check_database_health
        db_health = await check_database_health()
        
        # Check services health
        from app.core.mock_services import health_check as deps_health
        services_health = await deps_health()
        
        return {
            "status": "healthy",
            "database": db_health.get("status", "unknown"),
            "services": services_health,
            "websocket": "connected" if websocket_manager.running else "disconnected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


# WebSocket endpoints are now handled in the API router


if __name__ == "__main__":
    # Set multiprocessing start method for better compatibility
    mp.set_start_method("spawn", force=True)
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1,  # Use 1 worker for development, scale in production
        log_level="info" if not settings.DEBUG else "debug",
        access_log=True
    )