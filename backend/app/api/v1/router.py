"""
Main API router configuration
"""

from fastapi import APIRouter
from app.api.v1.endpoints import models, data, monitoring, websocket, data_drift, fairness, model_monitoring

# Create API router
api_router = APIRouter()

# Include endpoint routers
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(data.router, prefix="/data", tags=["data"])
api_router.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
api_router.include_router(websocket.router, prefix="/ws", tags=["websocket"])
api_router.include_router(data_drift.router, tags=["data-drift"])
api_router.include_router(fairness.router, tags=["fairness"])
api_router.include_router(model_monitoring.router, tags=["model-monitoring"])