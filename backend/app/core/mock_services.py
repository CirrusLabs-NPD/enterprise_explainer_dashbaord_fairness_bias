"""
Mock services for development
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import structlog

logger = structlog.get_logger(__name__)

class MockWorkerPool:
    """Mock worker pool"""
    
    def __init__(self, max_workers=4, max_cpu_workers=2, max_io_workers=4):
        self.max_workers = max_workers
        self.max_cpu_workers = max_cpu_workers
        self.max_io_workers = max_io_workers
        self.running = False
    
    async def start(self):
        self.running = True
        logger.info("Mock worker pool started")
    
    async def stop(self):
        self.running = False
        logger.info("Mock worker pool stopped")
    
    async def get_stats(self):
        return {
            "status": "running" if self.running else "stopped",
            "active_workers": 2,
            "max_workers": self.max_workers
        }

class MockWebSocketManager:
    """Mock WebSocket manager"""
    
    def __init__(self):
        self.running = False
        self.connections = []
    
    async def start(self):
        self.running = True
        logger.info("Mock WebSocket manager started")
    
    async def stop(self):
        self.running = False
        logger.info("Mock WebSocket manager stopped")

class MockModelService:
    """Mock model service"""
    
    def __init__(self, worker_pool):
        self.worker_pool = worker_pool
    
    async def get_models(self):
        return [
            {
                "id": "model_1",
                "name": "Credit Risk Model",
                "type": "classification",
                "status": "active",
                "accuracy": 0.87
            },
            {
                "id": "model_2", 
                "name": "Fraud Detection",
                "type": "classification",
                "status": "active",
                "accuracy": 0.92
            }
        ]

class MockExplanationService:
    """Mock explanation service"""
    
    def __init__(self, worker_pool, model_service):
        self.worker_pool = worker_pool
        self.model_service = model_service

class MockMonitoringService:
    """Mock monitoring service"""
    
    def __init__(self, websocket_manager):
        self.websocket_manager = websocket_manager
    
    async def start_drift_monitoring(self):
        logger.info("Mock drift monitoring started")
    
    async def stop(self):
        logger.info("Mock monitoring service stopped")

class MockDataService:
    """Mock data service"""
    
    def __init__(self, worker_pool):
        self.worker_pool = worker_pool

# Global instances
mock_worker_pool = MockWorkerPool()
mock_websocket_manager = MockWebSocketManager()
mock_model_service = MockModelService(mock_worker_pool)
mock_explanation_service = MockExplanationService(mock_worker_pool, mock_model_service)
mock_monitoring_service = MockMonitoringService(mock_websocket_manager)
mock_data_service = MockDataService(mock_worker_pool)

# Mock dependencies functions
async def get_worker_pool():
    if not mock_worker_pool.running:
        await mock_worker_pool.start()
    return mock_worker_pool

async def get_websocket_manager():
    return mock_websocket_manager

async def get_model_service():
    return mock_model_service

async def get_explanation_service():
    return mock_explanation_service

async def get_monitoring_service():
    return mock_monitoring_service

async def get_data_service():
    return mock_data_service

async def cleanup_dependencies():
    """Cleanup all service instances"""
    await mock_monitoring_service.stop()
    await mock_worker_pool.stop()
    await mock_websocket_manager.stop()

async def health_check():
    """Health check for all services"""
    return {
        "worker_pool": mock_worker_pool.running,
        "websocket_manager": mock_websocket_manager.running,
        "services": {
            "model_service": True,
            "explanation_service": True,
            "monitoring_service": True,
            "data_service": True
        }
    }