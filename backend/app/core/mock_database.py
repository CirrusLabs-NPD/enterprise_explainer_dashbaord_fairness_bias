"""
Mock database service for development without PostgreSQL
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import structlog

logger = structlog.get_logger(__name__)

class MockDatabase:
    """Mock database for development"""
    
    def __init__(self):
        self.data = {
            "models": {},
            "datasets": {},
            "explanations": {},
            "alerts": {},
            "drift_reports": {},
            "performance_metrics": {}
        }
        self.initialized = False
        
        # Pre-populate with sample data
        self._populate_sample_data()
    
    def _populate_sample_data(self):
        """Add sample data for development"""
        # Sample models
        for i in range(5):
            model_id = f"model_{i+1}"
            self.data["models"][model_id] = {
                "model_id": model_id,
                "name": f"ML Model {i+1}",
                "model_type": "classification" if i % 2 == 0 else "regression",
                "framework": "sklearn",
                "version": f"v1.{i}",
                "status": "active",
                "created_at": (datetime.now() - timedelta(days=i*10)).isoformat(),
                "accuracy": round(0.85 + (i * 0.02), 3),
                "feature_names": [f"feature_{j}" for j in range(10)],
                "training_metrics": {
                    "accuracy": round(0.85 + (i * 0.02), 3),
                    "precision": round(0.82 + (i * 0.02), 3),
                    "recall": round(0.80 + (i * 0.02), 3),
                    "f1_score": round(0.81 + (i * 0.02), 3)
                }
            }
        
        # Sample alerts
        for i in range(3):
            alert_id = str(uuid.uuid4())
            self.data["alerts"][alert_id] = {
                "alert_id": alert_id,
                "model_id": f"model_{i+1}",
                "alert_type": "drift_detected",
                "severity": "high" if i == 0 else "medium",
                "title": f"Data Drift Detected - Model {i+1}",
                "message": f"Significant drift detected in feature distribution",
                "status": "active",
                "created_at": (datetime.now() - timedelta(hours=i)).isoformat()
            }
    
    async def initialize(self):
        """Initialize mock database"""
        if self.initialized:
            return
        self.initialized = True
        logger.info("Mock database initialized")
    
    async def close(self):
        """Close mock database"""
        self.initialized = False
        logger.info("Mock database closed")
    
    async def get_models(self) -> List[Dict]:
        """Get all models"""
        return list(self.data["models"].values())
    
    async def get_model(self, model_id: str) -> Optional[Dict]:
        """Get specific model"""
        return self.data["models"].get(model_id)
    
    async def get_alerts(self, model_id: Optional[str] = None) -> List[Dict]:
        """Get alerts"""
        alerts = list(self.data["alerts"].values())
        if model_id:
            alerts = [a for a in alerts if a["model_id"] == model_id]
        return alerts
    
    async def create_model(self, model_data: Dict) -> str:
        """Create new model"""
        model_id = model_data.get("model_id", str(uuid.uuid4()))
        model_data["created_at"] = datetime.now().isoformat()
        self.data["models"][model_id] = model_data
        return model_id
    
    async def update_model(self, model_id: str, updates: Dict) -> bool:
        """Update model"""
        if model_id in self.data["models"]:
            self.data["models"][model_id].update(updates)
            self.data["models"][model_id]["updated_at"] = datetime.now().isoformat()
            return True
        return False
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete model"""
        if model_id in self.data["models"]:
            del self.data["models"][model_id]
            return True
        return False

# Global mock database instance
mock_database = MockDatabase()

# Mock database utility functions
async def init_db():
    """Initialize database"""
    await mock_database.initialize()

async def close_db():
    """Close database"""
    await mock_database.close()

async def check_database_health() -> dict:
    """Check database health"""
    return {
        "status": "healthy",
        "type": "mock",
        "models_count": len(mock_database.data["models"]),
        "alerts_count": len(mock_database.data["alerts"])
    }

# Mock connection context manager
class MockConnection:
    """Mock database connection"""
    
    def __init__(self):
        pass
    
    async def execute(self, query: str, *args):
        """Mock execute"""
        pass
    
    async def fetch(self, query: str, *args):
        """Mock fetch"""
        return []
    
    async def fetchrow(self, query: str, *args):
        """Mock fetchrow"""
        return None
    
    async def fetchval(self, query: str, *args):
        """Mock fetchval"""
        return None

async def get_db():
    """Get mock database connection"""
    class MockContextManager:
        async def __aenter__(self):
            return MockConnection()
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
    
    return MockContextManager()