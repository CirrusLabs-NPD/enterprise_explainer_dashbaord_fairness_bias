"""
Mock API router for development
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Create API router
api_router = APIRouter()

# Response models
class ModelResponse(BaseModel):
    id: str
    name: str
    type: str
    status: str
    accuracy: float
    created_at: str

class PerformanceMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float

class DriftResponse(BaseModel):
    drift_detected: bool
    drift_score: float
    threshold: float
    last_check: str

class BiasMetrics(BaseModel):
    overall_fairness_score: float
    demographic_parity: float
    equalized_odds: float
    protected_groups: List[str]

# Mock data generators
def generate_mock_models():
    """Generate mock model data"""
    models = []
    for i in range(5):
        models.append({
            "id": f"model_{i+1}",
            "name": f"ML Model {i+1}",
            "type": "classification" if i % 2 == 0 else "regression",
            "status": "active",
            "accuracy": round(0.85 + (i * 0.02), 3),
            "created_at": (datetime.now() - timedelta(days=i*10)).isoformat()
        })
    return models

# Models endpoints
@api_router.get("/models", response_model=List[ModelResponse])
async def get_models():
    """Get all models"""
    return generate_mock_models()

@api_router.get("/models/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str):
    """Get specific model"""
    models = generate_mock_models()
    model = next((m for m in models if m["id"] == model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@api_router.get("/models/{model_id}/performance", response_model=PerformanceMetrics)
async def get_model_performance(model_id: str):
    """Get model performance metrics"""
    return {
        "accuracy": round(random.uniform(0.80, 0.95), 3),
        "precision": round(random.uniform(0.75, 0.92), 3),
        "recall": round(random.uniform(0.78, 0.94), 3),
        "f1_score": round(random.uniform(0.76, 0.93), 3),
        "auc_roc": round(random.uniform(0.82, 0.96), 3)
    }

# Monitoring endpoints
@api_router.get("/monitoring/drift", response_model=DriftResponse)
async def get_drift_data():
    """Get data drift information"""
    return {
        "drift_detected": random.choice([True, False]),
        "drift_score": round(random.uniform(0.0, 0.3), 3),
        "threshold": 0.1,
        "last_check": datetime.now().isoformat()
    }

# Fairness endpoints  
@api_router.get("/fairness/bias", response_model=BiasMetrics)
async def get_bias_metrics():
    """Get fairness and bias metrics"""
    return {
        "overall_fairness_score": round(random.uniform(0.75, 0.95), 3),
        "demographic_parity": round(random.uniform(0.05, 0.15), 3),
        "equalized_odds": round(random.uniform(0.03, 0.12), 3),
        "protected_groups": ["gender", "race", "age"]
    }

# Explanations endpoints
@api_router.get("/explanations/{model_id}/global")
async def get_global_explanations(model_id: str):
    """Get global model explanations"""
    return {
        "feature_importance": [
            {"feature": f"feature_{i}", "importance": round(random.uniform(0.05, 0.25), 3)} 
            for i in range(10)
        ]
    }

@api_router.post("/explanations/{model_id}/local")
async def get_local_explanations(model_id: str, data: Dict[str, Any]):
    """Get local explanations for specific prediction"""
    return {
        "prediction": round(random.uniform(0, 1), 3),
        "shap_values": [round(random.uniform(-0.1, 0.1), 4) for _ in range(10)],
        "lime_explanation": [
            {"feature": f"feature_{i}", "value": round(random.uniform(-0.2, 0.2), 4)}
            for i in range(5)
        ]
    }

# Data endpoints
@api_router.get("/data/datasets")
async def get_datasets():
    """Get available datasets"""
    return {
        "datasets": [
            {
                "id": "dataset_1",
                "name": "Training Dataset",
                "size": 10000,
                "features": 20
            },
            {
                "id": "dataset_2", 
                "name": "Validation Dataset",
                "size": 2000,
                "features": 20
            }
        ]
    }

# Analysis endpoints
@api_router.post("/analysis/{model_id}/what-if")
async def analyze_what_if_scenarios(model_id: str, data: Dict[str, Any]):
    """What-if analysis for scenarios"""
    return {
        "scenarios": [
            {
                "scenario_id": 0,
                "original_prediction": 0.75,
                "modified_prediction": 0.82,
                "prediction_change": 0.07,
                "feature_changes": data.get("modifications", {}),
                "confidence_change": 0.05
            }
        ]
    }

@api_router.post("/analysis/{model_id}/feature-dependence")
async def analyze_feature_dependence(model_id: str, data: Dict[str, Any]):
    """Feature dependence analysis (PDP/ICE)"""
    feature_names = data.get("feature_names", ["feature_1", "feature_2"])
    method = data.get("method", "pdp")
    
    if method == "pdp":
        return {
            "method": "pdp",
            "results": {
                feature: {
                    "values": [i/10 for i in range(11)],
                    "partial_dependence": [random.uniform(-1, 1) for _ in range(11)]
                }
                for feature in feature_names
            }
        }
    else:
        return {
            "method": "ice",
            "results": {
                feature: {
                    "feature_range": [i/10 for i in range(11)],
                    "ice_curves": [[random.uniform(-1, 1) for _ in range(11)] for _ in range(5)],
                    "average_curve": [random.uniform(-1, 1) for _ in range(11)]
                }
                for feature in feature_names
            }
        }

@api_router.post("/analysis/{model_id}/decision-tree")
async def extract_decision_tree(model_id: str, data: Dict[str, Any]):
    """Extract decision tree structure"""
    return {
        "tree": {
            "type": "split",
            "feature": "credit_score",
            "threshold": 650.0,
            "samples": 1000,
            "impurity": 0.5,
            "depth": 0,
            "left": {
                "type": "leaf",
                "prediction": "high_risk",
                "samples": 400,
                "impurity": 0.2,
                "depth": 1
            },
            "right": {
                "type": "leaf", 
                "prediction": "low_risk",
                "samples": 600,
                "impurity": 0.1,
                "depth": 1
            }
        },
        "max_depth": 5,
        "n_leaves": 8,
        "n_nodes": 15
    }

@api_router.post("/analysis/{model_id}/counterfactuals")
async def generate_counterfactuals(model_id: str, data: Dict[str, Any]):
    """Generate counterfactual explanations"""
    instance = data.get("instance", {})
    return {
        "original_instance": instance,
        "original_prediction": 0.75,
        "counterfactuals": [
            {
                "modified_instance": {k: v * 1.1 for k, v in instance.items()},
                "prediction": 0.25,
                "changes": {k: v * 0.1 for k, v in instance.items()},
                "num_changes": 2,
                "prediction_change": -0.5
            }
        ]
    }

# Additional endpoints that the frontend might expect
@api_router.get("/data/{dataset_id}/statistics")
async def get_dataset_statistics(dataset_id: str):
    """Get dataset statistics"""
    return {
        "total_rows": random.randint(1000, 100000),
        "total_features": random.randint(10, 50),
        "missing_values": random.randint(0, 100),
        "duplicate_rows": random.randint(0, 50),
        "classification_stats": {
            "accuracy": round(random.uniform(0.80, 0.95), 3),
            "precision": round(random.uniform(0.75, 0.92), 3),
            "recall": round(random.uniform(0.78, 0.94), 3),
            "f1_score": round(random.uniform(0.76, 0.93), 3),
            "confusion_matrix": [[850, 45], [32, 873]]
        },
        "regression_stats": {
            "mse": round(random.uniform(0.01, 0.1), 4),
            "rmse": round(random.uniform(0.1, 0.3), 3),
            "mae": round(random.uniform(0.05, 0.2), 3),
            "r2_score": round(random.uniform(0.75, 0.95), 3)
        }
    }

@api_router.get("/monitoring/alerts")
async def get_alerts():
    """Get monitoring alerts"""
    return {
        "alerts": [
            {
                "id": str(uuid.uuid4()),
                "model_id": "model_1",
                "type": "drift_detected",
                "severity": "high",
                "message": "Data drift detected",
                "created_at": datetime.now().isoformat()
            }
        ]
    }