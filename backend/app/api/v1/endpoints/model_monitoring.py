"""
Advanced Model Monitoring API Endpoints
Provides enterprise-grade model monitoring, A/B testing, and business metrics tracking
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime

from app.services.model_monitoring_service import model_monitor
from app.services.data_drift_service import load_csv_flexible
import structlog

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/model-monitoring", tags=["Advanced Model Monitoring"])

# Pydantic models for request/response
class ModelRegistrationRequest(BaseModel):
    model_id: str = Field(..., description="Unique identifier for the model")
    model_type: str = Field("classification", description="Type of model (classification/regression)")
    business_metrics: Optional[List[str]] = Field(None, description="Business metrics to track")
    performance_thresholds: Optional[Dict[str, float]] = Field(None, description="Alert thresholds")
    model_version: Optional[str] = Field("1.0.0", description="Model version")

class PredictionTrackingRequest(BaseModel):
    model_id: str = Field(..., description="Model identifier")
    business_outcomes: Optional[Dict[str, float]] = Field(None, description="Business metrics")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class ABTestRequest(BaseModel):
    test_id: str = Field(..., description="Unique A/B test identifier")
    champion_model_id: str = Field(..., description="Current production model ID")
    challenger_model_id: str = Field(..., description="New model to test")
    traffic_split: float = Field(0.5, ge=0.0, le=1.0, description="Traffic percentage to challenger")
    success_metrics: Optional[List[str]] = Field(None, description="Metrics to optimize for")
    duration_days: int = Field(14, ge=1, le=90, description="Test duration in days")

class AlertConfigurationRequest(BaseModel):
    model_id: str = Field(..., description="Model identifier")
    alert_channels: List[str] = Field(..., description="Alert channels (email, slack, webhook)")
    thresholds: Dict[str, float] = Field(..., description="Custom alert thresholds")
    business_impact_threshold: Optional[float] = Field(None, description="Business impact alert threshold")

class ModelRegistrationResponse(BaseModel):
    status: str
    model_id: str
    monitoring_features: Dict[str, bool]
    registration_timestamp: str

class PredictionTrackingResponse(BaseModel):
    status: str
    tracking_id: str
    performance_metrics: Dict[str, float]
    business_metrics: Dict[str, float]
    drift_detected: bool
    alerts_triggered: List[Dict[str, Any]]
    explainability_available: bool

class ModelHealthResponse(BaseModel):
    model_info: Dict[str, Any]
    performance_summary: Dict[str, Any]
    business_impact: Dict[str, Any]
    data_quality: Dict[str, Any]
    alerts: Dict[str, Any]
    ab_tests: Dict[str, Any]
    feature_importance: Dict[str, Any]
    charts: Dict[str, str]
    recommendations: List[Dict[str, str]]

class ABTestResponse(BaseModel):
    status: str
    test_id: str
    configuration: Dict[str, Any]

@router.post("/register-model", response_model=ModelRegistrationResponse)
async def register_model_for_monitoring(
    model_file: UploadFile = File(..., description="Serialized model file (pickle/joblib)"),
    request_data: str = Form(..., description="JSON string of ModelRegistrationRequest")
):
    """
    Register a model for comprehensive monitoring
    """
    try:
        # Parse request parameters
        try:
            params = ModelRegistrationRequest.parse_raw(request_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request parameters: {str(e)}")
        
        # Load the model (mock implementation - in production, load actual model)
        # model_content = await model_file.read()
        # model = joblib.load(io.BytesIO(model_content))
        
        # For demo purposes, use a mock model
        mock_model = {"type": "mock", "filename": model_file.filename}
        
        # Register model
        registration_result = model_monitor.register_model(
            model_id=params.model_id,
            model_object=mock_model,
            model_type=params.model_type,
            business_metrics=params.business_metrics,
            performance_thresholds=params.performance_thresholds
        )
        
        logger.info(
            "Model registered for advanced monitoring",
            model_id=params.model_id,
            model_type=params.model_type
        )
        
        return ModelRegistrationResponse(
            status=registration_result["status"],
            model_id=params.model_id,
            monitoring_features=registration_result["monitoring_features"],
            registration_timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error registering model for monitoring", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/track-predictions", response_model=PredictionTrackingResponse)
async def track_prediction_batch(
    features_file: UploadFile = File(..., description="Features CSV file"),
    predictions_file: UploadFile = File(..., description="Predictions CSV file"),
    labels_file: Optional[UploadFile] = File(None, description="True labels CSV file (optional)"),
    request_data: str = Form(..., description="JSON string of PredictionTrackingRequest")
):
    """
    Track a batch of predictions with performance and business metrics
    """
    try:
        # Parse request parameters
        try:
            params = PredictionTrackingRequest.parse_raw(request_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request parameters: {str(e)}")
        
        # Read uploaded files
        features_content = await features_file.read()
        predictions_content = await predictions_file.read()
        
        try:
            features_df = load_csv_flexible(features_content)
            predictions_df = load_csv_flexible(predictions_content)
            predictions = predictions_df.iloc[:, 0].values  # Assume first column contains predictions
            
            true_labels = None
            if labels_file:
                labels_content = await labels_file.read()
                labels_df = load_csv_flexible(labels_content)
                true_labels = labels_df.iloc[:, 0].values
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading data files: {str(e)}")
        
        # Track predictions
        tracking_result = model_monitor.track_prediction_batch(
            model_id=params.model_id,
            features=features_df,
            predictions=predictions,
            true_labels=true_labels,
            business_outcomes=params.business_outcomes,
            metadata=params.metadata
        )
        
        logger.info(
            "Prediction batch tracked",
            model_id=params.model_id,
            batch_size=len(predictions),
            has_labels=true_labels is not None
        )
        
        return PredictionTrackingResponse(
            status=tracking_result["status"],
            tracking_id=tracking_result["tracking_id"],
            performance_metrics=tracking_result["performance_metrics"],
            business_metrics=tracking_result["business_metrics"],
            drift_detected=tracking_result["drift_detected"],
            alerts_triggered=tracking_result["alerts_triggered"],
            explainability_available=tracking_result["explainability_available"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error tracking predictions", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/setup-ab-test", response_model=ABTestResponse)
async def setup_ab_test(request: ABTestRequest):
    """
    Setup A/B test between champion and challenger models
    """
    try:
        ab_test_result = model_monitor.setup_ab_test(
            test_id=request.test_id,
            champion_model_id=request.champion_model_id,
            challenger_model_id=request.challenger_model_id,
            traffic_split=request.traffic_split,
            success_metrics=request.success_metrics,
            duration_days=request.duration_days
        )
        
        logger.info(
            "A/B test setup",
            test_id=request.test_id,
            champion=request.champion_model_id,
            challenger=request.challenger_model_id
        )
        
        return ABTestResponse(
            status=ab_test_result["status"],
            test_id=request.test_id,
            configuration=ab_test_result["configuration"]
        )
        
    except Exception as e:
        logger.error("Error setting up A/B test", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/model-health/{model_id}", response_model=ModelHealthResponse)
async def get_model_health_dashboard(model_id: str):
    """
    Get comprehensive model health dashboard
    """
    try:
        dashboard_data = model_monitor.get_model_health_dashboard(model_id)
        
        if dashboard_data.get("status") == "no_data":
            raise HTTPException(status_code=404, detail="No monitoring data available for this model")
        
        logger.info(
            "Model health dashboard generated",
            model_id=model_id
        )
        
        return ModelHealthResponse(**dashboard_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error generating model health dashboard", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/models")
async def list_monitored_models():
    """
    List all models under monitoring
    """
    try:
        models = {}
        for model_id, model_info in model_monitor.models.items():
            models[model_id] = {
                "model_id": model_id,
                "model_type": model_info["model_type"],
                "status": model_info["status"],
                "registered_at": model_info["registered_at"].isoformat(),
                "version": model_info["version"],
                "business_metrics_count": len(model_info["business_metrics"]),
                "has_tracking_data": len(model_monitor.performance_history.get(model_id, [])) > 0
            }
        
        return {
            "status": "success",
            "models": models,
            "total_models": len(models)
        }
        
    except Exception as e:
        logger.error("Error listing monitored models", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/alerts/{model_id}")
async def get_model_alerts(
    model_id: str,
    severity: Optional[str] = None,
    limit: int = 50
):
    """
    Get alerts for a specific model
    """
    try:
        model_alerts = [
            {
                **alert,
                "timestamp": alert["timestamp"].isoformat()
            }
            for alert in model_monitor.alerts 
            if alert["model_id"] == model_id
        ]
        
        if severity:
            model_alerts = [a for a in model_alerts if a["severity"] == severity]
        
        # Sort by timestamp (most recent first) and limit
        model_alerts = sorted(model_alerts, key=lambda x: x["timestamp"], reverse=True)[:limit]
        
        return {
            "status": "success",
            "model_id": model_id,
            "alerts": model_alerts,
            "total_alerts": len(model_alerts)
        }
        
    except Exception as e:
        logger.error("Error retrieving model alerts", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/ab-tests")
async def list_ab_tests():
    """
    List all A/B tests
    """
    try:
        ab_tests = {}
        for test_id, test_info in model_monitor.ab_tests.items():
            ab_tests[test_id] = {
                **test_info,
                "start_date": test_info["start_date"].isoformat(),
                "end_date": test_info["end_date"].isoformat()
            }
        
        return {
            "status": "success",
            "ab_tests": ab_tests,
            "total_tests": len(ab_tests),
            "active_tests": len([t for t in ab_tests.values() if t["status"] == "active"])
        }
        
    except Exception as e:
        logger.error("Error listing A/B tests", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/configure-alerts")
async def configure_model_alerts(request: AlertConfigurationRequest):
    """
    Configure custom alerts for a model
    """
    try:
        if request.model_id not in model_monitor.models:
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
        
        # Update model thresholds
        model_monitor.models[request.model_id]["performance_thresholds"].update(request.thresholds)
        
        # Store alert configuration (in production, integrate with notification services)
        alert_config = {
            "model_id": request.model_id,
            "channels": request.alert_channels,
            "thresholds": request.thresholds,
            "business_impact_threshold": request.business_impact_threshold,
            "configured_at": datetime.now()
        }
        
        logger.info(
            "Alert configuration updated",
            model_id=request.model_id,
            channels=request.alert_channels
        )
        
        return {
            "status": "configured",
            "model_id": request.model_id,
            "alert_configuration": alert_config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error configuring alerts", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/business-metrics/{model_id}")
async def get_business_metrics_dashboard(model_id: str, days: int = 30):
    """
    Get business metrics dashboard for a model
    """
    try:
        if model_id not in model_monitor.models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        history = model_monitor.performance_history.get(model_id, [])
        business_records = [r for r in history if r.get("business_metrics")]
        
        if not business_records:
            return {
                "status": "no_data",
                "message": "No business metrics data available"
            }
        
        # Calculate business impact metrics
        total_revenue = sum(
            record["business_metrics"].get("revenue_per_prediction", 0) 
            for record in business_records
        )
        
        avg_conversion = np.mean([
            record["business_metrics"].get("conversion_rate", 0) 
            for record in business_records
        ])
        
        total_predictions = sum(record["batch_size"] for record in business_records)
        
        return {
            "status": "success",
            "model_id": model_id,
            "business_impact": {
                "total_revenue": total_revenue,
                "average_conversion_rate": avg_conversion,
                "total_predictions": total_predictions,
                "revenue_per_prediction": total_revenue / total_predictions if total_predictions > 0 else 0
            },
            "time_period_days": days,
            "data_points": len(business_records)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error generating business metrics dashboard", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check endpoint for the model monitoring service
    """
    return {
        "status": "healthy",
        "service": "model-monitoring",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "monitored_models": len(model_monitor.models),
        "active_ab_tests": len([t for t in model_monitor.ab_tests.values() if t["status"] == "active"]),
        "total_alerts": len(model_monitor.alerts)
    }