"""
Monitoring API endpoints
"""

from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import structlog

from app.services.monitoring_service import MonitoringService, DriftDetectionMethod
from app.models.model_metadata import (
    ModelMonitoringConfig, DataDriftReport, ModelDriftReport, 
    DataQualityReport, Alert
)
from app.core.dependencies import get_monitoring_service
from app.core.security import get_current_user

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.post("/configure/{model_id}")
async def configure_monitoring(
    model_id: str,
    config: ModelMonitoringConfig,
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Configure monitoring for a model"""
    try:
        await monitoring_service.configure_monitoring(model_id, config)
        return {"message": "Monitoring configured successfully"}
    except Exception as e:
        logger.error(f"Error configuring monitoring for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reference-data/{model_id}")
async def set_reference_data(
    model_id: str,
    reference_data: List[List[float]],
    feature_names: List[str],
    reference_predictions: List[float] = None,
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Set reference data for drift detection"""
    try:
        import numpy as np
        
        ref_data = np.array(reference_data)
        ref_predictions = np.array(reference_predictions) if reference_predictions else None
        
        await monitoring_service.set_reference_data(
            model_id, ref_data, ref_predictions
        )
        
        return {"message": "Reference data set successfully"}
    except Exception as e:
        logger.error(f"Error setting reference data for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drift/detect/{model_id}", response_model=DataDriftReport)
async def detect_data_drift(
    model_id: str,
    current_data: List[List[float]],
    feature_names: List[str],
    method: DriftDetectionMethod = DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Detect data drift"""
    try:
        import numpy as np
        
        curr_data = np.array(current_data)
        
        report = await monitoring_service.detect_data_drift(
            model_id, curr_data, feature_names, method
        )
        
        return report
    except Exception as e:
        logger.error(f"Error detecting data drift for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drift/model/{model_id}", response_model=ModelDriftReport)
async def detect_model_drift(
    model_id: str,
    current_data: List[List[float]],
    current_labels: List[float],
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Detect model performance drift"""
    try:
        import numpy as np
        
        curr_data = np.array(current_data)
        curr_labels = np.array(current_labels)
        
        report = await monitoring_service.detect_model_drift(
            model_id, curr_data, curr_labels
        )
        
        return report
    except Exception as e:
        logger.error(f"Error detecting model drift for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quality/assess/{dataset_id}", response_model=DataQualityReport)
async def assess_data_quality(
    dataset_id: str,
    data: List[List[float]],
    feature_names: List[str],
    feature_types: Dict[str, str] = None,
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Assess data quality"""
    try:
        import numpy as np
        
        data_array = np.array(data)
        
        report = await monitoring_service.assess_data_quality(
            dataset_id, data_array, feature_names, feature_types
        )
        
        return report
    except Exception as e:
        logger.error(f"Error assessing data quality for {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/{model_id}")
async def get_model_alerts(
    model_id: str,
    limit: int = 100,
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Get alerts for a model"""
    try:
        # This would fetch from database in real implementation
        alerts = [
            {
                "alert_id": "alert_1",
                "model_id": model_id,
                "alert_type": "drift",
                "severity": "high",
                "title": "Data Drift Detected",
                "message": "Significant drift detected in feature importance",
                "status": "active",
                "created_at": "2024-01-15T10:30:00Z"
            }
        ]
        
        return alerts
    except Exception as e:
        logger.error(f"Error getting alerts for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Acknowledge an alert"""
    try:
        # This would update alert status in database
        return {"message": "Alert acknowledged"}
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Resolve an alert"""
    try:
        # This would update alert status in database
        return {"message": "Alert resolved"}
    except Exception as e:
        logger.error(f"Error resolving alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def monitoring_health(
    monitoring_service: MonitoringService = Depends(get_monitoring_service)
):
    """Get monitoring service health"""
    try:
        return {
            "status": "healthy",
            "monitoring_active": monitoring_service.running,
            "active_models": len(monitoring_service.monitoring_configs),
            "active_alerts": sum(
                len(alerts) for alerts in monitoring_service.active_alerts.values()
            )
        }
    except Exception as e:
        logger.error(f"Error checking monitoring health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_monitoring_stats(
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Get monitoring statistics"""
    try:
        return {
            "total_models_monitored": len(monitoring_service.monitoring_configs),
            "drift_detection_enabled": sum(
                1 for config in monitoring_service.monitoring_configs.values()
                if config.drift_detection_enabled
            ),
            "performance_monitoring_enabled": sum(
                1 for config in monitoring_service.monitoring_configs.values()
                if config.performance_monitoring_enabled
            ),
            "data_quality_monitoring_enabled": sum(
                1 for config in monitoring_service.monitoring_configs.values()
                if config.data_quality_monitoring_enabled
            ),
            "total_active_alerts": sum(
                len(alerts) for alerts in monitoring_service.active_alerts.values()
            ),
            "reference_data_loaded": len(monitoring_service.reference_data_cache)
        }
    except Exception as e:
        logger.error(f"Error getting monitoring stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_monitoring(
    background_tasks: BackgroundTasks,
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Start monitoring services"""
    try:
        if not monitoring_service.running:
            background_tasks.add_task(monitoring_service.start_drift_monitoring)
            return {"message": "Monitoring services started"}
        else:
            return {"message": "Monitoring services already running"}
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_monitoring(
    background_tasks: BackgroundTasks,
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: str = Depends(get_current_user)
):
    """Stop monitoring services"""
    try:
        if monitoring_service.running:
            background_tasks.add_task(monitoring_service.stop)
            return {"message": "Monitoring services stopped"}
        else:
            return {"message": "Monitoring services not running"}
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))