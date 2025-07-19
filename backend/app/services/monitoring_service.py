"""
Real-time Model Monitoring Service
Handles drift detection, performance monitoring, and alerting
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from enum import Enum

# Statistical libraries
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from app.core.websocket_manager import WebSocketManager
from app.models.model_metadata import (
    ModelMetadata, DataDriftReport, ModelDriftReport, 
    DataQualityReport, Alert, AlertConfig, ModelMonitoringConfig
)
from app.services.model_service import ModelService
from app.core.database import get_db
from app.config import settings

logger = structlog.get_logger(__name__)


class DriftDetectionMethod(Enum):
    """Drift detection methods"""
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    JENSEN_SHANNON = "jensen_shannon"
    WASSERSTEIN = "wasserstein"
    CHI_SQUARE = "chi_square"
    POPULATION_STABILITY_INDEX = "psi"
    ADVERSARIAL_ACCURACY = "adversarial_accuracy"


@dataclass
class MonitoringAlert:
    """Monitoring alert data"""
    alert_type: str
    severity: str
    model_id: str
    title: str
    message: str
    data: Dict[str, Any]
    timestamp: datetime


class MonitoringService:
    """
    Advanced monitoring service for ML models
    
    Features:
    - Real-time drift detection
    - Performance monitoring
    - Data quality assessment
    - Alerting and notifications
    - WebSocket updates
    """
    
    def __init__(self, websocket_manager: WebSocketManager):
        self.websocket_manager = websocket_manager
        self.model_service: Optional[ModelService] = None
        
        # Reference data storage
        self.reference_data_cache: Dict[str, np.ndarray] = {}
        self.reference_predictions_cache: Dict[str, np.ndarray] = {}
        
        # Monitoring state
        self.monitoring_configs: Dict[str, ModelMonitoringConfig] = {}
        self.alert_configs: Dict[str, List[AlertConfig]] = {}
        self.active_alerts: Dict[str, List[Alert]] = {}
        
        # Background tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        self.running = False
        
        logger.info("MonitoringService initialized")
    
    async def start_drift_monitoring(self):
        """Start drift monitoring background task"""
        if self.running:
            return
        
        self.running = True
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._drift_monitoring_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._data_quality_monitoring_loop()),
            asyncio.create_task(self._alert_processing_loop())
        ]
        
        logger.info("Drift monitoring started")
    
    async def start_performance_monitoring(self):
        """Start performance monitoring background task"""
        # This is handled by the drift monitoring loop
        pass
    
    async def stop(self):
        """Stop all monitoring tasks"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        logger.info("Monitoring service stopped")
    
    async def configure_monitoring(
        self,
        model_id: str,
        config: ModelMonitoringConfig
    ):
        """Configure monitoring for a model"""
        self.monitoring_configs[model_id] = config
        
        # Save to database
        await self._save_monitoring_config(config)
        
        logger.info(f"Monitoring configured for model {model_id}")
    
    async def set_reference_data(
        self,
        model_id: str,
        reference_data: np.ndarray,
        reference_predictions: Optional[np.ndarray] = None
    ):
        """Set reference data for drift detection"""
        self.reference_data_cache[model_id] = reference_data
        
        if reference_predictions is not None:
            self.reference_predictions_cache[model_id] = reference_predictions
        
        logger.info(f"Reference data set for model {model_id}: {reference_data.shape}")
    
    async def detect_data_drift(
        self,
        model_id: str,
        current_data: np.ndarray,
        feature_names: List[str],
        method: DriftDetectionMethod = DriftDetectionMethod.KOLMOGOROV_SMIRNOV
    ) -> DataDriftReport:
        """
        Detect data drift between reference and current data
        """
        if model_id not in self.reference_data_cache:
            raise ValueError(f"No reference data for model {model_id}")
        
        reference_data = self.reference_data_cache[model_id]
        
        # Calculate drift for each feature
        feature_drifts = {}
        
        for i, feature_name in enumerate(feature_names):
            if i >= reference_data.shape[1] or i >= current_data.shape[1]:
                continue
            
            ref_feature = reference_data[:, i]
            curr_feature = current_data[:, i]
            
            # Calculate drift score based on method
            if method == DriftDetectionMethod.KOLMOGOROV_SMIRNOV:
                drift_score = self._ks_drift_score(ref_feature, curr_feature)
            elif method == DriftDetectionMethod.JENSEN_SHANNON:
                drift_score = self._js_drift_score(ref_feature, curr_feature)
            elif method == DriftDetectionMethod.WASSERSTEIN:
                drift_score = self._wasserstein_drift_score(ref_feature, curr_feature)
            elif method == DriftDetectionMethod.POPULATION_STABILITY_INDEX:
                drift_score = self._psi_drift_score(ref_feature, curr_feature)
            else:
                drift_score = self._ks_drift_score(ref_feature, curr_feature)
            
            feature_drifts[feature_name] = drift_score
        
        # Calculate overall drift score
        overall_drift_score = np.mean(list(feature_drifts.values()))
        
        # Get drift threshold
        config = self.monitoring_configs.get(model_id)
        drift_threshold = config.drift_threshold if config else settings.DRIFT_THRESHOLD
        
        # Create report
        report = DataDriftReport(
            model_id=model_id,
            feature_drifts=feature_drifts,
            overall_drift_score=overall_drift_score,
            drift_threshold=drift_threshold,
            drift_detected=overall_drift_score > drift_threshold,
            reference_dataset_size=len(reference_data),
            current_dataset_size=len(current_data),
            detection_method=method.value
        )
        
        # Send alert if drift detected
        if report.drift_detected:
            await self._send_drift_alert(model_id, report)
        
        # Send WebSocket update
        await self._send_websocket_update(model_id, "drift_report", report.dict())
        
        return report
    
    async def detect_model_drift(
        self,
        model_id: str,
        current_data: np.ndarray,
        current_labels: np.ndarray
    ) -> ModelDriftReport:
        """
        Detect model performance drift
        """
        if not self.model_service:
            raise ValueError("ModelService not available")
        
        # Get current predictions
        current_predictions = await self.model_service.predict(model_id, current_data)
        
        # Calculate current metrics
        current_metrics = self._calculate_metrics(current_labels, current_predictions)
        
        # Get reference metrics
        reference_metrics = await self._get_reference_metrics(model_id)
        
        # Calculate performance drift
        performance_drift = self._calculate_performance_drift(reference_metrics, current_metrics)
        
        # Get drift threshold
        config = self.monitoring_configs.get(model_id)
        drift_threshold = config.performance_threshold if config else settings.PERFORMANCE_THRESHOLD
        
        # Create report
        report = ModelDriftReport(
            model_id=model_id,
            performance_drift=performance_drift,
            drift_threshold=drift_threshold,
            drift_detected=performance_drift > drift_threshold,
            current_metrics=current_metrics,
            reference_metrics=reference_metrics,
            detection_method="performance_comparison"
        )
        
        # Send alert if drift detected
        if report.drift_detected:
            await self._send_performance_alert(model_id, report)
        
        # Send WebSocket update
        await self._send_websocket_update(model_id, "model_drift_report", report.dict())
        
        return report
    
    async def assess_data_quality(
        self,
        dataset_id: str,
        data: np.ndarray,
        feature_names: List[str],
        feature_types: Optional[Dict[str, str]] = None
    ) -> DataQualityReport:
        """
        Assess data quality
        """
        total_rows, total_features = data.shape
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(data, columns=feature_names[:total_features])
        
        # Check missing values
        missing_values = df.isnull().sum().to_dict()
        
        # Check duplicates
        duplicate_rows = df.duplicated().sum()
        
        # Detect outliers (using IQR method)
        outliers = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            outliers[col] = outlier_count
        
        # Data types
        data_types = df.dtypes.astype(str).to_dict()
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(df, missing_values, duplicate_rows, outliers)
        
        # Identify issues
        issues = []
        if sum(missing_values.values()) > 0:
            issues.append(f"Missing values detected in {len([k for k, v in missing_values.items() if v > 0])} columns")
        if duplicate_rows > 0:
            issues.append(f"{duplicate_rows} duplicate rows detected")
        if sum(outliers.values()) > total_rows * 0.1:
            issues.append("High number of outliers detected")
        
        # Create report
        report = DataQualityReport(
            dataset_id=dataset_id,
            total_rows=total_rows,
            total_features=total_features,
            missing_values=missing_values,
            duplicate_rows=duplicate_rows,
            outliers=outliers,
            data_types=data_types,
            quality_score=quality_score,
            issues=issues
        )
        
        # Send alert if quality is poor
        if quality_score < 0.7:
            await self._send_quality_alert(dataset_id, report)
        
        return report
    
    # Background monitoring loops
    
    async def _drift_monitoring_loop(self):
        """Background task for drift monitoring"""
        while self.running:
            try:
                # Get all models with monitoring enabled
                for model_id, config in self.monitoring_configs.items():
                    if not config.drift_detection_enabled:
                        continue
                    
                    # Check if it's time for drift detection
                    if await self._should_run_drift_check(model_id, config):
                        await self._run_drift_detection(model_id)
                
                # Sleep for monitoring interval
                await asyncio.sleep(settings.MONITORING_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in drift monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _performance_monitoring_loop(self):
        """Background task for performance monitoring"""
        while self.running:
            try:
                # Get all models with performance monitoring enabled
                for model_id, config in self.monitoring_configs.items():
                    if not config.performance_monitoring_enabled:
                        continue
                    
                    # Check if it's time for performance monitoring
                    if await self._should_run_performance_check(model_id, config):
                        await self._run_performance_monitoring(model_id)
                
                # Sleep for monitoring interval
                await asyncio.sleep(settings.MONITORING_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _data_quality_monitoring_loop(self):
        """Background task for data quality monitoring"""
        while self.running:
            try:
                # Get all models with data quality monitoring enabled
                for model_id, config in self.monitoring_configs.items():
                    if not config.data_quality_monitoring_enabled:
                        continue
                    
                    # Check if it's time for data quality monitoring
                    if await self._should_run_quality_check(model_id, config):
                        await self._run_data_quality_monitoring(model_id)
                
                # Sleep for monitoring interval
                await asyncio.sleep(settings.MONITORING_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in data quality monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _alert_processing_loop(self):
        """Background task for processing alerts"""
        while self.running:
            try:
                # Process pending alerts
                await self._process_pending_alerts()
                
                # Sleep for alert processing interval
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(60)
    
    # Drift detection methods
    
    def _ks_drift_score(self, ref_data: np.ndarray, curr_data: np.ndarray) -> float:
        """Kolmogorov-Smirnov drift score"""
        try:
            statistic, p_value = stats.ks_2samp(ref_data, curr_data)
            return statistic
        except Exception as e:
            logger.warning(f"Error calculating KS drift score: {e}")
            return 0.0
    
    def _js_drift_score(self, ref_data: np.ndarray, curr_data: np.ndarray) -> float:
        """Jensen-Shannon drift score"""
        try:
            # Create histograms
            bins = np.histogram_bin_edges(np.concatenate([ref_data, curr_data]), bins=50)
            ref_hist, _ = np.histogram(ref_data, bins=bins, density=True)
            curr_hist, _ = np.histogram(curr_data, bins=bins, density=True)
            
            # Normalize to probability distributions
            ref_hist = ref_hist / np.sum(ref_hist)
            curr_hist = curr_hist / np.sum(curr_hist)
            
            # Calculate Jensen-Shannon distance
            return jensenshannon(ref_hist, curr_hist)
        except Exception as e:
            logger.warning(f"Error calculating JS drift score: {e}")
            return 0.0
    
    def _wasserstein_drift_score(self, ref_data: np.ndarray, curr_data: np.ndarray) -> float:
        """Wasserstein (Earth Mover's) drift score"""
        try:
            return stats.wasserstein_distance(ref_data, curr_data)
        except Exception as e:
            logger.warning(f"Error calculating Wasserstein drift score: {e}")
            return 0.0
    
    def _psi_drift_score(self, ref_data: np.ndarray, curr_data: np.ndarray, bins: int = 10) -> float:
        """Population Stability Index drift score"""
        try:
            # Create bins based on reference data
            bin_edges = np.histogram_bin_edges(ref_data, bins=bins)
            
            # Calculate expected (reference) and actual (current) distributions
            expected, _ = np.histogram(ref_data, bins=bin_edges, density=True)
            actual, _ = np.histogram(curr_data, bins=bin_edges, density=True)
            
            # Normalize
            expected = expected / np.sum(expected)
            actual = actual / np.sum(actual)
            
            # Avoid division by zero
            expected = np.where(expected == 0, 1e-10, expected)
            actual = np.where(actual == 0, 1e-10, actual)
            
            # Calculate PSI
            psi = np.sum((actual - expected) * np.log(actual / expected))
            
            return psi
        except Exception as e:
            logger.warning(f"Error calculating PSI drift score: {e}")
            return 0.0
    
    # Utility methods
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics"""
        metrics = {}
        
        # Determine if it's classification or regression
        if len(np.unique(y_true)) <= 10:  # Classification
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["f1_score"] = f1_score(y_true, y_pred, average='weighted')
            metrics["precision"] = precision_score(y_true, y_pred, average='weighted')
            metrics["recall"] = recall_score(y_true, y_pred, average='weighted')
        else:  # Regression
            metrics["mse"] = mean_squared_error(y_true, y_pred)
            metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics["r2"] = r2_score(y_true, y_pred)
        
        return metrics
    
    def _calculate_performance_drift(
        self,
        reference_metrics: Dict[str, float],
        current_metrics: Dict[str, float]
    ) -> float:
        """Calculate performance drift score"""
        drift_scores = []
        
        for metric_name in reference_metrics.keys():
            if metric_name in current_metrics:
                ref_value = reference_metrics[metric_name]
                curr_value = current_metrics[metric_name]
                
                # Calculate relative change
                if ref_value != 0:
                    drift_score = abs(curr_value - ref_value) / abs(ref_value)
                else:
                    drift_score = abs(curr_value)
                
                drift_scores.append(drift_score)
        
        return np.mean(drift_scores) if drift_scores else 0.0
    
    def _calculate_quality_score(
        self,
        df: pd.DataFrame,
        missing_values: Dict[str, int],
        duplicate_rows: int,
        outliers: Dict[str, int]
    ) -> float:
        """Calculate overall data quality score"""
        total_rows = len(df)
        total_cells = df.size
        
        # Missing values score
        missing_score = 1.0 - (sum(missing_values.values()) / total_cells)
        
        # Duplicate rows score
        duplicate_score = 1.0 - (duplicate_rows / total_rows)
        
        # Outliers score
        outlier_score = 1.0 - (sum(outliers.values()) / total_rows)
        
        # Weighted average
        quality_score = (missing_score * 0.4 + duplicate_score * 0.3 + outlier_score * 0.3)
        
        return max(0.0, min(1.0, quality_score))
    
    # Alert methods
    
    async def _send_drift_alert(self, model_id: str, report: DataDriftReport):
        """Send drift detection alert"""
        alert = MonitoringAlert(
            alert_type="drift",
            severity="high",
            model_id=model_id,
            title=f"Data Drift Detected - Model {model_id}",
            message=f"Data drift detected with score {report.overall_drift_score:.3f} (threshold: {report.drift_threshold:.3f})",
            data=report.dict(),
            timestamp=datetime.utcnow()
        )
        
        await self._process_alert(alert)
    
    async def _send_performance_alert(self, model_id: str, report: ModelDriftReport):
        """Send performance drift alert"""
        alert = MonitoringAlert(
            alert_type="performance",
            severity="high",
            model_id=model_id,
            title=f"Performance Drift Detected - Model {model_id}",
            message=f"Performance drift detected with score {report.performance_drift:.3f} (threshold: {report.drift_threshold:.3f})",
            data=report.dict(),
            timestamp=datetime.utcnow()
        )
        
        await self._process_alert(alert)
    
    async def _send_quality_alert(self, dataset_id: str, report: DataQualityReport):
        """Send data quality alert"""
        alert = MonitoringAlert(
            alert_type="data_quality",
            severity="medium",
            model_id=dataset_id,
            title=f"Data Quality Issues - Dataset {dataset_id}",
            message=f"Data quality score: {report.quality_score:.3f}. Issues: {', '.join(report.issues)}",
            data=report.dict(),
            timestamp=datetime.utcnow()
        )
        
        await self._process_alert(alert)
    
    async def _process_alert(self, alert: MonitoringAlert):
        """Process and send alert"""
        # Add to active alerts
        if alert.model_id not in self.active_alerts:
            self.active_alerts[alert.model_id] = []
        
        self.active_alerts[alert.model_id].append(alert)
        
        # Send WebSocket notification
        await self._send_websocket_update(alert.model_id, "alert", alert.__dict__)
        
        # Send to other channels (email, Slack, etc.)
        await self._send_external_alert(alert)
    
    async def _send_external_alert(self, alert: MonitoringAlert):
        """Send alert to external channels"""
        # This would integrate with external alerting systems
        logger.info(f"Alert sent: {alert.title}")
    
    async def _send_websocket_update(self, model_id: str, event_type: str, data: Dict[str, Any]):
        """Send WebSocket update"""
        message = {
            "type": event_type,
            "model_id": model_id,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.websocket_manager.broadcast_to_model_subscribers(
            model_id, 
            json.dumps(message)
        )
    
    # Database methods
    
    async def _save_monitoring_config(self, config: ModelMonitoringConfig):
        """Save monitoring configuration to database"""
        # Implementation would save to database
        pass
    
    async def _get_reference_metrics(self, model_id: str) -> Dict[str, float]:
        """Get reference metrics from database"""
        # Implementation would retrieve from database
        return {"accuracy": 0.85, "f1_score": 0.83}  # Placeholder
    
    async def _should_run_drift_check(self, model_id: str, config: ModelMonitoringConfig) -> bool:
        """Check if it's time to run drift detection"""
        # Implementation would check last run time
        return True  # Placeholder
    
    async def _should_run_performance_check(self, model_id: str, config: ModelMonitoringConfig) -> bool:
        """Check if it's time to run performance monitoring"""
        # Implementation would check last run time
        return True  # Placeholder
    
    async def _should_run_quality_check(self, model_id: str, config: ModelMonitoringConfig) -> bool:
        """Check if it's time to run data quality monitoring"""
        # Implementation would check last run time
        return True  # Placeholder
    
    async def _run_drift_detection(self, model_id: str):
        """Run drift detection for a model"""
        # Implementation would fetch current data and run drift detection
        pass
    
    async def _run_performance_monitoring(self, model_id: str):
        """Run performance monitoring for a model"""
        # Implementation would fetch current data and run performance monitoring
        pass
    
    async def _run_data_quality_monitoring(self, model_id: str):
        """Run data quality monitoring for a model"""
        # Implementation would fetch current data and run quality assessment
        pass
    
    async def _process_pending_alerts(self):
        """Process pending alerts"""
        # Implementation would process queued alerts
        pass