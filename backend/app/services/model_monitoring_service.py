"""
Advanced Model Monitoring Service
Provides comprehensive model performance monitoring, A/B testing, and business metrics tracking
"""

import pandas as pd
import numpy as np
import json
import tempfile
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from io import BytesIO
import asyncio
from concurrent.futures import ThreadPoolExecutor

import shap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import structlog

warnings.filterwarnings("ignore", category=FutureWarning)
logger = structlog.get_logger(__name__)

class ModelMonitor:
    """
    Advanced model monitoring with business metrics tracking
    """
    
    def __init__(self):
        self.models = {}
        self.monitoring_configs = {}
        self.performance_history = {}
        self.business_metrics_history = {}
        self.ab_tests = {}
        self.alerts = []
        
    def register_model(
        self, 
        model_id: str, 
        model_object: Any,
        model_type: str = "classification",
        business_metrics: List[str] = None,
        performance_thresholds: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Register a model for monitoring
        
        Args:
            model_id: Unique identifier for the model
            model_object: The trained model object
            model_type: "classification" or "regression"
            business_metrics: List of business metrics to track
            performance_thresholds: Alert thresholds for performance metrics
            
        Returns:
            Registration confirmation with model metadata
        """
        
        default_business_metrics = [
            "conversion_rate", "revenue_per_prediction", "customer_satisfaction",
            "click_through_rate", "retention_rate"
        ]
        
        default_thresholds = {
            "accuracy": 0.8 if model_type == "classification" else None,
            "f1_score": 0.75 if model_type == "classification" else None,
            "r2_score": 0.7 if model_type == "regression" else None,
            "mse": 1000.0 if model_type == "regression" else None,
            "data_drift_threshold": 0.1,
            "prediction_drift_threshold": 0.15
        }
        
        self.models[model_id] = {
            "model": model_object,
            "model_type": model_type,
            "business_metrics": business_metrics or default_business_metrics,
            "performance_thresholds": {**default_thresholds, **(performance_thresholds or {})},
            "registered_at": datetime.now(),
            "status": "active",
            "version": "1.0.0"
        }
        
        self.performance_history[model_id] = []
        self.business_metrics_history[model_id] = []
        
        logger.info(
            "Model registered for monitoring",
            model_id=model_id,
            model_type=model_type,
            business_metrics=len(business_metrics or default_business_metrics)
        )
        
        return {
            "status": "registered",
            "model_id": model_id,
            "monitoring_features": {
                "performance_monitoring": True,
                "business_metrics": True,
                "explainability": True,
                "data_drift": True,
                "model_drift": True,
                "alerting": True
            }
        }
    
    def track_prediction_batch(
        self,
        model_id: str,
        features: pd.DataFrame,
        predictions: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        business_outcomes: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Track a batch of predictions with performance and business metrics
        
        Args:
            model_id: ID of the model making predictions
            features: Input features used for predictions
            predictions: Model predictions
            true_labels: Ground truth labels (when available)
            business_outcomes: Business metrics for this batch
            metadata: Additional metadata (user_id, campaign_id, etc.)
            
        Returns:
            Tracking results with performance metrics
        """
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not registered")
        
        model_info = self.models[model_id]
        timestamp = datetime.now()
        
        # Calculate performance metrics if labels available
        performance_metrics = {}
        if true_labels is not None:
            performance_metrics = self._calculate_performance_metrics(
                predictions, true_labels, model_info["model_type"]
            )
        
        # Track business metrics
        business_metrics = business_outcomes or {}
        
        # Generate explainability insights for sample
        sample_explanations = self._generate_sample_explanations(
            model_info["model"], features.head(100), model_id
        )
        
        # Detect data drift
        drift_analysis = self._detect_realtime_drift(model_id, features)
        
        # Store tracking data
        tracking_record = {
            "timestamp": timestamp,
            "batch_size": len(predictions),
            "performance_metrics": performance_metrics,
            "business_metrics": business_metrics,
            "drift_analysis": drift_analysis,
            "sample_explanations": sample_explanations,
            "metadata": metadata or {}
        }
        
        self.performance_history[model_id].append(tracking_record)
        
        # Check for alerts
        alerts_triggered = self._check_performance_alerts(model_id, tracking_record)
        
        logger.info(
            "Prediction batch tracked",
            model_id=model_id,
            batch_size=len(predictions),
            performance_available=true_labels is not None,
            alerts_triggered=len(alerts_triggered)
        )
        
        return {
            "status": "tracked",
            "tracking_id": f"{model_id}_{timestamp.isoformat()}",
            "performance_metrics": performance_metrics,
            "business_metrics": business_metrics,
            "drift_detected": drift_analysis.get("drift_detected", False),
            "alerts_triggered": alerts_triggered,
            "explainability_available": len(sample_explanations) > 0
        }
    
    def setup_ab_test(
        self,
        test_id: str,
        champion_model_id: str,
        challenger_model_id: str,
        traffic_split: float = 0.5,
        success_metrics: List[str] = None,
        duration_days: int = 14
    ) -> Dict[str, Any]:
        """
        Setup A/B test between champion and challenger models
        
        Args:
            test_id: Unique identifier for the A/B test
            champion_model_id: Current production model
            challenger_model_id: New model to test
            traffic_split: Percentage of traffic to challenger (0.0-1.0)
            success_metrics: Metrics to compare models on
            duration_days: How long to run the test
            
        Returns:
            A/B test configuration
        """
        
        if champion_model_id not in self.models or challenger_model_id not in self.models:
            raise ValueError("Both champion and challenger models must be registered")
        
        default_metrics = ["accuracy", "conversion_rate", "revenue_per_prediction"]
        
        ab_test_config = {
            "test_id": test_id,
            "champion_model_id": champion_model_id,
            "challenger_model_id": challenger_model_id,
            "traffic_split": traffic_split,
            "success_metrics": success_metrics or default_metrics,
            "start_date": datetime.now(),
            "end_date": datetime.now() + timedelta(days=duration_days),
            "status": "active",
            "results": {
                "champion": {"traffic": 0, "metrics": {}},
                "challenger": {"traffic": 0, "metrics": {}}
            }
        }
        
        self.ab_tests[test_id] = ab_test_config
        
        logger.info(
            "A/B test setup completed",
            test_id=test_id,
            champion=champion_model_id,
            challenger=challenger_model_id,
            traffic_split=traffic_split
        )
        
        return {
            "status": "ab_test_created",
            "test_id": test_id,
            "configuration": ab_test_config
        }
    
    def get_model_health_dashboard(self, model_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive model health dashboard
        
        Args:
            model_id: ID of the model to analyze
            
        Returns:
            Complete dashboard data with visualizations
        """
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.models[model_id]
        history = self.performance_history.get(model_id, [])
        
        if not history:
            return {"status": "no_data", "message": "No tracking data available"}
        
        # Performance trends over time
        performance_trends = self._calculate_performance_trends(history)
        
        # Business impact analysis
        business_impact = self._calculate_business_impact(history)
        
        # Drift analysis summary
        drift_summary = self._analyze_drift_trends(history)
        
        # Feature importance evolution
        feature_importance = self._analyze_feature_importance_trends(model_id, history)
        
        # Alert summary
        alert_summary = self._get_alert_summary(model_id)
        
        # Model comparison (if A/B tests exist)
        ab_test_results = self._get_ab_test_results(model_id)
        
        # Generate visualizations
        charts = self._generate_dashboard_charts(
            performance_trends, business_impact, drift_summary, feature_importance
        )
        
        dashboard = {
            "model_info": {
                "model_id": model_id,
                "model_type": model_info["model_type"],
                "status": model_info["status"],
                "registered_at": model_info["registered_at"].isoformat(),
                "last_prediction": history[-1]["timestamp"].isoformat() if history else None
            },
            "performance_summary": {
                "current_performance": performance_trends.get("current", {}),
                "performance_trend": performance_trends.get("trend", "stable"),
                "performance_change": performance_trends.get("change_percentage", 0)
            },
            "business_impact": business_impact,
            "data_quality": {
                "drift_status": drift_summary.get("current_status", "stable"),
                "features_affected": drift_summary.get("affected_features", []),
                "drift_score": drift_summary.get("latest_score", 0)
            },
            "alerts": alert_summary,
            "ab_tests": ab_test_results,
            "feature_importance": feature_importance,
            "charts": charts,
            "recommendations": self._generate_recommendations(
                performance_trends, business_impact, drift_summary, alert_summary
            )
        }
        
        return dashboard
    
    def _calculate_performance_metrics(
        self, 
        predictions: np.ndarray, 
        true_labels: np.ndarray, 
        model_type: str
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        metrics = {}
        
        if model_type == "classification":
            metrics.update({
                "accuracy": accuracy_score(true_labels, predictions),
                "precision": precision_score(true_labels, predictions, average='weighted', zero_division=0),
                "recall": recall_score(true_labels, predictions, average='weighted', zero_division=0),
                "f1_score": f1_score(true_labels, predictions, average='weighted', zero_division=0)
            })
            
            # ROC AUC for binary classification
            if len(np.unique(true_labels)) == 2:
                try:
                    metrics["roc_auc"] = roc_auc_score(true_labels, predictions)
                except:
                    metrics["roc_auc"] = 0.5
                    
        elif model_type == "regression":
            metrics.update({
                "mse": mean_squared_error(true_labels, predictions),
                "mae": mean_absolute_error(true_labels, predictions),
                "rmse": np.sqrt(mean_squared_error(true_labels, predictions)),
                "r2_score": r2_score(true_labels, predictions)
            })
        
        return metrics
    
    def _generate_sample_explanations(
        self, 
        model: Any, 
        features: pd.DataFrame, 
        model_id: str
    ) -> Dict[str, Any]:
        """Generate SHAP explanations for sample predictions"""
        
        try:
            # Create SHAP explainer
            explainer = shap.Explainer(model, features.sample(min(100, len(features))))
            
            # Generate explanations for sample
            sample_size = min(20, len(features))
            sample_features = features.sample(sample_size)
            shap_values = explainer(sample_features)
            
            # Extract key insights
            feature_importance = {
                feature: float(np.abs(shap_values.values[:, i]).mean())
                for i, feature in enumerate(features.columns)
            }
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "feature_importance": dict(sorted_features[:10]),  # Top 10 features
                "sample_size": sample_size,
                "explanation_type": "shap",
                "top_features": [item[0] for item in sorted_features[:5]],
                "explanation_quality": "high" if sample_size >= 10 else "medium"
            }
            
        except Exception as e:
            logger.warning("Failed to generate SHAP explanations", error=str(e))
            return {
                "error": str(e),
                "explanation_quality": "unavailable"
            }
    
    def _detect_realtime_drift(self, model_id: str, current_features: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift in real-time using statistical tests"""
        
        history = self.performance_history.get(model_id, [])
        if not history:
            return {"drift_detected": False, "reason": "no_historical_data"}
        
        # Get recent historical data for comparison
        recent_history = history[-5:]  # Last 5 batches
        if not recent_history:
            return {"drift_detected": False, "reason": "insufficient_history"}
        
        try:
            # Simple drift detection using feature mean shifts
            drift_scores = {}
            drift_detected = False
            
            for feature in current_features.columns:
                if feature in current_features.select_dtypes(include=[np.number]).columns:
                    current_mean = current_features[feature].mean()
                    historical_means = [
                        record.get("features", {}).get(feature, current_mean) 
                        for record in recent_history
                    ]
                    
                    if historical_means:
                        historical_mean = np.mean(historical_means)
                        drift_score = abs(current_mean - historical_mean) / (historical_mean + 1e-8)
                        drift_scores[feature] = float(drift_score)
                        
                        if drift_score > 0.1:  # 10% threshold
                            drift_detected = True
            
            return {
                "drift_detected": drift_detected,
                "drift_scores": drift_scores,
                "affected_features": [f for f, score in drift_scores.items() if score > 0.1],
                "max_drift_score": max(drift_scores.values()) if drift_scores else 0
            }
            
        except Exception as e:
            logger.warning("Drift detection failed", error=str(e))
            return {"drift_detected": False, "error": str(e)}
    
    def _check_performance_alerts(
        self, 
        model_id: str, 
        tracking_record: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check if any performance thresholds are breached"""
        
        alerts_triggered = []
        model_info = self.models[model_id]
        thresholds = model_info["performance_thresholds"]
        performance = tracking_record["performance_metrics"]
        
        for metric, threshold in thresholds.items():
            if threshold is not None and metric in performance:
                value = performance[metric]
                
                # Determine if alert should trigger based on metric type
                alert_triggered = False
                if metric in ["accuracy", "f1_score", "r2_score"]:
                    alert_triggered = value < threshold
                elif metric in ["mse", "mae"]:
                    alert_triggered = value > threshold
                
                if alert_triggered:
                    alert = {
                        "alert_id": f"{model_id}_{metric}_{datetime.now().isoformat()}",
                        "model_id": model_id,
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "severity": "high" if abs(value - threshold) > threshold * 0.1 else "medium",
                        "timestamp": datetime.now(),
                        "message": f"{metric} ({value:.3f}) breached threshold ({threshold})"
                    }
                    
                    alerts_triggered.append(alert)
                    self.alerts.append(alert)
        
        # Check drift alerts
        drift_analysis = tracking_record.get("drift_analysis", {})
        if drift_analysis.get("drift_detected", False):
            alert = {
                "alert_id": f"{model_id}_drift_{datetime.now().isoformat()}",
                "model_id": model_id,
                "metric": "data_drift",
                "value": drift_analysis.get("max_drift_score", 0),
                "threshold": thresholds.get("data_drift_threshold", 0.1),
                "severity": "high",
                "timestamp": datetime.now(),
                "message": f"Data drift detected in features: {', '.join(drift_analysis.get('affected_features', []))}"
            }
            
            alerts_triggered.append(alert)
            self.alerts.append(alert)
        
        return alerts_triggered
    
    def _calculate_performance_trends(self, history: List[Dict]) -> Dict[str, Any]:
        """Calculate performance trends over time"""
        
        if len(history) < 2:
            return {"trend": "insufficient_data"}
        
        # Get recent performance metrics
        recent_records = [r for r in history if r["performance_metrics"]]
        if not recent_records:
            return {"trend": "no_performance_data"}
        
        # Calculate trends for key metrics
        trends = {}
        current_performance = recent_records[-1]["performance_metrics"]
        
        if len(recent_records) >= 2:
            previous_performance = recent_records[-2]["performance_metrics"]
            
            for metric in current_performance:
                if metric in previous_performance:
                    current_val = current_performance[metric]
                    previous_val = previous_performance[metric]
                    change = ((current_val - previous_val) / previous_val) * 100
                    trends[metric] = {
                        "current": current_val,
                        "previous": previous_val,
                        "change_percentage": change,
                        "trend": "improving" if change > 1 else "declining" if change < -1 else "stable"
                    }
        
        return {
            "current": current_performance,
            "trends": trends,
            "trend": "stable",  # Overall trend
            "data_points": len(recent_records)
        }
    
    def _calculate_business_impact(self, history: List[Dict]) -> Dict[str, Any]:
        """Calculate business impact metrics"""
        
        business_records = [r for r in history if r["business_metrics"]]
        if not business_records:
            return {"status": "no_business_data"}
        
        # Aggregate business metrics
        total_impact = {}
        for record in business_records:
            for metric, value in record["business_metrics"].items():
                if metric not in total_impact:
                    total_impact[metric] = []
                total_impact[metric].append(value)
        
        # Calculate summaries
        business_summary = {}
        for metric, values in total_impact.items():
            business_summary[metric] = {
                "total": sum(values),
                "average": np.mean(values),
                "trend": "positive" if len(values) > 1 and values[-1] > values[-2] else "stable"
            }
        
        return {
            "metrics": business_summary,
            "total_predictions": sum(len(r.get("business_metrics", {})) for r in business_records),
            "business_value": business_summary.get("revenue_per_prediction", {}).get("total", 0)
        }
    
    def _analyze_drift_trends(self, history: List[Dict]) -> Dict[str, Any]:
        """Analyze drift trends over time"""
        
        drift_records = [r for r in history if r.get("drift_analysis")]
        if not drift_records:
            return {"status": "no_drift_data"}
        
        recent_drift = drift_records[-1]["drift_analysis"]
        drift_events = [r for r in drift_records if r["drift_analysis"].get("drift_detected")]
        
        return {
            "current_status": "drift_detected" if recent_drift.get("drift_detected") else "stable",
            "drift_events_count": len(drift_events),
            "affected_features": recent_drift.get("affected_features", []),
            "latest_score": recent_drift.get("max_drift_score", 0),
            "drift_frequency": len(drift_events) / len(drift_records) if drift_records else 0
        }
    
    def _analyze_feature_importance_trends(self, model_id: str, history: List[Dict]) -> Dict[str, Any]:
        """Analyze how feature importance changes over time"""
        
        importance_records = [r for r in history if r.get("sample_explanations", {}).get("feature_importance")]
        if not importance_records:
            return {"status": "no_importance_data"}
        
        # Get latest feature importance
        latest_importance = importance_records[-1]["sample_explanations"]["feature_importance"]
        
        # Calculate stability of feature importance
        if len(importance_records) > 1:
            previous_importance = importance_records[-2]["sample_explanations"]["feature_importance"]
            
            # Calculate feature importance changes
            importance_changes = {}
            for feature in latest_importance:
                if feature in previous_importance:
                    change = latest_importance[feature] - previous_importance[feature]
                    importance_changes[feature] = change
        else:
            importance_changes = {}
        
        return {
            "current_importance": latest_importance,
            "importance_changes": importance_changes,
            "stable_features": [f for f, change in importance_changes.items() if abs(change) < 0.01],
            "top_features": list(latest_importance.keys())[:5]
        }
    
    def _get_alert_summary(self, model_id: str) -> Dict[str, Any]:
        """Get alert summary for a model"""
        
        model_alerts = [a for a in self.alerts if a["model_id"] == model_id]
        recent_alerts = [a for a in model_alerts if 
                        (datetime.now() - a["timestamp"]).days <= 7]
        
        return {
            "total_alerts": len(model_alerts),
            "recent_alerts": len(recent_alerts),
            "critical_alerts": len([a for a in recent_alerts if a["severity"] == "high"]),
            "alert_types": list(set(a["metric"] for a in recent_alerts)),
            "latest_alerts": recent_alerts[-5:] if recent_alerts else []
        }
    
    def _get_ab_test_results(self, model_id: str) -> Dict[str, Any]:
        """Get A/B test results for a model"""
        
        model_tests = {
            test_id: test for test_id, test in self.ab_tests.items()
            if test["champion_model_id"] == model_id or test["challenger_model_id"] == model_id
        }
        
        return {
            "active_tests": len([t for t in model_tests.values() if t["status"] == "active"]),
            "completed_tests": len([t for t in model_tests.values() if t["status"] == "completed"]),
            "test_results": model_tests
        }
    
    def _generate_dashboard_charts(
        self, 
        performance_trends: Dict, 
        business_impact: Dict, 
        drift_summary: Dict, 
        feature_importance: Dict
    ) -> Dict[str, str]:
        """Generate chart data for the dashboard"""
        
        charts = {}
        
        try:
            # Performance trend chart
            if "trends" in performance_trends:
                fig = go.Figure()
                for metric, trend_data in performance_trends["trends"].items():
                    fig.add_trace(go.Scatter(
                        x=["Previous", "Current"],
                        y=[trend_data["previous"], trend_data["current"]],
                        mode='lines+markers',
                        name=metric.title()
                    ))
                
                fig.update_layout(title="Performance Trends", xaxis_title="Time", yaxis_title="Value")
                charts["performance_trends"] = fig.to_json()
            
            # Feature importance chart
            if "current_importance" in feature_importance:
                importance_data = feature_importance["current_importance"]
                fig = go.Figure(data=[
                    go.Bar(x=list(importance_data.keys()), y=list(importance_data.values()))
                ])
                fig.update_layout(title="Feature Importance", xaxis_title="Features", yaxis_title="Importance")
                charts["feature_importance"] = fig.to_json()
            
        except Exception as e:
            logger.warning("Chart generation failed", error=str(e))
            charts["error"] = str(e)
        
        return charts
    
    def _generate_recommendations(
        self, 
        performance_trends: Dict, 
        business_impact: Dict, 
        drift_summary: Dict, 
        alert_summary: Dict
    ) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on model health"""
        
        recommendations = []
        
        # Performance-based recommendations
        if alert_summary.get("critical_alerts", 0) > 0:
            recommendations.append({
                "type": "critical",
                "title": "Address Critical Performance Issues",
                "description": "Model has critical performance alerts that need immediate attention",
                "action": "Review alert details and consider model retraining or rollback"
            })
        
        # Drift-based recommendations
        if drift_summary.get("current_status") == "drift_detected":
            recommendations.append({
                "type": "warning",
                "title": "Data Drift Detected",
                "description": f"Drift detected in {len(drift_summary.get('affected_features', []))} features",
                "action": "Investigate data sources and consider model retraining with recent data"
            })
        
        # Business impact recommendations
        if business_impact.get("business_value", 0) < 0:
            recommendations.append({
                "type": "business",
                "title": "Negative Business Impact",
                "description": "Model predictions are generating negative business value",
                "action": "Review prediction strategy and model performance"
            })
        
        # Default recommendation if no issues
        if not recommendations:
            recommendations.append({
                "type": "success",
                "title": "Model Health Good",
                "description": "Model is performing within expected parameters",
                "action": "Continue monitoring and maintain current setup"
            })
        
        return recommendations

# Global monitor instance
model_monitor = ModelMonitor()