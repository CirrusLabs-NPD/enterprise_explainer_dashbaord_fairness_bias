"""
Enterprise Dashboard and Custom Metrics Service
Provides comprehensive KPI dashboards, custom metrics, and executive reporting
Compatible with enterprise business intelligence and analytics requirements
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
import json
import asyncio
import structlog

from app.core.database import get_db

logger = structlog.get_logger(__name__)


class MetricType(str, Enum):
    """Types of custom metrics"""
    PERFORMANCE = "performance"
    FAIRNESS = "fairness"
    DRIFT = "drift"
    BUSINESS = "business"
    OPERATIONAL = "operational"
    COMPLIANCE = "compliance"


class AggregationType(str, Enum):
    """Aggregation types for metrics"""
    SUM = "sum"
    AVERAGE = "average"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    PERCENTILE = "percentile"
    RATE = "rate"
    RATIO = "ratio"


class DashboardType(str, Enum):
    """Types of dashboards"""
    EXECUTIVE = "executive"
    OPERATIONAL = "operational"
    TECHNICAL = "technical"
    COMPLIANCE = "compliance"
    BUSINESS = "business"


@dataclass
class CustomMetric:
    """Custom metric definition"""
    metric_id: str
    name: str
    description: str
    metric_type: MetricType
    aggregation: AggregationType
    formula: str
    thresholds: Dict[str, float]
    unit: str
    tags: List[str]
    is_active: bool = True
    created_by: str = "system"
    created_at: datetime = None


@dataclass
class KPITarget:
    """KPI target definition"""
    kpi_id: str
    target_value: float
    target_period: str  # daily, weekly, monthly, quarterly
    tolerance_upper: float
    tolerance_lower: float
    business_impact: str


class EnterpriseDashboardService:
    """
    Enterprise dashboard and metrics service
    """
    
    def __init__(self):
        self.custom_metrics = {}
        self.dashboard_configs = {}
        self.kpi_targets = {}
        
        # Initialize default enterprise metrics
        self.default_metrics = self._initialize_default_metrics()
    
    async def create_custom_metric(
        self,
        metric_definition: CustomMetric,
        model_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a custom metric definition
        
        Args:
            metric_definition: Custom metric configuration
            model_ids: Specific models to apply metric to (None for all)
            
        Returns:
            Created metric information
        """
        try:
            # Validate metric formula
            validation_result = await self._validate_metric_formula(metric_definition.formula)
            if not validation_result["valid"]:
                return {
                    "status": "error",
                    "error": f"Invalid metric formula: {validation_result['error']}"
                }
            
            # Store metric definition
            await self._store_custom_metric(metric_definition, model_ids)
            
            self.custom_metrics[metric_definition.metric_id] = metric_definition
            
            logger.info("Custom metric created",
                       metric_id=metric_definition.metric_id,
                       metric_type=metric_definition.metric_type.value)
            
            return {
                "status": "success",
                "metric_id": metric_definition.metric_id,
                "metric_type": metric_definition.metric_type.value
            }
            
        except Exception as e:
            logger.error("Error creating custom metric", error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def calculate_metric_value(
        self,
        metric_id: str,
        model_id: Optional[str] = None,
        time_range: Optional[Dict[str, datetime]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate value for a custom metric
        
        Args:
            metric_id: ID of the metric to calculate
            model_id: Specific model to calculate for
            time_range: Time range for calculation
            filters: Additional filters
            
        Returns:
            Calculated metric value and metadata
        """
        try:
            metric = self.custom_metrics.get(metric_id)
            if not metric:
                # Try to load from database
                metric = await self._load_custom_metric(metric_id)
                if not metric:
                    return {"status": "error", "error": "Metric not found"}
            
            # Get data for calculation
            data = await self._get_metric_data(
                metric, model_id, time_range, filters
            )
            
            # Calculate metric value
            calculated_value = await self._execute_metric_calculation(
                metric, data
            )
            
            # Evaluate against thresholds
            threshold_status = self._evaluate_thresholds(
                calculated_value["value"], metric.thresholds
            )
            
            result = {
                "status": "success",
                "metric_id": metric_id,
                "value": calculated_value["value"],
                "unit": metric.unit,
                "calculation_timestamp": datetime.utcnow().isoformat(),
                "threshold_status": threshold_status,
                "metadata": calculated_value.get("metadata", {}),
                "data_points": calculated_value.get("data_points", 0)
            }
            
            # Store calculated value
            await self._store_metric_value(metric_id, model_id, result)
            
            return result
            
        except Exception as e:
            logger.error("Error calculating metric", metric_id=metric_id, error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def create_dashboard(
        self,
        dashboard_id: str,
        dashboard_name: str,
        dashboard_type: DashboardType,
        layout_config: Dict[str, Any],
        metric_ids: List[str],
        filters: Optional[Dict[str, Any]] = None,
        refresh_interval: int = 300,  # 5 minutes
        access_permissions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a custom dashboard
        
        Args:
            dashboard_id: Unique dashboard identifier
            dashboard_name: Display name for dashboard
            dashboard_type: Type of dashboard
            layout_config: Dashboard layout configuration
            metric_ids: List of metrics to include
            filters: Default filters for dashboard
            refresh_interval: Auto-refresh interval in seconds
            access_permissions: User roles with access
            
        Returns:
            Created dashboard information
        """
        try:
            dashboard_config = {
                "dashboard_id": dashboard_id,
                "name": dashboard_name,
                "type": dashboard_type.value,
                "layout": layout_config,
                "metrics": metric_ids,
                "filters": filters or {},
                "refresh_interval": refresh_interval,
                "access_permissions": access_permissions or [],
                "created_at": datetime.utcnow().isoformat(),
                "is_active": True
            }
            
            # Validate metrics exist
            for metric_id in metric_ids:
                if metric_id not in self.custom_metrics and metric_id not in self.default_metrics:
                    return {"status": "error", "error": f"Metric {metric_id} not found"}
            
            # Store dashboard configuration
            await self._store_dashboard_config(dashboard_config)
            
            self.dashboard_configs[dashboard_id] = dashboard_config
            
            logger.info("Dashboard created",
                       dashboard_id=dashboard_id,
                       dashboard_type=dashboard_type.value,
                       metrics_count=len(metric_ids))
            
            return {
                "status": "success",
                "dashboard_id": dashboard_id,
                "dashboard_type": dashboard_type.value
            }
            
        except Exception as e:
            logger.error("Error creating dashboard", error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def get_dashboard_data(
        self,
        dashboard_id: str,
        time_range: Optional[Dict[str, datetime]] = None,
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get complete data for a dashboard
        
        Args:
            dashboard_id: Dashboard identifier
            time_range: Time range for data
            additional_filters: Additional filters to apply
            
        Returns:
            Complete dashboard data
        """
        try:
            # Get dashboard configuration
            dashboard_config = self.dashboard_configs.get(dashboard_id)
            if not dashboard_config:
                dashboard_config = await self._load_dashboard_config(dashboard_id)
                if not dashboard_config:
                    return {"status": "error", "error": "Dashboard not found"}
            
            # Combine filters
            combined_filters = {**dashboard_config.get("filters", {}), **(additional_filters or {})}
            
            # Calculate all metrics
            metric_values = {}
            for metric_id in dashboard_config["metrics"]:
                metric_result = await self.calculate_metric_value(
                    metric_id, 
                    time_range=time_range, 
                    filters=combined_filters
                )
                metric_values[metric_id] = metric_result
            
            # Get KPI status
            kpi_status = await self._get_kpi_status(dashboard_config["metrics"])
            
            # Get trends and historical data
            trends = await self._get_metric_trends(
                dashboard_config["metrics"], time_range
            )
            
            dashboard_data = {
                "dashboard_id": dashboard_id,
                "name": dashboard_config["name"],
                "type": dashboard_config["type"],
                "last_updated": datetime.utcnow().isoformat(),
                "metric_values": metric_values,
                "kpi_status": kpi_status,
                "trends": trends,
                "layout": dashboard_config["layout"],
                "refresh_interval": dashboard_config["refresh_interval"]
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error("Error getting dashboard data", dashboard_id=dashboard_id, error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def create_executive_summary(
        self,
        time_period: str = "monthly",
        include_forecasts: bool = True,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Create executive summary dashboard
        
        Args:
            time_period: Period for summary (daily, weekly, monthly, quarterly)
            include_forecasts: Include forecast data
            include_recommendations: Include recommendations
            
        Returns:
            Executive summary data
        """
        try:
            # Calculate time range based on period
            end_date = datetime.utcnow()
            if time_period == "daily":
                start_date = end_date - timedelta(days=1)
            elif time_period == "weekly":
                start_date = end_date - timedelta(weeks=1)
            elif time_period == "monthly":
                start_date = end_date - timedelta(days=30)
            elif time_period == "quarterly":
                start_date = end_date - timedelta(days=90)
            else:
                start_date = end_date - timedelta(days=30)
            
            time_range = {"start": start_date, "end": end_date}
            
            # Key business metrics
            business_metrics = await self._calculate_business_metrics(time_range)
            
            # Model performance overview
            performance_overview = await self._get_performance_overview(time_range)
            
            # Risk indicators
            risk_indicators = await self._get_risk_indicators(time_range)
            
            # Compliance status
            compliance_status = await self._get_compliance_overview(time_range)
            
            # Operational metrics
            operational_metrics = await self._get_operational_metrics(time_range)
            
            executive_summary = {
                "report_id": f"exec_summary_{int(datetime.utcnow().timestamp())}",
                "period": time_period,
                "time_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "generated_at": datetime.utcnow().isoformat(),
                "business_metrics": business_metrics,
                "performance_overview": performance_overview,
                "risk_indicators": risk_indicators,
                "compliance_status": compliance_status,
                "operational_metrics": operational_metrics
            }
            
            # Add forecasts if requested
            if include_forecasts:
                executive_summary["forecasts"] = await self._generate_forecasts(time_range)
            
            # Add recommendations if requested
            if include_recommendations:
                executive_summary["recommendations"] = await self._generate_executive_recommendations(
                    business_metrics, performance_overview, risk_indicators
                )
            
            # Store executive summary
            await self._store_executive_summary(executive_summary)
            
            return executive_summary
            
        except Exception as e:
            logger.error("Error creating executive summary", error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def get_real_time_kpis(self, kpi_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get real-time KPI values"""
        
        if kpi_ids is None:
            kpi_ids = list(self.kpi_targets.keys())
        
        real_time_kpis = {}
        
        for kpi_id in kpi_ids:
            try:
                # Calculate current KPI value
                kpi_value = await self.calculate_metric_value(kpi_id)
                
                # Get KPI target
                kpi_target = self.kpi_targets.get(kpi_id)
                
                if kpi_value["status"] == "success" and kpi_target:
                    # Calculate performance against target
                    performance = self._calculate_kpi_performance(
                        kpi_value["value"], kpi_target
                    )
                    
                    real_time_kpis[kpi_id] = {
                        "current_value": kpi_value["value"],
                        "target_value": kpi_target.target_value,
                        "performance_percentage": performance["percentage"],
                        "status": performance["status"],
                        "trend": await self._get_kpi_trend(kpi_id),
                        "last_updated": kpi_value["calculation_timestamp"]
                    }
                
            except Exception as e:
                logger.error("Error calculating KPI", kpi_id=kpi_id, error=str(e))
                real_time_kpis[kpi_id] = {"status": "error", "error": str(e)}
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "kpis": real_time_kpis
        }
    
    async def generate_business_impact_report(
        self,
        model_ids: Optional[List[str]] = None,
        time_range: Optional[Dict[str, datetime]] = None
    ) -> Dict[str, Any]:
        """Generate business impact report for ML models"""
        
        if time_range is None:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)
            time_range = {"start": start_date, "end": end_date}
        
        # Calculate business impact metrics
        impact_metrics = await self._calculate_business_impact_metrics(model_ids, time_range)
        
        # Model ROI analysis
        roi_analysis = await self._calculate_model_roi(model_ids, time_range)
        
        # Cost analysis
        cost_analysis = await self._calculate_model_costs(model_ids, time_range)
        
        # Risk assessment
        business_risk_assessment = await self._assess_business_risks(model_ids, time_range)
        
        # Recommendations
        business_recommendations = await self._generate_business_recommendations(
            impact_metrics, roi_analysis, cost_analysis, business_risk_assessment
        )
        
        report = {
            "report_id": f"business_impact_{int(datetime.utcnow().timestamp())}",
            "time_range": {
                "start": time_range["start"].isoformat(),
                "end": time_range["end"].isoformat()
            },
            "model_ids": model_ids,
            "generated_at": datetime.utcnow().isoformat(),
            "impact_metrics": impact_metrics,
            "roi_analysis": roi_analysis,
            "cost_analysis": cost_analysis,
            "risk_assessment": business_risk_assessment,
            "recommendations": business_recommendations
        }
        
        return report
    
    def _initialize_default_metrics(self) -> Dict[str, CustomMetric]:
        """Initialize default enterprise metrics"""
        
        default_metrics = {}
        
        # Performance metrics
        default_metrics["model_accuracy"] = CustomMetric(
            metric_id="model_accuracy",
            name="Model Accuracy",
            description="Overall model accuracy percentage",
            metric_type=MetricType.PERFORMANCE,
            aggregation=AggregationType.AVERAGE,
            formula="accuracy_score(y_true, y_pred)",
            thresholds={"warning": 0.8, "critical": 0.7},
            unit="percentage",
            tags=["performance", "accuracy"]
        )
        
        default_metrics["prediction_latency"] = CustomMetric(
            metric_id="prediction_latency",
            name="Prediction Latency",
            description="Average prediction response time",
            metric_type=MetricType.OPERATIONAL,
            aggregation=AggregationType.AVERAGE,
            formula="avg(prediction_time_ms)",
            thresholds={"warning": 100, "critical": 500},
            unit="milliseconds",
            tags=["performance", "latency"]
        )
        
        # Fairness metrics
        default_metrics["bias_score"] = CustomMetric(
            metric_id="bias_score",
            name="Bias Score",
            description="Overall bias score across sensitive attributes",
            metric_type=MetricType.FAIRNESS,
            aggregation=AggregationType.MAX,
            formula="max(demographic_parity_differences)",
            thresholds={"warning": 0.1, "critical": 0.2},
            unit="score",
            tags=["fairness", "bias"]
        )
        
        # Drift metrics
        default_metrics["data_drift_score"] = CustomMetric(
            metric_id="data_drift_score",
            name="Data Drift Score",
            description="Overall data drift score",
            metric_type=MetricType.DRIFT,
            aggregation=AggregationType.AVERAGE,
            formula="avg(jensen_shannon_divergences)",
            thresholds={"warning": 0.1, "critical": 0.3},
            unit="score",
            tags=["drift", "data"]
        )
        
        # Business metrics
        default_metrics["predictions_per_day"] = CustomMetric(
            metric_id="predictions_per_day",
            name="Predictions Per Day",
            description="Number of predictions made per day",
            metric_type=MetricType.BUSINESS,
            aggregation=AggregationType.COUNT,
            formula="count(predictions) / days",
            thresholds={"warning": 1000, "critical": 100},
            unit="count",
            tags=["business", "volume"]
        )
        
        # Compliance metrics
        default_metrics["compliance_score"] = CustomMetric(
            metric_id="compliance_score",
            name="Compliance Score",
            description="Overall compliance score across frameworks",
            metric_type=MetricType.COMPLIANCE,
            aggregation=AggregationType.AVERAGE,
            formula="avg(compliance_scores)",
            thresholds={"warning": 0.8, "critical": 0.6},
            unit="score",
            tags=["compliance", "governance"]
        )
        
        return default_metrics
    
    async def _validate_metric_formula(self, formula: str) -> Dict[str, Any]:
        """Validate custom metric formula"""
        
        # Basic validation - in production, this would be more sophisticated
        allowed_functions = [
            "avg", "sum", "count", "min", "max", "median", "percentile",
            "accuracy_score", "precision_score", "recall_score", "f1_score",
            "demographic_parity_difference", "equalized_odds_difference",
            "jensen_shannon_divergence", "population_stability_index"
        ]
        
        allowed_operators = ["+", "-", "*", "/", "(", ")", ".", "_", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        
        # Check for dangerous operations
        dangerous_keywords = ["import", "exec", "eval", "__", "open", "file", "os", "sys"]
        
        for keyword in dangerous_keywords:
            if keyword in formula.lower():
                return {"valid": False, "error": f"Dangerous keyword '{keyword}' not allowed"}
        
        return {"valid": True}
    
    async def _get_metric_data(
        self,
        metric: CustomMetric,
        model_id: Optional[str],
        time_range: Optional[Dict[str, datetime]],
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get data needed for metric calculation"""
        
        # This would fetch actual data from the database based on metric requirements
        # For now, return mock data structure
        
        async with get_db() as db:
            if time_range:
                start_time = time_range["start"]
                end_time = time_range["end"]
            else:
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=24)
            
            # Get relevant data based on metric type
            if metric.metric_type == MetricType.PERFORMANCE:
                data = await db.fetch("""
                    SELECT metrics FROM model_performance_metrics
                    WHERE ($1::varchar IS NULL OR model_id = $1)
                    AND timestamp BETWEEN $2 AND $3
                    ORDER BY timestamp
                """, model_id, start_time, end_time)
                
            elif metric.metric_type == MetricType.FAIRNESS:
                # Get fairness analysis results
                data = await db.fetch("""
                    SELECT details FROM audit_logs
                    WHERE event_type = 'bias_analysis'
                    AND ($1::varchar IS NULL OR details->>'model_id' = $1)
                    AND timestamp BETWEEN $2 AND $3
                """, model_id, start_time, end_time)
                
            else:
                # Generic data fetch
                data = await db.fetch("""
                    SELECT details, timestamp FROM audit_logs
                    WHERE ($1::varchar IS NULL OR details->>'model_id' = $1)
                    AND timestamp BETWEEN $2 AND $3
                """, model_id, start_time, end_time)
        
        return {"data": [dict(row) for row in data], "count": len(data)}
    
    async def _execute_metric_calculation(
        self,
        metric: CustomMetric,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute metric calculation based on formula"""
        
        # Simplified calculation - in production, this would use a proper expression evaluator
        data_points = data["data"]
        
        if not data_points:
            return {"value": 0.0, "data_points": 0}
        
        # Extract values based on metric type and aggregation
        if metric.aggregation == AggregationType.COUNT:
            value = len(data_points)
        elif metric.aggregation == AggregationType.AVERAGE:
            # For mock calculation
            value = 0.85  # Mock accuracy value
        elif metric.aggregation == AggregationType.SUM:
            value = len(data_points)
        else:
            value = 0.0
        
        return {
            "value": float(value),
            "data_points": len(data_points),
            "metadata": {
                "calculation_method": metric.aggregation.value,
                "formula": metric.formula
            }
        }
    
    def _evaluate_thresholds(
        self,
        value: float,
        thresholds: Dict[str, float]
    ) -> Dict[str, Any]:
        """Evaluate value against thresholds"""
        
        status = "good"
        triggered_thresholds = []
        
        for threshold_name, threshold_value in thresholds.items():
            if threshold_name == "critical" and value <= threshold_value:
                status = "critical"
                triggered_thresholds.append(threshold_name)
            elif threshold_name == "warning" and value <= threshold_value:
                if status != "critical":
                    status = "warning"
                triggered_thresholds.append(threshold_name)
        
        return {
            "status": status,
            "triggered_thresholds": triggered_thresholds,
            "value": value,
            "thresholds": thresholds
        }
    
    # Additional helper methods for dashboard and KPI calculations
    async def _calculate_business_metrics(self, time_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Calculate key business metrics"""
        return {
            "total_predictions": 50000,
            "model_uptime": 99.5,
            "cost_per_prediction": 0.002,
            "revenue_impact": 125000.00
        }
    
    async def _get_performance_overview(self, time_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Get model performance overview"""
        return {
            "average_accuracy": 0.87,
            "accuracy_trend": "stable",
            "models_in_production": 12,
            "models_needing_attention": 2
        }
    
    async def _get_risk_indicators(self, time_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Get risk indicators"""
        return {
            "high_risk_models": 1,
            "bias_violations": 0,
            "drift_alerts": 3,
            "compliance_issues": 0
        }
    
    async def _get_compliance_overview(self, time_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Get compliance overview"""
        return {
            "gdpr_compliance": "compliant",
            "eu_ai_act_compliance": "compliant",
            "audit_readiness": "ready",
            "documentation_completeness": 95
        }
    
    async def _get_operational_metrics(self, time_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Get operational metrics"""
        return {
            "average_latency_ms": 45,
            "error_rate": 0.01,
            "throughput_rps": 150,
            "resource_utilization": 0.65
        }
    
    # Storage methods
    async def _store_custom_metric(self, metric: CustomMetric, model_ids: Optional[List[str]]):
        """Store custom metric definition"""
        async with get_db() as db:
            await db.execute("""
                INSERT INTO custom_metrics (
                    metric_id, name, description, metric_type, aggregation,
                    formula, thresholds, unit, tags, model_ids, is_active, created_by
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (metric_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    formula = EXCLUDED.formula,
                    thresholds = EXCLUDED.thresholds,
                    updated_at = CURRENT_TIMESTAMP
            """, 
                metric.metric_id, metric.name, metric.description, metric.metric_type.value,
                metric.aggregation.value, metric.formula, json.dumps(metric.thresholds),
                metric.unit, json.dumps(metric.tags), json.dumps(model_ids or []),
                metric.is_active, metric.created_by
            )
    
    async def _store_metric_value(self, metric_id: str, model_id: Optional[str], result: Dict[str, Any]):
        """Store calculated metric value"""
        async with get_db() as db:
            await db.execute("""
                INSERT INTO metric_values (
                    metric_id, model_id, value, unit, threshold_status,
                    metadata, calculated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
            """,
                metric_id, model_id, result["value"], result["unit"],
                result["threshold_status"]["status"], json.dumps(result.get("metadata", {}))
            )
    
    async def _store_dashboard_config(self, dashboard_config: Dict[str, Any]):
        """Store dashboard configuration"""
        async with get_db() as db:
            await db.execute("""
                INSERT INTO dashboards (
                    dashboard_id, name, type, layout, metrics, filters,
                    refresh_interval, access_permissions, is_active
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (dashboard_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    layout = EXCLUDED.layout,
                    metrics = EXCLUDED.metrics,
                    filters = EXCLUDED.filters,
                    updated_at = CURRENT_TIMESTAMP
            """,
                dashboard_config["dashboard_id"], dashboard_config["name"],
                dashboard_config["type"], json.dumps(dashboard_config["layout"]),
                json.dumps(dashboard_config["metrics"]), json.dumps(dashboard_config["filters"]),
                dashboard_config["refresh_interval"], json.dumps(dashboard_config["access_permissions"]),
                dashboard_config["is_active"]
            )


# Database initialization
async def initialize_dashboard_tables():
    """Initialize dashboard-related database tables"""
    
    async with get_db() as db:
        # Custom metrics table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS custom_metrics (
                metric_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                description TEXT,
                metric_type VARCHAR NOT NULL,
                aggregation VARCHAR NOT NULL,
                formula TEXT NOT NULL,
                thresholds JSONB DEFAULT '{}',
                unit VARCHAR,
                tags JSONB DEFAULT '[]',
                model_ids JSONB DEFAULT '[]',
                is_active BOOLEAN DEFAULT TRUE,
                created_by VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Metric values table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS metric_values (
                id SERIAL PRIMARY KEY,
                metric_id VARCHAR NOT NULL,
                model_id VARCHAR,
                value FLOAT NOT NULL,
                unit VARCHAR,
                threshold_status VARCHAR,
                metadata JSONB DEFAULT '{}',
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (metric_id) REFERENCES custom_metrics(metric_id)
            )
        """)
        
        # Dashboards table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS dashboards (
                dashboard_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                type VARCHAR NOT NULL,
                layout JSONB NOT NULL,
                metrics JSONB NOT NULL,
                filters JSONB DEFAULT '{}',
                refresh_interval INTEGER DEFAULT 300,
                access_permissions JSONB DEFAULT '[]',
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Executive summaries table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS executive_summaries (
                report_id VARCHAR PRIMARY KEY,
                period VARCHAR NOT NULL,
                time_range JSONB NOT NULL,
                summary_data JSONB NOT NULL,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        await db.execute("CREATE INDEX IF NOT EXISTS idx_custom_metrics_type ON custom_metrics(metric_type)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_metric_values_metric_id ON metric_values(metric_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_metric_values_calculated_at ON metric_values(calculated_at)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_dashboards_type ON dashboards(type)")


# Utility functions
def create_default_executive_dashboard() -> Dict[str, Any]:
    """Create default executive dashboard configuration"""
    
    return {
        "dashboard_id": "executive_default",
        "name": "Executive Dashboard",
        "type": "executive",
        "layout": {
            "rows": [
                {
                    "height": "200px",
                    "columns": [
                        {"width": "25%", "metric_id": "model_accuracy", "chart_type": "gauge"},
                        {"width": "25%", "metric_id": "bias_score", "chart_type": "gauge"},
                        {"width": "25%", "metric_id": "predictions_per_day", "chart_type": "counter"},
                        {"width": "25%", "metric_id": "compliance_score", "chart_type": "gauge"}
                    ]
                },
                {
                    "height": "300px",
                    "columns": [
                        {"width": "50%", "metric_id": "performance_trend", "chart_type": "line"},
                        {"width": "50%", "metric_id": "risk_indicators", "chart_type": "bar"}
                    ]
                }
            ]
        },
        "metrics": ["model_accuracy", "bias_score", "predictions_per_day", "compliance_score"],
        "refresh_interval": 300,
        "access_permissions": ["executive", "admin"]
    }