"""
Enterprise Root Cause Analysis Service
Advanced automated root cause analysis for ML model performance issues, bias, and drift
Compatible with enterprise-grade debugging and incident response workflows
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from dataclasses import dataclass
import json
import structlog

logger = structlog.get_logger(__name__)


class IssueType(str, Enum):
    """Types of issues for root cause analysis"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    BIAS_DETECTION = "bias_detection"
    DATA_DRIFT = "data_drift"
    PREDICTION_ANOMALY = "prediction_anomaly"
    FAIRNESS_VIOLATION = "fairness_violation"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    SYSTEM_ERROR = "system_error"


class RootCauseSeverity(str, Enum):
    """Severity levels for root causes"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RootCauseHypothesis:
    """Hypothesis for root cause analysis"""
    hypothesis_id: str
    category: str
    description: str
    confidence_score: float
    evidence: List[Dict[str, Any]]
    severity: RootCauseSeverity
    suggested_actions: List[str]
    related_features: List[str]
    temporal_correlation: Optional[float] = None


class EnterpriseRootCauseAnalyzer:
    """
    Enterprise root cause analysis engine
    """
    
    def __init__(self):
        self.analysis_strategies = {
            IssueType.PERFORMANCE_DEGRADATION: self._analyze_performance_degradation,
            IssueType.BIAS_DETECTION: self._analyze_bias_issues,
            IssueType.DATA_DRIFT: self._analyze_data_drift_causes,
            IssueType.PREDICTION_ANOMALY: self._analyze_prediction_anomalies,
            IssueType.FAIRNESS_VIOLATION: self._analyze_fairness_violations,
            IssueType.DATA_QUALITY_ISSUE: self._analyze_data_quality_issues,
            IssueType.SYSTEM_ERROR: self._analyze_system_errors
        }
        
        self.feature_importance_threshold = 0.1
        self.correlation_threshold = 0.7
        self.drift_threshold = 0.1
    
    async def analyze_root_cause(
        self,
        issue_type: IssueType,
        model_id: str,
        issue_description: str,
        current_data: Optional[pd.DataFrame] = None,
        reference_data: Optional[pd.DataFrame] = None,
        model_predictions: Optional[np.ndarray] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
        historical_metrics: Optional[Dict[str, List[float]]] = None,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Perform comprehensive root cause analysis
        
        Args:
            issue_type: Type of issue to analyze
            model_id: Identifier of the affected model
            issue_description: Description of the observed issue
            current_data: Current dataset
            reference_data: Reference/training dataset
            model_predictions: Model predictions
            model_metadata: Model metadata and configuration
            historical_metrics: Historical performance metrics
            time_range_hours: Time range for historical analysis
            
        Returns:
            Comprehensive root cause analysis report
        """
        logger.info("Starting root cause analysis",
                   issue_type=issue_type.value,
                   model_id=model_id,
                   time_range_hours=time_range_hours)
        
        analysis_report = {
            "analysis_id": f"rca_{model_id}_{int(datetime.utcnow().timestamp())}",
            "model_id": model_id,
            "issue_type": issue_type.value,
            "issue_description": issue_description,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "time_range_analyzed": time_range_hours,
            "hypotheses": [],
            "primary_root_cause": None,
            "contributing_factors": [],
            "recommendations": [],
            "confidence_assessment": {},
            "supporting_evidence": {},
            "next_steps": []
        }
        
        # Get historical context
        historical_context = await self._gather_historical_context(
            model_id, time_range_hours
        )
        
        # Perform issue-specific analysis
        strategy = self.analysis_strategies.get(issue_type)
        if strategy:
            hypotheses = await strategy(
                model_id=model_id,
                current_data=current_data,
                reference_data=reference_data,
                model_predictions=model_predictions,
                model_metadata=model_metadata,
                historical_metrics=historical_metrics,
                historical_context=historical_context
            )
            
            analysis_report["hypotheses"] = [
                self._hypothesis_to_dict(h) for h in hypotheses
            ]
        
        # Cross-cutting analysis
        cross_cutting_hypotheses = await self._perform_cross_cutting_analysis(
            model_id, current_data, reference_data, historical_context
        )
        
        analysis_report["hypotheses"].extend([
            self._hypothesis_to_dict(h) for h in cross_cutting_hypotheses
        ])
        
        # Rank and prioritize hypotheses
        ranked_hypotheses = self._rank_hypotheses(analysis_report["hypotheses"])
        analysis_report["hypotheses"] = ranked_hypotheses
        
        # Identify primary root cause
        if ranked_hypotheses:
            analysis_report["primary_root_cause"] = ranked_hypotheses[0]
            analysis_report["contributing_factors"] = ranked_hypotheses[1:3]  # Top 3 contributing factors
        
        # Generate recommendations
        analysis_report["recommendations"] = await self._generate_recommendations(
            issue_type, ranked_hypotheses, model_metadata
        )
        
        # Assess overall confidence
        analysis_report["confidence_assessment"] = self._assess_analysis_confidence(
            ranked_hypotheses, current_data, reference_data
        )
        
        # Generate supporting evidence
        analysis_report["supporting_evidence"] = await self._generate_supporting_evidence(
            ranked_hypotheses, current_data, reference_data, historical_context
        )
        
        # Define next steps
        analysis_report["next_steps"] = self._define_next_steps(
            issue_type, ranked_hypotheses, analysis_report["confidence_assessment"]
        )
        
        # Store analysis results
        await self._store_analysis_results(analysis_report)
        
        logger.info("Root cause analysis completed",
                   analysis_id=analysis_report["analysis_id"],
                   primary_cause=analysis_report["primary_root_cause"]["category"] if analysis_report["primary_root_cause"] else None,
                   hypotheses_count=len(analysis_report["hypotheses"]))
        
        return analysis_report
    
    async def _analyze_performance_degradation(
        self,
        model_id: str,
        current_data: Optional[pd.DataFrame],
        reference_data: Optional[pd.DataFrame],
        model_predictions: Optional[np.ndarray],
        model_metadata: Optional[Dict[str, Any]],
        historical_metrics: Optional[Dict[str, List[float]]],
        historical_context: Dict[str, Any]
    ) -> List[RootCauseHypothesis]:
        """Analyze performance degradation issues"""
        
        hypotheses = []
        
        # Hypothesis 1: Data drift causing performance issues
        if current_data is not None and reference_data is not None:
            drift_analysis = await self._analyze_feature_drift_impact(
                current_data, reference_data, model_metadata
            )
            
            if drift_analysis["significant_drift_detected"]:
                hypotheses.append(RootCauseHypothesis(
                    hypothesis_id="perf_001",
                    category="data_drift",
                    description="Significant data drift detected in key features affecting model performance",
                    confidence_score=drift_analysis["confidence"],
                    evidence=[
                        {
                            "type": "feature_drift",
                            "features_affected": drift_analysis["drifted_features"],
                            "drift_scores": drift_analysis["drift_scores"]
                        }
                    ],
                    severity=RootCauseSeverity.HIGH,
                    suggested_actions=[
                        "Retrain model with recent data",
                        "Implement feature-specific drift monitoring",
                        "Review data collection pipeline"
                    ],
                    related_features=drift_analysis["drifted_features"]
                ))
        
        # Hypothesis 2: Feature importance changes
        if historical_context.get("feature_importance_history"):
            importance_analysis = self._analyze_feature_importance_changes(
                historical_context["feature_importance_history"]
            )
            
            if importance_analysis["significant_changes"]:
                hypotheses.append(RootCauseHypothesis(
                    hypothesis_id="perf_002",
                    category="feature_importance_shift",
                    description="Key feature importance has shifted significantly",
                    confidence_score=importance_analysis["confidence"],
                    evidence=[
                        {
                            "type": "importance_change",
                            "changed_features": importance_analysis["changed_features"],
                            "importance_deltas": importance_analysis["deltas"]
                        }
                    ],
                    severity=RootCauseSeverity.MEDIUM,
                    suggested_actions=[
                        "Investigate feature engineering pipeline",
                        "Review feature selection process",
                        "Consider feature reweighting"
                    ],
                    related_features=importance_analysis["changed_features"]
                ))
        
        # Hypothesis 3: Data quality degradation
        if current_data is not None:
            quality_analysis = self._analyze_data_quality_degradation(current_data, reference_data)
            
            if quality_analysis["quality_degraded"]:
                hypotheses.append(RootCauseHypothesis(
                    hypothesis_id="perf_003",
                    category="data_quality",
                    description="Data quality has degraded affecting model performance",
                    confidence_score=quality_analysis["confidence"],
                    evidence=[
                        {
                            "type": "quality_metrics",
                            "missing_values_increase": quality_analysis["missing_values_delta"],
                            "outliers_increase": quality_analysis["outliers_delta"],
                            "affected_columns": quality_analysis["affected_columns"]
                        }
                    ],
                    severity=RootCauseSeverity.HIGH,
                    suggested_actions=[
                        "Review data preprocessing pipeline",
                        "Implement data validation checks",
                        "Monitor data quality metrics"
                    ],
                    related_features=quality_analysis["affected_columns"]
                ))
        
        # Hypothesis 4: Temporal/seasonal patterns
        if historical_metrics:
            temporal_analysis = self._analyze_temporal_patterns(historical_metrics)
            
            if temporal_analysis["seasonal_pattern_detected"]:
                hypotheses.append(RootCauseHypothesis(
                    hypothesis_id="perf_004",
                    category="temporal_pattern",
                    description="Performance follows seasonal or temporal patterns",
                    confidence_score=temporal_analysis["confidence"],
                    evidence=[
                        {
                            "type": "temporal_correlation",
                            "pattern_type": temporal_analysis["pattern_type"],
                            "correlation_strength": temporal_analysis["correlation"]
                        }
                    ],
                    severity=RootCauseSeverity.MEDIUM,
                    suggested_actions=[
                        "Implement time-aware model features",
                        "Consider separate models for different time periods",
                        "Add temporal features to model"
                    ],
                    related_features=[],
                    temporal_correlation=temporal_analysis["correlation"]
                ))
        
        return hypotheses
    
    async def _analyze_bias_issues(
        self,
        model_id: str,
        current_data: Optional[pd.DataFrame],
        reference_data: Optional[pd.DataFrame],
        model_predictions: Optional[np.ndarray],
        model_metadata: Optional[Dict[str, Any]],
        historical_metrics: Optional[Dict[str, List[float]]],
        historical_context: Dict[str, Any]
    ) -> List[RootCauseHypothesis]:
        """Analyze bias detection issues"""
        
        hypotheses = []
        
        # Hypothesis 1: Training data bias
        if reference_data is not None:
            training_bias_analysis = self._analyze_training_data_bias(reference_data, model_metadata)
            
            if training_bias_analysis["bias_detected"]:
                hypotheses.append(RootCauseHypothesis(
                    hypothesis_id="bias_001",
                    category="training_data_bias",
                    description="Bias present in training data affecting model fairness",
                    confidence_score=training_bias_analysis["confidence"],
                    evidence=[
                        {
                            "type": "training_bias",
                            "biased_attributes": training_bias_analysis["biased_attributes"],
                            "representation_gaps": training_bias_analysis["representation_gaps"]
                        }
                    ],
                    severity=RootCauseSeverity.HIGH,
                    suggested_actions=[
                        "Collect more representative training data",
                        "Apply bias mitigation preprocessing",
                        "Use fairness-aware sampling techniques"
                    ],
                    related_features=training_bias_analysis["biased_attributes"]
                ))
        
        # Hypothesis 2: Feature correlation with protected attributes
        if current_data is not None and model_metadata:
            correlation_analysis = self._analyze_protected_attribute_correlations(
                current_data, model_metadata
            )
            
            if correlation_analysis["high_correlations_found"]:
                hypotheses.append(RootCauseHypothesis(
                    hypothesis_id="bias_002",
                    category="feature_correlation",
                    description="Model features are highly correlated with protected attributes",
                    confidence_score=correlation_analysis["confidence"],
                    evidence=[
                        {
                            "type": "feature_correlation",
                            "correlated_features": correlation_analysis["correlated_features"],
                            "correlation_strengths": correlation_analysis["correlations"]
                        }
                    ],
                    severity=RootCauseSeverity.MEDIUM,
                    suggested_actions=[
                        "Remove or transform correlated features",
                        "Apply bias mitigation algorithms",
                        "Use fairness constraints during training"
                    ],
                    related_features=correlation_analysis["correlated_features"]
                ))
        
        # Hypothesis 3: Population drift in sensitive groups
        if current_data is not None and reference_data is not None:
            population_drift_analysis = self._analyze_sensitive_group_drift(
                current_data, reference_data, model_metadata
            )
            
            if population_drift_analysis["drift_detected"]:
                hypotheses.append(RootCauseHypothesis(
                    hypothesis_id="bias_003",
                    category="population_drift",
                    description="Population distribution of sensitive groups has changed",
                    confidence_score=population_drift_analysis["confidence"],
                    evidence=[
                        {
                            "type": "population_shift",
                            "shifted_groups": population_drift_analysis["shifted_groups"],
                            "drift_magnitudes": population_drift_analysis["drift_magnitudes"]
                        }
                    ],
                    severity=RootCauseSeverity.HIGH,
                    suggested_actions=[
                        "Rebalance model training data",
                        "Apply post-processing bias correction",
                        "Monitor population distribution continuously"
                    ],
                    related_features=population_drift_analysis["sensitive_attributes"]
                ))
        
        return hypotheses
    
    async def _analyze_data_drift_causes(
        self,
        model_id: str,
        current_data: Optional[pd.DataFrame],
        reference_data: Optional[pd.DataFrame],
        model_predictions: Optional[np.ndarray],
        model_metadata: Optional[Dict[str, Any]],
        historical_metrics: Optional[Dict[str, List[float]]],
        historical_context: Dict[str, Any]
    ) -> List[RootCauseHypothesis]:
        """Analyze data drift root causes"""
        
        hypotheses = []
        
        # Hypothesis 1: External environmental changes
        if historical_context.get("external_events"):
            external_analysis = self._analyze_external_event_correlation(
                historical_context["external_events"], historical_metrics
            )
            
            if external_analysis["correlation_found"]:
                hypotheses.append(RootCauseHypothesis(
                    hypothesis_id="drift_001",
                    category="external_factors",
                    description="External environmental changes causing data distribution shifts",
                    confidence_score=external_analysis["confidence"],
                    evidence=[
                        {
                            "type": "external_correlation",
                            "correlated_events": external_analysis["correlated_events"],
                            "correlation_strengths": external_analysis["correlations"]
                        }
                    ],
                    severity=RootCauseSeverity.HIGH,
                    suggested_actions=[
                        "Monitor external factor indicators",
                        "Implement adaptive model strategies",
                        "Create event-specific model variants"
                    ],
                    related_features=[]
                ))
        
        # Hypothesis 2: Data pipeline changes
        pipeline_analysis = await self._analyze_data_pipeline_changes(model_id, historical_context)
        
        if pipeline_analysis["changes_detected"]:
            hypotheses.append(RootCauseHypothesis(
                hypothesis_id="drift_002",
                category="pipeline_changes",
                description="Data processing pipeline changes affecting data distribution",
                confidence_score=pipeline_analysis["confidence"],
                evidence=[
                    {
                        "type": "pipeline_modifications",
                        "changed_components": pipeline_analysis["changed_components"],
                        "change_timestamps": pipeline_analysis["change_times"]
                    }
                ],
                severity=RootCauseSeverity.HIGH,
                suggested_actions=[
                    "Review recent pipeline modifications",
                    "Implement pipeline versioning",
                    "Add data validation checkpoints"
                ],
                related_features=pipeline_analysis.get("affected_features", [])
            ))
        
        # Hypothesis 3: Upstream data source changes
        if current_data is not None and reference_data is not None:
            source_analysis = self._analyze_upstream_source_changes(
                current_data, reference_data
            )
            
            if source_analysis["source_changes_detected"]:
                hypotheses.append(RootCauseHypothesis(
                    hypothesis_id="drift_003",
                    category="upstream_sources",
                    description="Changes in upstream data sources affecting data quality",
                    confidence_score=source_analysis["confidence"],
                    evidence=[
                        {
                            "type": "source_changes",
                            "changed_patterns": source_analysis["changed_patterns"],
                            "affected_features": source_analysis["affected_features"]
                        }
                    ],
                    severity=RootCauseSeverity.MEDIUM,
                    suggested_actions=[
                        "Validate upstream data sources",
                        "Implement source monitoring",
                        "Create data lineage tracking"
                    ],
                    related_features=source_analysis["affected_features"]
                ))
        
        return hypotheses
    
    async def _perform_cross_cutting_analysis(
        self,
        model_id: str,
        current_data: Optional[pd.DataFrame],
        reference_data: Optional[pd.DataFrame],
        historical_context: Dict[str, Any]
    ) -> List[RootCauseHypothesis]:
        """Perform cross-cutting analysis across all issue types"""
        
        hypotheses = []
        
        # Infrastructure and system-level issues
        if historical_context.get("system_metrics"):
            system_analysis = self._analyze_system_performance_correlation(
                historical_context["system_metrics"]
            )
            
            if system_analysis["performance_issues_detected"]:
                hypotheses.append(RootCauseHypothesis(
                    hypothesis_id="cross_001",
                    category="infrastructure",
                    description="System performance issues affecting model behavior",
                    confidence_score=system_analysis["confidence"],
                    evidence=[
                        {
                            "type": "system_performance",
                            "affected_metrics": system_analysis["affected_metrics"],
                            "performance_degradation": system_analysis["degradation_level"]
                        }
                    ],
                    severity=RootCauseSeverity.MEDIUM,
                    suggested_actions=[
                        "Scale infrastructure resources",
                        "Optimize model inference pipeline",
                        "Monitor system performance continuously"
                    ],
                    related_features=[]
                ))
        
        # Model configuration issues
        config_analysis = await self._analyze_model_configuration_issues(model_id)
        
        if config_analysis["configuration_issues_found"]:
            hypotheses.append(RootCauseHypothesis(
                hypothesis_id="cross_002",
                category="model_configuration",
                description="Model configuration or hyperparameter issues",
                confidence_score=config_analysis["confidence"],
                evidence=[
                    {
                        "type": "configuration_issues",
                        "problematic_settings": config_analysis["problematic_settings"],
                        "recommended_changes": config_analysis["recommended_changes"]
                    }
                ],
                severity=RootCauseSeverity.MEDIUM,
                suggested_actions=[
                    "Review model hyperparameters",
                    "Perform hyperparameter optimization",
                    "Validate model configuration"
                ],
                related_features=[]
            ))
        
        return hypotheses
    
    def _rank_hypotheses(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank hypotheses by confidence and severity"""
        
        def hypothesis_score(h):
            confidence = h.get("confidence_score", 0)
            severity_weights = {
                "critical": 1.0,
                "high": 0.8,
                "medium": 0.6,
                "low": 0.4
            }
            severity_weight = severity_weights.get(h.get("severity", "low"), 0.4)
            return confidence * severity_weight
        
        return sorted(hypotheses, key=hypothesis_score, reverse=True)
    
    async def _generate_recommendations(
        self,
        issue_type: IssueType,
        hypotheses: List[Dict[str, Any]],
        model_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        if not hypotheses:
            return recommendations
        
        primary_hypothesis = hypotheses[0]
        
        # Immediate actions
        immediate_actions = primary_hypothesis.get("suggested_actions", [])
        if immediate_actions:
            recommendations.append({
                "priority": "immediate",
                "category": "incident_response",
                "actions": immediate_actions,
                "timeline": "0-24 hours",
                "rationale": f"Address primary root cause: {primary_hypothesis['description']}"
            })
        
        # Short-term improvements
        short_term_actions = []
        for hypothesis in hypotheses[1:3]:  # Next 2 hypotheses
            short_term_actions.extend(hypothesis.get("suggested_actions", []))
        
        if short_term_actions:
            recommendations.append({
                "priority": "short_term",
                "category": "process_improvement",
                "actions": list(set(short_term_actions)),  # Remove duplicates
                "timeline": "1-7 days",
                "rationale": "Address contributing factors and prevent recurrence"
            })
        
        # Long-term strategic improvements
        strategic_actions = [
            "Implement comprehensive monitoring and alerting",
            "Establish model governance and lifecycle management",
            "Create incident response playbooks",
            "Implement automated model validation"
        ]
        
        recommendations.append({
            "priority": "strategic",
            "category": "system_improvement",
            "actions": strategic_actions,
            "timeline": "1-4 weeks",
            "rationale": "Build robust systems to prevent similar issues"
        })
        
        return recommendations
    
    def _assess_analysis_confidence(
        self,
        hypotheses: List[Dict[str, Any]],
        current_data: Optional[pd.DataFrame],
        reference_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Assess overall confidence in the analysis"""
        
        if not hypotheses:
            return {
                "overall_confidence": 0.0,
                "confidence_level": "low",
                "limiting_factors": ["No hypotheses generated"],
                "data_quality_assessment": "insufficient"
            }
        
        # Calculate overall confidence
        hypothesis_confidences = [h.get("confidence_score", 0) for h in hypotheses]
        overall_confidence = np.mean(hypothesis_confidences) if hypothesis_confidences else 0.0
        
        # Assess limiting factors
        limiting_factors = []
        if current_data is None:
            limiting_factors.append("No current data available")
        if reference_data is None:
            limiting_factors.append("No reference data available")
        if len(hypotheses) < 3:
            limiting_factors.append("Limited number of hypotheses")
        
        # Confidence level
        if overall_confidence >= 0.8:
            confidence_level = "high"
        elif overall_confidence >= 0.6:
            confidence_level = "medium"
        elif overall_confidence >= 0.4:
            confidence_level = "moderate"
        else:
            confidence_level = "low"
        
        # Data quality assessment
        data_quality = "good"
        if current_data is not None:
            missing_rate = current_data.isnull().mean().mean()
            if missing_rate > 0.2:
                data_quality = "poor"
                limiting_factors.append("High missing data rate")
            elif missing_rate > 0.1:
                data_quality = "moderate"
        
        return {
            "overall_confidence": overall_confidence,
            "confidence_level": confidence_level,
            "limiting_factors": limiting_factors,
            "data_quality_assessment": data_quality,
            "hypothesis_count": len(hypotheses),
            "top_hypothesis_confidence": hypothesis_confidences[0] if hypothesis_confidences else 0.0
        }
    
    def _hypothesis_to_dict(self, hypothesis: RootCauseHypothesis) -> Dict[str, Any]:
        """Convert hypothesis object to dictionary"""
        return {
            "hypothesis_id": hypothesis.hypothesis_id,
            "category": hypothesis.category,
            "description": hypothesis.description,
            "confidence_score": hypothesis.confidence_score,
            "evidence": hypothesis.evidence,
            "severity": hypothesis.severity.value,
            "suggested_actions": hypothesis.suggested_actions,
            "related_features": hypothesis.related_features,
            "temporal_correlation": hypothesis.temporal_correlation
        }
    
    # Helper methods for specific analyses
    async def _analyze_feature_drift_impact(
        self, 
        current_data: pd.DataFrame, 
        reference_data: pd.DataFrame,
        model_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze feature drift impact on performance"""
        
        # Use the enterprise drift service for detailed analysis
        from app.services.enterprise_drift_service import EnterpriseDriftDetector
        
        drift_detector = EnterpriseDriftDetector()
        drift_results = drift_detector.detect_comprehensive_drift(
            reference_data, current_data, methods=["jensen_shannon_divergence", "population_stability_index"]
        )
        
        drifted_features = []
        drift_scores = {}
        
        for feature, result in drift_results["feature_drift_results"].items():
            if result["drift_detected"]:
                drifted_features.append(feature)
                drift_scores[feature] = result["drift_score"]
        
        return {
            "significant_drift_detected": len(drifted_features) > 0,
            "drifted_features": drifted_features,
            "drift_scores": drift_scores,
            "confidence": min(1.0, len(drifted_features) * 0.2)  # Simple confidence calculation
        }
    
    def _analyze_feature_importance_changes(
        self, 
        importance_history: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Analyze changes in feature importance over time"""
        
        if len(importance_history) < 2:
            return {"significant_changes": False, "confidence": 0.0}
        
        latest = importance_history[-1]
        previous = importance_history[-2]
        
        changed_features = []
        deltas = {}
        
        for feature in latest:
            if feature in previous:
                delta = abs(latest[feature] - previous[feature])
                if delta > self.feature_importance_threshold:
                    changed_features.append(feature)
                    deltas[feature] = delta
        
        return {
            "significant_changes": len(changed_features) > 0,
            "changed_features": changed_features,
            "deltas": deltas,
            "confidence": min(1.0, len(changed_features) * 0.3)
        }
    
    def _analyze_data_quality_degradation(
        self,
        current_data: pd.DataFrame,
        reference_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Analyze data quality degradation"""
        
        current_missing = current_data.isnull().mean()
        affected_columns = []
        missing_values_delta = {}
        
        if reference_data is not None:
            reference_missing = reference_data.isnull().mean()
            
            for col in current_data.columns:
                if col in reference_data.columns:
                    delta = current_missing[col] - reference_missing[col]
                    if delta > 0.05:  # 5% increase in missing values
                        affected_columns.append(col)
                        missing_values_delta[col] = delta
        else:
            # If no reference, check for high missing rates
            for col in current_data.columns:
                if current_missing[col] > 0.2:  # 20% missing
                    affected_columns.append(col)
                    missing_values_delta[col] = current_missing[col]
        
        # Simple outlier detection
        outliers_delta = {}
        for col in current_data.select_dtypes(include=[np.number]).columns:
            Q1 = current_data[col].quantile(0.25)
            Q3 = current_data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((current_data[col] < (Q1 - 1.5 * IQR)) | 
                       (current_data[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_rate = outliers / len(current_data)
            
            if outlier_rate > 0.1:  # 10% outliers
                outliers_delta[col] = outlier_rate
        
        return {
            "quality_degraded": len(affected_columns) > 0 or len(outliers_delta) > 0,
            "affected_columns": affected_columns,
            "missing_values_delta": missing_values_delta,
            "outliers_delta": outliers_delta,
            "confidence": min(1.0, (len(affected_columns) + len(outliers_delta)) * 0.25)
        }
    
    # Additional helper methods would be implemented for other specific analyses...
    
    async def _gather_historical_context(
        self, 
        model_id: str, 
        time_range_hours: int
    ) -> Dict[str, Any]:
        """Gather historical context for analysis"""
        
        from app.core.database import get_db
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_range_hours)
        
        context = {
            "alerts": [],
            "performance_metrics": {},
            "system_metrics": {},
            "external_events": [],
            "feature_importance_history": []
        }
        
        async with get_db() as db:
            # Get recent alerts
            alerts = await db.fetch("""
                SELECT alert_type, severity, triggered_at, details
                FROM alerts 
                WHERE model_id = $1 AND triggered_at BETWEEN $2 AND $3
                ORDER BY triggered_at DESC
            """, model_id, start_time, end_time)
            
            context["alerts"] = [dict(alert) for alert in alerts]
            
            # Get performance metrics
            metrics = await db.fetch("""
                SELECT metrics, timestamp
                FROM model_performance_metrics
                WHERE model_id = $1 AND timestamp BETWEEN $2 AND $3
                ORDER BY timestamp
            """, model_id, start_time, end_time)
            
            for metric in metrics:
                timestamp = metric["timestamp"].isoformat()
                context["performance_metrics"][timestamp] = json.loads(metric["metrics"])
        
        return context
    
    async def _store_analysis_results(self, analysis_report: Dict[str, Any]):
        """Store root cause analysis results"""
        
        from app.core.database import get_db
        
        async with get_db() as db:
            await db.execute("""
                INSERT INTO root_cause_analyses (
                    analysis_id, model_id, issue_type, analysis_report, 
                    primary_root_cause, confidence_level, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
            """, 
                analysis_report["analysis_id"],
                analysis_report["model_id"],
                analysis_report["issue_type"],
                json.dumps(analysis_report),
                analysis_report["primary_root_cause"]["category"] if analysis_report["primary_root_cause"] else None,
                analysis_report["confidence_assessment"]["confidence_level"]
            )
    
    # Placeholder methods for additional analyses
    def _analyze_temporal_patterns(self, historical_metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze temporal patterns in metrics"""
        return {"seasonal_pattern_detected": False, "confidence": 0.0, "pattern_type": None, "correlation": 0.0}
    
    def _analyze_training_data_bias(self, reference_data: pd.DataFrame, model_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze bias in training data"""
        return {"bias_detected": False, "confidence": 0.0, "biased_attributes": [], "representation_gaps": {}}
    
    def _analyze_protected_attribute_correlations(self, current_data: pd.DataFrame, model_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlations with protected attributes"""
        return {"high_correlations_found": False, "confidence": 0.0, "correlated_features": [], "correlations": {}}
    
    def _analyze_sensitive_group_drift(self, current_data: pd.DataFrame, reference_data: pd.DataFrame, model_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze drift in sensitive group populations"""
        return {"drift_detected": False, "confidence": 0.0, "shifted_groups": [], "drift_magnitudes": {}, "sensitive_attributes": []}
    
    def _analyze_external_event_correlation(self, external_events: List[Dict[str, Any]], historical_metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze correlation with external events"""
        return {"correlation_found": False, "confidence": 0.0, "correlated_events": [], "correlations": {}}
    
    async def _analyze_data_pipeline_changes(self, model_id: str, historical_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data pipeline changes"""
        return {"changes_detected": False, "confidence": 0.0, "changed_components": [], "change_times": [], "affected_features": []}
    
    def _analyze_upstream_source_changes(self, current_data: pd.DataFrame, reference_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze upstream data source changes"""
        return {"source_changes_detected": False, "confidence": 0.0, "changed_patterns": [], "affected_features": []}
    
    def _analyze_system_performance_correlation(self, system_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system performance correlation"""
        return {"performance_issues_detected": False, "confidence": 0.0, "affected_metrics": [], "degradation_level": 0.0}
    
    async def _analyze_model_configuration_issues(self, model_id: str) -> Dict[str, Any]:
        """Analyze model configuration issues"""
        return {"configuration_issues_found": False, "confidence": 0.0, "problematic_settings": [], "recommended_changes": []}
    
    async def _generate_supporting_evidence(self, hypotheses: List[Dict[str, Any]], current_data: Optional[pd.DataFrame], reference_data: Optional[pd.DataFrame], historical_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate supporting evidence for analysis"""
        return {"data_analysis": {}, "statistical_tests": {}, "visualizations": [], "historical_trends": {}}
    
    def _define_next_steps(self, issue_type: IssueType, hypotheses: List[Dict[str, Any]], confidence_assessment: Dict[str, Any]) -> List[str]:
        """Define next steps based on analysis"""
        
        next_steps = []
        
        if confidence_assessment["confidence_level"] == "low":
            next_steps.extend([
                "Collect additional data for more comprehensive analysis",
                "Validate findings with domain experts",
                "Consider alternative analysis approaches"
            ])
        
        if hypotheses:
            primary_category = hypotheses[0]["category"]
            
            if primary_category == "data_drift":
                next_steps.extend([
                    "Implement continuous drift monitoring",
                    "Schedule model retraining with recent data",
                    "Review data collection processes"
                ])
            elif primary_category == "bias_detection":
                next_steps.extend([
                    "Implement bias mitigation strategies",
                    "Review model training data for representation",
                    "Establish fairness monitoring"
                ])
            elif primary_category == "data_quality":
                next_steps.extend([
                    "Implement data validation pipelines",
                    "Review data preprocessing steps",
                    "Establish data quality monitoring"
                ])
        
        next_steps.append("Schedule follow-up analysis to validate implemented solutions")
        
        return next_steps


# Database initialization
async def initialize_rca_tables():
    """Initialize root cause analysis tables"""
    from app.core.database import get_db
    
    async with get_db() as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS root_cause_analyses (
                analysis_id VARCHAR PRIMARY KEY,
                model_id VARCHAR NOT NULL,
                issue_type VARCHAR NOT NULL,
                analysis_report JSONB NOT NULL,
                primary_root_cause VARCHAR,
                confidence_level VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        await db.execute("CREATE INDEX IF NOT EXISTS idx_rca_model_id ON root_cause_analyses(model_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_rca_issue_type ON root_cause_analyses(issue_type)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_rca_created_at ON root_cause_analyses(created_at)")


# Utility function for dashboard integration
def create_rca_dashboard_data(analysis_report: Dict[str, Any]) -> Dict[str, Any]:
    """Create dashboard data from root cause analysis report"""
    
    return {
        "analysis_summary": {
            "analysis_id": analysis_report["analysis_id"],
            "issue_type": analysis_report["issue_type"],
            "primary_cause": analysis_report["primary_root_cause"]["category"] if analysis_report["primary_root_cause"] else None,
            "confidence_level": analysis_report["confidence_assessment"]["confidence_level"],
            "analysis_timestamp": analysis_report["analysis_timestamp"]
        },
        "key_findings": {
            "primary_root_cause": analysis_report["primary_root_cause"],
            "contributing_factors": analysis_report["contributing_factors"],
            "affected_features": analysis_report["primary_root_cause"]["related_features"] if analysis_report["primary_root_cause"] else []
        },
        "recommendations": analysis_report["recommendations"],
        "confidence_metrics": analysis_report["confidence_assessment"],
        "next_steps": analysis_report["next_steps"]
    }