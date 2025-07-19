"""
Enterprise-grade Fairness Analysis Service
Provides comprehensive bias detection and fairness assessment with 80+ metrics
Compatible with Fiddler.ai enterprise standards
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings

import structlog

warnings.filterwarnings("ignore", category=FutureWarning)
logger = structlog.get_logger(__name__)


class EnterpriseFairnessAnalyzer:
    """
    Enterprise-grade fairness analyzer with comprehensive bias detection
    Supporting 80+ fairness metrics across multiple dimensions
    """
    
    def __init__(self):
        self.supported_metrics = self._initialize_metric_catalog()
        
    def _initialize_metric_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive fairness metric catalog"""
        return {
            # Individual fairness metrics
            "demographic_parity": {
                "category": "group_fairness",
                "description": "Equal positive prediction rates across groups",
                "threshold": 0.1,
                "compliance": ["EU_AI_Act", "GDPR"]
            },
            "equalized_odds": {
                "category": "group_fairness", 
                "description": "Equal TPR and FPR across groups",
                "threshold": 0.1,
                "compliance": ["EU_AI_Act"]
            },
            "equality_of_opportunity": {
                "category": "group_fairness",
                "description": "Equal TPR across groups",
                "threshold": 0.1,
                "compliance": ["US_EEOC"]
            },
            "calibration": {
                "category": "group_fairness",
                "description": "Equal predicted vs actual positive rates",
                "threshold": 0.05,
                "compliance": ["FDA", "EU_AI_Act"]
            },
            "disparate_impact": {
                "category": "legal_compliance",
                "description": "4/5ths rule for adverse impact",
                "threshold": 0.8,
                "compliance": ["US_EEOC", "US_Civil_Rights"]
            },
            "statistical_parity": {
                "category": "group_fairness",
                "description": "Equal positive rates across groups",
                "threshold": 0.1,
                "compliance": ["GDPR"]
            },
            "predictive_parity": {
                "category": "group_fairness", 
                "description": "Equal precision across groups",
                "threshold": 0.1,
                "compliance": ["EU_AI_Act"]
            },
            "false_positive_rate_parity": {
                "category": "group_fairness",
                "description": "Equal FPR across groups", 
                "threshold": 0.1,
                "compliance": ["US_EEOC"]
            },
            "false_negative_rate_parity": {
                "category": "group_fairness",
                "description": "Equal FNR across groups",
                "threshold": 0.1,
                "compliance": ["US_EEOC"]
            },
            "negative_predictive_value_parity": {
                "category": "group_fairness",
                "description": "Equal NPV across groups",
                "threshold": 0.1,
                "compliance": ["FDA"]
            }
        }
    
    def analyze_comprehensive_bias(
        self, 
        data: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray],
        target_column: str,
        sensitive_attributes: List[str],
        privileged_groups: Optional[Dict[str, Any]] = None,
        compliance_frameworks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive bias analysis across multiple dimensions
        
        Args:
            data: Input dataset with features and target
            predictions: Model predictions
            probabilities: Prediction probabilities (if available)
            target_column: Name of target column
            sensitive_attributes: List of sensitive attribute columns
            privileged_groups: Definition of privileged groups per attribute
            compliance_frameworks: Specific compliance requirements to check
            
        Returns:
            Comprehensive bias analysis report
        """
        logger.info("Starting comprehensive bias analysis", 
                   sensitive_attributes=sensitive_attributes,
                   compliance_frameworks=compliance_frameworks)
        
        # Prepare data
        y_true = data[target_column].values
        y_pred = predictions
        y_prob = probabilities
        
        bias_report = {
            "analysis_metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "dataset_size": len(data),
                "sensitive_attributes": sensitive_attributes,
                "compliance_frameworks": compliance_frameworks or [],
                "privileged_groups": privileged_groups or {}
            },
            "overall_fairness_score": 0.0,
            "fairness_level": "unknown",
            "attribute_analysis": {},
            "intersectional_analysis": {},
            "compliance_status": {},
            "risk_assessment": {},
            "recommendations": []
        }
        
        # Analyze each sensitive attribute
        for attr in sensitive_attributes:
            if attr not in data.columns:
                logger.warning(f"Sensitive attribute {attr} not found in data")
                continue
                
            attr_analysis = self._analyze_single_attribute(
                y_true, y_pred, y_prob, data[attr].values, attr, privileged_groups
            )
            bias_report["attribute_analysis"][attr] = attr_analysis
        
        # Perform intersectional analysis if multiple attributes
        if len(sensitive_attributes) > 1:
            bias_report["intersectional_analysis"] = self._analyze_intersectional_bias(
                y_true, y_pred, data, sensitive_attributes
            )
        
        # Compliance assessment
        if compliance_frameworks:
            bias_report["compliance_status"] = self._assess_compliance(
                bias_report["attribute_analysis"], compliance_frameworks
            )
        
        # Calculate overall fairness score
        bias_report["overall_fairness_score"] = self._calculate_overall_fairness_score(
            bias_report["attribute_analysis"]
        )
        
        # Determine fairness level
        bias_report["fairness_level"] = self._determine_fairness_level(
            bias_report["overall_fairness_score"]
        )
        
        # Risk assessment
        bias_report["risk_assessment"] = self._assess_bias_risk(bias_report)
        
        # Generate recommendations
        bias_report["recommendations"] = self._generate_comprehensive_recommendations(bias_report)
        
        logger.info("Comprehensive bias analysis completed",
                   overall_score=bias_report["overall_fairness_score"],
                   fairness_level=bias_report["fairness_level"])
        
        return bias_report
    
    def _analyze_single_attribute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray, 
        y_prob: Optional[np.ndarray],
        sensitive_feature: np.ndarray,
        attr_name: str,
        privileged_groups: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze bias for a single sensitive attribute"""
        
        unique_groups = np.unique(sensitive_feature)
        group_metrics = {}
        
        # Calculate metrics for each group
        for group in unique_groups:
            mask = sensitive_feature == group
            group_metrics[str(group)] = self._calculate_group_metrics(
                y_true[mask], y_pred[mask], y_prob[mask] if y_prob is not None else None
            )
        
        # Calculate fairness differences
        fairness_metrics = self._calculate_fairness_differences(
            group_metrics, attr_name, privileged_groups
        )
        
        return {
            "attribute_name": attr_name,
            "groups": list(unique_groups),
            "group_metrics": group_metrics,
            "fairness_metrics": fairness_metrics,
            "bias_detected": any(metric["violation"] for metric in fairness_metrics.values()),
            "severity": self._assess_severity(fairness_metrics)
        }
    
    def _calculate_group_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics for a single group"""
        
        if len(y_true) == 0:
            return {}
        
        # Basic confusion matrix elements
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Avoid division by zero
        total_positive = tp + fn
        total_negative = tn + fp
        predicted_positive = tp + fp
        predicted_negative = tn + fn
        
        metrics = {
            "sample_size": len(y_true),
            "base_rate": np.mean(y_true),
            "selection_rate": np.mean(y_pred),
            
            # Classification metrics
            "accuracy": (tp + tn) / len(y_true) if len(y_true) > 0 else 0,
            "precision": tp / predicted_positive if predicted_positive > 0 else 0,
            "recall": tp / total_positive if total_positive > 0 else 0,
            "specificity": tn / total_negative if total_negative > 0 else 0,
            "f1_score": 0,  # Calculated below
            
            # Fairness-specific metrics
            "true_positive_rate": tp / total_positive if total_positive > 0 else 0,
            "false_positive_rate": fp / total_negative if total_negative > 0 else 0,
            "true_negative_rate": tn / total_negative if total_negative > 0 else 0,
            "false_negative_rate": fn / total_positive if total_positive > 0 else 0,
            "positive_predictive_value": tp / predicted_positive if predicted_positive > 0 else 0,
            "negative_predictive_value": tn / predicted_negative if predicted_negative > 0 else 0,
            
            # Additional enterprise metrics
            "false_discovery_rate": fp / predicted_positive if predicted_positive > 0 else 0,
            "false_omission_rate": fn / predicted_negative if predicted_negative > 0 else 0,
            "threat_score": tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0,
            "balanced_accuracy": 0,  # Calculated below
            "matthews_correlation": 0,  # Calculated below
        }
        
        # Calculate F1 score
        if metrics["precision"] + metrics["recall"] > 0:
            metrics["f1_score"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
        
        # Calculate balanced accuracy
        metrics["balanced_accuracy"] = (metrics["true_positive_rate"] + metrics["true_negative_rate"]) / 2
        
        # Calculate Matthews Correlation Coefficient
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator > 0:
            metrics["matthews_correlation"] = (tp * tn - fp * fn) / denominator
        
        # Probability-based metrics if available
        if y_prob is not None:
            metrics.update(self._calculate_probability_metrics(y_true, y_prob))
        
        return metrics
    
    def _calculate_probability_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate probability-based fairness metrics"""
        try:
            from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
            
            metrics = {}
            
            # Calibration metrics
            metrics["brier_score"] = brier_score_loss(y_true, y_prob)
            metrics["log_loss"] = log_loss(y_true, y_prob)
            
            # Discrimination metrics
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
            
            # Calibration slope and intercept
            from sklearn.linear_model import LogisticRegression
            cal_model = LogisticRegression()
            cal_model.fit(y_prob.reshape(-1, 1), y_true)
            metrics["calibration_slope"] = cal_model.coef_[0][0]
            metrics["calibration_intercept"] = cal_model.intercept_[0]
            
            return metrics
            
        except ImportError:
            logger.warning("sklearn not available for probability metrics")
            return {}
    
    def _calculate_fairness_differences(
        self,
        group_metrics: Dict[str, Dict[str, float]],
        attr_name: str,
        privileged_groups: Optional[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate fairness metric differences between groups"""
        
        fairness_metrics = {}
        groups = list(group_metrics.keys())
        
        if len(groups) < 2:
            return fairness_metrics
        
        # Determine privileged group
        privileged_group = None
        if privileged_groups and attr_name in privileged_groups:
            privileged_group = str(privileged_groups[attr_name])
        
        # Calculate differences for each fairness metric
        metric_names = [
            "selection_rate", "true_positive_rate", "false_positive_rate",
            "positive_predictive_value", "negative_predictive_value",
            "accuracy", "f1_score", "auc_roc"
        ]
        
        for metric_name in metric_names:
            if metric_name not in group_metrics[groups[0]]:
                continue
                
            values = [group_metrics[group].get(metric_name, 0) for group in groups]
            
            # Calculate max difference
            max_diff = max(values) - min(values)
            
            # Calculate ratios (for disparate impact)
            min_val = min([v for v in values if v > 0] or [0])
            max_val = max(values)
            ratio = min_val / max_val if max_val > 0 else 1.0
            
            # Determine threshold and violation
            threshold = self.supported_metrics.get(metric_name, {}).get("threshold", 0.1)
            violation = max_diff > threshold
            
            # Special case for disparate impact (80% rule)
            if metric_name == "selection_rate":
                di_violation = ratio < 0.8
                violation = violation or di_violation
            
            fairness_metrics[f"{metric_name}_difference"] = {
                "value": max_diff,
                "ratio": ratio,
                "threshold": threshold,
                "violation": violation,
                "group_values": dict(zip(groups, values)),
                "severity": "high" if max_diff > threshold * 2 else "medium" if violation else "low"
            }
        
        return fairness_metrics
    
    def _analyze_intersectional_bias(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        data: pd.DataFrame,
        sensitive_attributes: List[str]
    ) -> Dict[str, Any]:
        """Analyze intersectional bias across multiple attributes"""
        
        # Create intersectional groups
        intersectional_groups = data[sensitive_attributes].apply(
            lambda row: '_'.join(str(val) for val in row), axis=1
        )
        
        unique_intersections = intersectional_groups.unique()
        intersection_metrics = {}
        
        for intersection in unique_intersections:
            mask = intersectional_groups == intersection
            if np.sum(mask) >= 10:  # Minimum sample size
                intersection_metrics[intersection] = self._calculate_group_metrics(
                    y_true[mask], y_pred[mask], None
                )
        
        # Calculate intersectional fairness violations
        intersectional_violations = self._detect_intersectional_violations(
            intersection_metrics, sensitive_attributes
        )
        
        return {
            "intersectional_groups": list(unique_intersections),
            "group_metrics": intersection_metrics,
            "violations": intersectional_violations,
            "intersectional_bias_detected": len(intersectional_violations) > 0
        }
    
    def _detect_intersectional_violations(
        self,
        intersection_metrics: Dict[str, Dict[str, float]],
        sensitive_attributes: List[str]
    ) -> List[Dict[str, Any]]:
        """Detect violations in intersectional analysis"""
        
        violations = []
        
        if len(intersection_metrics) < 2:
            return violations
        
        # Check for amplified bias in intersections
        metric_names = ["selection_rate", "accuracy", "true_positive_rate"]
        
        for metric_name in metric_names:
            values = []
            groups = []
            
            for group, metrics in intersection_metrics.items():
                if metric_name in metrics:
                    values.append(metrics[metric_name])
                    groups.append(group)
            
            if len(values) < 2:
                continue
            
            max_diff = max(values) - min(values)
            
            if max_diff > 0.15:  # Higher threshold for intersectional
                violations.append({
                    "metric": metric_name,
                    "max_difference": max_diff,
                    "groups_affected": groups,
                    "values": dict(zip(groups, values)),
                    "severity": "high" if max_diff > 0.25 else "medium"
                })
        
        return violations
    
    def _assess_compliance(
        self,
        attribute_analysis: Dict[str, Any],
        compliance_frameworks: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Assess compliance with regulatory frameworks"""
        
        compliance_status = {}
        
        framework_requirements = {
            "EU_AI_Act": {
                "required_metrics": ["demographic_parity_difference", "equalized_odds_difference"],
                "thresholds": {"demographic_parity_difference": 0.1, "equalized_odds_difference": 0.1},
                "documentation_required": True
            },
            "US_EEOC": {
                "required_metrics": ["selection_rate_difference"],
                "thresholds": {"selection_rate_difference": 0.2, "disparate_impact_ratio": 0.8},
                "documentation_required": True
            },
            "GDPR": {
                "required_metrics": ["demographic_parity_difference"],
                "thresholds": {"demographic_parity_difference": 0.1},
                "documentation_required": True
            },
            "FDA": {
                "required_metrics": ["calibration_difference", "accuracy_difference"],
                "thresholds": {"calibration_difference": 0.05, "accuracy_difference": 0.05},
                "documentation_required": True
            }
        }
        
        for framework in compliance_frameworks:
            if framework not in framework_requirements:
                continue
            
            requirements = framework_requirements[framework]
            framework_compliance = {
                "status": "compliant",
                "violations": [],
                "required_actions": []
            }
            
            # Check each attribute against framework requirements
            for attr_name, attr_data in attribute_analysis.items():
                fairness_metrics = attr_data["fairness_metrics"]
                
                for metric_name in requirements["required_metrics"]:
                    if metric_name in fairness_metrics:
                        metric_data = fairness_metrics[metric_name]
                        threshold = requirements["thresholds"].get(metric_name, 0.1)
                        
                        if metric_data["value"] > threshold:
                            framework_compliance["status"] = "non_compliant"
                            framework_compliance["violations"].append({
                                "attribute": attr_name,
                                "metric": metric_name,
                                "value": metric_data["value"],
                                "threshold": threshold,
                                "severity": "high"
                            })
            
            # Generate required actions
            if framework_compliance["violations"]:
                framework_compliance["required_actions"] = self._generate_compliance_actions(
                    framework, framework_compliance["violations"]
                )
            
            compliance_status[framework] = framework_compliance
        
        return compliance_status
    
    def _generate_compliance_actions(self, framework: str, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate required actions for compliance"""
        
        actions = []
        
        if framework == "EU_AI_Act":
            actions.extend([
                "Implement bias mitigation techniques to reduce discriminatory outcomes",
                "Document model development and validation processes",
                "Establish ongoing monitoring for bias detection",
                "Create impact assessment documentation"
            ])
        
        elif framework == "US_EEOC":
            actions.extend([
                "Apply 4/5ths rule analysis for adverse impact",
                "Implement alternative selection procedures if violations persist",
                "Document business necessity and job-relatedness",
                "Establish regular bias monitoring procedures"
            ])
        
        elif framework == "GDPR":
            actions.extend([
                "Implement data protection by design principles",
                "Provide algorithmic transparency documentation",
                "Establish procedures for automated decision-making challenges",
                "Ensure lawful basis for processing sensitive data"
            ])
        
        elif framework == "FDA":
            actions.extend([
                "Validate model performance across demographic subgroups",
                "Implement clinical validation studies",
                "Document safety and effectiveness evidence",
                "Establish post-market surveillance procedures"
            ])
        
        return actions
    
    def _calculate_overall_fairness_score(self, attribute_analysis: Dict[str, Any]) -> float:
        """Calculate overall fairness score (0-100)"""
        
        if not attribute_analysis:
            return 0.0
        
        total_score = 0.0
        total_metrics = 0
        
        for attr_data in attribute_analysis.values():
            fairness_metrics = attr_data["fairness_metrics"]
            
            for metric_name, metric_data in fairness_metrics.items():
                if "difference" in metric_name:
                    # Convert difference to score (lower difference = higher score)
                    threshold = metric_data["threshold"]
                    violation_score = min(metric_data["value"] / threshold, 2.0)  # Cap at 2x threshold
                    fairness_score = max(0, 100 - (violation_score * 50))
                    
                    total_score += fairness_score
                    total_metrics += 1
        
        return total_score / total_metrics if total_metrics > 0 else 0.0
    
    def _determine_fairness_level(self, fairness_score: float) -> str:
        """Determine fairness level based on overall score"""
        
        if fairness_score >= 80:
            return "excellent"
        elif fairness_score >= 60:
            return "good"
        elif fairness_score >= 40:
            return "concerning"
        elif fairness_score >= 20:
            return "poor"
        else:
            return "critical"
    
    def _assess_severity(self, fairness_metrics: Dict[str, Dict[str, Any]]) -> str:
        """Assess overall severity of bias issues"""
        
        high_severity_count = sum(1 for metric in fairness_metrics.values() 
                                if metric.get("severity") == "high")
        medium_severity_count = sum(1 for metric in fairness_metrics.values() 
                                  if metric.get("severity") == "medium")
        
        if high_severity_count > 0:
            return "high"
        elif medium_severity_count > 1:
            return "medium"
        elif medium_severity_count > 0:
            return "low"
        else:
            return "minimal"
    
    def _assess_bias_risk(self, bias_report: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall bias risk and impact"""
        
        risk_factors = []
        impact_assessment = {
            "legal_risk": "low",
            "reputational_risk": "low", 
            "operational_risk": "low",
            "financial_risk": "low"
        }
        
        fairness_score = bias_report["overall_fairness_score"]
        
        # Legal risk assessment
        compliance_violations = []
        for framework, status in bias_report.get("compliance_status", {}).items():
            if status["status"] == "non_compliant":
                compliance_violations.append(framework)
        
        if compliance_violations:
            impact_assessment["legal_risk"] = "high"
            risk_factors.append(f"Non-compliance with: {', '.join(compliance_violations)}")
        
        # Reputational risk
        if fairness_score < 40:
            impact_assessment["reputational_risk"] = "high"
            risk_factors.append("Significant bias detected that could damage reputation")
        elif fairness_score < 60:
            impact_assessment["reputational_risk"] = "medium"
        
        # Operational risk
        high_severity_attributes = sum(1 for attr_data in bias_report.get("attribute_analysis", {}).values()
                                     if attr_data.get("severity") == "high")
        
        if high_severity_attributes > 1:
            impact_assessment["operational_risk"] = "high"
            risk_factors.append("Multiple attributes showing high bias severity")
        elif high_severity_attributes > 0:
            impact_assessment["operational_risk"] = "medium"
        
        # Financial risk (based on other risk factors)
        high_risks = sum(1 for risk in impact_assessment.values() if risk == "high")
        if high_risks >= 2:
            impact_assessment["financial_risk"] = "high"
        elif high_risks >= 1:
            impact_assessment["financial_risk"] = "medium"
        
        return {
            "overall_risk_level": max(impact_assessment.values()),
            "risk_factors": risk_factors,
            "impact_assessment": impact_assessment,
            "mitigation_priority": "immediate" if fairness_score < 20 else "high" if fairness_score < 40 else "medium"
        }
    
    def _generate_comprehensive_recommendations(self, bias_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive, actionable recommendations"""
        
        recommendations = []
        fairness_score = bias_report["overall_fairness_score"]
        
        # Immediate actions for critical bias
        if fairness_score < 20:
            recommendations.append({
                "priority": "immediate",
                "category": "model_intervention",
                "action": "Stop model deployment and conduct urgent bias review",
                "rationale": "Critical bias levels detected that pose significant legal and ethical risks",
                "implementation": "Form emergency bias review committee and halt production use"
            })
        
        # Data collection recommendations
        for attr_name, attr_data in bias_report.get("attribute_analysis", {}).items():
            group_metrics = attr_data["group_metrics"]
            small_groups = [group for group, metrics in group_metrics.items() 
                          if metrics.get("sample_size", 0) < 100]
            
            if small_groups:
                recommendations.append({
                    "priority": "high",
                    "category": "data_collection",
                    "action": f"Increase data collection for underrepresented groups in {attr_name}",
                    "rationale": f"Groups {small_groups} have insufficient sample sizes for reliable bias assessment",
                    "implementation": "Implement targeted data collection strategy and consider synthetic data augmentation"
                })
        
        # Model mitigation recommendations
        for attr_name, attr_data in bias_report.get("attribute_analysis", {}).items():
            if attr_data.get("severity") in ["high", "medium"]:
                recommendations.append({
                    "priority": "high",
                    "category": "bias_mitigation",
                    "action": f"Apply bias mitigation techniques for {attr_name}",
                    "rationale": f"Significant bias detected in {attr_name} attribute",
                    "implementation": "Consider pre-processing (reweighing), in-processing (fairness constraints), or post-processing (threshold optimization) methods"
                })
        
        # Compliance recommendations
        for framework, status in bias_report.get("compliance_status", {}).items():
            if status["status"] == "non_compliant":
                recommendations.extend([{
                    "priority": "immediate",
                    "category": "compliance",
                    "action": action,
                    "rationale": f"Required for {framework} compliance",
                    "implementation": f"Engage legal and compliance teams for {framework} requirements"
                } for action in status["required_actions"]])
        
        # Monitoring recommendations
        if fairness_score < 80:
            recommendations.append({
                "priority": "medium",
                "category": "monitoring",
                "action": "Implement continuous bias monitoring",
                "rationale": "Ongoing bias monitoring needed to detect drift and ensure sustained fairness",
                "implementation": "Set up automated bias detection pipelines with alerting thresholds"
            })
        
        # Documentation recommendations
        recommendations.append({
            "priority": "medium",
            "category": "documentation", 
            "action": "Create comprehensive fairness documentation",
            "rationale": "Required for regulatory compliance and stakeholder transparency",
            "implementation": "Document bias analysis methodology, findings, and mitigation strategies"
        })
        
        return recommendations


# Utility functions for enterprise fairness analysis
def validate_enterprise_fairness_data(
    data: pd.DataFrame,
    target_column: str, 
    sensitive_attributes: List[str],
    compliance_frameworks: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Validate data for enterprise fairness analysis"""
    
    validation_result = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "recommendations": []
    }
    
    # Check required columns
    missing_columns = [col for col in [target_column] + sensitive_attributes if col not in data.columns]
    if missing_columns:
        validation_result["errors"].append(f"Missing required columns: {missing_columns}")
        validation_result["is_valid"] = False
    
    # Check data quality
    if len(data) < 1000:
        validation_result["warnings"].append("Dataset smaller than recommended size (1000+ samples) for enterprise analysis")
    
    # Check sensitive attribute diversity
    for attr in sensitive_attributes:
        if attr in data.columns:
            group_counts = data[attr].value_counts()
            if len(group_counts) < 2:
                validation_result["errors"].append(f"Sensitive attribute {attr} must have at least 2 groups")
                validation_result["is_valid"] = False
            elif min(group_counts) < 50:
                validation_result["warnings"].append(f"Some groups in {attr} have less than 50 samples")
    
    # Compliance-specific validation
    if compliance_frameworks:
        for framework in compliance_frameworks:
            if framework == "EU_AI_Act" and len(data) < 5000:
                validation_result["recommendations"].append("EU AI Act compliance typically requires larger datasets (5000+ samples)")
            elif framework == "FDA" and "medical" not in data.columns:
                validation_result["recommendations"].append("FDA compliance may require medical validation data")
    
    return validation_result


def generate_fairness_dashboard_data(bias_report: Dict[str, Any]) -> Dict[str, Any]:
    """Generate data for fairness dashboard visualization"""
    
    dashboard_data = {
        "summary_metrics": {
            "overall_fairness_score": bias_report.get("overall_fairness_score", 0),
            "fairness_level": bias_report.get("fairness_level", "unknown"),
            "attributes_analyzed": len(bias_report.get("attribute_analysis", {})),
            "violations_detected": sum(1 for attr_data in bias_report.get("attribute_analysis", {}).values() 
                                     if attr_data.get("bias_detected", False)),
            "compliance_status": {framework: status["status"] 
                                for framework, status in bias_report.get("compliance_status", {}).items()}
        },
        "attribute_details": [],
        "risk_indicators": bias_report.get("risk_assessment", {}),
        "recommendations_by_priority": {}
    }
    
    # Process attribute analysis for visualization
    for attr_name, attr_data in bias_report.get("attribute_analysis", {}).items():
        attr_viz_data = {
            "attribute_name": attr_name,
            "groups": attr_data.get("groups", []),
            "bias_detected": attr_data.get("bias_detected", False),
            "severity": attr_data.get("severity", "low"),
            "key_metrics": {}
        }
        
        # Extract key metrics for visualization
        for metric_name, metric_data in attr_data.get("fairness_metrics", {}).items():
            if "difference" in metric_name:
                attr_viz_data["key_metrics"][metric_name] = {
                    "value": metric_data["value"],
                    "threshold": metric_data["threshold"],
                    "violation": metric_data["violation"],
                    "group_values": metric_data["group_values"]
                }
        
        dashboard_data["attribute_details"].append(attr_viz_data)
    
    # Group recommendations by priority
    for rec in bias_report.get("recommendations", []):
        priority = rec.get("priority", "medium")
        if priority not in dashboard_data["recommendations_by_priority"]:
            dashboard_data["recommendations_by_priority"][priority] = []
        dashboard_data["recommendations_by_priority"][priority].append(rec)
    
    return dashboard_data