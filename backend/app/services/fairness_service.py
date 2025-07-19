"""
Fairness Analysis Service using Fairlearn and AIF360
Provides comprehensive fairness assessment and bias mitigation capabilities
"""

import pandas as pd
import numpy as np
import json
import tempfile
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from io import BytesIO
import pickle
import base64

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

# Fairlearn imports
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    equalized_odds_difference,
    demographic_parity_difference,
    count,
    false_negative_rate,
    true_negative_rate
)
from fairlearn.reductions import (
    ExponentiatedGradient,
    GridSearch,
    DemographicParity,
    EqualizedOdds
)
from fairlearn.postprocessing import ThresholdOptimizer

# AIF360 imports (optional, with fallback)
try:
    from aif360.algorithms.preprocessing import Reweighing
    from aif360.datasets import BinaryLabelDataset
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False
    
import matplotlib.pyplot as plt
import seaborn as sns
import structlog

warnings.filterwarnings("ignore", category=FutureWarning)
logger = structlog.get_logger(__name__)

class FairnessAnalyzer:
    """
    Comprehensive fairness analysis and bias mitigation
    """
    
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.sensitive_features_train = None
        self.sensitive_features_test = None
        self.model = None
        self.fairness_metrics = None
        
    def load_data(
        self, 
        data: pd.DataFrame, 
        target_column: str,
        sensitive_attribute: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Load and prepare data for fairness analysis
        
        Args:
            data: Input dataset
            target_column: Name of the target/label column
            sensitive_attribute: Name of the sensitive attribute column (e.g., 'gender', 'race')
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with data information
        """
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        if sensitive_attribute not in data.columns:
            raise ValueError(f"Sensitive attribute '{sensitive_attribute}' not found in dataset")
        
        # Prepare features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            if col != sensitive_attribute:  # Keep sensitive attribute as-is for now
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Extract sensitive features
        sensitive_features = X[sensitive_attribute].copy()
        
        # Remove sensitive attribute from features (optional - depends on use case)
        X_features = X.drop(columns=[sensitive_attribute])
        
        # Encode target if needed
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test, self.sensitive_features_train, self.sensitive_features_test = train_test_split(
            X_features, y, sensitive_features, test_size=test_size, random_state=random_state, stratify=y
        )
        
        data_info = {
            "total_samples": len(data),
            "training_samples": len(self.X_train),
            "test_samples": len(self.X_test),
            "features": list(self.X_train.columns),
            "sensitive_attribute": sensitive_attribute,
            "sensitive_groups": list(sensitive_features.unique()),
            "target_classes": list(np.unique(y)),
            "categorical_features": list(categorical_columns)
        }
        
        logger.info(
            "Data loaded for fairness analysis",
            **data_info
        )
        
        return data_info
    
    def train_baseline_model(self, model_type: str = "random_forest") -> Dict[str, Any]:
        """
        Train a baseline model for fairness evaluation
        
        Args:
            model_type: Type of model to train ('random_forest', 'logistic_regression')
            
        Returns:
            Dictionary with model performance metrics
        """
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Initialize model
        if model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate model
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        metrics = {
            "model_type": model_type,
            "train_accuracy": accuracy_score(self.y_train, train_pred),
            "test_accuracy": accuracy_score(self.y_test, test_pred),
            "classification_report": classification_report(self.y_test, test_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(self.y_test, test_pred).tolist()
        }
        
        logger.info(
            "Baseline model trained",
            model_type=model_type,
            test_accuracy=metrics["test_accuracy"]
        )
        
        return metrics
    
    def analyze_fairness(self) -> Dict[str, Any]:
        """
        Perform comprehensive fairness analysis
        
        Returns:
            Dictionary with fairness metrics and analysis
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_baseline_model() first.")
        
        # Get predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Calculate fairness metrics using Fairlearn
        metric_frame = MetricFrame(
            metrics={
                "accuracy": accuracy_score,
                "selection_rate": selection_rate,
                "count": count,
                "true_positive_rate": true_positive_rate,
                "false_positive_rate": false_positive_rate
            },
            y_true=self.y_test,
            y_pred=y_pred,
            sensitive_features=self.sensitive_features_test
        )
        
        # Calculate fairness differences
        demographic_parity_diff = demographic_parity_difference(
            self.y_test, y_pred, sensitive_features=self.sensitive_features_test
        )
        
        equalized_odds_diff = equalized_odds_difference(
            self.y_test, y_pred, sensitive_features=self.sensitive_features_test
        )
        
        # Organize results
        fairness_analysis = {
            "overall_metrics": {
                "demographic_parity_difference": float(demographic_parity_diff),
                "equalized_odds_difference": float(equalized_odds_diff)
            },
            "group_metrics": metric_frame.by_group.to_dict(),
            "overall_performance": metric_frame.overall.to_dict(),
            "sensitive_groups": list(self.sensitive_features_test.unique()),
            "fairness_assessment": self._assess_fairness_level(demographic_parity_diff, equalized_odds_diff),
            "recommendations": self._generate_fairness_recommendations(demographic_parity_diff, equalized_odds_diff)
        }
        
        self.fairness_metrics = fairness_analysis
        
        logger.info(
            "Fairness analysis completed",
            demographic_parity_diff=demographic_parity_diff,
            equalized_odds_diff=equalized_odds_diff,
            fairness_level=fairness_analysis["fairness_assessment"]["level"]
        )
        
        return fairness_analysis
    
    def apply_bias_mitigation(
        self, 
        method: str = "exponentiated_gradient",
        constraint: str = "demographic_parity"
    ) -> Dict[str, Any]:
        """
        Apply bias mitigation techniques
        
        Args:
            method: Mitigation method ('exponentiated_gradient', 'grid_search', 'threshold_optimizer')
            constraint: Fairness constraint ('demographic_parity', 'equalized_odds')
            
        Returns:
            Dictionary with mitigated model results
        """
        if self.X_train is None or self.model is None:
            raise ValueError("Data and baseline model required. Call load_data() and train_baseline_model() first.")
        
        # Set up constraint
        if constraint == "demographic_parity":
            fairness_constraint = DemographicParity()
        elif constraint == "equalized_odds":
            fairness_constraint = EqualizedOdds()
        else:
            raise ValueError(f"Unsupported constraint: {constraint}")
        
        # Apply mitigation method
        if method == "exponentiated_gradient":
            mitigator = ExponentiatedGradient(
                estimator=self.model.__class__(**self.model.get_params()),
                constraints=fairness_constraint
            )
            mitigated_model = mitigator.fit(self.X_train, self.y_train, sensitive_features=self.sensitive_features_train)
            
        elif method == "grid_search":
            mitigator = GridSearch(
                estimator=self.model.__class__(**self.model.get_params()),
                constraints=fairness_constraint
            )
            mitigated_model = mitigator.fit(self.X_train, self.y_train, sensitive_features=self.sensitive_features_train)
            
        elif method == "threshold_optimizer":
            mitigator = ThresholdOptimizer(
                estimator=self.model,
                constraints=constraint,
                prefit=True
            )
            mitigated_model = mitigator.fit(self.X_train, self.y_train, sensitive_features=self.sensitive_features_train)
            
        else:
            raise ValueError(f"Unsupported mitigation method: {method}")
        
        # Evaluate mitigated model
        y_pred_mitigated = mitigated_model.predict(self.X_test, sensitive_features=self.sensitive_features_test)
        
        # Calculate fairness metrics for mitigated model
        metric_frame_mitigated = MetricFrame(
            metrics={
                "accuracy": accuracy_score,
                "selection_rate": selection_rate,
                "count": count,
                "true_positive_rate": true_positive_rate,
                "false_positive_rate": false_positive_rate
            },
            y_true=self.y_test,
            y_pred=y_pred_mitigated,
            sensitive_features=self.sensitive_features_test
        )
        
        demographic_parity_diff_mitigated = demographic_parity_difference(
            self.y_test, y_pred_mitigated, sensitive_features=self.sensitive_features_test
        )
        
        equalized_odds_diff_mitigated = equalized_odds_difference(
            self.y_test, y_pred_mitigated, sensitive_features=self.sensitive_features_test
        )
        
        # Compare with baseline
        baseline_accuracy = accuracy_score(self.y_test, self.model.predict(self.X_test))
        mitigated_accuracy = accuracy_score(self.y_test, y_pred_mitigated)
        
        mitigation_results = {
            "method": method,
            "constraint": constraint,
            "mitigated_metrics": {
                "accuracy": float(mitigated_accuracy),
                "demographic_parity_difference": float(demographic_parity_diff_mitigated),
                "equalized_odds_difference": float(equalized_odds_diff_mitigated)
            },
            "baseline_comparison": {
                "accuracy_change": float(mitigated_accuracy - baseline_accuracy),
                "demographic_parity_improvement": float(abs(self.fairness_metrics["overall_metrics"]["demographic_parity_difference"]) - abs(demographic_parity_diff_mitigated)),
                "equalized_odds_improvement": float(abs(self.fairness_metrics["overall_metrics"]["equalized_odds_difference"]) - abs(equalized_odds_diff_mitigated))
            },
            "group_metrics": metric_frame_mitigated.by_group.to_dict(),
            "fairness_assessment": self._assess_fairness_level(demographic_parity_diff_mitigated, equalized_odds_diff_mitigated)
        }
        
        logger.info(
            "Bias mitigation applied",
            method=method,
            constraint=constraint,
            accuracy_change=mitigation_results["baseline_comparison"]["accuracy_change"],
            fairness_improvement=mitigation_results["baseline_comparison"]["demographic_parity_improvement"]
        )
        
        return mitigation_results
    
    def generate_fairness_report(self) -> str:
        """
        Generate a comprehensive HTML fairness report
        
        Returns:
            HTML report as string
        """
        if self.fairness_metrics is None:
            raise ValueError("Fairness analysis not performed. Call analyze_fairness() first.")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fairness Analysis Report', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy by group
        group_metrics = pd.DataFrame(self.fairness_metrics["group_metrics"])
        if 'accuracy' in group_metrics.index:
            group_metrics.loc['accuracy'].plot(kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('Accuracy by Sensitive Group')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Selection rate by group
        if 'selection_rate' in group_metrics.index:
            group_metrics.loc['selection_rate'].plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Selection Rate by Sensitive Group')
            axes[0, 1].set_ylabel('Selection Rate')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: True Positive Rate by group
        if 'true_positive_rate' in group_metrics.index:
            group_metrics.loc['true_positive_rate'].plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('True Positive Rate by Sensitive Group')
            axes[1, 0].set_ylabel('True Positive Rate')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: False Positive Rate by group
        if 'false_positive_rate' in group_metrics.index:
            group_metrics.loc['false_positive_rate'].plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('False Positive Rate by Sensitive Group')
            axes[1, 1].set_ylabel('False Positive Rate')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Generate HTML report
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fairness Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
                .metric-card {{ background-color: #fff; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin: 15px 0; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                .assessment {{ padding: 15px; border-radius: 8px; margin: 20px 0; }}
                .fair {{ background-color: #d4edda; border-color: #c3e6cb; color: #155724; }}
                .concerning {{ background-color: #fff3cd; border-color: #ffeaa7; color: #856404; }}
                .unfair {{ background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #dee2e6; padding: 8px; text-align: left; }}
                th {{ background-color: #f8f9fa; }}
                .plot {{ text-align: center; margin: 30px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Fairness Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="metric-card">
                <h2>Overall Fairness Metrics</h2>
                <p><strong>Demographic Parity Difference:</strong> 
                   <span class="metric-value">{self.fairness_metrics['overall_metrics']['demographic_parity_difference']:.4f}</span>
                </p>
                <p><strong>Equalized Odds Difference:</strong> 
                   <span class="metric-value">{self.fairness_metrics['overall_metrics']['equalized_odds_difference']:.4f}</span>
                </p>
            </div>
            
            <div class="assessment {self.fairness_metrics['fairness_assessment']['level']}">
                <h3>Fairness Assessment: {self.fairness_metrics['fairness_assessment']['level'].title()}</h3>
                <p>{self.fairness_metrics['fairness_assessment']['description']}</p>
            </div>
            
            <div class="metric-card">
                <h2>Group-wise Performance Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        {' '.join([f'<th>{group}</th>' for group in self.fairness_metrics['sensitive_groups']])}
                    </tr>
                    {self._generate_table_rows()}
                </table>
            </div>
            
            <div class="plot">
                <h2>Fairness Visualization</h2>
                <img src="data:image/png;base64,{plot_base64}" alt="Fairness Metrics Plot" style="max-width: 100%; height: auto;">
            </div>
            
            <div class="metric-card">
                <h2>Recommendations</h2>
                <ul>
                    {' '.join([f'<li>{rec}</li>' for rec in self.fairness_metrics['recommendations']])}
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _assess_fairness_level(self, demographic_parity_diff: float, equalized_odds_diff: float) -> Dict[str, str]:
        """
        Assess the fairness level based on metric thresholds
        """
        # Common fairness thresholds
        fair_threshold = 0.1
        concerning_threshold = 0.2
        
        max_diff = max(abs(demographic_parity_diff), abs(equalized_odds_diff))
        
        if max_diff <= fair_threshold:
            level = "fair"
            description = "The model shows acceptable levels of fairness across groups."
        elif max_diff <= concerning_threshold:
            level = "concerning"
            description = "The model shows some bias that should be addressed through mitigation techniques."
        else:
            level = "unfair"
            description = "The model exhibits significant bias and requires immediate attention."
        
        return {"level": level, "description": description}
    
    def _generate_fairness_recommendations(self, demographic_parity_diff: float, equalized_odds_diff: float) -> List[str]:
        """
        Generate actionable fairness recommendations
        """
        recommendations = []
        
        if abs(demographic_parity_diff) > 0.1:
            recommendations.append(
                "Consider applying demographic parity constraints using bias mitigation techniques."
            )
        
        if abs(equalized_odds_diff) > 0.1:
            recommendations.append(
                "Investigate equalized odds violations and consider post-processing methods."
            )
        
        if abs(demographic_parity_diff) > 0.2 or abs(equalized_odds_diff) > 0.2:
            recommendations.extend([
                "Review data collection process for potential bias sources.",
                "Consider collecting more representative training data.",
                "Implement regular fairness monitoring in production."
            ])
        
        if not recommendations:
            recommendations.append("Model shows good fairness properties. Continue monitoring.")
        
        return recommendations
    
    def _generate_table_rows(self) -> str:
        """
        Generate HTML table rows for group metrics
        """
        group_metrics = pd.DataFrame(self.fairness_metrics["group_metrics"])
        rows = []
        
        for metric in group_metrics.index:
            row_data = [metric]
            for group in self.fairness_metrics['sensitive_groups']:
                value = group_metrics.loc[metric, group]
                row_data.append(f"{value:.4f}" if isinstance(value, (int, float)) else str(value))
            
            rows.append(f"<tr><td>{'</td><td>'.join(map(str, row_data))}</td></tr>")
        
        return '\n'.join(rows)

# Utility functions
def validate_fairness_data(data: pd.DataFrame, target_column: str, sensitive_attribute: str) -> Dict[str, Any]:
    """
    Validate data for fairness analysis
    """
    issues = []
    warnings = []
    
    # Check required columns
    if target_column not in data.columns:
        issues.append(f"Target column '{target_column}' not found")
    
    if sensitive_attribute not in data.columns:
        issues.append(f"Sensitive attribute '{sensitive_attribute}' not found")
    
    # Check data quality
    if len(data) < 100:
        warnings.append("Dataset is small (< 100 samples), results may not be reliable")
    
    # Check class balance
    if target_column in data.columns:
        class_counts = data[target_column].value_counts()
        if len(class_counts) != 2:
            issues.append("Currently only binary classification is supported")
        elif min(class_counts) / max(class_counts) < 0.1:
            warnings.append("Severe class imbalance detected")
    
    # Check sensitive attribute groups
    if sensitive_attribute in data.columns:
        group_counts = data[sensitive_attribute].value_counts()
        if len(group_counts) < 2:
            issues.append("Sensitive attribute must have at least 2 groups")
        elif min(group_counts) < 10:
            warnings.append("Some sensitive groups have very few samples")
    
    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "data_summary": {
            "total_samples": len(data),
            "features": len(data.columns) - 1,  # Excluding target
            "missing_values": data.isnull().sum().sum(),
            "duplicate_rows": data.duplicated().sum()
        }
    }