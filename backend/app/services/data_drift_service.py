"""
Data Drift Detection Service using Evidently AI
Provides comprehensive drift analysis and reporting capabilities
"""

import pandas as pd
import numpy as np
import json
import tempfile
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from io import BytesIO, StringIO

from evidently import Report
from evidently.presets import DataDriftPreset
from evidently import Dataset
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric,
    DatasetSummaryMetric,
    ColumnSummaryMetric,
    RegressionQualityMetric,
    ClassificationQualityMetric,
    ColumnDistributionMetric
)

import structlog

warnings.filterwarnings("ignore", category=FutureWarning)
logger = structlog.get_logger(__name__)

class DataDriftAnalyzer:
    """
    Comprehensive data drift analysis using Evidently AI
    """
    
    def __init__(self):
        self.reference_df = None
        self.current_df = None
        self.reference_predictions = None
        self.current_predictions = None
        self.reference_labels = None
        self.current_labels = None
        self.model = None
        
    def load_data(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load reference and current datasets for drift analysis
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset to compare against reference
            
        Returns:
            Tuple of (reference_df, current_df)
        """
        # Ensure both datasets have same columns
        common_columns = list(set(reference_data.columns) & set(current_data.columns))
        
        if not common_columns:
            raise ValueError("No common columns found between reference and current datasets")
            
        self.reference_df = reference_data[common_columns].copy()
        self.current_df = current_data[common_columns].copy()
        
        logger.info(
            "Data loaded successfully",
            reference_shape=self.reference_df.shape,
            current_shape=self.current_df.shape,
            common_columns=len(common_columns)
        )
        
        return self.reference_df, self.current_df
    
    def split_single_dataset(self, data: pd.DataFrame, split_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split single dataset into reference and current portions
        
        Args:
            data: Input dataset
            split_ratio: Ratio for reference data (default: 0.7)
            
        Returns:
            Tuple of (reference_df, current_df)
        """
        split_idx = int(len(data) * split_ratio)
        
        self.reference_df = data.iloc[:split_idx].copy()
        self.current_df = data.iloc[split_idx:].copy()
        
        logger.info(
            "Dataset split completed",
            reference_shape=self.reference_df.shape,
            current_shape=self.current_df.shape,
            split_ratio=split_ratio
        )
        
        return self.reference_df, self.current_df
    
    def split_by_time(
        self, 
        data: pd.DataFrame, 
        time_column: str,
        reference_start: str,
        reference_end: str,
        current_start: str,
        current_end: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset by time periods
        
        Args:
            data: Input dataset with time column
            time_column: Name of the time column
            reference_start: Start date for reference period (YYYY-MM-DD)
            reference_end: End date for reference period (YYYY-MM-DD)
            current_start: Start date for current period (YYYY-MM-DD)
            current_end: End date for current period (YYYY-MM-DD)
            
        Returns:
            Tuple of (reference_df, current_df)
        """
        # Convert time column to datetime
        data[time_column] = pd.to_datetime(data[time_column])
        
        # Filter reference period
        ref_mask = (data[time_column] >= reference_start) & (data[time_column] <= reference_end)
        self.reference_df = data[ref_mask].copy()
        
        # Filter current period
        cur_mask = (data[time_column] >= current_start) & (data[time_column] <= current_end)
        self.current_df = data[cur_mask].copy()
        
        logger.info(
            "Time-based split completed",
            reference_period=f"{reference_start} to {reference_end}",
            current_period=f"{current_start} to {current_end}",
            reference_shape=self.reference_df.shape,
            current_shape=self.current_df.shape
        )
        
        return self.reference_df, self.current_df
    
    def preprocess_data(self, drop_columns: Optional[List[str]] = None) -> None:
        """
        Preprocess datasets before drift analysis
        
        Args:
            drop_columns: List of columns to drop from analysis
        """
        if drop_columns:
            self.reference_df = self.reference_df.drop(columns=drop_columns, errors='ignore')
            self.current_df = self.current_df.drop(columns=drop_columns, errors='ignore')
            
        # Handle missing values
        self.reference_df = self.reference_df.dropna()
        self.current_df = self.current_df.dropna()
        
        logger.info(
            "Data preprocessing completed",
            dropped_columns=drop_columns,
            reference_shape=self.reference_df.shape,
            current_shape=self.current_df.shape
        )
    
    def generate_full_drift_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive drift report for all columns
        
        Returns:
            Dictionary containing HTML report and drift metrics
        """
        if self.reference_df is None or self.current_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Create Evidently datasets
        reference_dataset = Dataset.from_pandas(self.reference_df)
        current_dataset = Dataset.from_pandas(self.current_df)
        
        # Create comprehensive report
        report = Report(metrics=[
            DataDriftPreset(),
            DatasetDriftMetric(),
            DatasetSummaryMetric(),
            DataDriftTable(),
        ])
        
        # Run the report
        report.run(reference_data=reference_dataset, current_data=current_dataset)
        
        # Get HTML report
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp:
            report.save_html(tmp.name)
            with open(tmp.name, 'r', encoding='utf-8') as f:
                html_content = f.read()
            os.unlink(tmp.name)
        
        # Get JSON data for analysis
        json_data = json.loads(report.json())
        
        # Extract drift summary
        drift_summary = self._extract_drift_summary(json_data)
        
        result = {
            "html_report": html_content,
            "json_data": json_data,
            "drift_summary": drift_summary,
            "metadata": {
                "reference_shape": self.reference_df.shape,
                "current_shape": self.current_df.shape,
                "columns_analyzed": list(self.reference_df.columns),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info(
            "Full drift report generated",
            columns_analyzed=len(self.reference_df.columns),
            drift_detected=drift_summary.get("overall_drift_detected", False)
        )
        
        return result
    
    def generate_column_drift_report(self, column_name: str) -> Dict[str, Any]:
        """
        Generate drift report for a specific column
        
        Args:
            column_name: Name of the column to analyze
            
        Returns:
            Dictionary containing HTML report and column-specific drift metrics
        """
        if column_name not in self.reference_df.columns:
            raise ValueError(f"Column '{column_name}' not found in the dataset")
        
        # Create single-column datasets
        ref_col_df = self.reference_df[[column_name]].copy()
        cur_col_df = self.current_df[[column_name]].copy()
        
        reference_dataset = Dataset.from_pandas(ref_col_df)
        current_dataset = Dataset.from_pandas(cur_col_df)
        
        # Create column-specific report
        report = Report(metrics=[
            ColumnDriftMetric(column_name=column_name),
            ColumnSummaryMetric(column_name=column_name),
            DataDriftPreset()
        ])
        
        # Run the report
        report.run(reference_data=reference_dataset, current_data=current_dataset)
        
        # Get HTML report
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp:
            report.save_html(tmp.name)
            with open(tmp.name, 'r', encoding='utf-8') as f:
                html_content = f.read()
            os.unlink(tmp.name)
        
        # Get JSON data
        json_data = json.loads(report.json())
        
        # Extract column-specific summary
        column_summary = self._extract_column_drift_summary(json_data, column_name)
        
        result = {
            "html_report": html_content,
            "json_data": json_data,
            "column_summary": column_summary,
            "column_name": column_name,
            "metadata": {
                "reference_count": len(ref_col_df),
                "current_count": len(cur_col_df),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info(
            "Column drift report generated",
            column_name=column_name,
            drift_detected=column_summary.get("drift_detected", False)
        )
        
        return result
    
    def get_data_preview(self, n_rows: int = 5) -> Dict[str, Any]:
        """
        Get preview of reference and current datasets
        
        Args:
            n_rows: Number of rows to include in preview
            
        Returns:
            Dictionary with data previews
        """
        if self.reference_df is None or self.current_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Clean data for JSON serialization
        ref_preview = self.reference_df.head(n_rows).replace([np.inf, -np.inf, np.nan], None)
        cur_preview = self.current_df.head(n_rows).replace([np.inf, -np.inf, np.nan], None)
        
        return {
            "reference_preview": ref_preview.to_dict(orient='records'),
            "current_preview": cur_preview.to_dict(orient='records'),
            "reference_info": {
                "shape": self.reference_df.shape,
                "columns": list(self.reference_df.columns),
                "dtypes": self.reference_df.dtypes.astype(str).to_dict()
            },
            "current_info": {
                "shape": self.current_df.shape,
                "columns": list(self.current_df.columns),
                "dtypes": self.current_df.dtypes.astype(str).to_dict()
            }
        }
    
    def _extract_drift_summary(self, json_data: Dict) -> Dict[str, Any]:
        """
        Extract high-level drift summary from Evidently JSON output
        """
        summary = {
            "overall_drift_detected": False,
            "total_columns": 0,
            "drifted_columns": 0,
            "drift_percentage": 0.0,
            "column_results": []
        }
        
        try:
            # Navigate through Evidently's JSON structure
            if "metrics" in json_data:
                for metric in json_data["metrics"]:
                    if metric.get("metric") == "DatasetDriftMetric":
                        result = metric.get("result", {})
                        summary["overall_drift_detected"] = result.get("dataset_drift", False)
                        summary["total_columns"] = result.get("number_of_columns", 0)
                        summary["drifted_columns"] = result.get("number_of_drifted_columns", 0)
                        
                        if summary["total_columns"] > 0:
                            summary["drift_percentage"] = (summary["drifted_columns"] / summary["total_columns"]) * 100
                    
                    elif metric.get("metric") == "DataDriftTable":
                        result = metric.get("result", {})
                        if "drift_by_columns" in result:
                            for col_name, col_result in result["drift_by_columns"].items():
                                summary["column_results"].append({
                                    "column": col_name,
                                    "drift_detected": col_result.get("drift_detected", False),
                                    "drift_score": col_result.get("drift_score"),
                                    "threshold": col_result.get("threshold"),
                                    "stattest_name": col_result.get("stattest_name")
                                })
                                
        except Exception as e:
            logger.error("Error extracting drift summary", error=str(e))
        
        return summary
    
    def _extract_column_drift_summary(self, json_data: Dict, column_name: str) -> Dict[str, Any]:
        """
        Extract drift summary for a specific column
        """
        summary = {
            "column_name": column_name,
            "drift_detected": False,
            "drift_score": None,
            "threshold": None,
            "stattest_name": None,
            "p_value": None
        }
        
        try:
            if "metrics" in json_data:
                for metric in json_data["metrics"]:
                    if (metric.get("metric") == "ColumnDriftMetric" and 
                        metric.get("parameters", {}).get("column_name") == column_name):
                        result = metric.get("result", {})
                        summary.update({
                            "drift_detected": result.get("drift_detected", False),
                            "drift_score": result.get("drift_score"),
                            "threshold": result.get("threshold"),
                            "stattest_name": result.get("stattest_name"),
                            "p_value": result.get("p_value")
                        })
                        break
                        
        except Exception as e:
            logger.error("Error extracting column drift summary", error=str(e), column=column_name)
        
        return summary
    
    def load_model_predictions(
        self, 
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
        reference_labels: Optional[np.ndarray] = None,
        current_labels: Optional[np.ndarray] = None,
        model=None
    ) -> None:
        """
        Load model predictions for model drift analysis
        
        Args:
            reference_predictions: Model predictions on reference data
            current_predictions: Model predictions on current data
            reference_labels: True labels for reference data (optional)
            current_labels: True labels for current data (optional)
            model: The trained model object (optional)
        """
        self.reference_predictions = reference_predictions
        self.current_predictions = current_predictions
        self.reference_labels = reference_labels
        self.current_labels = current_labels
        self.model = model
        
        logger.info(
            "Model predictions loaded",
            reference_predictions_shape=reference_predictions.shape,
            current_predictions_shape=current_predictions.shape,
            has_labels=reference_labels is not None and current_labels is not None
        )
    
    def analyze_model_drift(self, task_type: str = "classification") -> Dict[str, Any]:
        """
        Analyze model drift by comparing model predictions and performance
        
        Args:
            task_type: Type of ML task ("classification" or "regression")
            
        Returns:
            Dictionary containing model drift analysis
        """
        if self.reference_predictions is None or self.current_predictions is None:
            raise ValueError("Model predictions not loaded. Call load_model_predictions() first.")
        
        # Create datasets with predictions
        ref_pred_df = self.reference_df.copy()
        cur_pred_df = self.current_df.copy()
        
        ref_pred_df['prediction'] = self.reference_predictions
        cur_pred_df['prediction'] = self.current_predictions
        
        # Add true labels if available
        if self.reference_labels is not None and self.current_labels is not None:
            ref_pred_df['target'] = self.reference_labels
            cur_pred_df['target'] = self.current_labels
        
        # Create Evidently datasets
        reference_dataset = Dataset.from_pandas(ref_pred_df)
        current_dataset = Dataset.from_pandas(cur_pred_df)
        
        # Create model drift report
        metrics = [
            DataDriftPreset(),
            DatasetDriftMetric(),
            ColumnDriftMetric(column_name='prediction'),
        ]
        
        # Add performance metrics if labels are available
        if self.reference_labels is not None and self.current_labels is not None:
            if task_type == "classification":
                metrics.append(ClassificationQualityMetric())
            elif task_type == "regression":
                metrics.append(RegressionQualityMetric())
        
        report = Report(metrics=metrics)
        
        # Run the report
        report.run(reference_data=reference_dataset, current_data=current_dataset)
        
        # Get HTML report
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp:
            report.save_html(tmp.name)
            with open(tmp.name, 'r', encoding='utf-8') as f:
                html_content = f.read()
            os.unlink(tmp.name)
        
        # Get JSON data
        json_data = json.loads(report.json())
        
        # Extract model drift insights
        model_drift_summary = self._extract_model_drift_summary(json_data, task_type)
        
        result = {
            "html_report": html_content,
            "json_data": json_data,
            "model_drift_summary": model_drift_summary,
            "prediction_distribution_changes": self._analyze_prediction_distribution_changes(),
            "performance_degradation": self._calculate_performance_degradation(task_type),
            "metadata": {
                "task_type": task_type,
                "reference_predictions_count": len(self.reference_predictions),
                "current_predictions_count": len(self.current_predictions),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info(
            "Model drift analysis completed",
            task_type=task_type,
            prediction_drift_detected=model_drift_summary.get("prediction_drift_detected", False)
        )
        
        return result
    
    def _extract_model_drift_summary(self, json_data: Dict, task_type: str) -> Dict[str, Any]:
        """
        Extract model drift summary from Evidently JSON output
        """
        summary = {
            "prediction_drift_detected": False,
            "prediction_drift_score": None,
            "prediction_distribution_change": None,
            "performance_change": None,
            "drift_explanation": ""
        }
        
        try:
            # Look for prediction column drift
            if "metrics" in json_data:
                for metric in json_data["metrics"]:
                    if (metric.get("metric") == "ColumnDriftMetric" and 
                        metric.get("parameters", {}).get("column_name") == "prediction"):
                        result = metric.get("result", {})
                        summary["prediction_drift_detected"] = result.get("drift_detected", False)
                        summary["prediction_drift_score"] = result.get("drift_score")
                        
                        if summary["prediction_drift_detected"]:
                            summary["drift_explanation"] = (
                                f"Model predictions show significant distribution changes. "
                                f"Drift score: {summary['prediction_drift_score']:.4f}. "
                                f"This indicates the model is behaving differently on current data compared to reference data."
                            )
                        else:
                            summary["drift_explanation"] = (
                                "Model predictions remain consistent between reference and current data. "
                                "No significant model drift detected."
                            )
                        break
                    
                    # Check for performance metrics
                    elif metric.get("metric") in ["ClassificationQualityMetric", "RegressionQualityMetric"]:
                        result = metric.get("result", {})
                        summary["performance_change"] = result
                        
        except Exception as e:
            logger.error("Error extracting model drift summary", error=str(e))
            summary["drift_explanation"] = "Error analyzing model drift"
        
        return summary
    
    def _analyze_prediction_distribution_changes(self) -> Dict[str, Any]:
        """
        Analyze changes in prediction distributions
        """
        try:
            from scipy import stats
            
            # Calculate distribution statistics
            ref_stats = {
                "mean": float(np.mean(self.reference_predictions)),
                "std": float(np.std(self.reference_predictions)),
                "min": float(np.min(self.reference_predictions)),
                "max": float(np.max(self.reference_predictions)),
                "median": float(np.median(self.reference_predictions))
            }
            
            cur_stats = {
                "mean": float(np.mean(self.current_predictions)),
                "std": float(np.std(self.current_predictions)),
                "min": float(np.min(self.current_predictions)),
                "max": float(np.max(self.current_predictions)),
                "median": float(np.median(self.current_predictions))
            }
            
            # Calculate statistical tests
            ks_statistic, ks_p_value = stats.ks_2samp(self.reference_predictions, self.current_predictions)
            
            # Calculate prediction shift metrics
            mean_shift = cur_stats["mean"] - ref_stats["mean"]
            std_ratio = cur_stats["std"] / ref_stats["std"] if ref_stats["std"] > 0 else 1.0
            
            return {
                "reference_stats": ref_stats,
                "current_stats": cur_stats,
                "distribution_tests": {
                    "kolmogorov_smirnov": {
                        "statistic": float(ks_statistic),
                        "p_value": float(ks_p_value),
                        "significant_change": ks_p_value < 0.05
                    }
                },
                "shift_metrics": {
                    "mean_shift": float(mean_shift),
                    "std_ratio": float(std_ratio),
                    "relative_mean_change": float(mean_shift / ref_stats["mean"]) if ref_stats["mean"] != 0 else 0
                },
                "interpretation": self._interpret_prediction_changes(mean_shift, std_ratio, ks_p_value)
            }
            
        except Exception as e:
            logger.error("Error analyzing prediction distribution changes", error=str(e))
            return {"error": str(e)}
    
    def _calculate_performance_degradation(self, task_type: str) -> Dict[str, Any]:
        """
        Calculate performance degradation if true labels are available
        """
        if self.reference_labels is None or self.current_labels is None:
            return {
                "available": False,
                "message": "True labels not provided - cannot calculate performance degradation"
            }
        
        try:
            from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
            
            if task_type == "classification":
                ref_accuracy = accuracy_score(self.reference_labels, self.reference_predictions)
                cur_accuracy = accuracy_score(self.current_labels, self.current_predictions)
                
                ref_f1 = f1_score(self.reference_labels, self.reference_predictions, average='weighted')
                cur_f1 = f1_score(self.current_labels, self.current_predictions, average='weighted')
                
                return {
                    "available": True,
                    "task_type": task_type,
                    "reference_performance": {
                        "accuracy": float(ref_accuracy),
                        "f1_score": float(ref_f1)
                    },
                    "current_performance": {
                        "accuracy": float(cur_accuracy),
                        "f1_score": float(cur_f1)
                    },
                    "degradation": {
                        "accuracy_change": float(cur_accuracy - ref_accuracy),
                        "f1_change": float(cur_f1 - ref_f1),
                        "relative_accuracy_change": float((cur_accuracy - ref_accuracy) / ref_accuracy) if ref_accuracy > 0 else 0
                    },
                    "interpretation": self._interpret_performance_change(cur_accuracy - ref_accuracy, task_type)
                }
                
            elif task_type == "regression":
                ref_mse = mean_squared_error(self.reference_labels, self.reference_predictions)
                cur_mse = mean_squared_error(self.current_labels, self.current_predictions)
                
                ref_r2 = r2_score(self.reference_labels, self.reference_predictions)
                cur_r2 = r2_score(self.current_labels, self.current_predictions)
                
                return {
                    "available": True,
                    "task_type": task_type,
                    "reference_performance": {
                        "mse": float(ref_mse),
                        "r2_score": float(ref_r2)
                    },
                    "current_performance": {
                        "mse": float(cur_mse),
                        "r2_score": float(cur_r2)
                    },
                    "degradation": {
                        "mse_change": float(cur_mse - ref_mse),
                        "r2_change": float(cur_r2 - ref_r2),
                        "relative_mse_change": float((cur_mse - ref_mse) / ref_mse) if ref_mse > 0 else 0
                    },
                    "interpretation": self._interpret_performance_change(cur_r2 - ref_r2, task_type)
                }
                
        except Exception as e:
            logger.error("Error calculating performance degradation", error=str(e))
            return {"available": False, "error": str(e)}
    
    def _interpret_prediction_changes(self, mean_shift: float, std_ratio: float, ks_p_value: float) -> str:
        """
        Interpret prediction distribution changes
        """
        interpretations = []
        
        if abs(mean_shift) > 0.1:
            direction = "increased" if mean_shift > 0 else "decreased"
            interpretations.append(f"Model predictions have {direction} on average by {abs(mean_shift):.3f}")
        
        if std_ratio > 1.2:
            interpretations.append("Prediction variance has increased, indicating less consistent model behavior")
        elif std_ratio < 0.8:
            interpretations.append("Prediction variance has decreased, indicating more consistent but potentially overconfident model behavior")
        
        if ks_p_value < 0.05:
            interpretations.append("Statistical tests confirm significant changes in prediction distributions")
        
        if not interpretations:
            return "Model predictions remain stable between reference and current periods"
        
        return ". ".join(interpretations) + "."
    
    def _interpret_performance_change(self, performance_change: float, task_type: str) -> str:
        """
        Interpret performance changes
        """
        if task_type == "classification":
            if performance_change < -0.05:
                return f"Significant performance degradation detected (accuracy dropped by {abs(performance_change):.3f})"
            elif performance_change > 0.05:
                return f"Model performance has improved (accuracy increased by {performance_change:.3f})"
            else:
                return "Model performance remains stable"
        
        elif task_type == "regression":
            if performance_change < -0.05:  # R² decreased
                return f"Model performance has degraded (R² score dropped by {abs(performance_change):.3f})"
            elif performance_change > 0.05:
                return f"Model performance has improved (R² score increased by {performance_change:.3f})"
            else:
                return "Model performance remains stable"
        
        return "Performance change analysis not available"

# Utility functions for data loading
def load_csv_flexible(file_content: bytes) -> pd.DataFrame:
    """
    Load CSV with flexible delimiter detection
    """
    import chardet
    
    # Detect encoding
    encoding = chardet.detect(file_content)["encoding"]
    
    # Try comma first
    try:
        df = pd.read_csv(BytesIO(file_content), encoding=encoding)
        if df.shape[1] > 1:
            return df
    except:
        pass
    
    # Try semicolon
    try:
        df = pd.read_csv(BytesIO(file_content), sep=";", encoding=encoding)
        if df.shape[1] > 1:
            return df
    except:
        pass
    
    # Try tab
    try:
        df = pd.read_csv(BytesIO(file_content), sep="\t", encoding=encoding)
        if df.shape[1] > 1:
            return df
    except:
        pass
    
    raise ValueError("Could not parse CSV file with any common delimiter")

def validate_datasets_compatibility(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that reference and current datasets are compatible for drift analysis
    """
    issues = []
    warnings = []
    
    # Check if datasets have common columns
    ref_columns = set(ref_df.columns)
    cur_columns = set(cur_df.columns)
    common_columns = ref_columns & cur_columns
    
    if not common_columns:
        issues.append("No common columns found between datasets")
    
    missing_in_current = ref_columns - cur_columns
    missing_in_reference = cur_columns - ref_columns
    
    if missing_in_current:
        warnings.append(f"Columns in reference but not in current: {list(missing_in_current)}")
    
    if missing_in_reference:
        warnings.append(f"Columns in current but not in reference: {list(missing_in_reference)}")
    
    # Check data types compatibility
    for col in common_columns:
        ref_dtype = ref_df[col].dtype
        cur_dtype = cur_df[col].dtype
        
        if ref_dtype != cur_dtype:
            warnings.append(f"Column '{col}' has different data types: {ref_dtype} vs {cur_dtype}")
    
    # Check for minimum data requirements
    if len(ref_df) < 10:
        warnings.append("Reference dataset has fewer than 10 rows")
    
    if len(cur_df) < 10:
        warnings.append("Current dataset has fewer than 10 rows")
    
    return {
        "is_compatible": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "common_columns": list(common_columns),
        "total_common_columns": len(common_columns)
    }