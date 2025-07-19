"""
Enterprise Data Drift Detection Service
Advanced drift detection with Jensen-Shannon Divergence, PSI, and statistical tests
Compatible with Fiddler.ai enterprise standards with 15+ drift detection methods
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, chi2_contingency, mannwhitneyu, wasserstein_distance
import warnings
import structlog

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = structlog.get_logger(__name__)


class EnterpriseDriftDetector:
    """
    Enterprise-grade drift detection with comprehensive statistical methods
    Supporting 15+ drift detection algorithms used by industry leaders
    """
    
    def __init__(self):
        self.supported_methods = {
            # Statistical tests
            "kolmogorov_smirnov": self._ks_test,
            "anderson_darling": self._anderson_darling_test,
            "mann_whitney_u": self._mann_whitney_test,
            "chi_square": self._chi_square_test,
            "wasserstein": self._wasserstein_distance,
            
            # Information-theoretic measures
            "jensen_shannon_divergence": self._jensen_shannon_divergence,
            "kl_divergence": self._kl_divergence,
            "hellinger_distance": self._hellinger_distance,
            
            # Population stability measures
            "population_stability_index": self._population_stability_index,
            "characteristic_stability_index": self._characteristic_stability_index,
            
            # Distribution-based measures
            "earth_movers_distance": self._earth_movers_distance,
            "energy_distance": self._energy_distance,
            "maximum_mean_discrepancy": self._maximum_mean_discrepancy,
            
            # Ensemble methods
            "drift_ensemble": self._ensemble_drift_detection,
            "adaptive_windowing": self._adaptive_windowing_drift
        }
        
        self.thresholds = {
            "kolmogorov_smirnov": 0.05,
            "jensen_shannon_divergence": 0.1,
            "population_stability_index": 0.1,
            "wasserstein": 0.1,
            "chi_square": 0.05,
            "mann_whitney_u": 0.05,
            "anderson_darling": 0.05,
            "kl_divergence": 0.1,
            "hellinger_distance": 0.1,
            "characteristic_stability_index": 0.1,
            "earth_movers_distance": 0.1,
            "energy_distance": 0.1,
            "maximum_mean_discrepancy": 0.1
        }
    
    def detect_comprehensive_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
        custom_thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive drift detection using multiple methods
        
        Args:
            reference_data: Reference/training dataset
            current_data: Current/production dataset
            feature_columns: Specific features to analyze
            categorical_features: List of categorical feature names
            methods: Specific drift detection methods to use
            custom_thresholds: Custom thresholds for drift detection
            
        Returns:
            Comprehensive drift analysis report
        """
        logger.info("Starting comprehensive drift detection",
                   reference_size=len(reference_data),
                   current_size=len(current_data),
                   methods=methods)
        
        # Prepare data
        if feature_columns is None:
            feature_columns = list(reference_data.columns)
        
        if categorical_features is None:
            categorical_features = list(reference_data.select_dtypes(include=['object', 'category']).columns)
        
        numerical_features = [col for col in feature_columns if col not in categorical_features]
        
        # Use custom thresholds if provided
        thresholds = {**self.thresholds, **(custom_thresholds or {})}
        
        # Use all methods if none specified
        if methods is None:
            methods = list(self.supported_methods.keys())
        
        drift_report = {
            "analysis_metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "reference_size": len(reference_data),
                "current_size": len(current_data),
                "features_analyzed": feature_columns,
                "categorical_features": categorical_features,
                "numerical_features": numerical_features,
                "methods_used": methods
            },
            "overall_drift_detected": False,
            "overall_drift_score": 0.0,
            "feature_drift_results": {},
            "method_summary": {},
            "enterprise_metrics": {},
            "recommendations": []
        }
        
        # Analyze each feature
        for feature in feature_columns:
            if feature not in reference_data.columns or feature not in current_data.columns:
                logger.warning(f"Feature {feature} not found in data")
                continue
            
            feature_results = self._analyze_feature_drift(
                reference_data[feature],
                current_data[feature],
                feature,
                feature in categorical_features,
                methods,
                thresholds
            )
            
            drift_report["feature_drift_results"][feature] = feature_results
        
        # Calculate overall metrics
        drift_report["overall_drift_score"] = self._calculate_overall_drift_score(
            drift_report["feature_drift_results"]
        )
        
        drift_report["overall_drift_detected"] = drift_report["overall_drift_score"] > 0.1
        
        # Method-level summary
        drift_report["method_summary"] = self._calculate_method_summary(
            drift_report["feature_drift_results"], methods
        )
        
        # Enterprise-specific metrics
        drift_report["enterprise_metrics"] = self._calculate_enterprise_metrics(
            reference_data, current_data, drift_report["feature_drift_results"]
        )
        
        # Generate recommendations
        drift_report["recommendations"] = self._generate_drift_recommendations(drift_report)
        
        logger.info("Comprehensive drift detection completed",
                   overall_score=drift_report["overall_drift_score"],
                   drift_detected=drift_report["overall_drift_detected"])
        
        return drift_report
    
    def _analyze_feature_drift(
        self,
        reference_series: pd.Series,
        current_series: pd.Series,
        feature_name: str,
        is_categorical: bool,
        methods: List[str],
        thresholds: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze drift for a single feature using multiple methods"""
        
        feature_results = {
            "feature_name": feature_name,
            "is_categorical": is_categorical,
            "drift_detected": False,
            "drift_methods": {},
            "drift_score": 0.0,
            "data_quality": self._assess_feature_data_quality(reference_series, current_series)
        }
        
        # Clean data
        ref_clean = reference_series.dropna()
        curr_clean = current_series.dropna()
        
        if len(ref_clean) == 0 or len(curr_clean) == 0:
            logger.warning(f"No valid data for feature {feature_name}")
            return feature_results
        
        # Apply each method
        for method in methods:
            if method not in self.supported_methods:
                logger.warning(f"Unknown drift detection method: {method}")
                continue
            
            try:
                method_func = self.supported_methods[method]
                
                # Check if method is applicable
                if is_categorical and method in ["anderson_darling", "mann_whitney_u", "wasserstein"]:
                    continue  # Skip methods not suitable for categorical data
                
                if not is_categorical and method in ["chi_square"]:
                    continue  # Skip methods only for categorical data
                
                result = method_func(ref_clean, curr_clean, is_categorical)
                
                # Determine if drift is detected
                threshold = thresholds.get(method, 0.1)
                drift_detected = self._evaluate_drift_threshold(result, method, threshold)
                
                feature_results["drift_methods"][method] = {
                    "result": result,
                    "threshold": threshold,
                    "drift_detected": drift_detected,
                    "score": self._normalize_drift_score(result, method)
                }
                
                if drift_detected:
                    feature_results["drift_detected"] = True
                
            except Exception as e:
                logger.error(f"Error in drift method {method} for feature {feature_name}: {str(e)}")
                continue
        
        # Calculate overall feature drift score
        if feature_results["drift_methods"]:
            scores = [m["score"] for m in feature_results["drift_methods"].values()]
            feature_results["drift_score"] = np.mean(scores)
        
        return feature_results
    
    def _ks_test(self, ref_data: pd.Series, curr_data: pd.Series, is_categorical: bool) -> Dict[str, float]:
        """Kolmogorov-Smirnov test for distribution comparison"""
        if is_categorical:
            # Convert to numerical for KS test
            combined = pd.concat([ref_data, curr_data])
            categories = combined.unique()
            ref_encoded = pd.Categorical(ref_data, categories=categories).codes
            curr_encoded = pd.Categorical(curr_data, categories=categories).codes
            statistic, p_value = ks_2samp(ref_encoded, curr_encoded)
        else:
            statistic, p_value = ks_2samp(ref_data, curr_data)
        
        return {"statistic": float(statistic), "p_value": float(p_value)}
    
    def _anderson_darling_test(self, ref_data: pd.Series, curr_data: pd.Series, is_categorical: bool) -> Dict[str, float]:
        """Anderson-Darling test for distribution comparison"""
        try:
            from scipy.stats import anderson_ksamp
            statistic, critical_values, p_value = anderson_ksamp([ref_data.values, curr_data.values])
            return {"statistic": float(statistic), "p_value": float(p_value) if p_value is not None else 0.0}
        except ImportError:
            logger.warning("Anderson-Darling test not available, falling back to KS test")
            return self._ks_test(ref_data, curr_data, is_categorical)
    
    def _mann_whitney_test(self, ref_data: pd.Series, curr_data: pd.Series, is_categorical: bool) -> Dict[str, float]:
        """Mann-Whitney U test for distribution comparison"""
        if is_categorical:
            # Convert to numerical
            combined = pd.concat([ref_data, curr_data])
            categories = combined.unique()
            ref_encoded = pd.Categorical(ref_data, categories=categories).codes
            curr_encoded = pd.Categorical(curr_data, categories=categories).codes
            statistic, p_value = mannwhitneyu(ref_encoded, curr_encoded, alternative='two-sided')
        else:
            statistic, p_value = mannwhitneyu(ref_data, curr_data, alternative='two-sided')
        
        return {"statistic": float(statistic), "p_value": float(p_value)}
    
    def _chi_square_test(self, ref_data: pd.Series, curr_data: pd.Series, is_categorical: bool) -> Dict[str, float]:
        """Chi-square test for categorical distribution comparison"""
        if not is_categorical:
            # Bin numerical data
            combined = pd.concat([ref_data, curr_data])
            bins = np.histogram_bin_edges(combined, bins='auto')
            ref_binned = pd.cut(ref_data, bins=bins, include_lowest=True)
            curr_binned = pd.cut(curr_data, bins=bins, include_lowest=True)
        else:
            ref_binned = ref_data
            curr_binned = curr_data
        
        # Create contingency table
        ref_counts = ref_binned.value_counts().sort_index()
        curr_counts = curr_binned.value_counts().sort_index()
        
        # Align indices
        all_categories = ref_counts.index.union(curr_counts.index)
        ref_aligned = ref_counts.reindex(all_categories, fill_value=0)
        curr_aligned = curr_counts.reindex(all_categories, fill_value=0)
        
        # Perform chi-square test
        contingency_table = np.array([ref_aligned.values, curr_aligned.values])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        return {"statistic": float(chi2), "p_value": float(p_value), "dof": int(dof)}
    
    def _wasserstein_distance(self, ref_data: pd.Series, curr_data: pd.Series, is_categorical: bool) -> Dict[str, float]:
        """Wasserstein (Earth Mover's) distance"""
        if is_categorical:
            # Convert to numerical
            combined = pd.concat([ref_data, curr_data])
            categories = combined.unique()
            ref_encoded = pd.Categorical(ref_data, categories=categories).codes
            curr_encoded = pd.Categorical(curr_data, categories=categories).codes
            distance = wasserstein_distance(ref_encoded, curr_encoded)
        else:
            distance = wasserstein_distance(ref_data, curr_data)
        
        return {"distance": float(distance)}
    
    def _jensen_shannon_divergence(self, ref_data: pd.Series, curr_data: pd.Series, is_categorical: bool) -> Dict[str, float]:
        """Jensen-Shannon Divergence calculation"""
        
        if is_categorical:
            # Calculate probability distributions for categorical data
            ref_probs = ref_data.value_counts(normalize=True).sort_index()
            curr_probs = curr_data.value_counts(normalize=True).sort_index()
            
            # Align distributions
            all_categories = ref_probs.index.union(curr_probs.index)
            ref_aligned = ref_probs.reindex(all_categories, fill_value=1e-10)
            curr_aligned = curr_probs.reindex(all_categories, fill_value=1e-10)
            
            # Normalize to ensure they sum to 1
            ref_aligned = ref_aligned / ref_aligned.sum()
            curr_aligned = curr_aligned / curr_aligned.sum()
            
            js_distance = jensenshannon(ref_aligned.values, curr_aligned.values)
        else:
            # For numerical data, create histograms
            combined = pd.concat([ref_data, curr_data])
            bins = np.histogram_bin_edges(combined, bins=50)
            
            ref_hist, _ = np.histogram(ref_data, bins=bins, density=True)
            curr_hist, _ = np.histogram(curr_data, bins=bins, density=True)
            
            # Normalize histograms
            ref_hist = ref_hist / np.sum(ref_hist)
            curr_hist = curr_hist / np.sum(curr_hist)
            
            # Add small epsilon to avoid log(0)
            ref_hist = ref_hist + 1e-10
            curr_hist = curr_hist + 1e-10
            
            js_distance = jensenshannon(ref_hist, curr_hist)
        
        return {"js_divergence": float(js_distance)}
    
    def _kl_divergence(self, ref_data: pd.Series, curr_data: pd.Series, is_categorical: bool) -> Dict[str, float]:
        """Kullback-Leibler Divergence calculation"""
        
        if is_categorical:
            ref_probs = ref_data.value_counts(normalize=True).sort_index()
            curr_probs = curr_data.value_counts(normalize=True).sort_index()
            
            all_categories = ref_probs.index.union(curr_probs.index)
            ref_aligned = ref_probs.reindex(all_categories, fill_value=1e-10)
            curr_aligned = curr_probs.reindex(all_categories, fill_value=1e-10)
            
            ref_aligned = ref_aligned / ref_aligned.sum()
            curr_aligned = curr_aligned / curr_aligned.sum()
            
            kl_div = stats.entropy(curr_aligned.values, ref_aligned.values)
        else:
            combined = pd.concat([ref_data, curr_data])
            bins = np.histogram_bin_edges(combined, bins=50)
            
            ref_hist, _ = np.histogram(ref_data, bins=bins, density=True)
            curr_hist, _ = np.histogram(curr_data, bins=bins, density=True)
            
            ref_hist = ref_hist / np.sum(ref_hist) + 1e-10
            curr_hist = curr_hist / np.sum(curr_hist) + 1e-10
            
            kl_div = stats.entropy(curr_hist, ref_hist)
        
        return {"kl_divergence": float(kl_div)}
    
    def _hellinger_distance(self, ref_data: pd.Series, curr_data: pd.Series, is_categorical: bool) -> Dict[str, float]:
        """Hellinger distance calculation"""
        
        if is_categorical:
            ref_probs = ref_data.value_counts(normalize=True).sort_index()
            curr_probs = curr_data.value_counts(normalize=True).sort_index()
            
            all_categories = ref_probs.index.union(curr_probs.index)
            ref_aligned = ref_probs.reindex(all_categories, fill_value=0)
            curr_aligned = curr_probs.reindex(all_categories, fill_value=0)
            
            ref_aligned = ref_aligned / ref_aligned.sum()
            curr_aligned = curr_aligned / curr_aligned.sum()
            
            hellinger = np.sqrt(0.5 * np.sum((np.sqrt(ref_aligned) - np.sqrt(curr_aligned)) ** 2))
        else:
            combined = pd.concat([ref_data, curr_data])
            bins = np.histogram_bin_edges(combined, bins=50)
            
            ref_hist, _ = np.histogram(ref_data, bins=bins, density=True)
            curr_hist, _ = np.histogram(curr_data, bins=bins, density=True)
            
            ref_hist = ref_hist / np.sum(ref_hist)
            curr_hist = curr_hist / np.sum(curr_hist)
            
            hellinger = np.sqrt(0.5 * np.sum((np.sqrt(ref_hist) - np.sqrt(curr_hist)) ** 2))
        
        return {"hellinger_distance": float(hellinger)}
    
    def _population_stability_index(self, ref_data: pd.Series, curr_data: pd.Series, is_categorical: bool) -> Dict[str, float]:
        """Population Stability Index (PSI) calculation"""
        
        if is_categorical:
            # For categorical data
            ref_probs = ref_data.value_counts(normalize=True).sort_index()
            curr_probs = curr_data.value_counts(normalize=True).sort_index()
            
            all_categories = ref_probs.index.union(curr_probs.index)
            ref_aligned = ref_probs.reindex(all_categories, fill_value=0.001)  # Small value to avoid log(0)
            curr_aligned = curr_probs.reindex(all_categories, fill_value=0.001)
            
            ref_aligned = ref_aligned / ref_aligned.sum()
            curr_aligned = curr_aligned / curr_aligned.sum()
            
            psi = np.sum((curr_aligned - ref_aligned) * np.log(curr_aligned / ref_aligned))
        else:
            # For numerical data, create bins
            # Use quantile-based binning for better stability
            ref_quantiles = ref_data.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).values
            bins = [-np.inf] + list(ref_quantiles) + [np.inf]
            
            ref_binned = pd.cut(ref_data, bins=bins, include_lowest=True)
            curr_binned = pd.cut(curr_data, bins=bins, include_lowest=True)
            
            ref_probs = ref_binned.value_counts(normalize=True).sort_index()
            curr_probs = curr_binned.value_counts(normalize=True).sort_index()
            
            # Ensure same categories
            all_bins = ref_probs.index.union(curr_probs.index)
            ref_aligned = ref_probs.reindex(all_bins, fill_value=0.001)
            curr_aligned = curr_probs.reindex(all_bins, fill_value=0.001)
            
            ref_aligned = ref_aligned / ref_aligned.sum()
            curr_aligned = curr_aligned / curr_aligned.sum()
            
            psi = np.sum((curr_aligned - ref_aligned) * np.log(curr_aligned / ref_aligned))
        
        return {"psi": float(psi)}
    
    def _characteristic_stability_index(self, ref_data: pd.Series, curr_data: pd.Series, is_categorical: bool) -> Dict[str, float]:
        """Characteristic Stability Index (CSI) calculation"""
        
        # CSI is similar to PSI but focuses on the characteristic (variable) distribution
        # For this implementation, we'll use a modified PSI approach
        psi_result = self._population_stability_index(ref_data, curr_data, is_categorical)
        
        # CSI typically includes additional stability metrics
        ref_mean = ref_data.mean() if not is_categorical else len(ref_data.unique())
        curr_mean = curr_data.mean() if not is_categorical else len(curr_data.unique())
        
        mean_shift = abs(curr_mean - ref_mean) / abs(ref_mean) if ref_mean != 0 else 0
        
        csi = psi_result["psi"] + 0.1 * mean_shift  # Weighted combination
        
        return {"csi": float(csi), "mean_shift": float(mean_shift)}
    
    def _earth_movers_distance(self, ref_data: pd.Series, curr_data: pd.Series, is_categorical: bool) -> Dict[str, float]:
        """Earth Mover's Distance (same as Wasserstein distance)"""
        return self._wasserstein_distance(ref_data, curr_data, is_categorical)
    
    def _energy_distance(self, ref_data: pd.Series, curr_data: pd.Series, is_categorical: bool) -> Dict[str, float]:
        """Energy distance calculation"""
        try:
            from scipy.stats import energy_distance
            
            if is_categorical:
                # Convert to numerical
                combined = pd.concat([ref_data, curr_data])
                categories = combined.unique()
                ref_encoded = pd.Categorical(ref_data, categories=categories).codes
                curr_encoded = pd.Categorical(curr_data, categories=categories).codes
                distance = energy_distance(ref_encoded, curr_encoded)
            else:
                distance = energy_distance(ref_data.values, curr_data.values)
            
            return {"energy_distance": float(distance)}
        except ImportError:
            logger.warning("Energy distance not available, falling back to Wasserstein")
            return self._wasserstein_distance(ref_data, curr_data, is_categorical)
    
    def _maximum_mean_discrepancy(self, ref_data: pd.Series, curr_data: pd.Series, is_categorical: bool) -> Dict[str, float]:
        """Maximum Mean Discrepancy (MMD) calculation"""
        
        def rbf_kernel(x, y, gamma=1.0):
            """RBF kernel function"""
            return np.exp(-gamma * np.sum((x - y) ** 2))
        
        if is_categorical:
            # Convert to one-hot encoding for MMD
            combined = pd.concat([ref_data, curr_data])
            categories = combined.unique()
            ref_encoded = pd.get_dummies(ref_data.astype(str)).values
            curr_encoded = pd.get_dummies(curr_data.astype(str)).values
            
            # Ensure same shape
            all_cols = set(pd.get_dummies(combined.astype(str)).columns)
            ref_df = pd.get_dummies(ref_data.astype(str)).reindex(columns=all_cols, fill_value=0)
            curr_df = pd.get_dummies(curr_data.astype(str)).reindex(columns=all_cols, fill_value=0)
            
            ref_encoded = ref_df.values
            curr_encoded = curr_df.values
        else:
            ref_encoded = ref_data.values.reshape(-1, 1)
            curr_encoded = curr_data.values.reshape(-1, 1)
        
        # Sample for computational efficiency
        n_samples = min(1000, len(ref_encoded), len(curr_encoded))
        ref_sample = ref_encoded[:n_samples]
        curr_sample = curr_encoded[:n_samples]
        
        # Calculate MMD
        def kernel_matrix(X, Y, gamma=1.0):
            """Calculate kernel matrix"""
            XX = np.sum(X**2, axis=1)[:, np.newaxis]
            YY = np.sum(Y**2, axis=1)[np.newaxis, :]
            XY = np.dot(X, Y.T)
            return np.exp(-gamma * (XX + YY - 2 * XY))
        
        K_XX = kernel_matrix(ref_sample, ref_sample)
        K_YY = kernel_matrix(curr_sample, curr_sample)
        K_XY = kernel_matrix(ref_sample, curr_sample)
        
        mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
        
        return {"mmd": float(mmd)}
    
    def _ensemble_drift_detection(self, ref_data: pd.Series, curr_data: pd.Series, is_categorical: bool) -> Dict[str, float]:
        """Ensemble method combining multiple drift detection approaches"""
        
        methods_to_combine = ["jensen_shannon_divergence", "population_stability_index", "kolmogorov_smirnov"]
        if not is_categorical:
            methods_to_combine.extend(["wasserstein", "mann_whitney_u"])
        else:
            methods_to_combine.append("chi_square")
        
        results = {}
        scores = []
        
        for method in methods_to_combine:
            if method in self.supported_methods:
                try:
                    result = self.supported_methods[method](ref_data, curr_data, is_categorical)
                    score = self._normalize_drift_score(result, method)
                    results[method] = result
                    scores.append(score)
                except:
                    continue
        
        ensemble_score = np.mean(scores) if scores else 0.0
        ensemble_std = np.std(scores) if len(scores) > 1 else 0.0
        
        return {
            "ensemble_score": float(ensemble_score),
            "ensemble_std": float(ensemble_std),
            "methods_used": methods_to_combine,
            "individual_results": results
        }
    
    def _adaptive_windowing_drift(self, ref_data: pd.Series, curr_data: pd.Series, is_categorical: bool) -> Dict[str, float]:
        """Adaptive windowing drift detection"""
        
        # Use ADWIN-like approach for concept drift detection
        # This is a simplified version focusing on distribution changes
        
        window_sizes = [50, 100, 200, 500]
        drift_scores = []
        
        for window_size in window_sizes:
            if len(curr_data) < window_size:
                continue
            
            # Take recent window
            recent_window = curr_data.tail(window_size)
            
            # Compare with reference using JS divergence
            js_result = self._jensen_shannon_divergence(ref_data, recent_window, is_categorical)
            drift_scores.append(js_result["js_divergence"])
        
        if not drift_scores:
            return {"adaptive_score": 0.0}
        
        # Calculate adaptive score (higher weight on smaller windows)
        weights = [1.0 / (i + 1) for i in range(len(drift_scores))]
        weighted_score = np.average(drift_scores, weights=weights)
        
        return {
            "adaptive_score": float(weighted_score),
            "window_scores": drift_scores,
            "window_sizes": window_sizes[:len(drift_scores)]
        }
    
    def _assess_feature_data_quality(self, ref_data: pd.Series, curr_data: pd.Series) -> Dict[str, Any]:
        """Assess data quality metrics for a feature"""
        
        return {
            "reference_missing_rate": ref_data.isnull().mean(),
            "current_missing_rate": curr_data.isnull().mean(),
            "reference_unique_values": ref_data.nunique(),
            "current_unique_values": curr_data.nunique(),
            "reference_zero_rate": (ref_data == 0).mean() if ref_data.dtype in ['int64', 'float64'] else 0,
            "current_zero_rate": (curr_data == 0).mean() if curr_data.dtype in ['int64', 'float64'] else 0
        }
    
    def _evaluate_drift_threshold(self, result: Dict[str, Any], method: str, threshold: float) -> bool:
        """Evaluate if drift is detected based on method result and threshold"""
        
        # P-value based methods
        if "p_value" in result:
            return result["p_value"] < threshold
        
        # Distance/divergence based methods
        distance_metrics = ["js_divergence", "kl_divergence", "hellinger_distance", 
                          "distance", "psi", "csi", "energy_distance", "mmd"]
        
        for metric in distance_metrics:
            if metric in result:
                return result[metric] > threshold
        
        # Ensemble methods
        if "ensemble_score" in result:
            return result["ensemble_score"] > threshold
        
        if "adaptive_score" in result:
            return result["adaptive_score"] > threshold
        
        return False
    
    def _normalize_drift_score(self, result: Dict[str, Any], method: str) -> float:
        """Normalize drift detection result to 0-1 score"""
        
        # P-value based: convert to drift score (1 - p_value)
        if "p_value" in result:
            return min(1.0, max(0.0, 1.0 - result["p_value"]))
        
        # Distance/divergence based: cap at 1.0
        distance_metrics = ["js_divergence", "kl_divergence", "hellinger_distance", 
                          "distance", "psi", "csi", "energy_distance", "mmd"]
        
        for metric in distance_metrics:
            if metric in result:
                # Different normalization strategies for different metrics
                if metric in ["js_divergence", "hellinger_distance"]:
                    return min(1.0, result[metric])  # Already 0-1 bounded
                elif metric == "psi":
                    return min(1.0, result[metric] / 0.5)  # PSI > 0.5 is very high drift
                elif metric == "kl_divergence":
                    return min(1.0, result[metric] / 2.0)  # KL can be unbounded
                else:
                    return min(1.0, result[metric])
        
        # Ensemble methods
        if "ensemble_score" in result:
            return min(1.0, result["ensemble_score"])
        
        if "adaptive_score" in result:
            return min(1.0, result["adaptive_score"])
        
        return 0.0
    
    def _calculate_overall_drift_score(self, feature_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall drift score across all features"""
        
        if not feature_results:
            return 0.0
        
        feature_scores = [result["drift_score"] for result in feature_results.values()]
        return np.mean(feature_scores)
    
    def _calculate_method_summary(self, feature_results: Dict[str, Dict[str, Any]], methods: List[str]) -> Dict[str, Any]:
        """Calculate summary statistics for each drift detection method"""
        
        method_summary = {}
        
        for method in methods:
            method_results = []
            for feature_result in feature_results.values():
                if method in feature_result["drift_methods"]:
                    method_results.append(feature_result["drift_methods"][method]["score"])
            
            if method_results:
                method_summary[method] = {
                    "average_score": np.mean(method_results),
                    "max_score": np.max(method_results),
                    "features_with_drift": sum(1 for score in method_results if score > 0.1),
                    "total_features": len(method_results)
                }
        
        return method_summary
    
    def _calculate_enterprise_metrics(
        self, 
        reference_data: pd.DataFrame, 
        current_data: pd.DataFrame,
        feature_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate enterprise-specific drift metrics"""
        
        return {
            "data_volume_change": {
                "reference_size": len(reference_data),
                "current_size": len(current_data),
                "size_ratio": len(current_data) / len(reference_data) if len(reference_data) > 0 else 0,
                "volume_drift_detected": abs(len(current_data) / len(reference_data) - 1) > 0.2 if len(reference_data) > 0 else False
            },
            "feature_coverage": {
                "total_features": len(feature_results),
                "features_with_drift": sum(1 for result in feature_results.values() if result["drift_detected"]),
                "drift_percentage": sum(1 for result in feature_results.values() if result["drift_detected"]) / len(feature_results) * 100 if feature_results else 0
            },
            "data_quality_impact": {
                "avg_missing_rate_change": np.mean([
                    result["data_quality"]["current_missing_rate"] - result["data_quality"]["reference_missing_rate"]
                    for result in feature_results.values()
                ]),
                "features_with_quality_degradation": sum(1 for result in feature_results.values() 
                    if result["data_quality"]["current_missing_rate"] > result["data_quality"]["reference_missing_rate"] + 0.05)
            },
            "business_impact_indicators": {
                "high_impact_features": [
                    feature for feature, result in feature_results.items()
                    if result["drift_detected"] and result["drift_score"] > 0.5
                ],
                "critical_drift_count": sum(1 for result in feature_results.values() 
                    if result["drift_detected"] and result["drift_score"] > 0.7)
            }
        }
    
    def _generate_drift_recommendations(self, drift_report: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on drift analysis"""
        
        recommendations = []
        overall_score = drift_report["overall_drift_score"]
        features_with_drift = [
            feature for feature, result in drift_report["feature_drift_results"].items()
            if result["drift_detected"]
        ]
        
        # Critical drift recommendations
        if overall_score > 0.7:
            recommendations.append({
                "priority": "critical",
                "category": "model_intervention",
                "action": "Consider model retraining or replacement",
                "rationale": f"Critical drift detected (score: {overall_score:.3f}) across multiple features",
                "features_affected": features_with_drift
            })
        
        # Data collection recommendations
        enterprise_metrics = drift_report.get("enterprise_metrics", {})
        volume_change = enterprise_metrics.get("data_volume_change", {})
        
        if volume_change.get("volume_drift_detected", False):
            recommendations.append({
                "priority": "high",
                "category": "data_collection",
                "action": "Investigate data volume changes",
                "rationale": f"Significant change in data volume detected (ratio: {volume_change.get('size_ratio', 0):.2f})",
                "features_affected": []
            })
        
        # Feature-specific recommendations
        for feature, result in drift_report["feature_drift_results"].items():
            if result["drift_detected"] and result["drift_score"] > 0.5:
                recommendations.append({
                    "priority": "high",
                    "category": "feature_monitoring",
                    "action": f"Monitor and potentially retrain feature {feature}",
                    "rationale": f"High drift detected in {feature} (score: {result['drift_score']:.3f})",
                    "features_affected": [feature]
                })
        
        # Data quality recommendations
        quality_impact = enterprise_metrics.get("data_quality_impact", {})
        if quality_impact.get("features_with_quality_degradation", 0) > 0:
            recommendations.append({
                "priority": "medium",
                "category": "data_quality",
                "action": "Address data quality degradation",
                "rationale": f"{quality_impact.get('features_with_quality_degradation', 0)} features show quality degradation",
                "features_affected": []
            })
        
        # Monitoring recommendations
        if overall_score > 0.3:
            recommendations.append({
                "priority": "medium",
                "category": "monitoring",
                "action": "Increase monitoring frequency",
                "rationale": "Moderate drift detected, increased monitoring recommended",
                "features_affected": []
            })
        
        return recommendations


# Utility functions for enterprise drift analysis
def create_drift_monitoring_config(
    features: List[str],
    methods: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    monitoring_frequency: str = "daily"
) -> Dict[str, Any]:
    """Create a drift monitoring configuration"""
    
    if methods is None:
        methods = ["jensen_shannon_divergence", "population_stability_index", "kolmogorov_smirnov"]
    
    if thresholds is None:
        thresholds = {
            "jensen_shannon_divergence": 0.1,
            "population_stability_index": 0.1,
            "kolmogorov_smirnov": 0.05
        }
    
    return {
        "features": features,
        "methods": methods,
        "thresholds": thresholds,
        "monitoring_frequency": monitoring_frequency,
        "alert_on_drift": True,
        "auto_retrain_threshold": 0.5,
        "created_at": datetime.utcnow().isoformat()
    }


def generate_drift_dashboard_data(drift_report: Dict[str, Any]) -> Dict[str, Any]:
    """Generate data for drift monitoring dashboard"""
    
    dashboard_data = {
        "summary": {
            "overall_drift_score": drift_report.get("overall_drift_score", 0),
            "drift_detected": drift_report.get("overall_drift_detected", False),
            "features_analyzed": len(drift_report.get("feature_drift_results", {})),
            "features_with_drift": sum(1 for result in drift_report.get("feature_drift_results", {}).values() 
                                     if result.get("drift_detected", False)),
            "timestamp": drift_report.get("analysis_metadata", {}).get("timestamp")
        },
        "feature_details": [
            {
                "feature_name": feature,
                "drift_detected": result["drift_detected"],
                "drift_score": result["drift_score"],
                "is_categorical": result["is_categorical"],
                "methods_triggered": [method for method, method_result in result["drift_methods"].items() 
                                    if method_result["drift_detected"]]
            }
            for feature, result in drift_report.get("feature_drift_results", {}).items()
        ],
        "method_performance": drift_report.get("method_summary", {}),
        "enterprise_metrics": drift_report.get("enterprise_metrics", {}),
        "recommendations": drift_report.get("recommendations", [])
    }
    
    return dashboard_data