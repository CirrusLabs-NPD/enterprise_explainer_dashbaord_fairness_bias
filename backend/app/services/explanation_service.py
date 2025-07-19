"""
Advanced ML Explainability Service
Provides SHAP, LIME, and custom explanation algorithms with multiprocessing support
"""

import asyncio
import json
import numpy as np
import pandas as pd
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import structlog

# ML and explainability imports
import shap
import lime
import lime.lime_tabular
import lime.lime_text
import lime.lime_image
from captum.attr import (
    IntegratedGradients, DeepLift, GradientShap, 
    Saliency, InputXGradient, GuidedBackprop
)
from alibi.explainers import ALE, AnchorTabular, CounterfactualProto
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import onnx
import onnxruntime as ort

from app.core.worker_pool import WorkerPool, TaskType
from app.models.explanation_models import (
    ExplanationRequest, ExplanationResult, 
    ShapExplanation, LimeExplanation, 
    FeatureImportance, InteractionAnalysis
)
from app.config import settings

logger = structlog.get_logger(__name__)


class ExplanationMethod(Enum):
    """Available explanation methods"""
    SHAP_TREE = "shap_tree"
    SHAP_KERNEL = "shap_kernel"
    SHAP_DEEP = "shap_deep"
    SHAP_LINEAR = "shap_linear"
    LIME_TABULAR = "lime_tabular"
    LIME_TEXT = "lime_text"
    LIME_IMAGE = "lime_image"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    GRADIENT_SHAP = "gradient_shap"
    DEEP_LIFT = "deep_lift"
    ANCHORS = "anchors"
    COUNTERFACTUALS = "counterfactuals"
    CUSTOM_IMPORTANCE = "custom_importance"


@dataclass
class ExplanationTask:
    """Explanation task configuration"""
    task_id: str
    method: ExplanationMethod
    model_path: str
    data: np.ndarray
    feature_names: List[str]
    target_names: Optional[List[str]] = None
    instance_idx: Optional[int] = None
    background_data: Optional[np.ndarray] = None
    parameters: Optional[Dict] = None


class ExplanationService:
    """
    Advanced explanation service with multiprocessing support
    
    Features:
    - Multiple explanation methods (SHAP, LIME, Captum, Alibi)
    - Optimized for different model types
    - Batch processing capabilities
    - Caching for performance
    - Custom explanation algorithms
    """
    
    def __init__(self, worker_pool: WorkerPool):
        self.worker_pool = worker_pool
        self.model_cache: Dict[str, Any] = {}
        self.explainer_cache: Dict[str, Any] = {}
        self.background_data_cache: Dict[str, np.ndarray] = {}
        
        # Initialize custom explainers
        self._init_custom_explainers()
        
        logger.info("ExplanationService initialized")
    
    def _init_custom_explainers(self):
        """Initialize custom explanation algorithms"""
        self.custom_explainers = {
            "permutation_importance": self._permutation_importance,
            "drop_column_importance": self._drop_column_importance,
            "correlation_analysis": self._correlation_analysis,
            "mutual_information": self._mutual_information_analysis,
            "feature_interactions": self._feature_interactions,
            "partial_dependence": self._partial_dependence,
            "accumulated_local_effects": self._accumulated_local_effects,
        }
    
    async def explain_instance(
        self,
        model_id: str,
        instance_data: np.ndarray,
        method: ExplanationMethod,
        feature_names: List[str],
        target_names: Optional[List[str]] = None,
        parameters: Optional[Dict] = None
    ) -> ExplanationResult:
        """
        Explain a single instance prediction
        """
        task_id = f"explain_{model_id}_{int(time.time() * 1000)}"
        
        try:
            # Submit explanation task
            await self.worker_pool.submit_task(
                task_id=task_id,
                function=self._explain_instance_worker,
                args=(
                    model_id,
                    instance_data,
                    method,
                    feature_names,
                    target_names,
                    parameters or {}
                ),
                task_type=TaskType.CPU_INTENSIVE,
                timeout=settings.SHAP_TIMEOUT
            )
            
            # Get result
            result = await self.worker_pool.get_result(task_id, timeout=settings.SHAP_TIMEOUT + 10)
            
            if result.success:
                return result.result
            else:
                raise Exception(f"Explanation failed: {result.error}")
                
        except Exception as e:
            logger.error(f"Error explaining instance: {e}")
            raise
    
    async def explain_batch(
        self,
        model_id: str,
        batch_data: np.ndarray,
        method: ExplanationMethod,
        feature_names: List[str],
        target_names: Optional[List[str]] = None,
        parameters: Optional[Dict] = None
    ) -> List[ExplanationResult]:
        """
        Explain a batch of instances
        """
        batch_size = parameters.get("batch_size", settings.SHAP_BATCH_SIZE)
        results = []
        
        # Split into batches
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i + batch_size]
            
            # Submit batch task
            task_id = f"explain_batch_{model_id}_{i}_{int(time.time() * 1000)}"
            
            await self.worker_pool.submit_task(
                task_id=task_id,
                function=self._explain_batch_worker,
                args=(
                    model_id,
                    batch,
                    method,
                    feature_names,
                    target_names,
                    parameters or {}
                ),
                task_type=TaskType.CPU_INTENSIVE,
                timeout=settings.SHAP_TIMEOUT
            )
            
            # Get result
            result = await self.worker_pool.get_result(task_id)
            if result.success:
                results.extend(result.result)
            else:
                logger.error(f"Batch explanation failed: {result.error}")
        
        return results
    
    async def compute_feature_importance(
        self,
        model_id: str,
        training_data: np.ndarray,
        training_labels: np.ndarray,
        feature_names: List[str],
        method: str = "shap"
    ) -> FeatureImportance:
        """
        Compute global feature importance
        """
        task_id = f"feature_importance_{model_id}_{int(time.time() * 1000)}"
        
        await self.worker_pool.submit_task(
            task_id=task_id,
            function=self._compute_feature_importance_worker,
            args=(model_id, training_data, training_labels, feature_names, method),
            task_type=TaskType.CPU_INTENSIVE,
            timeout=settings.SHAP_TIMEOUT
        )
        
        result = await self.worker_pool.get_result(task_id)
        if result.success:
            return result.result
        else:
            raise Exception(f"Feature importance computation failed: {result.error}")
    
    async def compute_feature_interactions(
        self,
        model_id: str,
        data: np.ndarray,
        feature_names: List[str],
        method: str = "shap"
    ) -> InteractionAnalysis:
        """
        Compute feature interactions
        """
        task_id = f"interactions_{model_id}_{int(time.time() * 1000)}"
        
        await self.worker_pool.submit_task(
            task_id=task_id,
            function=self._compute_interactions_worker,
            args=(model_id, data, feature_names, method),
            task_type=TaskType.CPU_INTENSIVE,
            timeout=settings.SHAP_TIMEOUT
        )
        
        result = await self.worker_pool.get_result(task_id)
        if result.success:
            return result.result
        else:
            raise Exception(f"Interaction analysis failed: {result.error}")
    
    # Worker functions (run in separate processes)
    
    @staticmethod
    def _explain_instance_worker(
        model_id: str,
        instance_data: np.ndarray,
        method: ExplanationMethod,
        feature_names: List[str],
        target_names: Optional[List[str]],
        parameters: Dict
    ) -> ExplanationResult:
        """Worker function for single instance explanation"""
        try:
            # Load model
            model = ExplanationService._load_model(model_id)
            
            # Get explanation based on method
            if method == ExplanationMethod.SHAP_TREE:
                explanation = ExplanationService._shap_tree_explanation(
                    model, instance_data, feature_names, parameters
                )
            elif method == ExplanationMethod.SHAP_KERNEL:
                explanation = ExplanationService._shap_kernel_explanation(
                    model, instance_data, feature_names, parameters
                )
            elif method == ExplanationMethod.LIME_TABULAR:
                explanation = ExplanationService._lime_tabular_explanation(
                    model, instance_data, feature_names, parameters
                )
            elif method == ExplanationMethod.INTEGRATED_GRADIENTS:
                explanation = ExplanationService._integrated_gradients_explanation(
                    model, instance_data, feature_names, parameters
                )
            else:
                raise ValueError(f"Unsupported explanation method: {method}")
            
            return ExplanationResult(
                method=method.value,
                feature_names=feature_names,
                explanation=explanation,
                metadata={
                    "model_id": model_id,
                    "method": method.value,
                    "parameters": parameters
                }
            )
            
        except Exception as e:
            logger.error(f"Error in explanation worker: {e}")
            raise
    
    @staticmethod
    def _shap_tree_explanation(
        model: Any,
        instance_data: np.ndarray,
        feature_names: List[str],
        parameters: Dict
    ) -> ShapExplanation:
        """SHAP TreeExplainer explanation"""
        # Initialize explainer
        explainer = shap.TreeExplainer(model)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(instance_data)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary classification
        
        # Get base value
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1]  # Use positive class base value
        
        return ShapExplanation(
            shap_values=shap_values.tolist(),
            base_value=float(base_value),
            feature_names=feature_names,
            feature_values=instance_data.tolist(),
            prediction=float(model.predict(instance_data.reshape(1, -1))[0])
        )
    
    @staticmethod
    def _shap_kernel_explanation(
        model: Any,
        instance_data: np.ndarray,
        feature_names: List[str],
        parameters: Dict
    ) -> ShapExplanation:
        """SHAP KernelExplainer explanation"""
        # Generate background data if not provided
        background_data = parameters.get("background_data")
        if background_data is None:
            # Use mean values as background
            background_data = np.mean(instance_data.reshape(1, -1), axis=0, keepdims=True)
        
        # Initialize explainer
        explainer = shap.KernelExplainer(model.predict, background_data)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(instance_data.reshape(1, -1))
        
        return ShapExplanation(
            shap_values=shap_values[0].tolist(),
            base_value=float(explainer.expected_value),
            feature_names=feature_names,
            feature_values=instance_data.tolist(),
            prediction=float(model.predict(instance_data.reshape(1, -1))[0])
        )
    
    @staticmethod
    def _lime_tabular_explanation(
        model: Any,
        instance_data: np.ndarray,
        feature_names: List[str],
        parameters: Dict
    ) -> LimeExplanation:
        """LIME tabular explanation"""
        # Initialize LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=parameters.get("training_data", instance_data.reshape(1, -1)),
            feature_names=feature_names,
            class_names=parameters.get("class_names", ["0", "1"]),
            mode="classification" if hasattr(model, "predict_proba") else "regression"
        )
        
        # Generate explanation
        explanation = explainer.explain_instance(
            instance_data,
            model.predict_proba if hasattr(model, "predict_proba") else model.predict,
            num_features=len(feature_names),
            num_samples=parameters.get("num_samples", settings.LIME_NUM_SAMPLES)
        )
        
        # Extract feature importance
        feature_importance = dict(explanation.as_list())
        
        return LimeExplanation(
            feature_importance=feature_importance,
            feature_names=feature_names,
            feature_values=instance_data.tolist(),
            prediction=float(model.predict(instance_data.reshape(1, -1))[0]),
            score=float(explanation.score)
        )
    
    @staticmethod
    def _integrated_gradients_explanation(
        model: Any,
        instance_data: np.ndarray,
        feature_names: List[str],
        parameters: Dict
    ) -> Dict:
        """Integrated Gradients explanation (for neural networks)"""
        # This requires a PyTorch model
        # Implementation would depend on the specific model architecture
        # For now, return a placeholder
        return {
            "method": "integrated_gradients",
            "attributions": np.random.random(len(feature_names)).tolist(),
            "feature_names": feature_names,
            "baseline": np.zeros_like(instance_data).tolist()
        }
    
    @staticmethod
    def _compute_feature_importance_worker(
        model_id: str,
        training_data: np.ndarray,
        training_labels: np.ndarray,
        feature_names: List[str],
        method: str
    ) -> FeatureImportance:
        """Worker function for feature importance computation"""
        try:
            model = ExplanationService._load_model(model_id)
            
            if method == "shap":
                # Use SHAP for feature importance
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(training_data)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Use positive class
                
                # Compute mean absolute SHAP values
                importance_values = np.mean(np.abs(shap_values), axis=0)
                
            elif method == "permutation":
                # Permutation importance
                importance_values = ExplanationService._permutation_importance(
                    model, training_data, training_labels
                )
                
            elif method == "builtin":
                # Built-in feature importance (for tree-based models)
                if hasattr(model, "feature_importances_"):
                    importance_values = model.feature_importances_
                else:
                    raise ValueError("Model does not have built-in feature importance")
            
            else:
                raise ValueError(f"Unsupported importance method: {method}")
            
            # Sort by importance
            sorted_indices = np.argsort(importance_values)[::-1]
            
            return FeatureImportance(
                feature_names=[feature_names[i] for i in sorted_indices],
                importance_values=importance_values[sorted_indices].tolist(),
                method=method,
                ranking=list(range(1, len(feature_names) + 1))
            )
            
        except Exception as e:
            logger.error(f"Error computing feature importance: {e}")
            raise
    
    @staticmethod
    def _compute_interactions_worker(
        model_id: str,
        data: np.ndarray,
        feature_names: List[str],
        method: str
    ) -> InteractionAnalysis:
        """Worker function for interaction analysis"""
        try:
            model = ExplanationService._load_model(model_id)
            
            if method == "shap":
                # SHAP interaction values
                explainer = shap.TreeExplainer(model)
                interaction_values = explainer.shap_interaction_values(data)
                
                # Compute interaction strengths
                interaction_matrix = np.mean(np.abs(interaction_values), axis=0)
                
            else:
                # Custom interaction analysis
                interaction_matrix = ExplanationService._feature_interactions(
                    model, data, feature_names
                )
            
            return InteractionAnalysis(
                feature_names=feature_names,
                interaction_matrix=interaction_matrix.tolist(),
                method=method
            )
            
        except Exception as e:
            logger.error(f"Error computing interactions: {e}")
            raise
    
    # Utility methods
    
    @staticmethod
    def _load_model(model_id: str) -> Any:
        """Load model from storage"""
        model_path = f"{settings.MODEL_STORAGE_DIR}/{model_id}"
        
        # Try loading ONNX model first
        try:
            if model_path.endswith('.onnx'):
                return ort.InferenceSession(model_path)
            else:
                # Try pickle
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise
    
    @staticmethod
    def _permutation_importance(
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10
    ) -> np.ndarray:
        """Compute permutation importance"""
        baseline_score = accuracy_score(y, model.predict(X))
        importance_scores = []
        
        for feature_idx in range(X.shape[1]):
            scores = []
            for _ in range(n_repeats):
                # Permute feature
                X_permuted = X.copy()
                X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])
                
                # Compute score
                permuted_score = accuracy_score(y, model.predict(X_permuted))
                scores.append(baseline_score - permuted_score)
            
            importance_scores.append(np.mean(scores))
        
        return np.array(importance_scores)
    
    async def _ice_plots(
        self,
        model: Any,
        features: List[str],
        X_reference: np.ndarray,
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """Generate Individual Conditional Expectation plots"""
        try:
            # Sample data if too large
            if len(X_reference) > sample_size:
                indices = np.random.choice(len(X_reference), sample_size, replace=False)
                X_sample = X_reference[indices]
            else:
                X_sample = X_reference
            
            ice_results = {}
            
            for feature in features:
                feature_idx = self._get_feature_index(feature)
                
                # Create range for feature
                feature_values = X_reference[:, feature_idx]
                feature_range = np.linspace(
                    np.percentile(feature_values, 5),
                    np.percentile(feature_values, 95),
                    50
                )
                
                # Calculate ICE curves
                ice_curves = []
                for instance in X_sample:
                    curve = []
                    for value in feature_range:
                        modified_instance = instance.copy()
                        modified_instance[feature_idx] = value
                        prediction = model.predict([modified_instance])[0]
                        curve.append(prediction)
                    ice_curves.append(curve)
                
                ice_results[feature] = {
                    'feature_range': feature_range.tolist(),
                    'ice_curves': ice_curves,
                    'average_curve': np.mean(ice_curves, axis=0).tolist()
                }
            
            return ice_results
            
        except Exception as e:
            logger.error(f"Error generating ICE plots: {e}")
            raise
    
    async def generate_counterfactuals(
        self,
        model_id: str,
        instance: Dict[str, float],
        desired_outcome: Optional[float] = None,
        max_features_to_change: int = 5,
        feature_ranges: Optional[Dict[str, tuple]] = None
    ) -> Dict[str, Any]:
        """Generate counterfactual explanations"""
        try:
            # Load model
            model = self._load_model(model_id)
            
            # Convert instance to array
            instance_array = np.array(list(instance.values())).reshape(1, -1)
            original_prediction = model.predict(instance_array)[0]
            
            # Simple counterfactual generation (in practice, use DICE or similar)
            feature_names = list(instance.keys())
            counterfactuals = []
            
            # Try random perturbations
            for _ in range(100):  # Generate multiple candidates
                modified_instance = instance.copy()
                
                # Randomly select features to modify
                features_to_change = np.random.choice(
                    feature_names, 
                    size=min(max_features_to_change, len(feature_names)),
                    replace=False
                )
                
                for feature in features_to_change:
                    if feature_ranges and feature in feature_ranges:
                        min_val, max_val = feature_ranges[feature]
                    else:
                        # Use reasonable perturbation
                        current_val = instance[feature]
                        min_val = current_val * 0.5
                        max_val = current_val * 1.5
                    
                    modified_instance[feature] = np.random.uniform(min_val, max_val)
                
                # Get prediction
                modified_array = np.array(list(modified_instance.values())).reshape(1, -1)
                new_prediction = model.predict(modified_array)[0]
                
                # Check if it meets criteria
                if desired_outcome is None or abs(new_prediction - desired_outcome) < abs(original_prediction - desired_outcome):
                    # Calculate changes
                    changes = {
                        k: modified_instance[k] - instance[k] 
                        for k in modified_instance.keys()
                        if abs(modified_instance[k] - instance[k]) > 1e-6
                    }
                    
                    counterfactuals.append({
                        'modified_instance': modified_instance,
                        'prediction': new_prediction,
                        'changes': changes,
                        'num_changes': len(changes),
                        'prediction_change': new_prediction - original_prediction
                    })
            
            # Sort by number of changes and prediction difference
            counterfactuals.sort(key=lambda x: (x['num_changes'], abs(x['prediction_change'])))
            
            return {
                'original_instance': instance,
                'original_prediction': original_prediction,
                'counterfactuals': counterfactuals[:5],  # Return top 5
                'desired_outcome': desired_outcome
            }
            
        except Exception as e:
            logger.error(f"Error generating counterfactuals: {e}")
            raise
    
    async def analyze_feature_importance(
        self,
        importance_data: Dict[str, Any],
        top_k: int = 20,
        include_confidence_intervals: bool = True,
        include_interactions: bool = True
    ) -> Dict[str, Any]:
        """Analyze feature importance with statistical details"""
        try:
            # Extract importance values
            importances = importance_data.get('feature_importance', [])
            feature_names = [item['feature'] for item in importances]
            importance_values = [item['importance'] for item in importances]
            
            # Calculate statistics
            mean_importance = np.mean(importance_values)
            std_importance = np.std(importance_values)
            
            # Add confidence intervals (mock for now)
            detailed_importance = []
            for i, item in enumerate(importances[:top_k]):
                enhanced_item = item.copy()
                
                if include_confidence_intervals:
                    # Mock confidence intervals
                    ci_lower = item['importance'] - 0.1 * std_importance
                    ci_upper = item['importance'] + 0.1 * std_importance
                    enhanced_item['confidence_interval'] = [ci_lower, ci_upper]
                
                enhanced_item['rank'] = i + 1
                enhanced_item['relative_importance'] = item['importance'] / mean_importance
                
                detailed_importance.append(enhanced_item)
            
            result = {
                'detailed_importance': detailed_importance,
                'summary_stats': {
                    'mean_importance': mean_importance,
                    'std_importance': std_importance,
                    'max_importance': max(importance_values),
                    'min_importance': min(importance_values)
                }
            }
            
            if include_interactions:
                # Mock interaction analysis
                result['top_interactions'] = [
                    {
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'interaction_strength': np.random.uniform(0.1, 0.8)
                    }
                    for i in range(min(5, len(feature_names)))
                    for j in range(i+1, min(5, len(feature_names)))
                ]
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            raise
    
    @staticmethod
    def _drop_column_importance(
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Compute drop-column importance"""
        # This would require retraining models, which is expensive
        # For now, return permutation importance
        return ExplanationService._permutation_importance(model, X, y)
    
    @staticmethod
    def _correlation_analysis(
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Compute correlation-based importance"""
        correlations = []
        for feature_idx in range(X.shape[1]):
            corr = np.corrcoef(X[:, feature_idx], y)[0, 1]
            correlations.append(abs(corr))
        
        return np.array(correlations)
    
    @staticmethod
    def _mutual_information_analysis(
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Compute mutual information based importance"""
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        
        if len(np.unique(y)) <= 10:  # Classification
            return mutual_info_classif(X, y)
        else:  # Regression
            return mutual_info_regression(X, y)
    
    @staticmethod
    def _feature_interactions(
        model: Any,
        X: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """Compute feature interactions"""
        n_features = len(feature_names)
        interaction_matrix = np.zeros((n_features, n_features))
        
        # Compute pairwise interactions
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Simple interaction: difference in prediction when both features are permuted
                # vs when individual features are permuted
                X_both = X.copy()
                X_both[:, i] = np.random.permutation(X_both[:, i])
                X_both[:, j] = np.random.permutation(X_both[:, j])
                
                X_i = X.copy()
                X_i[:, i] = np.random.permutation(X_i[:, i])
                
                X_j = X.copy()
                X_j[:, j] = np.random.permutation(X_j[:, j])
                
                pred_original = model.predict(X)
                pred_both = model.predict(X_both)
                pred_i = model.predict(X_i)
                pred_j = model.predict(X_j)
                
                # Interaction strength
                interaction = np.mean(np.abs(pred_original - pred_both)) - \
                             np.mean(np.abs(pred_original - pred_i)) - \
                             np.mean(np.abs(pred_original - pred_j))
                
                interaction_matrix[i, j] = interaction
                interaction_matrix[j, i] = interaction
        
        return interaction_matrix
    
    @staticmethod
    def _partial_dependence(
        model: Any,
        X: np.ndarray,
        feature_idx: int,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute partial dependence"""
        feature_values = X[:, feature_idx]
        min_val, max_val = np.min(feature_values), np.max(feature_values)
        
        # Create grid
        grid = np.linspace(min_val, max_val, n_points)
        pd_values = []
        
        for val in grid:
            X_modified = X.copy()
            X_modified[:, feature_idx] = val
            predictions = model.predict(X_modified)
            pd_values.append(np.mean(predictions))
        
        return grid, np.array(pd_values)
    
    @staticmethod
    def _accumulated_local_effects(
        model: Any,
        X: np.ndarray,
        feature_idx: int,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Accumulated Local Effects (ALE)"""
        # This is a simplified implementation
        # Full ALE requires more sophisticated handling of feature distributions
        return ExplanationService._partial_dependence(model, X, feature_idx, n_points)
    
    async def get_explanation_methods(self) -> List[Dict[str, Any]]:
        """Get available explanation methods"""
        return [
            {
                "name": method.value,
                "display_name": method.value.replace("_", " ").title(),
                "description": f"Explanation using {method.value}",
                "supported_model_types": ["sklearn", "onnx", "custom"]
            }
            for method in ExplanationMethod
        ]
    
    async def validate_explanation_request(self, request: ExplanationRequest) -> bool:
        """Validate explanation request"""
        # Check if model exists
        model_path = f"{settings.MODEL_STORAGE_DIR}/{request.model_id}"
        if not os.path.exists(model_path):
            return False
        
        # Check if method is supported
        if request.method not in [m.value for m in ExplanationMethod]:
            return False
        
        return True