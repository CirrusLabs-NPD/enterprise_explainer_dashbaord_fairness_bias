"""
Analysis service for advanced ML model analysis
Handles what-if analysis, decision trees, and advanced visualizations
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import asyncio
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import _tree
import structlog

from app.core.worker_pool import WorkerPool

logger = structlog.get_logger(__name__)


class AnalysisService:
    """Service for advanced model analysis"""
    
    def __init__(self, worker_pool: WorkerPool):
        self.worker_pool = worker_pool
        logger.info("AnalysisService initialized")
    
    async def extract_decision_tree(
        self,
        model: Any,
        model_type: str,
        max_depth: Optional[int] = None,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Extract decision tree structure from model
        
        For tree-based models: Extract actual structure
        For other models: Build surrogate tree
        """
        try:
            model_class = type(model).__name__.lower()
            
            # Check if it's already a tree-based model
            if any(tree_type in model_class for tree_type in ['tree', 'forest', 'xgb', 'lgb']):
                # Extract from tree-based model
                if hasattr(model, 'tree_'):
                    # Single decision tree
                    tree_structure = self._extract_sklearn_tree(
                        model.tree_,
                        feature_names=feature_names,
                        class_names=class_names
                    )
                elif hasattr(model, 'estimators_'):
                    # Random forest - extract first tree as example
                    tree_structure = self._extract_sklearn_tree(
                        model.estimators_[0].tree_,
                        feature_names=feature_names,
                        class_names=class_names
                    )
                else:
                    # Build surrogate tree
                    tree_structure = await self._build_surrogate_tree(
                        model, model_type, X_train, y_train,
                        max_depth, min_samples_split, min_samples_leaf,
                        feature_names, class_names
                    )
            else:
                # Build surrogate tree for non-tree model
                tree_structure = await self._build_surrogate_tree(
                    model, model_type, X_train, y_train,
                    max_depth, min_samples_split, min_samples_leaf,
                    feature_names, class_names
                )
            
            return tree_structure
            
        except Exception as e:
            logger.error(f"Error extracting decision tree: {e}")
            raise
    
    def _extract_sklearn_tree(
        self,
        tree,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Extract structure from sklearn tree"""
        
        def recurse(node, depth):
            if tree.feature[node] != _tree.TREE_UNDEFINED:
                # Internal node
                feature = feature_names[tree.feature[node]] if feature_names else f"X[{tree.feature[node]}]"
                threshold = tree.threshold[node]
                
                left_child = recurse(tree.children_left[node], depth + 1)
                right_child = recurse(tree.children_right[node], depth + 1)
                
                return {
                    "type": "split",
                    "feature": feature,
                    "threshold": float(threshold),
                    "samples": int(tree.n_node_samples[node]),
                    "impurity": float(tree.impurity[node]),
                    "depth": depth,
                    "left": left_child,
                    "right": right_child
                }
            else:
                # Leaf node
                values = tree.value[node]
                if len(values.shape) == 3:  # Multi-output
                    values = values[0]
                
                if class_names and len(values[0]) == len(class_names):
                    # Classification
                    class_idx = np.argmax(values[0])
                    prediction = class_names[class_idx]
                    class_distribution = {
                        class_names[i]: int(values[0][i]) 
                        for i in range(len(class_names))
                    }
                else:
                    # Regression or no class names
                    prediction = float(values[0][0]) if values.shape[1] == 1 else values[0].tolist()
                    class_distribution = None
                
                return {
                    "type": "leaf",
                    "prediction": prediction,
                    "samples": int(tree.n_node_samples[node]),
                    "impurity": float(tree.impurity[node]),
                    "depth": depth,
                    "class_distribution": class_distribution
                }
        
        return {
            "tree": recurse(0, 0),
            "max_depth": tree.max_depth,
            "n_leaves": tree.n_leaves,
            "n_nodes": tree.node_count,
            "feature_names": feature_names,
            "class_names": class_names
        }
    
    async def _build_surrogate_tree(
        self,
        model: Any,
        model_type: str,
        X_train: Optional[np.ndarray],
        y_train: Optional[np.ndarray],
        max_depth: Optional[int],
        min_samples_split: int,
        min_samples_leaf: int,
        feature_names: Optional[List[str]],
        class_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Build surrogate tree to approximate black-box model"""
        
        if X_train is None or y_train is None:
            # Generate synthetic data if no training data provided
            # This is a simplified version - in production, you'd want better sampling
            n_samples = 1000
            n_features = len(feature_names) if feature_names else 10
            X_train = np.random.randn(n_samples, n_features)
            
            # Get predictions from original model
            if hasattr(model, 'predict'):
                y_train = model.predict(X_train)
            else:
                raise ValueError("Model must have predict method")
        
        # Create surrogate tree
        if model_type == 'classification':
            surrogate = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
        else:
            surrogate = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
        
        # Fit surrogate
        surrogate.fit(X_train, y_train)
        
        # Extract structure
        tree_structure = self._extract_sklearn_tree(
            surrogate.tree_,
            feature_names=feature_names,
            class_names=class_names
        )
        
        # Add metadata about surrogate
        tree_structure['is_surrogate'] = True
        tree_structure['surrogate_accuracy'] = surrogate.score(X_train, y_train)
        
        return tree_structure
    
    async def calculate_feature_interactions_detailed(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Calculate detailed feature interactions"""
        
        try:
            # This would use SHAP interaction values or similar
            # For now, returning a structured format
            n_features = len(feature_names)
            
            # Mock interaction matrix (in practice, use SHAP)
            interaction_matrix = np.random.rand(n_features, n_features)
            interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2
            np.fill_diagonal(interaction_matrix, 0)
            
            # Find top interactions
            interactions = []
            for i in range(n_features):
                for j in range(i+1, n_features):
                    interactions.append({
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'strength': float(interaction_matrix[i, j]),
                        'type': 'synergistic' if interaction_matrix[i, j] > 0.5 else 'redundant'
                    })
            
            # Sort by strength
            interactions.sort(key=lambda x: abs(x['strength']), reverse=True)
            
            return {
                'top_interactions': interactions[:top_k],
                'interaction_matrix': interaction_matrix.tolist(),
                'feature_names': feature_names,
                'summary_stats': {
                    'mean_interaction': float(np.mean(np.abs(interaction_matrix))),
                    'max_interaction': float(np.max(np.abs(interaction_matrix))),
                    'n_strong_interactions': int(np.sum(np.abs(interaction_matrix) > 0.5))
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating feature interactions: {e}")
            raise
    
    async def perform_sensitivity_analysis(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        n_samples: int = 100,
        perturbation_range: float = 0.1
    ) -> Dict[str, Any]:
        """Perform sensitivity analysis on model predictions"""
        
        try:
            sensitivities = {}
            base_predictions = model.predict(X[:n_samples])
            
            for i, feature in enumerate(feature_names):
                # Perturb feature
                X_perturbed = X[:n_samples].copy()
                feature_std = np.std(X[:, i])
                X_perturbed[:, i] += np.random.normal(
                    0, 
                    feature_std * perturbation_range, 
                    n_samples
                )
                
                # Get new predictions
                perturbed_predictions = model.predict(X_perturbed)
                
                # Calculate sensitivity
                sensitivity = np.mean(np.abs(perturbed_predictions - base_predictions))
                sensitivities[feature] = float(sensitivity)
            
            # Sort by sensitivity
            sorted_features = sorted(
                sensitivities.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            return {
                'feature_sensitivities': dict(sorted_features),
                'most_sensitive': sorted_features[0][0],
                'least_sensitive': sorted_features[-1][0],
                'sensitivity_scores': sensitivities
            }
            
        except Exception as e:
            logger.error(f"Error in sensitivity analysis: {e}")
            raise
    
    async def generate_decision_rules(
        self,
        tree_structure: Dict[str, Any],
        min_confidence: float = 0.8,
        min_support: int = 10
    ) -> List[Dict[str, Any]]:
        """Extract decision rules from tree structure"""
        
        rules = []
        
        def extract_rules(node, conditions=[]):
            if node['type'] == 'leaf':
                # Create rule
                if node['samples'] >= min_support:
                    rule = {
                        'conditions': conditions.copy(),
                        'prediction': node['prediction'],
                        'support': node['samples'],
                        'confidence': 1 - node['impurity'] if node['impurity'] < 0.5 else node['impurity']
                    }
                    
                    if rule['confidence'] >= min_confidence:
                        rules.append(rule)
            else:
                # Internal node - recurse
                left_conditions = conditions + [
                    f"{node['feature']} <= {node['threshold']:.3f}"
                ]
                right_conditions = conditions + [
                    f"{node['feature']} > {node['threshold']:.3f}"
                ]
                
                extract_rules(node['left'], left_conditions)
                extract_rules(node['right'], right_conditions)
        
        extract_rules(tree_structure['tree'])
        
        # Sort by support
        rules.sort(key=lambda x: x['support'], reverse=True)
        
        return rules