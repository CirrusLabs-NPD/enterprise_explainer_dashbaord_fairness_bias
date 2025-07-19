"""
Model Management Service
Handles ONNX model loading, inference, and metadata management
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import hashlib
import asyncio
import structlog

# ONNX and ML imports
import onnx
import onnxruntime as ort
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from app.core.worker_pool import WorkerPool, TaskType
from app.models.model_metadata import ModelMetadata, ModelType, ModelStatus
from app.config import settings
from app.core.database import get_db

logger = structlog.get_logger(__name__)


class ModelService:
    """
    Comprehensive model management service
    
    Features:
    - ONNX model loading and inference
    - Model metadata management
    - Performance monitoring
    - Batch inference
    - Model validation
    """
    
    def __init__(self, worker_pool: WorkerPool):
        self.worker_pool = worker_pool
        self.model_cache: Dict[str, Any] = {}
        self.metadata_cache: Dict[str, ModelMetadata] = {}
        self.session_cache: Dict[str, ort.InferenceSession] = {}
        
        # Ensure model directory exists
        os.makedirs(settings.MODEL_STORAGE_DIR, exist_ok=True)
        
        logger.info("ModelService initialized")
    
    async def upload_model(
        self,
        model_file: bytes,
        model_name: str,
        model_type: ModelType,
        metadata: Dict[str, Any],
        training_data: Optional[np.ndarray] = None,
        validation_data: Optional[np.ndarray] = None
    ) -> str:
        """
        Upload and register a new model
        """
        try:
            # Generate model ID
            model_id = self._generate_model_id(model_name, model_file)
            
            # Save model file
            model_path = f"{settings.MODEL_STORAGE_DIR}/{model_id}"
            
            # Detect model format and save accordingly
            if model_file.startswith(b'ONNX') or model_file.startswith(b'\x08'):
                # ONNX model
                model_path += ".onnx"
                with open(model_path, 'wb') as f:
                    f.write(model_file)
                
                # Validate ONNX model
                await self._validate_onnx_model(model_path)
                
            else:
                # Assume pickle/joblib model
                model_path += ".pkl"
                with open(model_path, 'wb') as f:
                    f.write(model_file)
                
                # Validate pickle model
                await self._validate_pickle_model(model_path)
            
            # Create model metadata
            model_metadata = ModelMetadata(
                model_id=model_id,
                name=model_name,
                model_type=model_type,
                file_path=model_path,
                version="1.0.0",
                description=metadata.get("description", ""),
                feature_names=metadata.get("feature_names", []),
                target_names=metadata.get("target_names", []),
                preprocessing_steps=metadata.get("preprocessing_steps", []),
                hyperparameters=metadata.get("hyperparameters", {}),
                training_metrics=metadata.get("training_metrics", {}),
                status=ModelStatus.ACTIVE,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Save metadata to database
            await self._save_model_metadata(model_metadata)
            
            # Cache metadata
            self.metadata_cache[model_id] = model_metadata
            
            # Perform initial validation if training data is provided
            if training_data is not None:
                await self._validate_model_performance(model_id, training_data, validation_data)
            
            logger.info(f"Model {model_id} uploaded successfully")
            return model_id
            
        except Exception as e:
            logger.error(f"Error uploading model: {e}")
            raise
    
    async def load_model(self, model_id: str) -> Any:
        """
        Load model from storage with caching
        """
        if model_id in self.model_cache:
            return self.model_cache[model_id]
        
        try:
            metadata = await self.get_model_metadata(model_id)
            if not metadata:
                raise ValueError(f"Model {model_id} not found")
            
            model_path = metadata.file_path
            
            if model_path.endswith('.onnx'):
                # Load ONNX model
                session = ort.InferenceSession(model_path)
                model = ONNXModelWrapper(session, metadata)
                self.session_cache[model_id] = session
                
            else:
                # Load pickle model
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                # Wrap sklearn models
                if isinstance(model, BaseEstimator):
                    model = SklearnModelWrapper(model, metadata)
            
            # Cache model
            self.model_cache[model_id] = model
            
            logger.info(f"Model {model_id} loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise
    
    async def predict(
        self,
        model_id: str,
        data: np.ndarray,
        return_probabilities: bool = False
    ) -> np.ndarray:
        """
        Make predictions using the model
        """
        try:
            model = await self.load_model(model_id)
            
            # Submit prediction task
            task_id = f"predict_{model_id}_{hash(data.tobytes())}"
            
            await self.worker_pool.submit_task(
                task_id=task_id,
                function=self._predict_worker,
                args=(model, data, return_probabilities),
                task_type=TaskType.CPU_INTENSIVE
            )
            
            result = await self.worker_pool.get_result(task_id)
            
            if result.success:
                return result.result
            else:
                raise Exception(f"Prediction failed: {result.error}")
                
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    async def predict_batch(
        self,
        model_id: str,
        data: np.ndarray,
        batch_size: int = 32,
        return_probabilities: bool = False
    ) -> np.ndarray:
        """
        Make batch predictions with parallel processing
        """
        try:
            model = await self.load_model(model_id)
            
            # Split data into batches
            batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
            
            # Submit batch tasks
            tasks = []
            for i, batch in enumerate(batches):
                task_id = f"predict_batch_{model_id}_{i}"
                await self.worker_pool.submit_task(
                    task_id=task_id,
                    function=self._predict_worker,
                    args=(model, batch, return_probabilities),
                    task_type=TaskType.CPU_INTENSIVE
                )
                tasks.append(task_id)
            
            # Collect results
            results = []
            for task_id in tasks:
                result = await self.worker_pool.get_result(task_id)
                if result.success:
                    results.append(result.result)
                else:
                    raise Exception(f"Batch prediction failed: {result.error}")
            
            # Concatenate results
            return np.concatenate(results, axis=0)
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise
    
    async def evaluate_model(
        self,
        model_id: str,
        test_data: np.ndarray,
        test_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        """
        try:
            # Make predictions
            predictions = await self.predict(model_id, test_data)
            
            # Get model metadata
            metadata = await self.get_model_metadata(model_id)
            
            # Calculate metrics based on model type
            if metadata.model_type == ModelType.CLASSIFICATION:
                # Get probabilities for additional metrics
                try:
                    probabilities = await self.predict(model_id, test_data, return_probabilities=True)
                except:
                    probabilities = None
                
                metrics = self._calculate_classification_metrics(
                    test_labels, predictions, probabilities
                )
            else:
                metrics = self._calculate_regression_metrics(test_labels, predictions)
            
            # Update model metadata with evaluation metrics
            metadata.validation_metrics = metrics
            await self._save_model_metadata(metadata)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    async def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Get model metadata
        """
        if model_id in self.metadata_cache:
            return self.metadata_cache[model_id]
        
        try:
            # Load from database
            async with get_db() as db:
                query = "SELECT * FROM model_metadata WHERE model_id = :model_id"
                result = await db.fetch_one(query, {"model_id": model_id})
                
                if result:
                    metadata = ModelMetadata(**dict(result))
                    self.metadata_cache[model_id] = metadata
                    return metadata
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting model metadata: {e}")
            return None
    
    async def list_models(self) -> List[ModelMetadata]:
        """
        List all registered models
        """
        try:
            async with get_db() as db:
                query = "SELECT * FROM model_metadata ORDER BY created_at DESC"
                results = await db.fetch_all(query)
                
                return [ModelMetadata(**dict(result)) for result in results]
                
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def delete_model(self, model_id: str) -> bool:
        """
        Delete model and associated files
        """
        try:
            # Get metadata
            metadata = await self.get_model_metadata(model_id)
            if not metadata:
                return False
            
            # Delete model file
            if os.path.exists(metadata.file_path):
                os.remove(metadata.file_path)
            
            # Delete from database
            async with get_db() as db:
                query = "DELETE FROM model_metadata WHERE model_id = :model_id"
                await db.execute(query, {"model_id": model_id})
            
            # Remove from caches
            self.model_cache.pop(model_id, None)
            self.metadata_cache.pop(model_id, None)
            self.session_cache.pop(model_id, None)
            
            logger.info(f"Model {model_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return False
    
    # Worker functions
    
    @staticmethod
    def _predict_worker(model: Any, data: np.ndarray, return_probabilities: bool) -> np.ndarray:
        """Worker function for model prediction"""
        try:
            if return_probabilities and hasattr(model, 'predict_proba'):
                return model.predict_proba(data)
            else:
                return model.predict(data)
        except Exception as e:
            logger.error(f"Error in prediction worker: {e}")
            raise
    
    # Utility methods
    
    def _generate_model_id(self, model_name: str, model_file: bytes) -> str:
        """Generate unique model ID"""
        content_hash = hashlib.md5(model_file).hexdigest()[:8]
        timestamp = int(datetime.utcnow().timestamp())
        return f"{model_name}_{timestamp}_{content_hash}"
    
    async def _validate_onnx_model(self, model_path: str):
        """Validate ONNX model"""
        try:
            # Load and check ONNX model
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
            
            # Test inference session
            session = ort.InferenceSession(model_path)
            
            # Verify inputs and outputs
            input_names = [input.name for input in session.get_inputs()]
            output_names = [output.name for output in session.get_outputs()]
            
            logger.info(f"ONNX model validated: {len(input_names)} inputs, {len(output_names)} outputs")
            
        except Exception as e:
            logger.error(f"ONNX model validation failed: {e}")
            raise
    
    async def _validate_pickle_model(self, model_path: str):
        """Validate pickle model"""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Check if it's a valid sklearn model
            if not hasattr(model, 'predict'):
                raise ValueError("Model does not have predict method")
            
            logger.info(f"Pickle model validated: {type(model).__name__}")
            
        except Exception as e:
            logger.error(f"Pickle model validation failed: {e}")
            raise
    
    async def _validate_model_performance(
        self,
        model_id: str,
        training_data: np.ndarray,
        validation_data: Optional[np.ndarray] = None
    ):
        """Validate model performance on training data"""
        try:
            # This would involve making predictions and calculating metrics
            # For now, just log that validation was requested
            logger.info(f"Model performance validation requested for {model_id}")
            
        except Exception as e:
            logger.error(f"Error validating model performance: {e}")
    
    def _calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate classification metrics"""
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average='weighted')),
            "recall": float(recall_score(y_true, y_pred, average='weighted')),
            "f1_score": float(f1_score(y_true, y_pred, average='weighted'))
        }
        
        # Add AUC if probabilities are available
        if y_prob is not None:
            try:
                from sklearn.metrics import roc_auc_score
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                else:  # Multi-class
                    metrics["auc"] = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
            except Exception as e:
                logger.warning(f"Could not calculate AUC: {e}")
        
        return metrics
    
    def _calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics"""
        return {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))
        }
    
    async def _save_model_metadata(self, metadata: ModelMetadata):
        """Save model metadata to database"""
        try:
            async with get_db() as db:
                query = """
                INSERT INTO model_metadata (
                    model_id, name, model_type, file_path, version, description,
                    feature_names, target_names, preprocessing_steps, hyperparameters,
                    training_metrics, validation_metrics, status, created_at, updated_at
                ) VALUES (
                    :model_id, :name, :model_type, :file_path, :version, :description,
                    :feature_names, :target_names, :preprocessing_steps, :hyperparameters,
                    :training_metrics, :validation_metrics, :status, :created_at, :updated_at
                )
                ON CONFLICT (model_id) DO UPDATE SET
                    name = :name,
                    model_type = :model_type,
                    file_path = :file_path,
                    version = :version,
                    description = :description,
                    feature_names = :feature_names,
                    target_names = :target_names,
                    preprocessing_steps = :preprocessing_steps,
                    hyperparameters = :hyperparameters,
                    training_metrics = :training_metrics,
                    validation_metrics = :validation_metrics,
                    status = :status,
                    updated_at = :updated_at
                """
                
                await db.execute(query, {
                    "model_id": metadata.model_id,
                    "name": metadata.name,
                    "model_type": metadata.model_type.value,
                    "file_path": metadata.file_path,
                    "version": metadata.version,
                    "description": metadata.description,
                    "feature_names": json.dumps(metadata.feature_names),
                    "target_names": json.dumps(metadata.target_names),
                    "preprocessing_steps": json.dumps(metadata.preprocessing_steps),
                    "hyperparameters": json.dumps(metadata.hyperparameters),
                    "training_metrics": json.dumps(metadata.training_metrics),
                    "validation_metrics": json.dumps(metadata.validation_metrics),
                    "status": metadata.status.value,
                    "created_at": metadata.created_at,
                    "updated_at": metadata.updated_at
                })
                
        except Exception as e:
            logger.error(f"Error saving model metadata: {e}")
            raise


class ONNXModelWrapper:
    """Wrapper for ONNX models to provide sklearn-like interface"""
    
    def __init__(self, session: ort.InferenceSession, metadata: ModelMetadata):
        self.session = session
        self.metadata = metadata
        self.input_names = [input.name for input in session.get_inputs()]
        self.output_names = [output.name for output in session.get_outputs()]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        # Ensure input is float32
        X = X.astype(np.float32)
        
        # Create input dictionary
        input_dict = {self.input_names[0]: X}
        
        # Run inference
        outputs = self.session.run(self.output_names, input_dict)
        
        # Return predictions
        return outputs[0]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities (if available)"""
        # This assumes the model outputs probabilities
        # For some models, this might need to be implemented differently
        return self.predict(X)


class SklearnModelWrapper:
    """Wrapper for sklearn models to add metadata"""
    
    def __init__(self, model: BaseEstimator, metadata: ModelMetadata):
        self.model = model
        self.metadata = metadata
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("Model does not support probability predictions")
    
    @property
    def feature_importances_(self) -> np.ndarray:
        """Get feature importances if available"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            raise NotImplementedError("Model does not have feature importances")