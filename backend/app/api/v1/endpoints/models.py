"""
Model management API endpoints
"""

import asyncio
import numpy as np
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import structlog

from app.services.model_service import ModelService
from app.services.explanation_service import ExplanationService
from app.models.model_metadata import (
    ModelMetadata, ModelType, PredictionRequest, PredictionResult,
    BatchPredictionRequest, BatchPredictionResult, ExplanationRequest, ExplanationResult
)
from app.core.dependencies import get_model_service, get_explanation_service
from app.core.security import get_current_user

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


@router.post("/upload", response_model=Dict[str, str])
async def upload_model(
    background_tasks: BackgroundTasks,
    model_file: UploadFile = File(...),
    model_name: str = Form(...),
    model_type: ModelType = Form(...),
    description: str = Form(""),
    feature_names: str = Form("[]"),  # JSON string
    target_names: str = Form("[]"),   # JSON string
    model_service: ModelService = Depends(get_model_service),
    current_user: str = Depends(get_current_user)
):
    """
    Upload a new model (ONNX or pickle format)
    """
    try:
        # Read model file
        model_content = await model_file.read()
        
        # Parse metadata
        import json
        feature_names_list = json.loads(feature_names)
        target_names_list = json.loads(target_names)
        
        metadata = {
            "description": description,
            "feature_names": feature_names_list,
            "target_names": target_names_list,
            "uploaded_by": current_user
        }
        
        # Upload model
        model_id = await model_service.upload_model(
            model_file=model_content,
            model_name=model_name,
            model_type=model_type,
            metadata=metadata
        )
        
        return {"model_id": model_id, "message": "Model uploaded successfully"}
        
    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[ModelMetadata])
async def list_models(
    model_service: ModelService = Depends(get_model_service),
    current_user: str = Depends(get_current_user)
):
    """
    List all available models
    """
    try:
        models = await model_service.list_models()
        return models
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}", response_model=ModelMetadata)
async def get_model(
    model_id: str,
    model_service: ModelService = Depends(get_model_service),
    current_user: str = Depends(get_current_user)
):
    """
    Get model metadata
    """
    try:
        metadata = await model_service.get_model_metadata(model_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Model not found")
        return metadata
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    model_service: ModelService = Depends(get_model_service),
    current_user: str = Depends(get_current_user)
):
    """
    Delete a model
    """
    try:
        success = await model_service.delete_model(model_id)
        if not success:
            raise HTTPException(status_code=404, detail="Model not found")
        return {"message": "Model deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/predict", response_model=PredictionResult)
async def predict(
    model_id: str,
    request: PredictionRequest,
    model_service: ModelService = Depends(get_model_service),
    current_user: str = Depends(get_current_user)
):
    """
    Make predictions using a model
    """
    try:
        import time
        start_time = time.time()
        
        # Convert input data
        data = np.array(request.data)
        
        # Make predictions
        predictions = await model_service.predict(
            model_id=model_id,
            data=data,
            return_probabilities=request.return_probabilities
        )
        
        # Get probabilities if requested
        probabilities = None
        if request.return_probabilities:
            try:
                probabilities = await model_service.predict(
                    model_id=model_id,
                    data=data,
                    return_probabilities=True
                )
            except:
                pass  # Model might not support probabilities
        
        # Get explanations if requested
        explanations = None
        if request.return_explanations and request.explanation_method:
            explanation_service = get_explanation_service()
            # This would be implemented with the explanation service
            pass
        
        prediction_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Get model metadata for version
        metadata = await model_service.get_model_metadata(model_id)
        
        return PredictionResult(
            predictions=predictions.tolist(),
            probabilities=probabilities.tolist() if probabilities is not None else None,
            explanations=explanations,
            prediction_time_ms=prediction_time,
            model_version=metadata.version if metadata else "unknown"
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/predict/batch", response_model=BatchPredictionResult)
async def predict_batch(
    model_id: str,
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    model_service: ModelService = Depends(get_model_service),
    current_user: str = Depends(get_current_user)
):
    """
    Make batch predictions
    """
    try:
        # This would be implemented with batch processing
        # For now, return a placeholder
        batch_id = f"batch_{model_id}_{int(time.time())}"
        
        result = BatchPredictionResult(
            batch_id=batch_id,
            model_id=model_id,
            status="processing",
            total_rows=0,
            processed_rows=0
        )
        
        # Add background task for processing
        background_tasks.add_task(
            _process_batch_prediction,
            batch_id, model_id, request, model_service
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error starting batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/evaluate")
async def evaluate_model(
    model_id: str,
    test_data: List[List[float]],
    test_labels: List[float],
    model_service: ModelService = Depends(get_model_service),
    current_user: str = Depends(get_current_user)
):
    """
    Evaluate model performance
    """
    try:
        # Convert input data
        data = np.array(test_data)
        labels = np.array(test_labels)
        
        # Evaluate model
        metrics = await model_service.evaluate_model(model_id, data, labels)
        
        return {"metrics": metrics}
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}/stats")
async def get_model_stats(
    model_id: str,
    model_service: ModelService = Depends(get_model_service),
    current_user: str = Depends(get_current_user)
):
    """
    Get model statistics and usage metrics
    """
    try:
        # This would fetch stats from database
        stats = {
            "total_predictions": 1000,
            "avg_prediction_time_ms": 50,
            "last_prediction": "2024-01-15T10:30:00Z",
            "accuracy": 0.85,
            "model_size_mb": 12.5
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting model stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/explain", response_model=ExplanationResult)
async def explain_prediction(
    model_id: str,
    request: ExplanationRequest,
    explanation_service: ExplanationService = Depends(get_explanation_service),
    current_user: str = Depends(get_current_user)
):
    """
    Generate explanation for a prediction
    """
    try:
        # Convert input data
        data = np.array(request.data)
        
        # Get explanation
        if request.instance_indices:
            # Explain specific instances
            explanations = []
            for idx in request.instance_indices:
                if idx < len(data):
                    explanation = await explanation_service.explain_instance(
                        model_id=model_id,
                        instance_data=data[idx],
                        method=request.method,
                        feature_names=request.feature_names,
                        target_names=request.target_names,
                        parameters=request.parameters
                    )
                    explanations.append(explanation)
            
            return explanations[0] if explanations else None
        else:
            # Explain all instances
            explanation = await explanation_service.explain_batch(
                model_id=model_id,
                batch_data=data,
                method=request.method,
                feature_names=request.feature_names,
                target_names=request.target_names,
                parameters=request.parameters
            )
            
            return explanation[0] if explanation else None
        
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}/feature-importance")
async def get_feature_importance(
    model_id: str,
    method: str = "shap",
    explanation_service: ExplanationService = Depends(get_explanation_service),
    current_user: str = Depends(get_current_user)
):
    """
    Get global feature importance
    """
    try:
        # This would use training data to compute importance
        # For now, return mock data
        importance = {
            "feature_names": ["feature_1", "feature_2", "feature_3"],
            "importance_values": [0.5, 0.3, 0.2],
            "method": method
        }
        
        return importance
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}/interactions")
async def get_feature_interactions(
    model_id: str,
    method: str = "shap",
    explanation_service: ExplanationService = Depends(get_explanation_service),
    current_user: str = Depends(get_current_user)
):
    """
    Get feature interactions
    """
    try:
        # This would compute feature interactions
        # For now, return mock data
        interactions = {
            "feature_names": ["feature_1", "feature_2", "feature_3"],
            "interaction_matrix": [
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.2],
                [0.3, 0.2, 1.0]
            ],
            "method": method
        }
        
        return interactions
        
    except Exception as e:
        logger.error(f"Error getting feature interactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _process_batch_prediction(
    batch_id: str,
    model_id: str,
    request: BatchPredictionRequest,
    model_service: ModelService
):
    """
    Background task for processing batch predictions
    """
    try:
        # This would implement actual batch processing
        logger.info(f"Processing batch prediction {batch_id}")
        
        # Simulate processing
        await asyncio.sleep(10)
        
        logger.info(f"Batch prediction {batch_id} completed")
        
    except Exception as e:
        logger.error(f"Error processing batch prediction {batch_id}: {e}")