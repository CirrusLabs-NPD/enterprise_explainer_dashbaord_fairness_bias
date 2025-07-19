"""
Data management API endpoints
"""

import os
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import structlog

from app.services.data_service import DataService
from app.models.model_metadata import DatasetMetadata, DataQualityReport
from app.core.dependencies import get_data_service
from app.core.security import get_current_user

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/data", tags=["data"])


@router.post("/upload", response_model=Dict[str, str])
async def upload_dataset(
    background_tasks: BackgroundTasks,
    data_file: UploadFile = File(...),
    dataset_name: str = Form(...),
    description: str = Form(""),
    data_service: DataService = Depends(get_data_service),
    current_user: str = Depends(get_current_user)
):
    """
    Upload a dataset (CSV, Parquet, JSON)
    """
    try:
        # Validate file type
        if not data_file.filename.endswith(('.csv', '.parquet', '.json')):
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file format. Please upload CSV, Parquet, or JSON files."
            )
        
        # Read file content
        content = await data_file.read()
        
        # Upload dataset
        dataset_id = await data_service.upload_dataset(
            file_content=content,
            filename=data_file.filename,
            dataset_name=dataset_name,
            description=description,
            uploaded_by=current_user
        )
        
        # Start background processing
        background_tasks.add_task(
            _process_dataset,
            dataset_id, data_service
        )
        
        return {"dataset_id": dataset_id, "message": "Dataset uploaded successfully"}
        
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[DatasetMetadata])
async def list_datasets(
    data_service: DataService = Depends(get_data_service),
    current_user: str = Depends(get_current_user)
):
    """
    List all available datasets
    """
    try:
        datasets = await data_service.list_datasets()
        return datasets
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}", response_model=DatasetMetadata)
async def get_dataset(
    dataset_id: str,
    data_service: DataService = Depends(get_data_service),
    current_user: str = Depends(get_current_user)
):
    """
    Get dataset metadata
    """
    try:
        metadata = await data_service.get_dataset_metadata(dataset_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return metadata
    except Exception as e:
        logger.error(f"Error getting dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    data_service: DataService = Depends(get_data_service),
    current_user: str = Depends(get_current_user)
):
    """
    Delete a dataset
    """
    try:
        success = await data_service.delete_dataset(dataset_id)
        if not success:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return {"message": "Dataset deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/sample")
async def get_dataset_sample(
    dataset_id: str,
    n_rows: int = 100,
    data_service: DataService = Depends(get_data_service),
    current_user: str = Depends(get_current_user)
):
    """
    Get a sample of the dataset
    """
    try:
        sample = await data_service.get_dataset_sample(dataset_id, n_rows)
        return {"sample": sample}
    except Exception as e:
        logger.error(f"Error getting dataset sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/statistics")
async def get_dataset_statistics(
    dataset_id: str,
    data_service: DataService = Depends(get_data_service),
    current_user: str = Depends(get_current_user)
):
    """
    Get dataset statistics
    """
    try:
        stats = await data_service.compute_statistics(dataset_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting dataset statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/quality", response_model=DataQualityReport)
async def assess_data_quality(
    dataset_id: str,
    data_service: DataService = Depends(get_data_service),
    current_user: str = Depends(get_current_user)
):
    """
    Assess data quality
    """
    try:
        quality_report = await data_service.assess_data_quality(dataset_id)
        return quality_report
    except Exception as e:
        logger.error(f"Error assessing data quality: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dataset_id}/preprocess")
async def preprocess_dataset(
    dataset_id: str,
    preprocessing_steps: List[Dict[str, Any]],
    background_tasks: BackgroundTasks,
    data_service: DataService = Depends(get_data_service),
    current_user: str = Depends(get_current_user)
):
    """
    Apply preprocessing steps to dataset
    """
    try:
        # Start background preprocessing
        background_tasks.add_task(
            _preprocess_dataset,
            dataset_id, preprocessing_steps, data_service
        )
        
        return {"message": "Preprocessing started"}
        
    except Exception as e:
        logger.error(f"Error starting preprocessing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dataset_id}/split")
async def split_dataset(
    dataset_id: str,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
    data_service: DataService = Depends(get_data_service),
    current_user: str = Depends(get_current_user)
):
    """
    Split dataset into train/validation/test sets
    """
    try:
        # Validate ratios
        if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
            raise HTTPException(
                status_code=400,
                detail="Train, validation, and test ratios must sum to 1.0"
            )
        
        splits = await data_service.split_dataset(
            dataset_id=dataset_id,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            random_state=random_state
        )
        
        return splits
        
    except Exception as e:
        logger.error(f"Error splitting dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/profile")
async def profile_dataset(
    dataset_id: str,
    data_service: DataService = Depends(get_data_service),
    current_user: str = Depends(get_current_user)
):
    """
    Generate comprehensive data profile
    """
    try:
        profile = await data_service.profile_dataset(dataset_id)
        return profile
    except Exception as e:
        logger.error(f"Error profiling dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dataset_id}/validate")
async def validate_dataset(
    dataset_id: str,
    validation_rules: List[Dict[str, Any]],
    data_service: DataService = Depends(get_data_service),
    current_user: str = Depends(get_current_user)
):
    """
    Validate dataset against rules
    """
    try:
        validation_result = await data_service.validate_dataset(
            dataset_id, validation_rules
        )
        return validation_result
    except Exception as e:
        logger.error(f"Error validating dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream/connect")
async def connect_data_stream(
    stream_config: Dict[str, Any],
    data_service: DataService = Depends(get_data_service),
    current_user: str = Depends(get_current_user)
):
    """
    Connect to a data stream (Kafka, Kinesis, etc.)
    """
    try:
        stream_id = await data_service.connect_stream(stream_config)
        return {"stream_id": stream_id, "message": "Stream connected successfully"}
    except Exception as e:
        logger.error(f"Error connecting to stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/stream/{stream_id}")
async def disconnect_data_stream(
    stream_id: str,
    data_service: DataService = Depends(get_data_service),
    current_user: str = Depends(get_current_user)
):
    """
    Disconnect from a data stream
    """
    try:
        success = await data_service.disconnect_stream(stream_id)
        if not success:
            raise HTTPException(status_code=404, detail="Stream not found")
        return {"message": "Stream disconnected successfully"}
    except Exception as e:
        logger.error(f"Error disconnecting stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream/{stream_id}/status")
async def get_stream_status(
    stream_id: str,
    data_service: DataService = Depends(get_data_service),
    current_user: str = Depends(get_current_user)
):
    """
    Get data stream status
    """
    try:
        status = await data_service.get_stream_status(stream_id)
        return status
    except Exception as e:
        logger.error(f"Error getting stream status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _process_dataset(dataset_id: str, data_service: DataService):
    """
    Background task for processing uploaded dataset
    """
    try:
        logger.info(f"Processing dataset {dataset_id}")
        
        # Compute statistics
        await data_service.compute_statistics(dataset_id)
        
        # Assess data quality
        await data_service.assess_data_quality(dataset_id)
        
        logger.info(f"Dataset {dataset_id} processed successfully")
        
    except Exception as e:
        logger.error(f"Error processing dataset {dataset_id}: {e}")


async def _preprocess_dataset(
    dataset_id: str,
    preprocessing_steps: List[Dict[str, Any]],
    data_service: DataService
):
    """
    Background task for preprocessing dataset
    """
    try:
        logger.info(f"Preprocessing dataset {dataset_id}")
        
        # Apply preprocessing steps
        await data_service.apply_preprocessing(dataset_id, preprocessing_steps)
        
        logger.info(f"Dataset {dataset_id} preprocessing completed")
        
    except Exception as e:
        logger.error(f"Error preprocessing dataset {dataset_id}: {e}")