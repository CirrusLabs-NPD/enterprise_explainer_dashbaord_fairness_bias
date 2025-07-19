"""
Data Drift Detection API Endpoints
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import pandas as pd
import io
import json
from datetime import datetime

from app.services.data_drift_service import (
    DataDriftAnalyzer, 
    load_csv_flexible, 
    validate_datasets_compatibility
)
import structlog

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/data-drift", tags=["Data Drift Analysis"])

# Pydantic models for request/response
class DriftAnalysisRequest(BaseModel):
    split_ratio: Optional[float] = Field(0.7, ge=0.1, le=0.9, description="Ratio for splitting single dataset")
    drop_columns: Optional[List[str]] = Field(None, description="Columns to exclude from analysis")
    time_column: Optional[str] = Field(None, description="Column name for time-based splitting")
    reference_start: Optional[str] = Field(None, description="Start date for reference period (YYYY-MM-DD)")
    reference_end: Optional[str] = Field(None, description="End date for reference period (YYYY-MM-DD)")
    current_start: Optional[str] = Field(None, description="Start date for current period (YYYY-MM-DD)")
    current_end: Optional[str] = Field(None, description="End date for current period (YYYY-MM-DD)")

class ColumnDriftRequest(BaseModel):
    column_name: str = Field(..., description="Name of the column to analyze")
    drop_columns: Optional[List[str]] = Field(None, description="Columns to exclude from analysis")

class ModelDriftRequest(BaseModel):
    task_type: str = Field("classification", description="Type of ML task (classification or regression)")
    drop_columns: Optional[List[str]] = Field(None, description="Columns to exclude from analysis")

class DriftAnalysisResponse(BaseModel):
    status: str
    analysis_id: str
    html_report: str
    drift_summary: Dict[str, Any]
    data_preview: Dict[str, Any]
    metadata: Dict[str, Any]

class ColumnDriftResponse(BaseModel):
    status: str
    analysis_id: str
    html_report: str
    column_summary: Dict[str, Any]
    column_name: str
    metadata: Dict[str, Any]

class ModelDriftResponse(BaseModel):
    status: str
    analysis_id: str
    html_report: str
    model_drift_summary: Dict[str, Any]
    prediction_distribution_changes: Dict[str, Any]
    performance_degradation: Dict[str, Any]
    metadata: Dict[str, Any]

# Global analyzer instance (in production, consider using dependency injection)
drift_analyzer = DataDriftAnalyzer()

@router.post("/analyze/upload-datasets", response_model=DriftAnalysisResponse)
async def analyze_drift_from_uploads(
    reference_file: UploadFile = File(..., description="Reference dataset CSV file"),
    current_file: UploadFile = File(..., description="Current dataset CSV file"),
    request_data: str = Form("{}", description="JSON string of DriftAnalysisRequest")
):
    """
    Analyze data drift between two uploaded CSV files
    """
    try:
        # Parse request parameters
        try:
            params = DriftAnalysisRequest.parse_raw(request_data)
        except:
            params = DriftAnalysisRequest()
        
        # Validate file types
        if not reference_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Reference file must be a CSV")
        if not current_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Current file must be a CSV")
        
        # Read files
        ref_content = await reference_file.read()
        cur_content = await current_file.read()
        
        # Load as DataFrames
        try:
            ref_df = load_csv_flexible(ref_content)
            cur_df = load_csv_flexible(cur_content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading CSV files: {str(e)}")
        
        # Validate compatibility
        compatibility = validate_datasets_compatibility(ref_df, cur_df)
        if not compatibility["is_compatible"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Datasets are not compatible: {'; '.join(compatibility['issues'])}"
            )
        
        # Load data into analyzer
        drift_analyzer.load_data(ref_df, cur_df)
        
        # Preprocess if needed
        if params.drop_columns:
            drift_analyzer.preprocess_data(drop_columns=params.drop_columns)
        
        # Generate analysis
        analysis_result = drift_analyzer.generate_full_drift_report()
        data_preview = drift_analyzer.get_data_preview()
        
        # Generate analysis ID
        analysis_id = f"drift_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(
            "Drift analysis completed from uploads",
            analysis_id=analysis_id,
            reference_file=reference_file.filename,
            current_file=current_file.filename,
            drift_detected=analysis_result["drift_summary"].get("overall_drift_detected", False)
        )
        
        return DriftAnalysisResponse(
            status="success",
            analysis_id=analysis_id,
            html_report=analysis_result["html_report"],
            drift_summary=analysis_result["drift_summary"],
            data_preview=data_preview,
            metadata=analysis_result["metadata"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in drift analysis from uploads", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/analyze/single-dataset", response_model=DriftAnalysisResponse)
async def analyze_drift_from_single_dataset(
    dataset_file: UploadFile = File(..., description="Dataset CSV file to split and analyze"),
    request_data: str = Form("{}", description="JSON string of DriftAnalysisRequest")
):
    """
    Analyze data drift by splitting a single dataset into reference and current portions
    """
    try:
        # Parse request parameters
        try:
            params = DriftAnalysisRequest.parse_raw(request_data)
        except:
            params = DriftAnalysisRequest()
        
        # Validate file type
        if not dataset_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Dataset file must be a CSV")
        
        # Read file
        content = await dataset_file.read()
        
        # Load as DataFrame
        try:
            df = load_csv_flexible(content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading CSV file: {str(e)}")
        
        # Check if we have enough data
        if len(df) < 20:
            raise HTTPException(status_code=400, detail="Dataset too small for meaningful analysis (minimum 20 rows)")
        
        # Split data based on method
        if params.time_column and params.reference_start and params.current_start:
            # Time-based splitting
            if params.time_column not in df.columns:
                raise HTTPException(status_code=400, detail=f"Time column '{params.time_column}' not found")
            
            drift_analyzer.split_by_time(
                df, 
                params.time_column,
                params.reference_start, 
                params.reference_end,
                params.current_start, 
                params.current_end
            )
        else:
            # Ratio-based splitting
            drift_analyzer.split_single_dataset(df, params.split_ratio)
        
        # Preprocess if needed
        if params.drop_columns:
            drift_analyzer.preprocess_data(drop_columns=params.drop_columns)
        
        # Generate analysis
        analysis_result = drift_analyzer.generate_full_drift_report()
        data_preview = drift_analyzer.get_data_preview()
        
        # Generate analysis ID
        analysis_id = f"drift_single_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(
            "Drift analysis completed from single dataset",
            analysis_id=analysis_id,
            dataset_file=dataset_file.filename,
            split_method="time" if params.time_column else "ratio",
            drift_detected=analysis_result["drift_summary"].get("overall_drift_detected", False)
        )
        
        return DriftAnalysisResponse(
            status="success",
            analysis_id=analysis_id,
            html_report=analysis_result["html_report"],
            drift_summary=analysis_result["drift_summary"],
            data_preview=data_preview,
            metadata=analysis_result["metadata"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in drift analysis from single dataset", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/analyze/column-drift", response_model=ColumnDriftResponse)
async def analyze_column_drift(
    reference_file: UploadFile = File(..., description="Reference dataset CSV file"),
    current_file: UploadFile = File(..., description="Current dataset CSV file"),
    request_data: str = Form(..., description="JSON string of ColumnDriftRequest")
):
    """
    Analyze data drift for a specific column
    """
    try:
        # Parse request parameters
        try:
            params = ColumnDriftRequest.parse_raw(request_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request parameters: {str(e)}")
        
        # Validate file types
        if not reference_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Reference file must be a CSV")
        if not current_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Current file must be a CSV")
        
        # Read files
        ref_content = await reference_file.read()
        cur_content = await current_file.read()
        
        # Load as DataFrames
        try:
            ref_df = load_csv_flexible(ref_content)
            cur_df = load_csv_flexible(cur_content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading CSV files: {str(e)}")
        
        # Check if column exists
        if params.column_name not in ref_df.columns or params.column_name not in cur_df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{params.column_name}' not found in both datasets")
        
        # Load data into analyzer
        drift_analyzer.load_data(ref_df, cur_df)
        
        # Preprocess if needed
        if params.drop_columns:
            drift_analyzer.preprocess_data(drop_columns=params.drop_columns)
        
        # Generate column-specific analysis
        analysis_result = drift_analyzer.generate_column_drift_report(params.column_name)
        
        # Generate analysis ID
        analysis_id = f"drift_column_{params.column_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(
            "Column drift analysis completed",
            analysis_id=analysis_id,
            column_name=params.column_name,
            drift_detected=analysis_result["column_summary"].get("drift_detected", False)
        )
        
        return ColumnDriftResponse(
            status="success",
            analysis_id=analysis_id,
            html_report=analysis_result["html_report"],
            column_summary=analysis_result["column_summary"],
            column_name=params.column_name,
            metadata=analysis_result["metadata"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in column drift analysis", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/analyze/model-drift", response_model=ModelDriftResponse)
async def analyze_model_drift(
    reference_file: UploadFile = File(..., description="Reference dataset CSV file"),
    current_file: UploadFile = File(..., description="Current dataset CSV file"),
    reference_predictions_file: UploadFile = File(..., description="Reference predictions CSV file"),
    current_predictions_file: UploadFile = File(..., description="Current predictions CSV file"),
    reference_labels_file: Optional[UploadFile] = File(None, description="Reference labels CSV file (optional)"),
    current_labels_file: Optional[UploadFile] = File(None, description="Current labels CSV file (optional)"),
    request_data: str = Form(..., description="JSON string of ModelDriftRequest")
):
    """
    Analyze model drift by comparing model predictions between reference and current periods
    """
    try:
        # Parse request parameters
        try:
            params = ModelDriftRequest.parse_raw(request_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request parameters: {str(e)}")
        
        # Validate file types
        files_to_check = [reference_file, current_file, reference_predictions_file, current_predictions_file]
        if reference_labels_file:
            files_to_check.append(reference_labels_file)
        if current_labels_file:
            files_to_check.append(current_labels_file)
            
        for file in files_to_check:
            if not file.filename.endswith('.csv'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} must be a CSV")
        
        # Read data files
        ref_content = await reference_file.read()
        cur_content = await current_file.read()
        ref_pred_content = await reference_predictions_file.read()
        cur_pred_content = await current_predictions_file.read()
        
        # Load datasets
        try:
            ref_df = load_csv_flexible(ref_content)
            cur_df = load_csv_flexible(cur_content)
            ref_pred_df = load_csv_flexible(ref_pred_content)
            cur_pred_df = load_csv_flexible(cur_pred_content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading CSV files: {str(e)}")
        
        # Extract predictions (assume first column contains predictions)
        reference_predictions = ref_pred_df.iloc[:, 0].values
        current_predictions = cur_pred_df.iloc[:, 0].values
        
        # Load labels if provided
        reference_labels = None
        current_labels = None
        
        if reference_labels_file and current_labels_file:
            try:
                ref_labels_content = await reference_labels_file.read()
                cur_labels_content = await current_labels_file.read()
                ref_labels_df = load_csv_flexible(ref_labels_content)
                cur_labels_df = load_csv_flexible(cur_labels_content)
                
                reference_labels = ref_labels_df.iloc[:, 0].values
                current_labels = cur_labels_df.iloc[:, 0].values
            except Exception as e:
                logger.warning("Error loading label files, proceeding without labels", error=str(e))
        
        # Validate compatibility
        compatibility = validate_datasets_compatibility(ref_df, cur_df)
        if not compatibility["is_compatible"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Datasets are not compatible: {'; '.join(compatibility['issues'])}"
            )
        
        # Load data into analyzer
        drift_analyzer.load_data(ref_df, cur_df)
        
        # Preprocess if needed
        if params.drop_columns:
            drift_analyzer.preprocess_data(drop_columns=params.drop_columns)
        
        # Load model predictions
        drift_analyzer.load_model_predictions(
            reference_predictions=reference_predictions,
            current_predictions=current_predictions,
            reference_labels=reference_labels,
            current_labels=current_labels
        )
        
        # Perform model drift analysis
        analysis_result = drift_analyzer.analyze_model_drift(params.task_type)
        
        # Generate analysis ID
        analysis_id = f"model_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(
            "Model drift analysis completed",
            analysis_id=analysis_id,
            task_type=params.task_type,
            prediction_drift_detected=analysis_result["model_drift_summary"].get("prediction_drift_detected", False)
        )
        
        return ModelDriftResponse(
            status="success",
            analysis_id=analysis_id,
            html_report=analysis_result["html_report"],
            model_drift_summary=analysis_result["model_drift_summary"],
            prediction_distribution_changes=analysis_result["prediction_distribution_changes"],
            performance_degradation=analysis_result["performance_degradation"],
            metadata=analysis_result["metadata"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in model drift analysis", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/columns")
async def get_available_columns():
    """
    Get list of columns available in the currently loaded datasets
    """
    try:
        if drift_analyzer.reference_df is None or drift_analyzer.current_df is None:
            raise HTTPException(status_code=400, detail="No datasets loaded. Upload datasets first.")
        
        ref_columns = list(drift_analyzer.reference_df.columns)
        cur_columns = list(drift_analyzer.current_df.columns)
        common_columns = list(set(ref_columns) & set(cur_columns))
        
        return {
            "status": "success",
            "reference_columns": ref_columns,
            "current_columns": cur_columns,
            "common_columns": common_columns,
            "total_common": len(common_columns)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting available columns", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check endpoint for the data drift service
    """
    return {
        "status": "healthy",
        "service": "data-drift",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@router.post("/validate-datasets")
async def validate_uploaded_datasets(
    reference_file: UploadFile = File(..., description="Reference dataset CSV file"),
    current_file: UploadFile = File(..., description="Current dataset CSV file")
):
    """
    Validate that uploaded datasets are compatible for drift analysis
    """
    try:
        # Read files
        ref_content = await reference_file.read()
        cur_content = await current_file.read()
        
        # Load as DataFrames
        try:
            ref_df = load_csv_flexible(ref_content)
            cur_df = load_csv_flexible(cur_content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading CSV files: {str(e)}")
        
        # Validate compatibility
        compatibility = validate_datasets_compatibility(ref_df, cur_df)
        
        logger.info(
            "Dataset validation completed",
            reference_file=reference_file.filename,
            current_file=current_file.filename,
            is_compatible=compatibility["is_compatible"]
        )
        
        return {
            "status": "success",
            "validation": compatibility,
            "reference_info": {
                "filename": reference_file.filename,
                "shape": ref_df.shape,
                "columns": list(ref_df.columns)
            },
            "current_info": {
                "filename": current_file.filename,
                "shape": cur_df.shape,
                "columns": list(cur_df.columns)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error validating datasets", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")