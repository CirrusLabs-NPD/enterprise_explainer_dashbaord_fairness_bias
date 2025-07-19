"""
Fairness Analysis API Endpoints
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import pandas as pd
import io
import json
from datetime import datetime

from app.services.fairness_service import (
    FairnessAnalyzer, 
    validate_fairness_data
)
from app.services.data_drift_service import load_csv_flexible
import structlog

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/fairness", tags=["Fairness Analysis"])

# Pydantic models for request/response
class FairnessAnalysisRequest(BaseModel):
    target_column: str = Field(..., description="Name of the target/label column")
    sensitive_attribute: str = Field(..., description="Name of the sensitive attribute column")
    test_size: Optional[float] = Field(0.2, ge=0.1, le=0.5, description="Proportion of data for testing")
    model_type: Optional[str] = Field("random_forest", description="Type of model to train")
    random_state: Optional[int] = Field(42, description="Random seed for reproducibility")

class BiaseMitigationRequest(BaseModel):
    method: str = Field(..., description="Mitigation method")
    constraint: str = Field(..., description="Fairness constraint")

class FairnessAnalysisResponse(BaseModel):
    status: str
    analysis_id: str
    data_info: Dict[str, Any]
    model_performance: Dict[str, Any]
    fairness_analysis: Dict[str, Any]
    html_report: str

class MitigationResponse(BaseModel):
    status: str
    mitigation_results: Dict[str, Any]
    comparison: Dict[str, Any]

# Global analyzer instance
fairness_analyzer = FairnessAnalyzer()

@router.post("/analyze", response_model=FairnessAnalysisResponse)
async def analyze_fairness(
    dataset_file: UploadFile = File(..., description="Dataset CSV file"),
    request_data: str = Form(..., description="JSON string of FairnessAnalysisRequest")
):
    """
    Perform comprehensive fairness analysis on uploaded dataset
    """
    try:
        # Parse request parameters
        try:
            params = FairnessAnalysisRequest.parse_raw(request_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request parameters: {str(e)}")
        
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
        
        # Validate data for fairness analysis
        validation = validate_fairness_data(df, params.target_column, params.sensitive_attribute)
        if not validation["is_valid"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Data validation failed: {'; '.join(validation['issues'])}"
            )
        
        # Load data into analyzer
        data_info = fairness_analyzer.load_data(
            df, 
            params.target_column,
            params.sensitive_attribute,
            params.test_size,
            params.random_state
        )
        
        # Train baseline model
        model_performance = fairness_analyzer.train_baseline_model(params.model_type)
        
        # Perform fairness analysis
        fairness_analysis = fairness_analyzer.analyze_fairness()
        
        # Generate HTML report
        html_report = fairness_analyzer.generate_fairness_report()
        
        # Generate analysis ID
        analysis_id = f"fairness_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(
            "Fairness analysis completed",
            analysis_id=analysis_id,
            dataset_file=dataset_file.filename,
            target_column=params.target_column,
            sensitive_attribute=params.sensitive_attribute,
            fairness_level=fairness_analysis["fairness_assessment"]["level"]
        )
        
        return FairnessAnalysisResponse(
            status="success",
            analysis_id=analysis_id,
            data_info=data_info,
            model_performance=model_performance,
            fairness_analysis=fairness_analysis,
            html_report=html_report
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in fairness analysis", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/mitigate-bias", response_model=MitigationResponse)
async def mitigate_bias(
    request_data: BiaseMitigationRequest
):
    """
    Apply bias mitigation techniques to the current model
    """
    try:
        # Check if fairness analysis has been performed
        if fairness_analyzer.fairness_metrics is None:
            raise HTTPException(
                status_code=400, 
                detail="Fairness analysis must be performed first. Call /analyze endpoint."
            )
        
        # Apply bias mitigation
        mitigation_results = fairness_analyzer.apply_bias_mitigation(
            method=request_data.method,
            constraint=request_data.constraint
        )
        
        # Compare with baseline
        baseline_metrics = fairness_analyzer.fairness_metrics
        comparison = {
            "baseline": {
                "demographic_parity_difference": baseline_metrics["overall_metrics"]["demographic_parity_difference"],
                "equalized_odds_difference": baseline_metrics["overall_metrics"]["equalized_odds_difference"],
                "fairness_level": baseline_metrics["fairness_assessment"]["level"]
            },
            "mitigated": {
                "demographic_parity_difference": mitigation_results["mitigated_metrics"]["demographic_parity_difference"],
                "equalized_odds_difference": mitigation_results["mitigated_metrics"]["equalized_odds_difference"],
                "fairness_level": mitigation_results["fairness_assessment"]["level"]
            },
            "improvements": mitigation_results["baseline_comparison"]
        }
        
        logger.info(
            "Bias mitigation applied",
            method=request_data.method,
            constraint=request_data.constraint,
            fairness_improvement=mitigation_results["baseline_comparison"]["demographic_parity_improvement"]
        )
        
        return MitigationResponse(
            status="success",
            mitigation_results=mitigation_results,
            comparison=comparison
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in bias mitigation", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/validate-dataset")
async def validate_dataset(
    dataset_file: UploadFile = File(..., description="Dataset CSV file"),
    target_column: str = Form(..., description="Name of the target column"),
    sensitive_attribute: str = Form(..., description="Name of the sensitive attribute")
):
    """
    Validate dataset for fairness analysis
    """
    try:
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
        
        # Validate data
        validation = validate_fairness_data(df, target_column, sensitive_attribute)
        
        # Additional info
        dataset_info = {
            "filename": dataset_file.filename,
            "shape": df.shape,
            "columns": list(df.columns),
            "target_column_values": df[target_column].value_counts().to_dict() if target_column in df.columns else {},
            "sensitive_attribute_values": df[sensitive_attribute].value_counts().to_dict() if sensitive_attribute in df.columns else {},
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }
        
        logger.info(
            "Dataset validation completed",
            filename=dataset_file.filename,
            is_valid=validation["is_valid"],
            target_column=target_column,
            sensitive_attribute=sensitive_attribute
        )
        
        return {
            "status": "success",
            "validation": validation,
            "dataset_info": dataset_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error validating dataset", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/methods")
async def get_available_methods():
    """
    Get available bias mitigation methods and constraints
    """
    return {
        "status": "success",
        "mitigation_methods": {
            "exponentiated_gradient": {
                "description": "Uses exponentiated gradient method to enforce fairness constraints",
                "suitable_for": "Large datasets, complex models",
                "constraints": ["demographic_parity", "equalized_odds"]
            },
            "grid_search": {
                "description": "Searches over a grid of models to find fair solutions",
                "suitable_for": "Small to medium datasets",
                "constraints": ["demographic_parity", "equalized_odds"]
            },
            "threshold_optimizer": {
                "description": "Post-processing method that optimizes decision thresholds",
                "suitable_for": "When model retraining is not possible",
                "constraints": ["demographic_parity", "equalized_odds"]
            }
        },
        "fairness_constraints": {
            "demographic_parity": {
                "description": "Equal positive prediction rates across groups",
                "when_to_use": "When equal representation in positive outcomes is important"
            },
            "equalized_odds": {
                "description": "Equal true positive and false positive rates across groups",
                "when_to_use": "When maintaining prediction accuracy is crucial"
            }
        },
        "model_types": {
            "random_forest": {
                "description": "Random Forest Classifier",
                "advantages": ["Good performance", "Feature importance", "Handles missing values"]
            },
            "logistic_regression": {
                "description": "Logistic Regression",
                "advantages": ["Interpretable", "Fast training", "Probabilistic outputs"]
            }
        }
    }

@router.get("/health")
async def health_check():
    """
    Health check endpoint for the fairness service
    """
    return {
        "status": "healthy",
        "service": "fairness",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@router.get("/metrics-info")
async def get_fairness_metrics_info():
    """
    Get information about fairness metrics
    """
    return {
        "status": "success",
        "metrics": {
            "demographic_parity_difference": {
                "description": "Difference in positive prediction rates between groups",
                "range": "[-1, 1]",
                "interpretation": "0 = perfectly fair, closer to 0 is better",
                "threshold": "Typically < 0.1 for acceptable fairness"
            },
            "equalized_odds_difference": {
                "description": "Difference in true positive rates and false positive rates between groups",
                "range": "[0, 1]",
                "interpretation": "0 = perfectly fair, closer to 0 is better",
                "threshold": "Typically < 0.1 for acceptable fairness"
            },
            "selection_rate": {
                "description": "Proportion of positive predictions for each group",
                "range": "[0, 1]",
                "interpretation": "Should be similar across groups for demographic parity"
            },
            "true_positive_rate": {
                "description": "Proportion of actual positives correctly identified",
                "range": "[0, 1]",
                "interpretation": "Higher is better, should be similar across groups"
            },
            "false_positive_rate": {
                "description": "Proportion of actual negatives incorrectly classified as positive",
                "range": "[0, 1]",
                "interpretation": "Lower is better, should be similar across groups"
            }
        },
        "fairness_levels": {
            "fair": "Demographic parity and equalized odds differences ≤ 0.1",
            "concerning": "Differences between 0.1 and 0.2",
            "unfair": "Differences > 0.2"
        }
    }