"""
Advanced analysis endpoints for ML models
Includes what-if analysis, feature dependence, and decision trees
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.dependencies import (
    get_model_service,
    get_explanation_service,
    get_current_user,
    validate_model_id
)
from app.services.model_service import ModelService
from app.services.explanation_service import ExplanationService
from app.services.analysis_service import AnalysisService

# Dependency function for AnalysisService
async def get_analysis_service() -> AnalysisService:
    """Get or create analysis service instance"""
    from app.core.dependencies import get_worker_pool
    worker_pool = await get_worker_pool()
    return AnalysisService(worker_pool)

router = APIRouter(prefix="/analysis", tags=["analysis"])


# Request/Response Models
class WhatIfScenario(BaseModel):
    """What-if scenario definition"""
    base_instance: Dict[str, float] = Field(..., description="Base instance values")
    modifications: Dict[str, float] = Field(..., description="Feature modifications to apply")
    
class WhatIfRequest(BaseModel):
    """What-if analysis request"""
    scenarios: List[WhatIfScenario] = Field(..., description="List of what-if scenarios")
    feature_names: List[str] = Field(..., description="Feature names")
    compare_with_original: bool = Field(default=True, description="Compare with original prediction")

class WhatIfResult(BaseModel):
    """What-if analysis result"""
    scenario_id: int
    original_prediction: Optional[float]
    original_probabilities: Optional[List[float]]
    modified_prediction: float
    modified_probabilities: Optional[List[float]]
    feature_changes: Dict[str, float]
    prediction_change: float
    confidence_change: Optional[float]

class FeatureDependenceRequest(BaseModel):
    """Feature dependence analysis request"""
    feature_names: List[str] = Field(..., description="Features to analyze")
    num_points: int = Field(default=50, ge=10, le=200, description="Number of points for PDP")
    method: str = Field(default="pdp", description="Method: 'pdp' or 'ice'")
    sample_size: Optional[int] = Field(None, description="Sample size for ICE plots")

class DecisionTreeRequest(BaseModel):
    """Decision tree visualization request"""
    max_depth: Optional[int] = Field(None, ge=1, le=20, description="Maximum tree depth")
    min_samples_split: int = Field(default=20, ge=2, description="Minimum samples to split")
    min_samples_leaf: int = Field(default=10, ge=1, description="Minimum samples in leaf")
    feature_names: Optional[List[str]] = Field(None, description="Feature names for visualization")
    class_names: Optional[List[str]] = Field(None, description="Class names for visualization")

class CounterfactualRequest(BaseModel):
    """Counterfactual explanation request"""
    instance: Dict[str, float] = Field(..., description="Instance to explain")
    desired_outcome: Optional[float] = Field(None, description="Desired prediction outcome")
    max_features_to_change: int = Field(default=5, ge=1, le=10, description="Max features to modify")
    feature_ranges: Optional[Dict[str, tuple]] = Field(None, description="Valid ranges for features")


@router.post("/{model_id}/what-if", response_model=List[WhatIfResult])
async def analyze_what_if_scenarios(
    model_id: str = Depends(validate_model_id),
    request: WhatIfRequest = ...,
    model_service: ModelService = Depends(get_model_service),
    analysis_service: AnalysisService = Depends(get_analysis_service),
    current_user: str = Depends(get_current_user)
):
    """
    Perform what-if analysis on multiple scenarios
    
    This endpoint allows you to:
    - Test how predictions change with modified features
    - Compare multiple scenarios side-by-side
    - Understand model sensitivity to feature changes
    """
    try:
        # Load model
        model_info = await model_service.get_model(model_id)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        results = []
        
        for idx, scenario in enumerate(request.scenarios):
            # Get original prediction if requested
            original_pred = None
            original_probs = None
            
            if request.compare_with_original:
                original_result = await model_service.predict(
                    model_id=model_id,
                    data=[list(scenario.base_instance.values())],
                    feature_names=request.feature_names,
                    return_probabilities=True
                )
                original_pred = original_result['predictions'][0]
                original_probs = original_result.get('probabilities', [None])[0]
            
            # Apply modifications
            modified_instance = scenario.base_instance.copy()
            modified_instance.update(scenario.modifications)
            
            # Get modified prediction
            modified_result = await model_service.predict(
                model_id=model_id,
                data=[list(modified_instance.values())],
                feature_names=request.feature_names,
                return_probabilities=True
            )
            modified_pred = modified_result['predictions'][0]
            modified_probs = modified_result.get('probabilities', [None])[0]
            
            # Calculate changes
            pred_change = modified_pred - original_pred if original_pred is not None else 0
            conf_change = None
            if original_probs and modified_probs:
                conf_change = max(modified_probs) - max(original_probs)
            
            results.append(WhatIfResult(
                scenario_id=idx,
                original_prediction=original_pred,
                original_probabilities=original_probs,
                modified_prediction=modified_pred,
                modified_probabilities=modified_probs,
                feature_changes=scenario.modifications,
                prediction_change=pred_change,
                confidence_change=conf_change
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"What-if analysis failed: {str(e)}"
        )


@router.post("/{model_id}/feature-dependence")
async def analyze_feature_dependence(
    model_id: str = Depends(validate_model_id),
    request: FeatureDependenceRequest = ...,
    model_service: ModelService = Depends(get_model_service),
    explanation_service: ExplanationService = Depends(get_explanation_service),
    current_user: str = Depends(get_current_user)
):
    """
    Analyze feature dependence using PDP or ICE plots
    
    - PDP (Partial Dependence Plot): Shows average effect of features
    - ICE (Individual Conditional Expectation): Shows individual instance effects
    """
    try:
        # Get model metadata
        model_info = await model_service.get_model(model_id)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        # Load model and reference data
        model = await model_service.load_model(model_id)
        
        if request.method == "pdp":
            # Calculate partial dependence
            pdp_results = {}
            for feature in request.feature_names:
                pdp_values = await explanation_service._partial_dependence(
                    model=model,
                    feature=feature,
                    feature_names=model_info.get('feature_names', []),
                    X_reference=None,  # Should load reference data
                    num_points=request.num_points
                )
                pdp_results[feature] = pdp_values
            
            return {
                "method": "pdp",
                "results": pdp_results,
                "feature_names": request.feature_names,
                "num_points": request.num_points
            }
            
        elif request.method == "ice":
            # Calculate ICE curves
            ice_results = await explanation_service._ice_plots(
                model=model,
                features=request.feature_names,
                X_reference=None,  # Should load reference data
                sample_size=request.sample_size or 100
            )
            
            return {
                "method": "ice",
                "results": ice_results,
                "feature_names": request.feature_names,
                "sample_size": request.sample_size or 100
            }
        
        else:
            raise ValueError(f"Unknown method: {request.method}")
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature dependence analysis failed: {str(e)}"
        )


@router.post("/{model_id}/decision-tree")
async def extract_decision_tree(
    model_id: str = Depends(validate_model_id),
    request: DecisionTreeRequest = ...,
    model_service: ModelService = Depends(get_model_service),
    analysis_service: AnalysisService = Depends(get_analysis_service),
    current_user: str = Depends(get_current_user)
):
    """
    Extract decision tree representation of the model
    
    For tree-based models: Returns actual tree structure
    For black-box models: Returns surrogate tree approximation
    """
    try:
        # Get model metadata
        model_info = await model_service.get_model(model_id)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        # Load model
        model = await model_service.load_model(model_id)
        
        # Extract or build decision tree
        tree_data = await analysis_service.extract_decision_tree(
            model=model,
            model_type=model_info.get('model_type'),
            max_depth=request.max_depth,
            min_samples_split=request.min_samples_split,
            min_samples_leaf=request.min_samples_leaf,
            feature_names=request.feature_names or model_info.get('feature_names'),
            class_names=request.class_names or model_info.get('target_names')
        )
        
        return tree_data
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Decision tree extraction failed: {str(e)}"
        )


@router.post("/{model_id}/counterfactuals")
async def generate_counterfactuals(
    model_id: str = Depends(validate_model_id),
    request: CounterfactualRequest = ...,
    model_service: ModelService = Depends(get_model_service),
    explanation_service: ExplanationService = Depends(get_explanation_service),
    current_user: str = Depends(get_current_user)
):
    """
    Generate counterfactual explanations
    
    Find minimal changes needed to achieve a different prediction outcome
    """
    try:
        # Get model metadata
        model_info = await model_service.get_model(model_id)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        # Generate counterfactuals
        counterfactuals = await explanation_service.generate_counterfactuals(
            model_id=model_id,
            instance=request.instance,
            desired_outcome=request.desired_outcome,
            max_features_to_change=request.max_features_to_change,
            feature_ranges=request.feature_ranges
        )
        
        return counterfactuals
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Counterfactual generation failed: {str(e)}"
        )


@router.get("/{model_id}/feature-importance/detailed")
async def get_detailed_feature_importance(
    model_id: str = Depends(validate_model_id),
    method: str = "shap",
    top_k: int = 20,
    model_service: ModelService = Depends(get_model_service),
    explanation_service: ExplanationService = Depends(get_explanation_service),
    current_user: str = Depends(get_current_user)
):
    """
    Get detailed feature importance with confidence intervals
    """
    try:
        # Get feature importance
        importance_data = await model_service.get_feature_importance(
            model_id=model_id,
            method=method
        )
        
        # Add statistical analysis
        detailed_importance = await explanation_service.analyze_feature_importance(
            importance_data=importance_data,
            top_k=top_k,
            include_confidence_intervals=True,
            include_interactions=True
        )
        
        return detailed_importance
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature importance analysis failed: {str(e)}"
        )