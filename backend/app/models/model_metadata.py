"""
Data models for model metadata and explanations
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
import numpy as np


class ModelType(str, Enum):
    """Model type enumeration"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    RECOMMENDATION = "recommendation"


class ModelStatus(str, Enum):
    """Model status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    VALIDATION = "validation"
    ARCHIVED = "archived"
    ERROR = "error"


class ModelFramework(str, Enum):
    """Model framework enumeration"""
    SKLEARN = "sklearn"
    ONNX = "onnx"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    CUSTOM = "custom"


class ModelMetadata(BaseModel):
    """Model metadata model"""
    model_id: str
    name: str
    model_type: ModelType
    framework: ModelFramework = ModelFramework.SKLEARN
    file_path: str
    version: str = "1.0.0"
    description: str = ""
    
    # Features and targets
    feature_names: List[str] = Field(default_factory=list)
    target_names: List[str] = Field(default_factory=list)
    feature_types: Dict[str, str] = Field(default_factory=dict)  # feature_name -> type
    
    # Preprocessing
    preprocessing_steps: List[Dict[str, Any]] = Field(default_factory=list)
    scaler_path: Optional[str] = None
    encoder_path: Optional[str] = None
    
    # Model parameters
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Performance metrics
    training_metrics: Dict[str, float] = Field(default_factory=dict)
    validation_metrics: Dict[str, float] = Field(default_factory=dict)
    test_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Model info
    model_size_bytes: Optional[int] = None
    training_time_seconds: Optional[float] = None
    inference_time_ms: Optional[float] = None
    
    # Monitoring
    drift_threshold: float = 0.1
    performance_threshold: float = 0.05
    last_evaluation: Optional[datetime] = None
    
    # Metadata
    status: ModelStatus = ModelStatus.ACTIVE
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Tags and labels
    tags: List[str] = Field(default_factory=list)
    labels: Dict[str, str] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class ExplanationRequest(BaseModel):
    """Request model for explanations"""
    model_id: str
    method: str
    data: List[List[float]]  # Input data
    feature_names: List[str]
    target_names: Optional[List[str]] = None
    instance_indices: Optional[List[int]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Background data for certain methods
    background_data: Optional[List[List[float]]] = None
    
    # Explanation options
    top_k_features: Optional[int] = None
    return_interactions: bool = False
    return_dependence: bool = False


class ExplanationResult(BaseModel):
    """Result model for explanations"""
    explanation_id: str
    model_id: str
    method: str
    feature_names: List[str]
    explanation: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    execution_time_ms: Optional[float] = None


class ShapExplanation(BaseModel):
    """SHAP explanation result"""
    shap_values: List[float]
    base_value: float
    feature_names: List[str]
    feature_values: List[Union[float, str]]
    prediction: float
    expected_value: Optional[float] = None
    
    # For interaction values
    interaction_values: Optional[List[List[float]]] = None


class LimeExplanation(BaseModel):
    """LIME explanation result"""
    feature_importance: Dict[str, float]
    feature_names: List[str]
    feature_values: List[Union[float, str]]
    prediction: float
    score: float
    local_model_r2: Optional[float] = None


class FeatureImportance(BaseModel):
    """Feature importance result"""
    feature_names: List[str]
    importance_values: List[float]
    method: str
    ranking: List[int]
    confidence_intervals: Optional[List[Tuple[float, float]]] = None


class InteractionAnalysis(BaseModel):
    """Feature interaction analysis result"""
    feature_names: List[str]
    interaction_matrix: List[List[float]]
    method: str
    top_interactions: Optional[List[Dict[str, Any]]] = None


class DependenceAnalysis(BaseModel):
    """Partial dependence analysis result"""
    feature_name: str
    feature_values: List[float]
    dependence_values: List[float]
    method: str
    confidence_intervals: Optional[List[Tuple[float, float]]] = None


class ModelPerformanceMetrics(BaseModel):
    """Model performance metrics"""
    model_id: str
    metrics: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    dataset_size: int
    prediction_time_ms: Optional[float] = None


class DataDriftReport(BaseModel):
    """Data drift detection report"""
    model_id: str
    feature_drifts: Dict[str, float]  # feature_name -> drift_score
    overall_drift_score: float
    drift_threshold: float
    drift_detected: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    reference_dataset_size: int
    current_dataset_size: int
    detection_method: str


class ModelDriftReport(BaseModel):
    """Model drift detection report"""
    model_id: str
    performance_drift: float
    drift_threshold: float
    drift_detected: bool
    current_metrics: Dict[str, float]
    reference_metrics: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    detection_method: str


class DataQualityReport(BaseModel):
    """Data quality assessment report"""
    dataset_id: str
    total_rows: int
    total_features: int
    missing_values: Dict[str, int]  # feature_name -> count
    duplicate_rows: int
    outliers: Dict[str, int]  # feature_name -> count
    data_types: Dict[str, str]  # feature_name -> type
    quality_score: float  # 0-1 score
    issues: List[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    model_id: str
    data: List[List[float]]
    feature_names: List[str]
    return_probabilities: bool = False
    return_explanations: bool = False
    explanation_method: Optional[str] = None


class PredictionResult(BaseModel):
    """Result model for predictions"""
    predictions: List[float]
    probabilities: Optional[List[List[float]]] = None
    explanations: Optional[List[ExplanationResult]] = None
    prediction_time_ms: float
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    model_id: str
    data_source: str  # file path or URL
    batch_size: int = 100
    return_probabilities: bool = False
    output_format: str = "json"  # json, csv, parquet


class BatchPredictionResult(BaseModel):
    """Result model for batch predictions"""
    batch_id: str
    model_id: str
    status: str
    total_rows: int
    processed_rows: int
    output_path: Optional[str] = None
    error_count: int = 0
    errors: List[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    processing_time_seconds: Optional[float] = None


class ModelMonitoringConfig(BaseModel):
    """Model monitoring configuration"""
    model_id: str
    monitoring_enabled: bool = True
    drift_detection_enabled: bool = True
    performance_monitoring_enabled: bool = True
    data_quality_monitoring_enabled: bool = True
    
    # Thresholds
    drift_threshold: float = 0.1
    performance_threshold: float = 0.05
    data_quality_threshold: float = 0.8
    
    # Monitoring intervals
    drift_check_interval_hours: int = 24
    performance_check_interval_hours: int = 6
    data_quality_check_interval_hours: int = 12
    
    # Alerting
    alert_on_drift: bool = True
    alert_on_performance_drop: bool = True
    alert_on_data_quality_issues: bool = True
    alert_channels: List[str] = Field(default_factory=list)  # email, slack, webhook
    
    # Data retention
    keep_monitoring_data_days: int = 30
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AlertConfig(BaseModel):
    """Alert configuration"""
    alert_id: str
    name: str
    model_id: str
    alert_type: str  # drift, performance, data_quality
    condition: str  # threshold condition
    threshold_value: float
    enabled: bool = True
    
    # Notification settings
    channels: List[str] = Field(default_factory=list)
    severity: str = "medium"  # low, medium, high, critical
    
    # Rate limiting
    max_alerts_per_hour: int = 1
    last_alert_sent: Optional[datetime] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Alert(BaseModel):
    """Alert instance"""
    alert_id: str
    config_id: str
    model_id: str
    alert_type: str
    severity: str
    title: str
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)
    
    # Status
    status: str = "active"  # active, acknowledged, resolved
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DatasetMetadata(BaseModel):
    """Dataset metadata"""
    dataset_id: str
    name: str
    description: str = ""
    file_path: str
    format: str  # csv, parquet, json, etc.
    size_bytes: int
    num_rows: int
    num_columns: int
    
    # Schema
    column_names: List[str] = Field(default_factory=list)
    column_types: Dict[str, str] = Field(default_factory=dict)
    
    # Statistics
    statistics: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    tags: List[str] = Field(default_factory=list)


class ExperimentConfig(BaseModel):
    """A/B testing experiment configuration"""
    experiment_id: str
    name: str
    description: str = ""
    
    # Models to compare
    control_model_id: str
    treatment_model_ids: List[str]
    
    # Traffic allocation
    traffic_allocation: Dict[str, float] = Field(default_factory=dict)  # model_id -> percentage
    
    # Experiment settings
    duration_days: int = 30
    minimum_sample_size: int = 1000
    significance_level: float = 0.05
    
    # Metrics to track
    success_metrics: List[str] = Field(default_factory=list)
    
    # Status
    status: str = "draft"  # draft, running, completed, stopped
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)