"""
Configuration settings for the ML Explainer Dashboard
"""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "ML Explainer Dashboard"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # CORS
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    
    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///./ml_explainer.db",
        env="DATABASE_URL"
    )
    DATABASE_MIN_CONNECTIONS: int = Field(default=1, env="DATABASE_MIN_CONNECTIONS")
    DATABASE_MAX_CONNECTIONS: int = Field(default=5, env="DATABASE_MAX_CONNECTIONS") 
    DATABASE_TIMEOUT: int = Field(default=60, env="DATABASE_TIMEOUT")
    
    # Redis
    REDIS_URL: str = Field(
        default="redis://localhost:6379",
        env="REDIS_URL"
    )
    
    # Worker Configuration
    MAX_WORKERS: int = Field(default=4, env="MAX_WORKERS")
    MAX_CPU_WORKERS: int = Field(default=2, env="MAX_CPU_WORKERS")
    MAX_IO_WORKERS: int = Field(default=4, env="MAX_IO_WORKERS")
    WORKER_QUEUE_SIZE: int = Field(default=100, env="WORKER_QUEUE_SIZE")
    
    # SHAP Configuration
    SHAP_MAX_EVALS: int = Field(default=100, env="SHAP_MAX_EVALS")
    SHAP_BATCH_SIZE: int = Field(default=32, env="SHAP_BATCH_SIZE")
    SHAP_TIMEOUT: int = Field(default=300, env="SHAP_TIMEOUT")  # seconds
    
    # LIME Configuration
    LIME_NUM_FEATURES: int = Field(default=10, env="LIME_NUM_FEATURES")
    LIME_NUM_SAMPLES: int = Field(default=5000, env="LIME_NUM_SAMPLES")
    LIME_TIMEOUT: int = Field(default=180, env="LIME_TIMEOUT")  # seconds
    
    # File Upload
    MAX_UPLOAD_SIZE: int = Field(default=500 * 1024 * 1024, env="MAX_UPLOAD_SIZE")  # 500MB
    UPLOAD_DIR: str = Field(default="/tmp/uploads", env="UPLOAD_DIR")
    
    # Model Storage
    MODEL_STORAGE_DIR: str = Field(default="/tmp/models", env="MODEL_STORAGE_DIR")
    
    # Monitoring
    MONITORING_INTERVAL: int = Field(default=60, env="MONITORING_INTERVAL")  # seconds
    DRIFT_THRESHOLD: float = Field(default=0.1, env="DRIFT_THRESHOLD")
    PERFORMANCE_THRESHOLD: float = Field(default=0.05, env="PERFORMANCE_THRESHOLD")
    
    # Caching
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # seconds
    
    # Security
    SECRET_KEY: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")
    
    # External Services
    PROMETHEUS_GATEWAY: Optional[str] = Field(default=None, env="PROMETHEUS_GATEWAY")
    GRAFANA_URL: Optional[str] = Field(default=None, env="GRAFANA_URL")
    
    # Feature Flags
    ENABLE_ADVANCED_EXPLANATIONS: bool = Field(default=True, env="ENABLE_ADVANCED_EXPLANATIONS")
    ENABLE_DRIFT_DETECTION: bool = Field(default=True, env="ENABLE_DRIFT_DETECTION")
    ENABLE_REAL_TIME_MONITORING: bool = Field(default=True, env="ENABLE_REAL_TIME_MONITORING")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()

# Ensure required directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.MODEL_STORAGE_DIR, exist_ok=True)