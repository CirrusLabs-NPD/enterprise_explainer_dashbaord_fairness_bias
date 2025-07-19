"""
Input validation and sanitization utilities
"""

import re
import json
from typing import Any, Dict, List, Optional, Union, Type
from pydantic import BaseModel, ValidationError, validator
from fastapi import HTTPException, status, UploadFile
import pandas as pd
import numpy as np
import structlog

from app.core.exceptions import ValidationError as CustomValidationError

logger = structlog.get_logger(__name__)

# File validation
ALLOWED_FILE_TYPES = {
    'model': ['pkl', 'joblib', 'onnx', 'h5', 'pb'],
    'data': ['csv', 'json', 'parquet', 'xlsx', 'xls'],
    'image': ['jpg', 'jpeg', 'png', 'gif', 'bmp'],
    'document': ['pdf', 'txt', 'md']
}

MAX_FILE_SIZES = {
    'model': 500 * 1024 * 1024,  # 500MB
    'data': 100 * 1024 * 1024,   # 100MB
    'image': 10 * 1024 * 1024,   # 10MB
    'document': 50 * 1024 * 1024  # 50MB
}

class FileValidator:
    """File validation utilities"""
    
    @staticmethod
    def validate_file_type(filename: str, allowed_category: str) -> bool:
        """Validate file type against allowed categories"""
        if not filename:
            return False
        
        extension = filename.lower().split('.')[-1]
        allowed_extensions = ALLOWED_FILE_TYPES.get(allowed_category, [])
        return extension in allowed_extensions
    
    @staticmethod
    def validate_file_size(file_size: int, category: str) -> bool:
        """Validate file size"""
        max_size = MAX_FILE_SIZES.get(category, 50 * 1024 * 1024)
        return file_size <= max_size
    
    @staticmethod
    async def validate_upload_file(
        file: UploadFile,
        category: str,
        required_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Comprehensive file validation"""
        
        # Basic file validation
        if not file.filename:
            raise CustomValidationError("Filename is required")
        
        if not FileValidator.validate_file_type(file.filename, category):
            allowed = ALLOWED_FILE_TYPES.get(category, [])
            raise CustomValidationError(
                f"Invalid file type. Allowed types for {category}: {', '.join(allowed)}"
            )
        
        # Read file content for size validation
        content = await file.read()
        await file.seek(0)  # Reset file pointer
        
        if not FileValidator.validate_file_size(len(content), category):
            max_size_mb = MAX_FILE_SIZES.get(category, 50 * 1024 * 1024) / (1024 * 1024)
            raise CustomValidationError(f"File too large. Maximum size: {max_size_mb}MB")
        
        validation_result = {
            "filename": file.filename,
            "size": len(content),
            "content_type": file.content_type,
            "valid": True
        }
        
        # Additional validation for data files
        if category == 'data' and file.filename.endswith('.csv'):
            try:
                # Validate CSV structure
                import io
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                
                validation_result.update({
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "dtypes": df.dtypes.to_dict()
                })
                
                # Check required columns
                if required_columns:
                    missing_columns = set(required_columns) - set(df.columns)
                    if missing_columns:
                        raise CustomValidationError(
                            f"Missing required columns: {', '.join(missing_columns)}"
                        )
                
                # Check for empty dataframe
                if df.empty:
                    raise CustomValidationError("Dataset is empty")
                
            except pd.errors.EmptyDataError:
                raise CustomValidationError("CSV file is empty")
            except pd.errors.ParserError as e:
                raise CustomValidationError(f"Invalid CSV format: {str(e)}")
            except UnicodeDecodeError:
                raise CustomValidationError("File encoding not supported. Please use UTF-8")
        
        return validation_result

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        min_rows: int = 1,
        max_rows: int = 1000000,
        required_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Validate pandas DataFrame"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "memory_usage": df.memory_usage(deep=True).sum()
            }
        }
        
        # Check row count
        if len(df) < min_rows:
            validation_result["errors"].append(f"Dataset has too few rows. Minimum: {min_rows}")
        
        if len(df) > max_rows:
            validation_result["errors"].append(f"Dataset has too many rows. Maximum: {max_rows}")
        
        # Check required columns
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                validation_result["errors"].append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Check numeric columns
        if numeric_columns:
            for col in numeric_columns:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    validation_result["errors"].append(f"Column '{col}' must be numeric")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        columns_with_missing = missing_counts[missing_counts > 0]
        if not columns_with_missing.empty:
            validation_result["warnings"].append({
                "type": "missing_values",
                "columns": columns_with_missing.to_dict()
            })
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_result["warnings"].append({
                "type": "duplicate_rows",
                "count": int(duplicate_count)
            })
        
        # Check for infinite values
        if df.select_dtypes(include=[np.number]).isin([np.inf, -np.inf]).any().any():
            validation_result["warnings"].append({
                "type": "infinite_values",
                "message": "Dataset contains infinite values"
            })
        
        validation_result["valid"] = len(validation_result["errors"]) == 0
        return validation_result

class ModelValidator:
    """Model validation utilities"""
    
    @staticmethod
    def validate_model_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model metadata"""
        
        required_fields = ["name", "model_type", "feature_names"]
        optional_fields = ["description", "version", "framework", "hyperparameters"]
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        for field in required_fields:
            if field not in metadata:
                validation_result["errors"].append(f"Missing required field: {field}")
            elif not metadata[field]:
                validation_result["errors"].append(f"Field '{field}' cannot be empty")
        
        # Validate model_type
        valid_model_types = ["classification", "regression", "clustering", "anomaly_detection"]
        if "model_type" in metadata and metadata["model_type"] not in valid_model_types:
            validation_result["errors"].append(
                f"Invalid model_type. Must be one of: {', '.join(valid_model_types)}"
            )
        
        # Validate feature_names
        if "feature_names" in metadata:
            if not isinstance(metadata["feature_names"], list):
                validation_result["errors"].append("feature_names must be a list")
            elif not metadata["feature_names"]:
                validation_result["errors"].append("feature_names cannot be empty")
        
        # Validate version format
        if "version" in metadata:
            version_pattern = r'^\d+\.\d+\.\d+$'
            if not re.match(version_pattern, str(metadata["version"])):
                validation_result["warnings"].append(
                    "Version should follow semantic versioning (e.g., 1.0.0)"
                )
        
        validation_result["valid"] = len(validation_result["errors"]) == 0
        return validation_result

class InputSanitizer:
    """Input sanitization utilities"""
    
    @staticmethod
    def sanitize_string(
        input_str: str,
        max_length: int = 1000,
        allow_html: bool = False,
        allow_special_chars: bool = True
    ) -> str:
        """Sanitize string input"""
        
        if not input_str:
            return ""
        
        # Remove null bytes
        input_str = input_str.replace('\x00', '')
        
        # Limit length
        if len(input_str) > max_length:
            input_str = input_str[:max_length]
        
        # Remove HTML if not allowed
        if not allow_html:
            html_pattern = re.compile(r'<[^>]+>')
            input_str = html_pattern.sub('', input_str)
        
        # Remove special characters if not allowed
        if not allow_special_chars:
            input_str = re.sub(r'[^\w\s-]', '', input_str)
        
        return input_str.strip()
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename"""
        if not filename:
            return ""
        
        # Remove path separators
        filename = filename.replace('/', '').replace('\\', '')
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"|?*]', '', filename)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            max_name_length = 255 - len(ext) - 1 if ext else 255
            filename = name[:max_name_length] + ('.' + ext if ext else '')
        
        return filename
    
    @staticmethod
    def sanitize_json(json_str: str, max_depth: int = 10) -> Any:
        """Safely parse and validate JSON"""
        try:
            data = json.loads(json_str)
            
            def check_depth(obj, current_depth=0):
                if current_depth > max_depth:
                    raise ValueError("JSON structure too deep")
                
                if isinstance(obj, dict):
                    for value in obj.values():
                        check_depth(value, current_depth + 1)
                elif isinstance(obj, list):
                    for item in obj:
                        check_depth(item, current_depth + 1)
            
            check_depth(data)
            return data
            
        except json.JSONDecodeError as e:
            raise CustomValidationError(f"Invalid JSON format: {str(e)}")
        except ValueError as e:
            raise CustomValidationError(str(e))

# Pydantic models for common validation
class ModelMetadataRequest(BaseModel):
    name: str
    model_type: str
    feature_names: List[str]
    description: Optional[str] = ""
    version: Optional[str] = "1.0.0"
    framework: Optional[str] = "sklearn"
    hyperparameters: Optional[Dict[str, Any]] = {}
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Name cannot be empty')
        return InputSanitizer.sanitize_string(v, max_length=100)
    
    @validator('model_type')
    def validate_model_type(cls, v):
        valid_types = ["classification", "regression", "clustering", "anomaly_detection"]
        if v not in valid_types:
            raise ValueError(f'Model type must be one of: {", ".join(valid_types)}')
        return v
    
    @validator('feature_names')
    def validate_feature_names(cls, v):
        if not v:
            raise ValueError('Feature names cannot be empty')
        
        # Sanitize feature names
        sanitized = [InputSanitizer.sanitize_string(name, max_length=50) for name in v]
        
        # Check for duplicates
        if len(set(sanitized)) != len(sanitized):
            raise ValueError('Feature names must be unique')
        
        return sanitized

class DataUploadRequest(BaseModel):
    dataset_name: str
    description: Optional[str] = ""
    target_column: Optional[str] = None
    
    @validator('dataset_name')
    def validate_dataset_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Dataset name cannot be empty')
        return InputSanitizer.sanitize_string(v, max_length=100)

# Validation decorators
def validate_request_size(max_size_mb: int = 50):
    """Decorator to validate request size"""
    def decorator(func):
        async def wrapper(request, *args, **kwargs):
            if hasattr(request, 'headers'):
                content_length = request.headers.get('content-length')
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    if size_mb > max_size_mb:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"Request too large. Maximum size: {max_size_mb}MB"
                        )
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

def validate_json_payload(max_size_kb: int = 1024):
    """Decorator to validate JSON payload size"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for arg in args:
                if isinstance(arg, (dict, list)):
                    json_str = json.dumps(arg)
                    size_kb = len(json_str.encode('utf-8')) / 1024
                    if size_kb > max_size_kb:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"JSON payload too large. Maximum size: {max_size_kb}KB"
                        )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator