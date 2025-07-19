"""
Custom exceptions and error handling
"""

from typing import Any, Dict, Optional
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import structlog

logger = structlog.get_logger(__name__)

class MLExplainerException(Exception):
    """Base exception for ML Explainer Dashboard"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "GENERAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)

class ValidationError(MLExplainerException):
    """Data validation error"""
    
    def __init__(self, message: str, field: str = None, details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={"field": field, **(details or {})},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )

class AuthenticationError(MLExplainerException):
    """Authentication error"""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=status.HTTP_401_UNAUTHORIZED
        )

class AuthorizationError(MLExplainerException):
    """Authorization error"""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=status.HTTP_403_FORBIDDEN
        )

class ModelError(MLExplainerException):
    """Model-related error"""
    
    def __init__(self, message: str, model_id: str = None, details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            error_code="MODEL_ERROR",
            details={"model_id": model_id, **(details or {})},
            status_code=status.HTTP_400_BAD_REQUEST
        )

class ModelNotFoundError(MLExplainerException):
    """Model not found error"""
    
    def __init__(self, model_id: str):
        super().__init__(
            message=f"Model with ID '{model_id}' not found",
            error_code="MODEL_NOT_FOUND",
            details={"model_id": model_id},
            status_code=status.HTTP_404_NOT_FOUND
        )

class DataError(MLExplainerException):
    """Data-related error"""
    
    def __init__(self, message: str, dataset_id: str = None, details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            error_code="DATA_ERROR",
            details={"dataset_id": dataset_id, **(details or {})},
            status_code=status.HTTP_400_BAD_REQUEST
        )

class DataNotFoundError(MLExplainerException):
    """Data not found error"""
    
    def __init__(self, dataset_id: str):
        super().__init__(
            message=f"Dataset with ID '{dataset_id}' not found",
            error_code="DATA_NOT_FOUND",
            details={"dataset_id": dataset_id},
            status_code=status.HTTP_404_NOT_FOUND
        )

class ExplanationError(MLExplainerException):
    """Explanation generation error"""
    
    def __init__(self, message: str, method: str = None, details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            error_code="EXPLANATION_ERROR",
            details={"method": method, **(details or {})},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

class DriftDetectionError(MLExplainerException):
    """Drift detection error"""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            error_code="DRIFT_DETECTION_ERROR",
            details=details or {},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

class FileUploadError(MLExplainerException):
    """File upload error"""
    
    def __init__(self, message: str, filename: str = None, details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            error_code="FILE_UPLOAD_ERROR",
            details={"filename": filename, **(details or {})},
            status_code=status.HTTP_400_BAD_REQUEST
        )

class ConfigurationError(MLExplainerException):
    """Configuration error"""
    
    def __init__(self, message: str, config_key: str = None):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

class ExternalServiceError(MLExplainerException):
    """External service error"""
    
    def __init__(self, message: str, service: str = None, details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            details={"service": service, **(details or {})},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

class RateLimitError(MLExplainerException):
    """Rate limit exceeded error"""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            details={"retry_after": retry_after},
            status_code=status.HTTP_429_TOO_MANY_REQUESTS
        )

# Error handlers
async def mlexplainer_exception_handler(request: Request, exc: MLExplainerException) -> JSONResponse:
    """Handle custom ML Explainer exceptions"""
    logger.error(
        "ML Explainer exception",
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
                "timestamp": "2024-01-01T00:00:00Z",  # In production, use actual timestamp
                "path": str(request.url.path)
            }
        },
        headers={"X-Error-Code": exc.error_code} if exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS else {}
    )

async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle HTTP exceptions"""
    logger.warning(
        "HTTP exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method
    )
    
    error_codes = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED", 
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        422: "UNPROCESSABLE_ENTITY",
        500: "INTERNAL_SERVER_ERROR",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE"
    }
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": error_codes.get(exc.status_code, "HTTP_ERROR"),
                "message": exc.detail,
                "timestamp": "2024-01-01T00:00:00Z",
                "path": str(request.url.path)
            }
        }
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle validation exceptions"""
    logger.warning(
        "Validation exception",
        errors=exc.errors(),
        path=request.url.path,
        method=request.method
    )
    
    # Format validation errors
    formatted_errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(x) for x in error["loc"])
        formatted_errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input")
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {
                    "validation_errors": formatted_errors
                },
                "timestamp": "2024-01-01T00:00:00Z",
                "path": str(request.url.path)
            }
        }
    )

async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions"""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path,
        method=request.method,
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "timestamp": "2024-01-01T00:00:00Z",
                "path": str(request.url.path)
            }
        }
    )

# Error response helpers
def create_error_response(
    error_code: str,
    message: str,
    status_code: int = status.HTTP_400_BAD_REQUEST,
    details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """Create standardized error response"""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": error_code,
                "message": message,
                "details": details or {},
                "timestamp": "2024-01-01T00:00:00Z"
            }
        }
    )

def handle_model_error(model_id: str, operation: str, error: Exception) -> MLExplainerException:
    """Handle model-related errors with context"""
    if "not found" in str(error).lower():
        return ModelNotFoundError(model_id)
    elif "permission" in str(error).lower() or "unauthorized" in str(error).lower():
        return AuthorizationError(f"Not authorized to {operation} model '{model_id}'")
    elif "validation" in str(error).lower():
        return ValidationError(f"Invalid data for model {operation}", details={"model_id": model_id})
    else:
        return ModelError(f"Failed to {operation} model: {str(error)}", model_id)

def handle_data_error(dataset_id: str, operation: str, error: Exception) -> MLExplainerException:
    """Handle data-related errors with context"""
    if "not found" in str(error).lower():
        return DataNotFoundError(dataset_id)
    elif "permission" in str(error).lower() or "unauthorized" in str(error).lower():
        return AuthorizationError(f"Not authorized to {operation} dataset '{dataset_id}'")
    elif "validation" in str(error).lower():
        return ValidationError(f"Invalid data for dataset {operation}", details={"dataset_id": dataset_id})
    else:
        return DataError(f"Failed to {operation} dataset: {str(error)}", dataset_id)

# Context manager for error handling
class ErrorContext:
    """Context manager for structured error handling"""
    
    def __init__(self, operation: str, resource_id: str = None, resource_type: str = None):
        self.operation = operation
        self.resource_id = resource_id
        self.resource_type = resource_type
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(
                "Operation failed",
                operation=self.operation,
                resource_type=self.resource_type,
                resource_id=self.resource_id,
                error=str(exc_val),
                error_type=exc_type.__name__
            )
        return False  # Don't suppress the exception