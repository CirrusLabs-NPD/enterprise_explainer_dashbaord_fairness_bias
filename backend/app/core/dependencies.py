"""
Dependency injection for FastAPI
"""

from functools import lru_cache
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import settings
from app.core.worker_pool import WorkerPool
from app.core.websocket_manager import WebSocketManager, websocket_manager
from app.services.model_service import ModelService
from app.services.explanation_service import ExplanationService
from app.services.monitoring_service import MonitoringService
from app.services.data_service import DataService


# Security scheme
security = HTTPBearer()

# Global instances
_worker_pool: Optional[WorkerPool] = None
_model_service: Optional[ModelService] = None
_explanation_service: Optional[ExplanationService] = None
_monitoring_service: Optional[MonitoringService] = None
_data_service: Optional[DataService] = None


async def get_worker_pool() -> WorkerPool:
    """Get or create worker pool instance"""
    global _worker_pool
    if _worker_pool is None:
        _worker_pool = WorkerPool(
            max_workers=settings.MAX_WORKERS,
            max_cpu_workers=settings.MAX_CPU_WORKERS,
            max_io_workers=settings.MAX_IO_WORKERS
        )
        await _worker_pool.start()
    return _worker_pool


async def get_websocket_manager() -> WebSocketManager:
    """Get WebSocket manager instance"""
    return websocket_manager


async def get_model_service(
    worker_pool: WorkerPool = Depends(get_worker_pool)
) -> ModelService:
    """Get or create model service instance"""
    global _model_service
    if _model_service is None:
        _model_service = ModelService(worker_pool)
    return _model_service


async def get_explanation_service(
    worker_pool: WorkerPool = Depends(get_worker_pool),
    model_service: ModelService = Depends(get_model_service)
) -> ExplanationService:
    """Get or create explanation service instance"""
    global _explanation_service
    if _explanation_service is None:
        _explanation_service = ExplanationService(worker_pool, model_service)
    return _explanation_service


async def get_monitoring_service(
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
) -> MonitoringService:
    """Get or create monitoring service instance"""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService(websocket_manager)
        await _monitoring_service.start_drift_monitoring()
    return _monitoring_service


async def get_data_service(
    worker_pool: WorkerPool = Depends(get_worker_pool)
) -> DataService:
    """Get or create data service instance"""
    global _data_service
    if _data_service is None:
        _data_service = DataService(worker_pool)
    return _data_service


# Authentication dependencies
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Get current user from token
    This is a simplified implementation - replace with proper JWT validation
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    # In a real implementation, you would:
    # 1. Validate the JWT token
    # 2. Extract user information
    # 3. Check token expiration
    # 4. Verify token signature
    
    # For now, return a mock user
    if token == "test_token":
        return "test_user"
    
    # For development, allow any token
    if settings.DEBUG:
        return "dev_user"
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid token",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_active_user(
    current_user: str = Depends(get_current_user)
) -> str:
    """Get current active user"""
    # In a real implementation, you would check if the user is active
    return current_user


# Optional dependencies (for endpoints that don't require authentication)
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[str]:
    """Get current user if authenticated, otherwise return None"""
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


# Rate limiting dependency
class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    async def __call__(self, current_user: str = Depends(get_current_user)):
        """Rate limit requests per user"""
        import time
        
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        if current_user in self.requests:
            self.requests[current_user] = [
                req_time for req_time in self.requests[current_user]
                if req_time > window_start
            ]
        else:
            self.requests[current_user] = []
        
        # Check rate limit
        if len(self.requests[current_user]) >= self.max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Add current request
        self.requests[current_user].append(now)
        
        return current_user


# Create rate limiter instances
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
strict_rate_limiter = RateLimiter(max_requests=10, window_seconds=60)


# Cleanup function
async def cleanup_dependencies():
    """Cleanup all service instances"""
    global _worker_pool, _model_service, _explanation_service, _monitoring_service, _data_service
    
    if _monitoring_service:
        await _monitoring_service.stop()
    
    if _worker_pool:
        await _worker_pool.stop()
    
    await websocket_manager.stop()
    
    # Reset global instances
    _worker_pool = None
    _model_service = None
    _explanation_service = None
    _monitoring_service = None
    _data_service = None


# Health check dependencies
async def health_check() -> dict:
    """Health check for all services"""
    status = {
        "worker_pool": False,
        "websocket_manager": False,
        "services": {
            "model_service": False,
            "explanation_service": False,
            "monitoring_service": False,
            "data_service": False
        }
    }
    
    try:
        # Check worker pool
        if _worker_pool:
            worker_stats = await _worker_pool.get_stats()
            status["worker_pool"] = worker_stats["status"] == "running"
        
        # Check WebSocket manager
        status["websocket_manager"] = websocket_manager.running
        
        # Check services
        status["services"]["model_service"] = _model_service is not None
        status["services"]["explanation_service"] = _explanation_service is not None
        status["services"]["monitoring_service"] = _monitoring_service is not None
        status["services"]["data_service"] = _data_service is not None
        
    except Exception as e:
        status["error"] = str(e)
    
    return status


# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance"""
    import time
    import functools
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Log performance metrics
            import structlog
            logger = structlog.get_logger(__name__)
            logger.info(
                f"Function {func.__name__} executed",
                execution_time_ms=execution_time,
                function_name=func.__name__
            )
            
            return result
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            import structlog
            logger = structlog.get_logger(__name__)
            logger.error(
                f"Function {func.__name__} failed",
                execution_time_ms=execution_time,
                function_name=func.__name__,
                error=str(e)
            )
            raise
    
    return wrapper


# Service factory functions
@lru_cache()
def get_settings():
    """Get application settings (cached)"""
    return settings


# Validation dependencies
async def validate_model_id(model_id: str) -> str:
    """Validate model ID format"""
    if not model_id or len(model_id) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid model ID"
        )
    return model_id


async def validate_dataset_id(dataset_id: str) -> str:
    """Validate dataset ID format"""
    if not dataset_id or len(dataset_id) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID"
        )
    return dataset_id