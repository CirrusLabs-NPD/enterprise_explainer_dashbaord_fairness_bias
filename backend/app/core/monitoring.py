"""
Monitoring and metrics collection
"""

import time
import psutil
from typing import Dict, Any, Optional
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
from prometheus_client.core import CollectorRegistry
import structlog

logger = structlog.get_logger(__name__)

# Create custom registry
REGISTRY = CollectorRegistry()

# Application metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=REGISTRY
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=REGISTRY
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections',
    registry=REGISTRY
)

# ML-specific metrics
MODEL_PREDICTIONS = Counter(
    'ml_model_predictions_total',
    'Total number of model predictions',
    ['model_id', 'model_type'],
    registry=REGISTRY
)

EXPLANATION_REQUESTS = Counter(
    'ml_explanation_requests_total',
    'Total number of explanation requests',
    ['method', 'model_id'],
    registry=REGISTRY
)

EXPLANATION_DURATION = Histogram(
    'ml_explanation_duration_seconds',
    'Time taken to generate explanations',
    ['method', 'model_id'],
    registry=REGISTRY
)

DRIFT_DETECTIONS = Counter(
    'ml_drift_detections_total',
    'Total number of drift detections',
    ['model_id', 'drift_type'],
    registry=REGISTRY
)

MODEL_ACCURACY = Gauge(
    'ml_model_accuracy',
    'Current model accuracy',
    ['model_id'],
    registry=REGISTRY
)

# System metrics
CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'CPU usage percentage',
    registry=REGISTRY
)

MEMORY_USAGE = Gauge(
    'system_memory_usage_percent',
    'Memory usage percentage',
    registry=REGISTRY
)

DISK_USAGE = Gauge(
    'system_disk_usage_percent',
    'Disk usage percentage',
    ['device'],
    registry=REGISTRY
)

# Database metrics
DB_CONNECTIONS = Gauge(
    'database_connections_active',
    'Number of active database connections',
    registry=REGISTRY
)

DB_QUERY_DURATION = Histogram(
    'database_query_duration_seconds',
    'Database query duration',
    ['operation'],
    registry=REGISTRY
)

# Cache metrics
CACHE_HITS = Counter(
    'cache_hits_total',
    'Total number of cache hits',
    ['cache_type'],
    registry=REGISTRY
)

CACHE_MISSES = Counter(
    'cache_misses_total',
    'Total number of cache misses',
    ['cache_type'],
    registry=REGISTRY
)

# Error metrics
ERROR_COUNT = Counter(
    'errors_total',
    'Total number of errors',
    ['error_type', 'endpoint'],
    registry=REGISTRY
)

# Rate limiting metrics
RATE_LIMIT_HITS = Counter(
    'rate_limit_hits_total',
    'Total number of rate limit hits',
    ['identifier_type'],
    registry=REGISTRY
)

class MetricsCollector:
    """Collect and update system metrics"""
    
    def __init__(self):
        self.last_update = 0
        self.update_interval = 30  # seconds
    
    def collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            CPU_USAGE.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            MEMORY_USAGE.set(memory.percent)
            
            # Disk usage
            for partition in psutil.disk_partitions():
                try:
                    disk_usage = psutil.disk_usage(partition.mountpoint)
                    DISK_USAGE.labels(device=partition.device).set(
                        (disk_usage.used / disk_usage.total) * 100
                    )
                except PermissionError:
                    # Skip partitions we can't access
                    continue
            
            logger.debug(
                "System metrics collected",
                cpu_percent=cpu_percent,
                memory_percent=memory.percent
            )
            
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
    
    def should_update(self) -> bool:
        """Check if metrics should be updated"""
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.last_update = current_time
            return True
        return False
    
    def update_metrics(self):
        """Update all metrics"""
        if self.should_update():
            self.collect_system_metrics()

# Global metrics collector
metrics_collector = MetricsCollector()

# Decorator for monitoring function execution
def monitor_execution(metric_name: str = None, labels: Dict[str, str] = None):
    """Decorator to monitor function execution time and count"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = metric_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = await func(*args, **kwargs)
                
                # Record success
                REQUEST_COUNT.labels(
                    method='async',
                    endpoint=function_name,
                    status='success'
                ).inc()
                
                return result
                
            except Exception as e:
                # Record error
                REQUEST_COUNT.labels(
                    method='async',
                    endpoint=function_name,
                    status='error'
                ).inc()
                
                ERROR_COUNT.labels(
                    error_type=type(e).__name__,
                    endpoint=function_name
                ).inc()
                
                raise
            
            finally:
                # Record duration
                duration = time.time() - start_time
                REQUEST_DURATION.labels(
                    method='async',
                    endpoint=function_name
                ).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = metric_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                
                # Record success
                REQUEST_COUNT.labels(
                    method='sync',
                    endpoint=function_name,
                    status='success'
                ).inc()
                
                return result
                
            except Exception as e:
                # Record error
                REQUEST_COUNT.labels(
                    method='sync',
                    endpoint=function_name,
                    status='error'
                ).inc()
                
                ERROR_COUNT.labels(
                    error_type=type(e).__name__,
                    endpoint=function_name
                ).inc()
                
                raise
            
            finally:
                # Record duration
                duration = time.time() - start_time
                REQUEST_DURATION.labels(
                    method='sync',
                    endpoint=function_name
                ).observe(duration)
        
        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# ML-specific monitoring functions
def record_prediction(model_id: str, model_type: str, duration: float = None):
    """Record a model prediction"""
    MODEL_PREDICTIONS.labels(model_id=model_id, model_type=model_type).inc()
    
    if duration is not None:
        # You could add a prediction duration histogram here
        logger.debug("Prediction recorded", model_id=model_id, duration=duration)

def record_explanation(method: str, model_id: str, duration: float):
    """Record an explanation generation"""
    EXPLANATION_REQUESTS.labels(method=method, model_id=model_id).inc()
    EXPLANATION_DURATION.labels(method=method, model_id=model_id).observe(duration)

def record_drift_detection(model_id: str, drift_type: str, drift_detected: bool):
    """Record drift detection"""
    DRIFT_DETECTIONS.labels(model_id=model_id, drift_type=drift_type).inc()
    
    logger.info(
        "Drift detection recorded",
        model_id=model_id,
        drift_type=drift_type,
        drift_detected=drift_detected
    )

def update_model_accuracy(model_id: str, accuracy: float):
    """Update model accuracy metric"""
    MODEL_ACCURACY.labels(model_id=model_id).set(accuracy)

# Database monitoring
def record_db_operation(operation: str, duration: float):
    """Record database operation"""
    DB_QUERY_DURATION.labels(operation=operation).observe(duration)

def update_db_connections(count: int):
    """Update database connection count"""
    DB_CONNECTIONS.set(count)

# Cache monitoring
def record_cache_hit(cache_type: str = "default"):
    """Record cache hit"""
    CACHE_HITS.labels(cache_type=cache_type).inc()

def record_cache_miss(cache_type: str = "default"):
    """Record cache miss"""
    CACHE_MISSES.labels(cache_type=cache_type).inc()

# Rate limiting monitoring
def record_rate_limit_hit(identifier_type: str):
    """Record rate limit hit"""
    RATE_LIMIT_HITS.labels(identifier_type=identifier_type).inc()

# Health check function
def get_health_metrics() -> Dict[str, Any]:
    """Get current health metrics"""
    try:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        return {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_mb": memory.available / (1024 * 1024),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error("Failed to get health metrics", error=str(e))
        return {"error": str(e)}

# Metrics endpoint
def get_metrics() -> str:
    """Get Prometheus metrics"""
    # Update system metrics before generating output
    metrics_collector.update_metrics()
    
    return generate_latest(REGISTRY).decode('utf-8')

# Application info
APP_INFO = Info(
    'ml_explainer_app_info',
    'Information about the ML Explainer Dashboard',
    registry=REGISTRY
)

def set_app_info(version: str, environment: str):
    """Set application information"""
    APP_INFO.info({
        'version': version,
        'environment': environment,
        'build_time': time.strftime('%Y-%m-%d %H:%M:%S')
    })