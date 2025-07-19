"""
Rate limiting and request throttling
"""

import time
import asyncio
from typing import Dict, Optional, Callable, Any
from collections import defaultdict, deque
from functools import wraps
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
import structlog

logger = structlog.get_logger(__name__)

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        burst_size: Optional[int] = None
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.burst_size = burst_size or max_requests
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.tokens: Dict[str, float] = defaultdict(lambda: self.burst_size)
        self.last_update: Dict[str, float] = defaultdict(time.time)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, identifier: str) -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed and return rate limit info"""
        async with self._lock:
            now = time.time()
            
            # Token bucket algorithm
            time_passed = now - self.last_update[identifier]
            self.last_update[identifier] = now
            
            # Add tokens based on time passed
            tokens_to_add = time_passed * (self.max_requests / self.window_seconds)
            self.tokens[identifier] = min(
                self.burst_size,
                self.tokens[identifier] + tokens_to_add
            )
            
            # Check if request can be processed
            if self.tokens[identifier] >= 1:
                self.tokens[identifier] -= 1
                allowed = True
            else:
                allowed = False
            
            # Calculate when next request will be allowed
            next_reset = now + (1 - self.tokens[identifier]) / (self.max_requests / self.window_seconds)
            
            # Rate limit info
            info = {
                "allowed": allowed,
                "limit": self.max_requests,
                "remaining": int(self.tokens[identifier]),
                "reset_time": int(next_reset),
                "retry_after": max(0, int(next_reset - now)) if not allowed else 0
            }
            
            return allowed, info

class SlidingWindowRateLimiter:
    """Sliding window rate limiter"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, identifier: str) -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed"""
        async with self._lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Remove old requests outside the window
            request_times = self.requests[identifier]
            while request_times and request_times[0] <= window_start:
                request_times.popleft()
            
            # Check if we can add this request
            allowed = len(request_times) < self.max_requests
            
            if allowed:
                request_times.append(now)
            
            # Calculate reset time (when oldest request will expire)
            reset_time = int(request_times[0] + self.window_seconds) if request_times else int(now)
            
            info = {
                "allowed": allowed,
                "limit": self.max_requests,
                "remaining": max(0, self.max_requests - len(request_times)),
                "reset_time": reset_time,
                "retry_after": max(0, reset_time - int(now)) if not allowed else 0
            }
            
            return allowed, info

class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system load"""
    
    def __init__(
        self,
        base_limit: int = 100,
        window_seconds: int = 60,
        load_threshold: float = 0.8
    ):
        self.base_limit = base_limit
        self.window_seconds = window_seconds
        self.load_threshold = load_threshold
        self.current_limit = base_limit
        self.rate_limiter = SlidingWindowRateLimiter(base_limit, window_seconds)
        self.last_adjustment = time.time()
    
    async def _adjust_limit(self):
        """Adjust rate limit based on system load"""
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # Calculate load factor
            load_factor = max(cpu_percent, memory_percent) / 100
            
            # Adjust limit based on load
            if load_factor > self.load_threshold:
                # Reduce limit when system is under load
                adjustment_factor = 1 - (load_factor - self.load_threshold)
                new_limit = int(self.base_limit * adjustment_factor)
            else:
                # Gradually increase limit when system is healthy
                new_limit = min(self.base_limit, self.current_limit + 5)
            
            if new_limit != self.current_limit:
                self.current_limit = new_limit
                self.rate_limiter = SlidingWindowRateLimiter(new_limit, self.window_seconds)
                logger.info("Rate limit adjusted", new_limit=new_limit, load_factor=load_factor)
        
        except ImportError:
            # psutil not available, use base limit
            pass
        except Exception as e:
            logger.warning("Failed to adjust rate limit", error=str(e))
    
    async def is_allowed(self, identifier: str) -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed with adaptive limiting"""
        now = time.time()
        
        # Adjust limit every 30 seconds
        if now - self.last_adjustment > 30:
            await self._adjust_limit()
            self.last_adjustment = now
        
        return await self.rate_limiter.is_allowed(identifier)

# Rate limiter instances
default_limiter = RateLimiter(max_requests=100, window_seconds=60)
strict_limiter = RateLimiter(max_requests=20, window_seconds=60)
upload_limiter = RateLimiter(max_requests=5, window_seconds=60)
auth_limiter = RateLimiter(max_requests=10, window_seconds=300)  # 10 attempts per 5 minutes
adaptive_limiter = AdaptiveRateLimiter(base_limit=100, window_seconds=60)

async def get_client_identifier(request: Request) -> str:
    """Get client identifier for rate limiting"""
    # Try to get user ID from authentication
    auth_header = request.headers.get("authorization")
    if auth_header:
        try:
            from app.core.auth import auth_handler
            token = auth_header.replace("Bearer ", "")
            payload = auth_handler.decode_token(token)
            return f"user:{payload.get('user_id', 'unknown')}"
        except Exception:
            pass
    
    # Fall back to IP address
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return f"ip:{forwarded_for.split(',')[0].strip()}"
    
    return f"ip:{request.client.host}"

def rate_limit(
    limiter: RateLimiter = default_limiter,
    identifier_func: Callable[[Request], str] = get_client_identifier
):
    """Rate limiting decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find request object in arguments
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                # No request object found, allow the request
                return await func(*args, **kwargs)
            
            # Get client identifier
            identifier = await identifier_func(request)
            
            # Check rate limit
            allowed, info = await limiter.is_allowed(identifier)
            
            if not allowed:
                logger.warning(
                    "Rate limit exceeded",
                    identifier=identifier,
                    limit=info["limit"],
                    retry_after=info["retry_after"]
                )
                
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={
                        "X-RateLimit-Limit": str(info["limit"]),
                        "X-RateLimit-Remaining": str(info["remaining"]),
                        "X-RateLimit-Reset": str(info["reset_time"]),
                        "Retry-After": str(info["retry_after"])
                    }
                )
            
            # Add rate limit headers to response
            response = await func(*args, **kwargs)
            
            if hasattr(response, 'headers'):
                response.headers["X-RateLimit-Limit"] = str(info["limit"])
                response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
                response.headers["X-RateLimit-Reset"] = str(info["reset_time"])
            
            return response
        
        return wrapper
    return decorator

def rate_limit_middleware(
    limiter: RateLimiter = default_limiter,
    exempt_paths: list = None
):
    """Rate limiting middleware"""
    exempt_paths = exempt_paths or ["/health", "/metrics", "/docs", "/redoc"]
    
    async def middleware(request: Request, call_next):
        # Skip rate limiting for exempt paths
        if any(request.url.path.startswith(path) for path in exempt_paths):
            return await call_next(request)
        
        # Get client identifier
        identifier = await get_client_identifier(request)
        
        # Check rate limit
        allowed, info = await limiter.is_allowed(identifier)
        
        if not allowed:
            logger.warning(
                "Rate limit exceeded",
                identifier=identifier,
                path=request.url.path,
                method=request.method,
                limit=info["limit"],
                retry_after=info["retry_after"]
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Rate limit exceeded",
                        "details": {
                            "limit": info["limit"],
                            "retry_after": info["retry_after"]
                        }
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(info["limit"]),
                    "X-RateLimit-Remaining": str(info["remaining"]),
                    "X-RateLimit-Reset": str(info["reset_time"]),
                    "Retry-After": str(info["retry_after"])
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(info["reset_time"])
        
        return response
    
    return middleware

# Decorator shortcuts
rate_limit_default = rate_limit(default_limiter)
rate_limit_strict = rate_limit(strict_limiter)
rate_limit_upload = rate_limit(upload_limiter)
rate_limit_auth = rate_limit(auth_limiter)
rate_limit_adaptive = rate_limit(adaptive_limiter)