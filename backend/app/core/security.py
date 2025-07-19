"""
Security utilities for authentication and authorization
"""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext

from app.config import settings


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()


class SecurityManager:
    """Security manager for authentication and authorization"""
    
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = settings.REFRESH_TOKEN_EXPIRE_DAYS
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire, "type": "access"})
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_api_key(self, user_id: str, permissions: List[str] = None) -> str:
        """Create API key for user"""
        data = {
            "user_id": user_id,
            "permissions": permissions or [],
            "created_at": datetime.utcnow().isoformat(),
            "type": "api_key"
        }
        # API keys don't expire
        return jwt.encode(data, self.secret_key, algorithm=self.algorithm)
    
    def verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """Verify API key"""
        try:
            payload = jwt.decode(api_key, self.secret_key, algorithms=[self.algorithm])
            if payload.get("type") != "api_key":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key",
                )
            return payload
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )


# Global security manager instance
security_manager = SecurityManager()


# Authentication functions
async def get_current_user(credentials: HTTPAuthorizationCredentials) -> str:
    """Get current user from JWT token"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    # Check if it's an API key (no expiration)
    try:
        payload = security_manager.verify_api_key(token)
        return payload["user_id"]
    except HTTPException:
        pass
    
    # Check if it's a regular JWT token
    try:
        payload = security_manager.verify_token(token)
        if payload.get("type") not in ["access", "refresh"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )
        return payload["user_id"]
    except HTTPException:
        pass
    
    # For development, allow test tokens
    if settings.DEBUG:
        if token == "test_token":
            return "test_user"
        if token.startswith("dev_"):
            return token.replace("dev_", "")
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid token",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = None
) -> Optional[str]:
    """Get current user if authenticated, otherwise return None"""
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


# Permission checking
class PermissionChecker:
    """Permission checker for role-based access control"""
    
    def __init__(self, required_permissions: List[str]):
        self.required_permissions = required_permissions
    
    async def __call__(self, current_user: str = None) -> str:
        """Check if user has required permissions"""
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )
        
        # In a real implementation, you would:
        # 1. Fetch user permissions from database
        # 2. Check if user has required permissions
        # 3. Handle role-based access control
        
        # For now, allow all authenticated users
        return current_user


# Role-based access control
class RoleChecker:
    """Role-based access control checker"""
    
    def __init__(self, required_roles: List[str]):
        self.required_roles = required_roles
    
    async def __call__(self, current_user: str = None) -> str:
        """Check if user has required roles"""
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )
        
        # In a real implementation, you would:
        # 1. Fetch user roles from database
        # 2. Check if user has required roles
        # 3. Handle hierarchical roles
        
        # For now, allow all authenticated users
        return current_user


# Security decorators
def require_permissions(permissions: List[str]):
    """Decorator to require specific permissions"""
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # In a real implementation, check permissions here
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_roles(roles: List[str]):
    """Decorator to require specific roles"""
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # In a real implementation, check roles here
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# CORS security
def get_cors_origins():
    """Get CORS origins from settings"""
    if settings.DEBUG:
        return ["*"]
    
    return settings.CORS_ORIGINS


# Rate limiting
class RateLimiter:
    """Rate limiter for API endpoints"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    async def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limit"""
        import time
        
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > window_start
            ]
        else:
            self.requests[identifier] = []
        
        # Check rate limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True
    
    async def __call__(self, current_user: str = None) -> str:
        """Rate limit requests per user"""
        identifier = current_user or "anonymous"
        
        if not await self.check_rate_limit(identifier):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(self.window_seconds)}
            )
        
        return current_user


# Input validation and sanitization
def sanitize_input(input_str: str) -> str:
    """Sanitize input string"""
    if not input_str:
        return ""
    
    # Remove potentially dangerous characters
    dangerous_chars = ["<", ">", "&", "\"", "'", "/", "\\"]
    for char in dangerous_chars:
        input_str = input_str.replace(char, "")
    
    # Limit length
    return input_str[:1000]


def validate_file_type(filename: str, allowed_types: List[str]) -> bool:
    """Validate file type"""
    if not filename:
        return False
    
    extension = filename.lower().split('.')[-1]
    return extension in allowed_types


def validate_file_size(file_size: int, max_size_mb: int = 100) -> bool:
    """Validate file size"""
    max_size_bytes = max_size_mb * 1024 * 1024
    return file_size <= max_size_bytes


# Security headers
def get_security_headers() -> Dict[str, str]:
    """Get security headers"""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }


# Create common rate limiters
default_rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
strict_rate_limiter = RateLimiter(max_requests=10, window_seconds=60)
upload_rate_limiter = RateLimiter(max_requests=5, window_seconds=60)


# Common permission checkers
admin_permission = PermissionChecker(["admin"])
model_permission = PermissionChecker(["model:read", "model:write"])
data_permission = PermissionChecker(["data:read", "data:write"])


# Common role checkers
admin_role = RoleChecker(["admin"])
user_role = RoleChecker(["user", "admin"])
readonly_role = RoleChecker(["readonly", "user", "admin"])