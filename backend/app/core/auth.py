"""
Authentication and Authorization system
"""

import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from app.config import settings

logger = structlog.get_logger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token handler
security = HTTPBearer()

class AuthHandler:
    """Handle authentication operations"""
    
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = "HS256"
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
    
    def get_password_hash(self, password: str) -> str:
        """Hash password"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=30)  # 30 days for refresh token
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

# Global auth handler instance
auth_handler = AuthHandler()

# User roles and permissions
class UserRole:
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    
class Permission:
    # Model permissions
    MODEL_CREATE = "model:create"
    MODEL_READ = "model:read"
    MODEL_UPDATE = "model:update"
    MODEL_DELETE = "model:delete"
    
    # Data permissions
    DATA_UPLOAD = "data:upload"
    DATA_READ = "data:read"
    DATA_DELETE = "data:delete"
    
    # Analysis permissions
    ANALYSIS_RUN = "analysis:run"
    ANALYSIS_READ = "analysis:read"
    
    # Admin permissions
    USER_MANAGE = "user:manage"
    SYSTEM_CONFIG = "system:config"

# Role-based permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.MODEL_CREATE, Permission.MODEL_READ, Permission.MODEL_UPDATE, Permission.MODEL_DELETE,
        Permission.DATA_UPLOAD, Permission.DATA_READ, Permission.DATA_DELETE,
        Permission.ANALYSIS_RUN, Permission.ANALYSIS_READ,
        Permission.USER_MANAGE, Permission.SYSTEM_CONFIG
    ],
    UserRole.ANALYST: [
        Permission.MODEL_CREATE, Permission.MODEL_READ, Permission.MODEL_UPDATE,
        Permission.DATA_UPLOAD, Permission.DATA_READ,
        Permission.ANALYSIS_RUN, Permission.ANALYSIS_READ
    ],
    UserRole.VIEWER: [
        Permission.MODEL_READ,
        Permission.DATA_READ,
        Permission.ANALYSIS_READ
    ]
}

class User:
    """User model"""
    
    def __init__(self, user_id: str, username: str, email: str, role: str, is_active: bool = True):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.role = role
        self.is_active = is_active
        self.permissions = ROLE_PERMISSIONS.get(role, [])
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        return permission in self.permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "permissions": self.permissions
        }

# Mock user database (replace with real database in production)
MOCK_USERS_DB = {
    "admin": {
        "user_id": "1",
        "username": "admin",
        "email": "admin@example.com",
        "hashed_password": auth_handler.get_password_hash("admin123"),
        "role": UserRole.ADMIN,
        "is_active": True
    },
    "analyst": {
        "user_id": "2",
        "username": "analyst",
        "email": "analyst@example.com",
        "hashed_password": auth_handler.get_password_hash("analyst123"),
        "role": UserRole.ANALYST,
        "is_active": True
    },
    "viewer": {
        "user_id": "3",
        "username": "viewer",
        "email": "viewer@example.com",
        "hashed_password": auth_handler.get_password_hash("viewer123"),
        "role": UserRole.VIEWER,
        "is_active": True
    }
}

def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate user with username and password"""
    user_data = MOCK_USERS_DB.get(username)
    if not user_data:
        return None
    
    if not auth_handler.verify_password(password, user_data["hashed_password"]):
        return None
    
    if not user_data["is_active"]:
        return None
    
    return User(
        user_id=user_data["user_id"],
        username=user_data["username"],
        email=user_data["email"],
        role=user_data["role"],
        is_active=user_data["is_active"]
    )

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user"""
    token = credentials.credentials
    payload = auth_handler.decode_token(token)
    
    username = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    user_data = MOCK_USERS_DB.get(username)
    if user_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    if not user_data["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User is inactive"
        )
    
    return User(
        user_id=user_data["user_id"],
        username=user_data["username"],
        email=user_data["email"],
        role=user_data["role"],
        is_active=user_data["is_active"]
    )

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if not current_user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission}"
            )
        return current_user
    return permission_checker

def require_role(role: str):
    """Decorator to require specific role"""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {role}"
            )
        return current_user
    return role_checker

# Convenience decorators for common permissions
require_admin = require_role(UserRole.ADMIN)
require_analyst_or_admin = lambda: require_permission(Permission.ANALYSIS_RUN)
require_model_read = lambda: require_permission(Permission.MODEL_READ)
require_model_write = lambda: require_permission(Permission.MODEL_CREATE)