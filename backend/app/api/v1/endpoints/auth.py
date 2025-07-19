"""
Authentication endpoints
"""

from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import structlog

from app.core.auth import (
    auth_handler, authenticate_user, get_current_user, 
    User, security, UserRole
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Request/Response models
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    role: str
    is_active: bool
    permissions: list

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest) -> TokenResponse:
    """Authenticate user and return JWT tokens"""
    try:
        # Authenticate user
        user = authenticate_user(request.username, request.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Create tokens
        token_data = {"sub": user.username, "user_id": user.user_id, "role": user.role}
        access_token = auth_handler.create_access_token(token_data)
        refresh_token = auth_handler.create_refresh_token(token_data)
        
        logger.info("User logged in", username=user.username, user_id=user.user_id)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=auth_handler.access_token_expire_minutes * 60,
            user=user.to_dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest) -> TokenResponse:
    """Refresh access token using refresh token"""
    try:
        # Decode refresh token
        payload = auth_handler.decode_token(request.refresh_token)
        username = payload.get("sub")
        
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Get user (this would typically come from database)
        from app.core.auth import MOCK_USERS_DB
        user_data = MOCK_USERS_DB.get(username)
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        user = User(
            user_id=user_data["user_id"],
            username=user_data["username"],
            email=user_data["email"],
            role=user_data["role"],
            is_active=user_data["is_active"]
        )
        
        # Create new tokens
        token_data = {"sub": user.username, "user_id": user.user_id, "role": user.role}
        access_token = auth_handler.create_access_token(token_data)
        new_refresh_token = auth_handler.create_refresh_token(token_data)
        
        logger.info("Token refreshed", username=user.username)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=auth_handler.access_token_expire_minutes * 60,
            user=user.to_dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Token refresh error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)) -> UserResponse:
    """Get current user information"""
    return UserResponse(
        user_id=current_user.user_id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        is_active=current_user.is_active,
        permissions=current_user.permissions
    )

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)) -> Dict[str, str]:
    """Logout user (in a real implementation, you would blacklist the token)"""
    logger.info("User logged out", username=current_user.username)
    return {"message": "Successfully logged out"}

@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Change user password"""
    try:
        # Get current user data
        from app.core.auth import MOCK_USERS_DB
        user_data = MOCK_USERS_DB.get(current_user.username)
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify current password
        if not auth_handler.verify_password(request.current_password, user_data["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Update password (in real implementation, update in database)
        new_hashed_password = auth_handler.get_password_hash(request.new_password)
        user_data["hashed_password"] = new_hashed_password
        
        logger.info("Password changed", username=current_user.username)
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Password change error", error=str(e), username=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )

@router.get("/check-token")
async def check_token(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Check if token is valid"""
    return {
        "valid": True,
        "user": current_user.to_dict(),
        "timestamp": datetime.utcnow().isoformat()
    }