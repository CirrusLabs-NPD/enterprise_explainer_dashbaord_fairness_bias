"""
Tests for authentication system
"""

import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient

from app.core.auth import auth_handler, authenticate_user


class TestAuthentication:
    """Test authentication functionality"""
    
    def test_password_hashing(self):
        """Test password hashing and verification"""
        password = "test_password_123"
        hashed = auth_handler.get_password_hash(password)
        
        assert auth_handler.verify_password(password, hashed)
        assert not auth_handler.verify_password("wrong_password", hashed)
    
    def test_token_creation_and_verification(self):
        """Test JWT token creation and verification"""
        data = {"sub": "test_user", "user_id": "123", "role": "admin"}
        
        # Create token
        token = auth_handler.create_access_token(data)
        assert token is not None
        
        # Verify token
        decoded_data = auth_handler.decode_token(token)
        assert decoded_data["sub"] == "test_user"
        assert decoded_data["user_id"] == "123"
        assert decoded_data["role"] == "admin"
        assert "exp" in decoded_data
    
    def test_user_authentication(self):
        """Test user authentication with valid credentials"""
        user = authenticate_user("admin", "admin123")
        assert user is not None
        assert user.username == "admin"
        assert user.role == "admin"
        assert user.is_active
    
    def test_user_authentication_invalid_credentials(self):
        """Test user authentication with invalid credentials"""
        # Wrong password
        user = authenticate_user("admin", "wrong_password")
        assert user is None
        
        # Non-existent user
        user = authenticate_user("non_existent", "password")
        assert user is None
    
    def test_user_permissions(self):
        """Test user permission system"""
        user = authenticate_user("admin", "admin123")
        assert user.has_permission("model:create")
        assert user.has_permission("user:manage")
        
        user = authenticate_user("viewer", "viewer123")
        assert user.has_permission("model:read")
        assert not user.has_permission("model:create")
        assert not user.has_permission("user:manage")


class TestAuthEndpoints:
    """Test authentication endpoints"""
    
    @pytest.mark.asyncio
    async def test_login_success(self, async_client: AsyncClient):
        """Test successful login"""
        response = await async_client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["username"] == "admin"
        assert data["user"]["role"] == "admin"
    
    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self, async_client: AsyncClient):
        """Test login with invalid credentials"""
        response = await async_client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "wrong_password"}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert data["error"]["message"] == "Invalid username or password"
    
    @pytest.mark.asyncio
    async def test_get_current_user(self, async_client: AsyncClient, admin_headers: dict):
        """Test getting current user information"""
        response = await async_client.get(
            "/api/v1/auth/me",
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["username"] == "admin"
        assert data["role"] == "admin"
        assert data["is_active"] is True
        assert "permissions" in data
    
    @pytest.mark.asyncio
    async def test_get_current_user_unauthorized(self, async_client: AsyncClient):
        """Test getting current user without authentication"""
        response = await async_client.get("/api/v1/auth/me")
        
        assert response.status_code == 403  # No auth header provided
    
    @pytest.mark.asyncio
    async def test_refresh_token(self, async_client: AsyncClient):
        """Test token refresh"""
        # First login to get tokens
        login_response = await async_client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        
        tokens = login_response.json()
        refresh_token = tokens["refresh_token"]
        
        # Use refresh token to get new access token
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["user"]["username"] == "admin"
    
    @pytest.mark.asyncio
    async def test_change_password(self, async_client: AsyncClient, admin_headers: dict):
        """Test password change"""
        response = await async_client.post(
            "/api/v1/auth/change-password",
            json={
                "current_password": "admin123",
                "new_password": "new_password_123"
            },
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Password changed successfully"
        
        # Reset password back for other tests
        await async_client.post(
            "/api/v1/auth/change-password",
            json={
                "current_password": "new_password_123",
                "new_password": "admin123"
            },
            headers=admin_headers
        )
    
    @pytest.mark.asyncio
    async def test_change_password_wrong_current(self, async_client: AsyncClient, admin_headers: dict):
        """Test password change with wrong current password"""
        response = await async_client.post(
            "/api/v1/auth/change-password",
            json={
                "current_password": "wrong_password",
                "new_password": "new_password_123"
            },
            headers=admin_headers
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Current password is incorrect" in data["error"]["message"]
    
    @pytest.mark.asyncio
    async def test_logout(self, async_client: AsyncClient, admin_headers: dict):
        """Test logout"""
        response = await async_client.post(
            "/api/v1/auth/logout",
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Successfully logged out"
    
    @pytest.mark.asyncio
    async def test_check_token(self, async_client: AsyncClient, admin_headers: dict):
        """Test token validation"""
        response = await async_client.get(
            "/api/v1/auth/check-token",
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["valid"] is True
        assert data["user"]["username"] == "admin"
        assert "timestamp" in data


class TestRoleBasedAccess:
    """Test role-based access control"""
    
    @pytest.mark.asyncio
    async def test_admin_access(self, async_client: AsyncClient, admin_headers: dict):
        """Test admin can access all endpoints"""
        response = await async_client.get(
            "/api/v1/auth/me",
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "model:create" in data["permissions"]
        assert "user:manage" in data["permissions"]
    
    @pytest.mark.asyncio
    async def test_analyst_access(self, async_client: AsyncClient, analyst_headers: dict):
        """Test analyst has limited permissions"""
        response = await async_client.get(
            "/api/v1/auth/me",
            headers=analyst_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "model:create" in data["permissions"]
        assert "analysis:run" in data["permissions"]
        assert "user:manage" not in data["permissions"]
    
    @pytest.mark.asyncio
    async def test_viewer_access(self, async_client: AsyncClient, viewer_headers: dict):
        """Test viewer has read-only permissions"""
        response = await async_client.get(
            "/api/v1/auth/me",
            headers=viewer_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "model:read" in data["permissions"]
        assert "model:create" not in data["permissions"]
        assert "user:manage" not in data["permissions"]