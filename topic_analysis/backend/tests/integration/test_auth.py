"""
Integration tests for authentication flow.

These tests verify the complete authentication workflow
including login, logout, and token refresh.
"""
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


class TestAuthenticationFlow:
    """Test complete authentication flow."""

    def test_complete_auth_flow(self):
        """Test complete authentication flow from login to logout."""
        # Step 1: Login with valid credentials
        login_data = {
            "email": "test@example.com",
            "password": "SecurePass123!"
        }
        
        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert "user" in data
        
        access_token = data["access_token"]
        refresh_token = data["refresh_token"]
        
        # Step 2: Access protected resource
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/api/v1/users/me", headers=headers)
        assert response.status_code == 200
        
        user_data = response.json()
        assert user_data["email"] == "test@example.com"
        
        # Step 3: Refresh token
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        assert response.status_code == 200
        
        refresh_data = response.json()
        assert "access_token" in refresh_data
        assert "refresh_token" in refresh_data
        
        new_access_token = refresh_data["access_token"]
        
        # Step 4: Use new token
        headers = {"Authorization": f"Bearer {new_access_token}"}
        response = client.get("/api/v1/users/me", headers=headers)
        assert response.status_code == 200
        
        # Step 5: Logout
        response = client.post("/api/v1/auth/logout", headers=headers)
        assert response.status_code == 200
        
        # Step 6: Verify token is invalidated
        response = client.get("/api/v1/users/me", headers=headers)
        assert response.status_code == 401

    def test_login_with_invalid_credentials(self):
        """Test login with invalid credentials."""
        login_data = {
            "email": "test@example.com",
            "password": "wrongpassword"
        }
        
        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 401

    def test_login_with_nonexistent_user(self):
        """Test login with non-existent user."""
        login_data = {
            "email": "nonexistent@example.com",
            "password": "SecurePass123!"
        }
        
        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 401

    def test_access_protected_resource_without_token(self):
        """Test accessing protected resource without token."""
        response = client.get("/api/v1/users/me")
        assert response.status_code == 401

    def test_access_protected_resource_with_invalid_token(self):
        """Test accessing protected resource with invalid token."""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/users/me", headers=headers)
        assert response.status_code == 401

    def test_refresh_with_invalid_token(self):
        """Test token refresh with invalid refresh token."""
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "invalid_refresh_token"}
        )
        assert response.status_code == 401

    def test_refresh_with_expired_token(self):
        """Test token refresh with expired refresh token."""
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "expired_refresh_token"}
        )
        assert response.status_code == 401

    def test_multiple_concurrent_sessions(self):
        """Test multiple concurrent sessions for same user."""
        # Login first time
        login_data = {
            "email": "test@example.com",
            "password": "SecurePass123!"
        }
        
        response1 = client.post("/api/v1/auth/login", json=login_data)
        assert response1.status_code == 200
        
        # Login second time (should create new session)
        response2 = client.post("/api/v1/auth/login", json=login_data)
        assert response2.status_code == 200
        
        # Both tokens should work
        token1 = response1.json()["access_token"]
        token2 = response2.json()["access_token"]
        
        headers1 = {"Authorization": f"Bearer {token1}"}
        headers2 = {"Authorization": f"Bearer {token2}"}
        
        response1 = client.get("/api/v1/users/me", headers=headers1)
        response2 = client.get("/api/v1/users/me", headers=headers2)
        
        assert response1.status_code == 200
        assert response2.status_code == 200

    def test_account_lockout_after_failed_attempts(self):
        """Test account lockout after multiple failed login attempts."""
        login_data = {
            "email": "test@example.com",
            "password": "wrongpassword"
        }
        
        # Make 5 failed attempts
        for _ in range(5):
            response = client.post("/api/v1/auth/login", json=login_data)
            assert response.status_code == 401
        
        # 6th attempt should lock account
        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 423  # Account locked
