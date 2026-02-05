"""
Contract tests for password reset endpoints.

These tests verify that the API contracts are properly implemented
and match the OpenAPI specification.
"""
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


class TestPasswordResetContracts:
    """Test password reset endpoint contracts."""

    def test_request_password_reset_contract(self):
        """Test POST /api/v1/auth/request-password-reset endpoint contract."""
        # This test should fail initially as the endpoint doesn't exist yet
        response = client.post(
            "/api/v1/auth/request-password-reset",
            json={"email": "test@example.com"}
        )
        
        # Should return 200 for successful password reset request
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert data["message"] == "Password reset email sent"

    def test_request_password_reset_validation_contract(self):
        """Test POST /api/v1/auth/request-password-reset validation contract."""
        # Test missing email
        response = client.post("/api/v1/auth/request-password-reset", json={})
        assert response.status_code == 422
        
        # Test invalid email format
        response = client.post(
            "/api/v1/auth/request-password-reset",
            json={"email": "invalid-email"}
        )
        assert response.status_code == 422

    def test_request_password_reset_user_not_found_contract(self):
        """Test POST /api/v1/auth/request-password-reset with non-existent user contract."""
        response = client.post(
            "/api/v1/auth/request-password-reset",
            json={"email": "nonexistent@example.com"}
        )
        
        # Should return 404 for non-existent user
        assert response.status_code == 404

    def test_reset_password_contract(self):
        """Test POST /api/v1/auth/reset-password endpoint contract."""
        # This test should fail initially as the endpoint doesn't exist yet
        response = client.post(
            "/api/v1/auth/reset-password",
            json={
                "token": "valid_reset_token",
                "new_password": "NewSecurePass123!"
            }
        )
        
        # Should return 200 for successful password reset
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert data["message"] == "Password reset successfully"

    def test_reset_password_validation_contract(self):
        """Test POST /api/v1/auth/reset-password validation contract."""
        # Test missing required fields
        response = client.post("/api/v1/auth/reset-password", json={})
        assert response.status_code == 422
        
        # Test weak new password
        response = client.post(
            "/api/v1/auth/reset-password",
            json={
                "token": "valid_reset_token",
                "new_password": "weak"  # Too weak
            }
        )
        assert response.status_code == 422

    def test_reset_password_invalid_token_contract(self):
        """Test POST /api/v1/auth/reset-password with invalid token contract."""
        response = client.post(
            "/api/v1/auth/reset-password",
            json={
                "token": "invalid_token",
                "new_password": "NewSecurePass123!"
            }
        )
        
        # Should return 400 for invalid token
        assert response.status_code == 400

    def test_reset_password_expired_token_contract(self):
        """Test POST /api/v1/auth/reset-password with expired token contract."""
        response = client.post(
            "/api/v1/auth/reset-password",
            json={
                "token": "expired_token",
                "new_password": "NewSecurePass123!"
            }
        )
        
        # Should return 400 for expired token
        assert response.status_code == 400

    def test_reset_password_used_token_contract(self):
        """Test POST /api/v1/auth/reset-password with already used token contract."""
        response = client.post(
            "/api/v1/auth/reset-password",
            json={
                "token": "used_token",
                "new_password": "NewSecurePass123!"
            }
        )
        
        # Should return 400 for already used token
        assert response.status_code == 400
