"""
Integration tests for password reset flow.

These tests verify the complete password reset workflow
from request to completion.
"""
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


class TestPasswordResetFlow:
    """Test complete password reset flow."""

    def test_complete_password_reset_flow(self):
        """Test complete password reset flow from request to completion."""
        # Step 1: Request password reset
        reset_request_data = {"email": "test@example.com"}
        
        response = client.post("/api/v1/auth/request-password-reset", json=reset_request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Password reset email sent"
        
        # Step 2: Reset password with token (this will fail until implemented)
        reset_data = {
            "token": "reset_token_from_email",
            "new_password": "NewSecurePass123!"
        }
        
        response = client.post("/api/v1/auth/reset-password", json=reset_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Password reset successfully"
        
        # Step 3: Login with new password
        login_data = {
            "email": "test@example.com",
            "password": "NewSecurePass123!"
        }
        
        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200

    def test_password_reset_with_nonexistent_user(self):
        """Test password reset request with non-existent user."""
        reset_request_data = {"email": "nonexistent@example.com"}
        
        response = client.post("/api/v1/auth/request-password-reset", json=reset_request_data)
        assert response.status_code == 404

    def test_password_reset_with_invalid_token(self):
        """Test password reset with invalid token."""
        reset_data = {
            "token": "invalid_token",
            "new_password": "NewSecurePass123!"
        }
        
        response = client.post("/api/v1/auth/reset-password", json=reset_data)
        assert response.status_code == 400

    def test_password_reset_with_expired_token(self):
        """Test password reset with expired token."""
        reset_data = {
            "token": "expired_token",
            "new_password": "NewSecurePass123!"
        }
        
        response = client.post("/api/v1/auth/reset-password", json=reset_data)
        assert response.status_code == 400

    def test_password_reset_with_used_token(self):
        """Test password reset with already used token."""
        reset_data = {
            "token": "used_token",
            "new_password": "NewSecurePass123!"
        }
        
        response = client.post("/api/v1/auth/reset-password", json=reset_data)
        assert response.status_code == 400

    def test_password_reset_validation(self):
        """Test password reset with invalid new password."""
        reset_data = {
            "token": "valid_token",
            "new_password": "weak"  # Too weak
        }
        
        response = client.post("/api/v1/auth/reset-password", json=reset_data)
        assert response.status_code == 422

    def test_multiple_password_reset_requests(self):
        """Test multiple password reset requests for same user."""
        reset_request_data = {"email": "test@example.com"}
        
        # First request
        response1 = client.post("/api/v1/auth/request-password-reset", json=reset_request_data)
        assert response1.status_code == 200
        
        # Second request (should invalidate first token)
        response2 = client.post("/api/v1/auth/request-password-reset", json=reset_request_data)
        assert response2.status_code == 200
        
        # First token should now be invalid
        reset_data = {
            "token": "first_reset_token",
            "new_password": "NewSecurePass123!"
        }
        
        response = client.post("/api/v1/auth/reset-password", json=reset_data)
        assert response.status_code == 400
