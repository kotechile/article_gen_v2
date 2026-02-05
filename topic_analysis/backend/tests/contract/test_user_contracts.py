"""
Contract tests for user management endpoints.

These tests verify that the API contracts are properly implemented
and match the OpenAPI specification.
"""
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


class TestUserContracts:
    """Test user management endpoint contracts."""

    def test_get_user_profile_contract(self):
        """Test GET /api/v1/users/me endpoint contract."""
        # This test should fail initially as the endpoint doesn't exist yet
        headers = {"Authorization": "Bearer valid_token"}
        response = client.get("/api/v1/users/me", headers=headers)
        
        # Should return 200 for successful profile retrieval
        assert response.status_code == 200
        
        data = response.json()
        assert "id" in data
        assert "email" in data
        assert "first_name" in data
        assert "last_name" in data
        assert "role" in data
        assert "is_active" in data
        assert "is_verified" in data
        assert "created_at" in data
        assert "updated_at" in data

    def test_get_user_profile_unauthorized_contract(self):
        """Test GET /api/v1/users/me without token contract."""
        response = client.get("/api/v1/users/me")
        assert response.status_code == 401

    def test_update_user_profile_contract(self):
        """Test PUT /api/v1/users/me endpoint contract."""
        # This test should fail initially as the endpoint doesn't exist yet
        headers = {"Authorization": "Bearer valid_token"}
        response = client.put(
            "/api/v1/users/me",
            headers=headers,
            json={
                "first_name": "Jane",
                "last_name": "Smith"
            }
        )
        
        # Should return 200 for successful profile update
        assert response.status_code == 200
        
        data = response.json()
        assert data["first_name"] == "Jane"
        assert data["last_name"] == "Smith"

    def test_update_user_profile_validation_contract(self):
        """Test PUT /api/v1/users/me validation contract."""
        headers = {"Authorization": "Bearer valid_token"}
        
        # Test invalid field lengths
        response = client.put(
            "/api/v1/users/me",
            headers=headers,
            json={
                "first_name": "",  # Too short
                "last_name": "A" * 51  # Too long
            }
        )
        assert response.status_code == 422

    def test_change_password_contract(self):
        """Test POST /api/v1/users/me/change-password endpoint contract."""
        # This test should fail initially as the endpoint doesn't exist yet
        headers = {"Authorization": "Bearer valid_token"}
        response = client.post(
            "/api/v1/users/me/change-password",
            headers=headers,
            json={
                "current_password": "OldPass123!",
                "new_password": "NewPass123!"
            }
        )
        
        # Should return 200 for successful password change
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert data["message"] == "Password changed successfully"

    def test_change_password_validation_contract(self):
        """Test POST /api/v1/users/me/change-password validation contract."""
        headers = {"Authorization": "Bearer valid_token"}
        
        # Test missing required fields
        response = client.post(
            "/api/v1/users/me/change-password",
            headers=headers,
            json={}
        )
        assert response.status_code == 422
        
        # Test weak new password
        response = client.post(
            "/api/v1/users/me/change-password",
            headers=headers,
            json={
                "current_password": "OldPass123!",
                "new_password": "weak"  # Too weak
            }
        )
        assert response.status_code == 422

    def test_change_password_wrong_current_contract(self):
        """Test POST /api/v1/users/me/change-password with wrong current password."""
        headers = {"Authorization": "Bearer valid_token"}
        response = client.post(
            "/api/v1/users/me/change-password",
            headers=headers,
            json={
                "current_password": "WrongPass123!",
                "new_password": "NewPass123!"
            }
        )
        
        # Should return 400 for wrong current password
        assert response.status_code == 400
