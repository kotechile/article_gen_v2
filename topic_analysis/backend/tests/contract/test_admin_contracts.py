"""
Contract tests for admin endpoints.

These tests verify that the API contracts are properly implemented
and match the OpenAPI specification.
"""
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


class TestAdminContracts:
    """Test admin endpoint contracts."""

    def test_list_users_contract(self):
        """Test GET /api/v1/admin/users endpoint contract."""
        # This test should fail initially as the endpoint doesn't exist yet
        headers = {"Authorization": "Bearer admin_token"}
        response = client.get("/api/v1/admin/users", headers=headers)
        
        # Should return 200 for successful user list retrieval
        assert response.status_code == 200
        
        data = response.json()
        assert "users" in data
        assert "pagination" in data
        assert isinstance(data["users"], list)
        
        # Check pagination structure
        pagination = data["pagination"]
        assert "page" in pagination
        assert "limit" in pagination
        assert "total" in pagination
        assert "pages" in pagination

    def test_list_users_with_filters_contract(self):
        """Test GET /api/v1/admin/users with query parameters contract."""
        headers = {"Authorization": "Bearer admin_token"}
        response = client.get(
            "/api/v1/admin/users?page=1&limit=10&role=user&is_active=true",
            headers=headers
        )
        
        # Should return 200 for successful filtered user list
        assert response.status_code == 200
        
        data = response.json()
        assert "users" in data
        assert "pagination" in data

    def test_list_users_unauthorized_contract(self):
        """Test GET /api/v1/admin/users without admin token contract."""
        # Test without token
        response = client.get("/api/v1/admin/users")
        assert response.status_code == 401
        
        # Test with regular user token
        headers = {"Authorization": "Bearer user_token"}
        response = client.get("/api/v1/admin/users", headers=headers)
        assert response.status_code == 403

    def test_get_user_by_id_contract(self):
        """Test GET /api/v1/admin/users/{user_id} endpoint contract."""
        # This test should fail initially as the endpoint doesn't exist yet
        headers = {"Authorization": "Bearer admin_token"}
        user_id = "123e4567-e89b-12d3-a456-426614174000"
        response = client.get(f"/api/v1/admin/users/{user_id}", headers=headers)
        
        # Should return 200 for successful user retrieval
        assert response.status_code == 200
        
        data = response.json()
        assert "id" in data
        assert "email" in data
        assert "first_name" in data
        assert "last_name" in data
        assert "role" in data
        assert "is_active" in data
        assert "is_verified" in data

    def test_get_user_by_id_not_found_contract(self):
        """Test GET /api/v1/admin/users/{user_id} with non-existent user contract."""
        headers = {"Authorization": "Bearer admin_token"}
        user_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/api/v1/admin/users/{user_id}", headers=headers)
        
        # Should return 404 for non-existent user
        assert response.status_code == 404

    def test_update_user_contract(self):
        """Test PUT /api/v1/admin/users/{user_id} endpoint contract."""
        # This test should fail initially as the endpoint doesn't exist yet
        headers = {"Authorization": "Bearer admin_token"}
        user_id = "123e4567-e89b-12d3-a456-426614174000"
        response = client.put(
            f"/api/v1/admin/users/{user_id}",
            headers=headers,
            json={
                "first_name": "Updated",
                "last_name": "User",
                "role": "admin",
                "is_active": True
            }
        )
        
        # Should return 200 for successful user update
        assert response.status_code == 200
        
        data = response.json()
        assert data["first_name"] == "Updated"
        assert data["last_name"] == "User"
        assert data["role"] == "admin"
        assert data["is_active"] is True

    def test_update_user_validation_contract(self):
        """Test PUT /api/v1/admin/users/{user_id} validation contract."""
        headers = {"Authorization": "Bearer admin_token"}
        user_id = "123e4567-e89b-12d3-a456-426614174000"
        
        # Test invalid role
        response = client.put(
            f"/api/v1/admin/users/{user_id}",
            headers=headers,
            json={
                "role": "invalid_role"
            }
        )
        assert response.status_code == 422

    def test_deactivate_user_contract(self):
        """Test DELETE /api/v1/admin/users/{user_id} endpoint contract."""
        # This test should fail initially as the endpoint doesn't exist yet
        headers = {"Authorization": "Bearer admin_token"}
        user_id = "123e4567-e89b-12d3-a456-426614174000"
        response = client.delete(f"/api/v1/admin/users/{user_id}", headers=headers)
        
        # Should return 200 for successful user deactivation
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert data["message"] == "User deactivated successfully"

    def test_deactivate_user_not_found_contract(self):
        """Test DELETE /api/v1/admin/users/{user_id} with non-existent user contract."""
        headers = {"Authorization": "Bearer admin_token"}
        user_id = "00000000-0000-0000-0000-000000000000"
        response = client.delete(f"/api/v1/admin/users/{user_id}", headers=headers)
        
        # Should return 404 for non-existent user
        assert response.status_code == 404
