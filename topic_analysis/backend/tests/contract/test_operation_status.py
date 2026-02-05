"""
Contract Tests for GET /database/operations/{id} Endpoint

These tests verify the API contract for retrieving operation status.
Tests MUST fail initially (no implementation) and pass after implementation.
"""

import pytest
import requests
import uuid
from typing import Dict, Any


class TestOperationStatusEndpoint:
    """Test cases for GET /database/operations/{id} endpoint."""
    
    def test_operation_status_endpoint_exists(self):
        """Test that the operation status endpoint exists and is accessible."""
        # This test will fail initially - endpoint doesn't exist yet
        operation_id = str(uuid.uuid4())
        
        response = requests.get(f"http://localhost:8000/api/v1/database/operations/{operation_id}")
        
        # Should return 401 (unauthorized) or 404 (not found), not 500
        assert response.status_code in [200, 401, 404], f"Expected 200, 401, or 404, got {response.status_code}"
    
    def test_operation_status_requires_auth(self):
        """Test that operation status requires authentication."""
        operation_id = str(uuid.uuid4())
        
        response = requests.get(f"http://localhost:8000/api/v1/database/operations/{operation_id}")
        
        # Should return 401 Unauthorized without auth
        assert response.status_code == 401, "Operation status should require authentication"
        
        # Should return proper error response
        data = response.json()
        assert "error" in data or "message" in data
    
    def test_operation_status_with_auth(self):
        """Test operation status with valid authentication."""
        operation_id = str(uuid.uuid4())
        
        headers = {"Authorization": "Bearer test-token"}
        
        response = requests.get(
            f"http://localhost:8000/api/v1/database/operations/{operation_id}",
            headers=headers
        )
        
        # Should return 200 or 404 with valid auth
        assert response.status_code in [200, 404, 401]
        
        if response.status_code == 200:
            data = response.json()
            assert "operation_id" in data
            assert "status" in data
    
    def test_operation_status_response_schema(self):
        """Test that the response schema matches the contract."""
        operation_id = str(uuid.uuid4())
        
        headers = {"Authorization": "Bearer test-token"}
        
        response = requests.get(
            f"http://localhost:8000/api/v1/database/operations/{operation_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Verify required fields exist
            required_fields = ["operation_id", "status", "created_at"]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
            
            # Verify field types
            assert isinstance(data["operation_id"], str)
            assert isinstance(data["status"], str)
            assert isinstance(data["created_at"], str)
            
            # Verify status values
            assert data["status"] in ["pending", "success", "error", "timeout"]
    
    def test_operation_status_pending_operation(self):
        """Test response for pending operation."""
        operation_id = str(uuid.uuid4())
        
        headers = {"Authorization": "Bearer test-token"}
        
        response = requests.get(
            f"http://localhost:8000/api/v1/database/operations/{operation_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data["status"] == "pending":
                # Pending operations should not have completed_at
                assert "completed_at" not in data or data["completed_at"] is None
                assert "execution_time_ms" not in data or data["execution_time_ms"] is None
    
    def test_operation_status_completed_operation(self):
        """Test response for completed operation."""
        operation_id = str(uuid.uuid4())
        
        headers = {"Authorization": "Bearer test-token"}
        
        response = requests.get(
            f"http://localhost:8000/api/v1/database/operations/{operation_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data["status"] in ["success", "error", "timeout"]:
                # Completed operations should have completed_at
                assert "completed_at" in data
                assert data["completed_at"] is not None
                
                # Should have execution time
                if "execution_time_ms" in data:
                    assert isinstance(data["execution_time_ms"], (int, float))
                    assert data["execution_time_ms"] >= 0
    
    def test_operation_status_error_operation(self):
        """Test response for error operation."""
        operation_id = str(uuid.uuid4())
        
        headers = {"Authorization": "Bearer test-token"}
        
        response = requests.get(
            f"http://localhost:8000/api/v1/database/operations/{operation_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data["status"] == "error":
                # Error operations should have error message
                assert "error_message" in data
                assert isinstance(data["error_message"], str)
                assert len(data["error_message"]) > 0
    
    def test_operation_status_not_found(self):
        """Test response for non-existent operation."""
        operation_id = str(uuid.uuid4())
        
        headers = {"Authorization": "Bearer test-token"}
        
        response = requests.get(
            f"http://localhost:8000/api/v1/database/operations/{operation_id}",
            headers=headers
        )
        
        if response.status_code == 404:
            data = response.json()
            
            # Should return proper 404 response
            assert "error" in data or "message" in data
            assert "timestamp" in data
    
    def test_operation_status_invalid_uuid(self):
        """Test response for invalid operation ID format."""
        invalid_id = "not-a-valid-uuid"
        
        headers = {"Authorization": "Bearer test-token"}
        
        response = requests.get(
            f"http://localhost:8000/api/v1/database/operations/{invalid_id}",
            headers=headers
        )
        
        # Should return 400 Bad Request for invalid UUID
        assert response.status_code == 400, "Should return 400 for invalid UUID format"
        
        data = response.json()
        assert "error" in data or "message" in data
    
    def test_operation_status_timestamp_format(self):
        """Test that timestamps are in ISO format."""
        operation_id = str(uuid.uuid4())
        
        headers = {"Authorization": "Bearer test-token"}
        
        response = requests.get(
            f"http://localhost:8000/api/v1/database/operations/{operation_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Verify timestamp format
            for timestamp_field in ["created_at", "completed_at"]:
                if timestamp_field in data and data[timestamp_field]:
                    timestamp = data[timestamp_field]
                    assert isinstance(timestamp, str)
                    
                    # Should be parseable as ISO datetime
                    from datetime import datetime
                    try:
                        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except ValueError:
                        pytest.fail(f"Invalid timestamp format: {timestamp}")
    
    def test_operation_status_content_type(self):
        """Test that endpoint returns JSON content type."""
        operation_id = str(uuid.uuid4())
        
        headers = {"Authorization": "Bearer test-token"}
        
        response = requests.get(
            f"http://localhost:8000/api/v1/database/operations/{operation_id}",
            headers=headers
        )
        
        assert response.status_code in [200, 404, 401, 400]
        assert "application/json" in response.headers.get("content-type", "")
    
    def test_operation_status_method_not_allowed(self):
        """Test that only GET method is allowed."""
        operation_id = str(uuid.uuid4())
        
        # Test POST method (should return 405 Method Not Allowed)
        response = requests.post(f"http://localhost:8000/api/v1/database/operations/{operation_id}")
        assert response.status_code == 405, "POST method should not be allowed"
        
        # Test PUT method (should return 405 Method Not Allowed)
        response = requests.put(f"http://localhost:8000/api/v1/database/operations/{operation_id}")
        assert response.status_code == 405, "PUT method should not be allowed"
        
        # Test DELETE method (should return 405 Method Not Allowed)
        response = requests.delete(f"http://localhost:8000/api/v1/database/operations/{operation_id}")
        assert response.status_code == 405, "DELETE method should not be allowed"
    
    def test_operation_status_performance(self):
        """Test that status check responds within acceptable time."""
        import time
        
        operation_id = str(uuid.uuid4())
        
        headers = {"Authorization": "Bearer test-token"}
        
        start_time = time.time()
        response = requests.get(
            f"http://localhost:8000/api/v1/database/operations/{operation_id}",
            headers=headers
        )
        end_time = time.time()
        
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Should respond within 200ms
        assert execution_time < 200, f"Status check took {execution_time:.2f}ms, expected < 200ms"
        
        # Should still return valid response
        assert response.status_code in [200, 404, 401, 400]
    
    def test_operation_status_unauthorized_token(self):
        """Test response with invalid or expired token."""
        operation_id = str(uuid.uuid4())
        
        headers = {"Authorization": "Bearer invalid-token"}
        
        response = requests.get(
            f"http://localhost:8000/api/v1/database/operations/{operation_id}",
            headers=headers
        )
        
        # Should return 401 Unauthorized for invalid token
        assert response.status_code == 401, "Should return 401 for invalid token"
        
        data = response.json()
        assert "error" in data or "message" in data
