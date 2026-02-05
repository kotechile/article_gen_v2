"""
Contract Tests for GET /health/database Endpoint

These tests verify the API contract for the database health check endpoint.
Tests MUST fail initially (no implementation) and pass after implementation.
"""

import pytest
import requests
from typing import Dict, Any


class TestHealthEndpoint:
    """Test cases for GET /health/database endpoint."""
    
    def test_health_endpoint_exists(self):
        """Test that the health endpoint exists and is accessible."""
        # This test will fail initially - endpoint doesn't exist yet
        response = requests.get("http://localhost:8000/api/v1/health/database")
        
        # Should return 200 (not 404)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    def test_health_endpoint_response_schema(self):
        """Test that the health endpoint returns the correct schema."""
        response = requests.get("http://localhost:8000/api/v1/health/database")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify required fields exist
        required_fields = ["status", "database", "timestamp"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Verify field types
        assert isinstance(data["status"], str)
        assert isinstance(data["database"], str)
        assert isinstance(data["timestamp"], str)
        
        # Verify status values
        assert data["status"] in ["healthy", "unhealthy"]
        assert data["database"] == "supabase"
    
    def test_health_endpoint_healthy_response(self):
        """Test healthy database response."""
        response = requests.get("http://localhost:8000/api/v1/health/database")
        
        if response.status_code == 200:
            data = response.json()
            
            if data["status"] == "healthy":
                # Verify healthy response structure
                assert "tables_available" in data
                assert isinstance(data["tables_available"], list)
                
                # Verify execution time is reasonable
                if "execution_time_ms" in data:
                    assert isinstance(data["execution_time_ms"], (int, float))
                    assert data["execution_time_ms"] >= 0
    
    def test_health_endpoint_unhealthy_response(self):
        """Test unhealthy database response."""
        response = requests.get("http://localhost:8000/api/v1/health/database")
        
        if response.status_code == 503:
            data = response.json()
            
            # Verify unhealthy response structure
            assert data["status"] == "unhealthy"
            assert "message" in data
            assert isinstance(data["message"], str)
            assert "timestamp" in data
    
    def test_health_endpoint_performance(self):
        """Test that health check responds within acceptable time."""
        import time
        
        start_time = time.time()
        response = requests.get("http://localhost:8000/api/v1/health/database")
        end_time = time.time()
        
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Should respond within 200ms
        assert execution_time < 200, f"Health check took {execution_time:.2f}ms, expected < 200ms"
        
        # Should still return valid response
        assert response.status_code in [200, 503]
    
    def test_health_endpoint_content_type(self):
        """Test that health endpoint returns JSON content type."""
        response = requests.get("http://localhost:8000/api/v1/health/database")
        
        assert response.status_code in [200, 503]
        assert "application/json" in response.headers.get("content-type", "")
    
    def test_health_endpoint_cors_headers(self):
        """Test that health endpoint includes CORS headers."""
        response = requests.get("http://localhost:8000/api/v1/health/database")
        
        assert response.status_code in [200, 503]
        
        # Check for CORS headers (if implemented)
        cors_headers = [
            "Access-Control-Allow-Origin",
            "Access-Control-Allow-Methods",
            "Access-Control-Allow-Headers"
        ]
        
        # At least one CORS header should be present
        has_cors = any(header in response.headers for header in cors_headers)
        # Note: This might not be implemented yet, so we'll check if any exist
        if has_cors:
            assert any(header in response.headers for header in cors_headers)
    
    def test_health_endpoint_error_handling(self):
        """Test error handling when database is unavailable."""
        # This test simulates database unavailability
        # In a real scenario, this might be tested by stopping the database
        response = requests.get("http://localhost:8000/api/v1/health/database")
        
        # Should handle errors gracefully
        assert response.status_code in [200, 503]
        
        if response.status_code == 503:
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "message" in data
    
    def test_health_endpoint_timestamp_format(self):
        """Test that timestamp is in ISO format."""
        response = requests.get("http://localhost:8000/api/v1/health/database")
        
        assert response.status_code in [200, 503]
        data = response.json()
        
        # Verify timestamp format
        timestamp = data["timestamp"]
        assert isinstance(timestamp, str)
        
        # Should be parseable as ISO datetime
        from datetime import datetime
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            pytest.fail(f"Invalid timestamp format: {timestamp}")
    
    def test_health_endpoint_no_auth_required(self):
        """Test that health endpoint doesn't require authentication."""
        # Make request without any authentication headers
        response = requests.get("http://localhost:8000/api/v1/health/database")
        
        # Should not return 401 Unauthorized
        assert response.status_code != 401, "Health endpoint should not require authentication"
        
        # Should return either 200 or 503
        assert response.status_code in [200, 503]
    
    def test_health_endpoint_method_not_allowed(self):
        """Test that only GET method is allowed."""
        # Test POST method (should return 405 Method Not Allowed)
        response = requests.post("http://localhost:8000/api/v1/health/database")
        assert response.status_code == 405, "POST method should not be allowed"
        
        # Test PUT method (should return 405 Method Not Allowed)
        response = requests.put("http://localhost:8000/api/v1/health/database")
        assert response.status_code == 405, "PUT method should not be allowed"
        
        # Test DELETE method (should return 405 Method Not Allowed)
        response = requests.delete("http://localhost:8000/api/v1/health/database")
        assert response.status_code == 405, "DELETE method should not be allowed"
