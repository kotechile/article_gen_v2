"""
Contract test for POST /api/enhanced-topics/compare-methods endpoint
Tests the API contract as defined in enhanced-topics-api.yaml
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock


class TestEnhancedTopicsCompareContract:
    """Contract tests for method comparison endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create test client - this will fail until implementation exists"""
        # This will fail until the actual FastAPI app includes the enhanced topics routes
        from main import app  # This import will fail until implementation
        return TestClient(app)
    
    def test_compare_methods_basic_request_contract(self, client):
        """Test basic method comparison request contract"""
        request_data = {
            "search_query": "digital marketing",
            "user_id": "user789",
            "max_subtopics": 6
        }
        
        response = client.post("/api/enhanced-topics/compare-methods", json=request_data)
        
        # Contract validation - these assertions will fail until implementation
        assert response.status_code == 200
        
        data = response.json()
        
        # Required fields from contract
        assert "success" in data
        assert "original_query" in data
        assert "comparison" in data
        assert "recommendation" in data
        
        # Type validation
        assert isinstance(data["success"], bool)
        assert isinstance(data["original_query"], str)
        assert isinstance(data["comparison"], dict)
        assert isinstance(data["recommendation"], str)
        
        # Comparison structure validation
        comparison = data["comparison"]
        assert "llm_only" in comparison
        assert "autocomplete_only" in comparison
        assert "hybrid" in comparison
        
        # Method result structure validation
        for method in ["llm_only", "autocomplete_only", "hybrid"]:
            method_result = comparison[method]
            assert "subtopics" in method_result
            assert "processing_time" in method_result
            assert "method" in method_result
            
            # Type validation for method result
            assert isinstance(method_result["subtopics"], list)
            assert isinstance(method_result["processing_time"], (int, float))
            assert isinstance(method_result["method"], str)
            
            # Processing time should be positive
            assert method_result["processing_time"] >= 0
    
    def test_compare_methods_default_parameters_contract(self, client):
        """Test method comparison with default parameters"""
        request_data = {
            "search_query": "fitness equipment",
            "user_id": "user123"
            # max_subtopics should default to 6
        }
        
        response = client.post("/api/enhanced-topics/compare-methods", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should work with default parameters
        assert data["success"] is True
        assert data["original_query"] == "fitness equipment"
    
    def test_compare_methods_validation_errors_contract(self, client):
        """Test validation error responses contract"""
        # Test missing required fields
        response = client.post("/api/enhanced-topics/compare-methods", json={
            "user_id": "user123"
            # Missing search_query
        })
        assert response.status_code == 400
        
        # Test invalid max_subtopics
        response = client.post("/api/enhanced-topics/compare-methods", json={
            "search_query": "test",
            "user_id": "user123",
            "max_subtopics": 15  # Exceeds maximum of 10
        })
        assert response.status_code == 400
        
        # Test query length limits
        response = client.post("/api/enhanced-topics/compare-methods", json={
            "search_query": "x" * 201,  # Exceeds max length of 200
            "user_id": "user123"
        })
        assert response.status_code == 400
    
    def test_compare_methods_server_error_contract(self, client):
        """Test server error response contract"""
        request_data = {
            "search_query": "test query",
            "user_id": "user123"
        }
        
        # Simulate server error
        with patch('enhanced_topic_routes.enhanced_service') as mock_service:
            mock_service.compare_methods.side_effect = Exception("Internal error")
            
            response = client.post("/api/enhanced-topics/compare-methods", json=request_data)
            
            assert response.status_code == 500
            data = response.json()
            assert data["success"] is False
            assert "error" in data
            assert "message" in data
    
    def test_compare_methods_performance_contract(self, client):
        """Test performance requirements from contract"""
        request_data = {
            "search_query": "digital marketing",
            "user_id": "user123",
            "max_subtopics": 6
        }
        
        response = client.post("/api/enhanced-topics/compare-methods", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Performance requirements - should complete within reasonable time
        comparison = data["comparison"]
        for method in ["llm_only", "autocomplete_only", "hybrid"]:
            processing_time = comparison[method]["processing_time"]
            assert processing_time < 5.0  # Each method should be reasonably fast
    
    def test_compare_methods_result_consistency_contract(self, client):
        """Test that comparison results are consistent"""
        request_data = {
            "search_query": "fitness equipment",
            "user_id": "user123",
            "max_subtopics": 4
        }
        
        response = client.post("/api/enhanced-topics/compare-methods", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        comparison = data["comparison"]
        
        # All methods should have results
        for method in ["llm_only", "autocomplete_only", "hybrid"]:
            method_result = comparison[method]
            assert len(method_result["subtopics"]) <= 4  # Respects max_subtopics
            assert method_result["processing_time"] >= 0
            assert method_result["method"] in ["LLM Only", "Autocomplete Only", "Hybrid (LLM + Autocomplete)"]
        
        # Recommendation should be present and meaningful
        assert len(data["recommendation"]) > 0
        assert "approach" in data["recommendation"].lower() or "method" in data["recommendation"].lower()
    
    def test_compare_methods_empty_results_contract(self, client):
        """Test handling of empty results gracefully"""
        request_data = {
            "search_query": "very obscure topic that should have no results",
            "user_id": "user123"
        }
        
        response = client.post("/api/enhanced-topics/compare-methods", json=request_data)
        
        # Should still return 200 even with empty results
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        comparison = data["comparison"]
        
        # All methods should return empty arrays rather than failing
        for method in ["llm_only", "autocomplete_only", "hybrid"]:
            method_result = comparison[method]
            assert isinstance(method_result["subtopics"], list)
            # Empty results are acceptable
            assert len(method_result["subtopics"]) >= 0
    
    def test_compare_methods_response_structure_contract(self, client):
        """Test response structure matches contract exactly"""
        request_data = {
            "search_query": "test query",
            "user_id": "user123"
        }
        
        response = client.post("/api/enhanced-topics/compare-methods", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Exact contract validation
        required_fields = ["success", "original_query", "comparison", "recommendation"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Comparison structure validation
        comparison = data["comparison"]
        required_methods = ["llm_only", "autocomplete_only", "hybrid"]
        for method in required_methods:
            assert method in comparison, f"Missing comparison method: {method}"
            
            method_result = comparison[method]
            required_method_fields = ["subtopics", "processing_time", "method"]
            for field in required_method_fields:
                assert field in method_result, f"Missing method field: {field}"
        
        # No extra fields beyond contract
        allowed_top_level = set(required_fields)
        actual_top_level = set(data.keys())
        assert actual_top_level.issubset(allowed_top_level), f"Unexpected top-level fields: {actual_top_level - allowed_top_level}"

