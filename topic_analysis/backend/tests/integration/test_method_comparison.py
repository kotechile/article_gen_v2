"""
Integration test for method comparison functionality
Tests the complete method comparison workflow
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock


class TestMethodComparisonIntegration:
    """Integration tests for method comparison"""
    
    @pytest.fixture
    def client(self):
        """Create test client - this will fail until implementation exists"""
        from main import app  # This import will fail until implementation
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_method_comparison_full_workflow(self, client):
        """Test complete method comparison workflow"""
        request_data = {
            "search_query": "digital marketing",
            "user_id": "test_user",
            "max_subtopics": 6
        }
        
        # Mock different results for each method
        mock_llm_subtopics = [
            "digital marketing basics",
            "advanced digital marketing",
            "digital marketing tools"
        ]
        
        mock_autocomplete_subtopics = [
            "digital marketing courses",
            "digital marketing agency",
            "digital marketing jobs"
        ]
        
        mock_hybrid_subtopics = [
            "digital marketing strategy 2024",
            "digital marketing automation",
            "digital marketing analytics"
        ]
        
        with patch('enhanced_topic_decomposition_service.llm_service') as mock_llm, \
             patch('enhanced_topic_decomposition_service.google_autocomplete_service') as mock_autocomplete:
            
            # Mock LLM-only results
            mock_llm.generate_subtopics.return_value = mock_llm_subtopics
            
            # Mock autocomplete-only results
            mock_autocomplete.get_suggestions.return_value = {
                "query": "digital marketing",
                "suggestions": ["digital marketing courses", "digital marketing agency"],
                "total_suggestions": 2,
                "processing_time": 0.3,
                "success": True
            }
            
            # Mock hybrid results (combination of both)
            mock_hybrid_service = MagicMock()
            mock_hybrid_service.decompose_topic_enhanced.return_value = {
                "subtopics": mock_hybrid_subtopics,
                "processing_time": 1.1,
                "enhancement_methods": ["autocomplete", "llm"]
            }
            
            with patch('enhanced_topic_routes.enhanced_service', mock_hybrid_service):
                response = client.post("/api/enhanced-topics/compare-methods", json=request_data)
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify complete comparison structure
                assert data["success"] is True
                assert data["original_query"] == "digital marketing"
                assert "comparison" in data
                assert "recommendation" in data
                
                comparison = data["comparison"]
                
                # Verify all three methods are present
                assert "llm_only" in comparison
                assert "autocomplete_only" in comparison
                assert "hybrid" in comparison
                
                # Verify each method has required fields
                for method in ["llm_only", "autocomplete_only", "hybrid"]:
                    method_result = comparison[method]
                    assert "subtopics" in method_result
                    assert "processing_time" in method_result
                    assert "method" in method_result
                    assert isinstance(method_result["subtopics"], list)
                    assert isinstance(method_result["processing_time"], (int, float))
                    assert isinstance(method_result["method"], str)
    
    @pytest.mark.asyncio
    async def test_method_comparison_performance_metrics(self, client):
        """Test performance metrics collection"""
        request_data = {
            "search_query": "fitness equipment",
            "user_id": "test_user"
        }
        
        with patch('enhanced_topic_decomposition_service.llm_service') as mock_llm, \
             patch('enhanced_topic_decomposition_service.google_autocomplete_service') as mock_autocomplete:
            
            # Mock different processing times
            mock_llm.generate_subtopics.return_value = ["fitness equipment basics"]
            mock_autocomplete.get_suggestions.return_value = {
                "query": "fitness equipment",
                "suggestions": ["fitness equipment"],
                "total_suggestions": 1,
                "processing_time": 0.2,
                "success": True
            }
            
            response = client.post("/api/enhanced-topics/compare-methods", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            comparison = data["comparison"]
            
            # Verify performance metrics are collected
            for method in ["llm_only", "autocomplete_only", "hybrid"]:
                processing_time = comparison[method]["processing_time"]
                assert processing_time >= 0
                assert processing_time < 5.0  # Reasonable upper bound
    
    @pytest.mark.asyncio
    async def test_method_comparison_error_handling(self, client):
        """Test error handling in method comparison"""
        request_data = {
            "search_query": "test query",
            "user_id": "test_user"
        }
        
        # Mock LLM failure
        with patch('enhanced_topic_decomposition_service.llm_service') as mock_llm, \
             patch('enhanced_topic_decomposition_service.google_autocomplete_service') as mock_autocomplete:
            
            mock_llm.generate_subtopics.side_effect = Exception("LLM service down")
            mock_autocomplete.get_suggestions.return_value = {
                "query": "test query",
                "suggestions": ["test query suggestion"],
                "total_suggestions": 1,
                "processing_time": 0.1,
                "success": True
            }
            
            response = client.post("/api/enhanced-topics/compare-methods", json=request_data)
            
            # Should still return 200 with partial results
            assert response.status_code == 200
            data = response.json()
            
            # LLM-only should have empty results due to failure
            comparison = data["comparison"]
            assert comparison["llm_only"]["subtopics"] == []
            
            # Autocomplete-only should still work
            assert len(comparison["autocomplete_only"]["subtopics"]) > 0
    
    @pytest.mark.asyncio
    async def test_method_comparison_recommendation_generation(self, client):
        """Test recommendation generation based on results"""
        request_data = {
            "search_query": "fitness equipment",
            "user_id": "test_user"
        }
        
        with patch('enhanced_topic_decomposition_service.llm_service') as mock_llm, \
             patch('enhanced_topic_decomposition_service.google_autocomplete_service') as mock_autocomplete:
            
            # Mock different quality results
            mock_llm.generate_subtopics.return_value = ["fitness equipment basics"]
            mock_autocomplete.get_suggestions.return_value = {
                "query": "fitness equipment",
                "suggestions": ["fitness equipment", "fitness equipment for home"],
                "total_suggestions": 2,
                "processing_time": 0.1,
                "success": True
            }
            
            response = client.post("/api/enhanced-topics/compare-methods", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Should have a meaningful recommendation
            assert len(data["recommendation"]) > 0
            assert "approach" in data["recommendation"].lower() or "method" in data["recommendation"].lower()
    
    @pytest.mark.asyncio
    async def test_method_comparison_empty_results_handling(self, client):
        """Test handling of empty results gracefully"""
        request_data = {
            "search_query": "very obscure topic with no results",
            "user_id": "test_user"
        }
        
        with patch('enhanced_topic_decomposition_service.llm_service') as mock_llm, \
             patch('enhanced_topic_decomposition_service.google_autocomplete_service') as mock_autocomplete:
            
            # Mock empty results
            mock_llm.generate_subtopics.return_value = []
            mock_autocomplete.get_suggestions.return_value = {
                "query": "very obscure topic with no results",
                "suggestions": [],
                "total_suggestions": 0,
                "processing_time": 0.1,
                "success": True
            }
            
            response = client.post("/api/enhanced-topics/compare-methods", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Should handle empty results gracefully
            comparison = data["comparison"]
            for method in ["llm_only", "autocomplete_only", "hybrid"]:
                assert isinstance(comparison[method]["subtopics"], list)
                # Empty results are acceptable
                assert len(comparison[method]["subtopics"]) >= 0
    
    @pytest.mark.asyncio
    async def test_method_comparison_concurrent_execution(self, client):
        """Test concurrent execution of different methods"""
        request_data = {
            "search_query": "digital marketing",
            "user_id": "test_user"
        }
        
        with patch('enhanced_topic_decomposition_service.llm_service') as mock_llm, \
             patch('enhanced_topic_decomposition_service.google_autocomplete_service') as mock_autocomplete:
            
            # Mock concurrent execution
            mock_llm.generate_subtopics.return_value = ["digital marketing basics"]
            mock_autocomplete.get_suggestions.return_value = {
                "query": "digital marketing",
                "suggestions": ["digital marketing courses"],
                "total_suggestions": 1,
                "processing_time": 0.1,
                "success": True
            }
            
            response = client.post("/api/enhanced-topics/compare-methods", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # All methods should have been executed
            comparison = data["comparison"]
            assert len(comparison) == 3  # llm_only, autocomplete_only, hybrid
    
    @pytest.mark.asyncio
    async def test_method_comparison_validation_errors(self, client):
        """Test validation error handling"""
        # Test missing required fields
        response = client.post("/api/enhanced-topics/compare-methods", json={
            "user_id": "test_user"
            # Missing search_query
        })
        assert response.status_code == 400
        
        # Test invalid parameters
        response = client.post("/api/enhanced-topics/compare-methods", json={
            "search_query": "test",
            "user_id": "test_user",
            "max_subtopics": 15  # Exceeds maximum
        })
        assert response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_method_comparison_performance_requirements(self, client):
        """Test performance requirements for method comparison"""
        request_data = {
            "search_query": "fitness equipment",
            "user_id": "test_user"
        }
        
        with patch('enhanced_topic_decomposition_service.llm_service') as mock_llm, \
             patch('enhanced_topic_decomposition_service.google_autocomplete_service') as mock_autocomplete:
            
            # Mock fast responses
            mock_llm.generate_subtopics.return_value = ["fitness equipment basics"]
            mock_autocomplete.get_suggestions.return_value = {
                "query": "fitness equipment",
                "suggestions": ["fitness equipment"],
                "total_suggestions": 1,
                "processing_time": 0.1,
                "success": True
            }
            
            response = client.post("/api/enhanced-topics/compare-methods", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Method comparison should complete within reasonable time
            comparison = data["comparison"]
            total_time = sum(method["processing_time"] for method in comparison.values())
            assert total_time < 3.0  # All methods combined should be fast

