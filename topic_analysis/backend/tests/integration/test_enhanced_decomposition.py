"""
Integration test for enhanced topic decomposition workflow
Tests the complete flow from API request to response
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


class TestEnhancedDecompositionIntegration:
    """Integration tests for enhanced topic decomposition"""
    
    @pytest.fixture
    def client(self):
        """Create test client - this will fail until implementation exists"""
        from main import app  # This import will fail until implementation
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_enhanced_decomposition_full_workflow(self, client):
        """Test complete enhanced decomposition workflow"""
        request_data = {
            "search_query": "fitness equipment",
            "user_id": "test_user",
            "max_subtopics": 6,
            "use_autocomplete": True,
            "use_llm": True
        }
        
        # Mock the Google Autocomplete service
        mock_autocomplete_data = {
            "query": "fitness equipment",
            "suggestions": [
                "fitness equipment",
                "fitness equipment for home",
                "fitness equipment store",
                "fitness equipment near me"
            ],
            "total_suggestions": 4,
            "processing_time": 0.5,
            "success": True
        }
        
        # Mock the LLM service
        mock_llm_subtopics = [
            "best home gym equipment 2024",
            "commercial fitness equipment",
            "fitness equipment reviews",
            "fitness equipment maintenance"
        ]
        
        with patch('enhanced_topic_decomposition_service.google_autocomplete_service') as mock_autocomplete, \
             patch('enhanced_topic_decomposition_service.llm_service') as mock_llm:
            
            mock_autocomplete.get_suggestions.return_value = mock_autocomplete_data
            mock_llm.generate_subtopics.return_value = mock_llm_subtopics
            
            response = client.post("/api/enhanced-topics/decompose", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify the complete workflow
            assert data["success"] is True
            assert data["original_query"] == "fitness equipment"
            assert len(data["subtopics"]) <= 6
            assert "autocomplete" in data["enhancement_methods"]
            assert "llm" in data["enhancement_methods"]
            
            # Verify subtopics have required fields
            for subtopic in data["subtopics"]:
                assert "title" in subtopic
                assert "relevance_score" in subtopic
                assert "source" in subtopic
                assert subtopic["source"] in ["llm", "autocomplete", "hybrid"]
    
    @pytest.mark.asyncio
    async def test_enhanced_decomposition_autocomplete_failure_fallback(self, client):
        """Test fallback when autocomplete fails"""
        request_data = {
            "search_query": "fitness equipment",
            "user_id": "test_user",
            "use_autocomplete": True,
            "use_llm": True
        }
        
        # Mock autocomplete failure
        mock_autocomplete_data = {
            "success": False,
            "error": "Google API unavailable",
            "suggestions": [],
            "total_suggestions": 0,
            "processing_time": 0.1
        }
        
        mock_llm_subtopics = [
            "fitness equipment basics",
            "advanced fitness equipment",
            "fitness equipment tools"
        ]
        
        with patch('enhanced_topic_decomposition_service.google_autocomplete_service') as mock_autocomplete, \
             patch('enhanced_topic_decomposition_service.llm_service') as mock_llm:
            
            mock_autocomplete.get_suggestions.return_value = mock_autocomplete_data
            mock_llm.generate_subtopics.return_value = mock_llm_subtopics
            
            response = client.post("/api/enhanced-topics/decompose", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Should still succeed with LLM-only approach
            assert data["success"] is True
            assert len(data["subtopics"]) > 0
            
            # Should indicate fallback occurred
            assert "llm" in data["enhancement_methods"]
            # May or may not include "autocomplete" depending on implementation
    
    @pytest.mark.asyncio
    async def test_enhanced_decomposition_llm_failure_fallback(self, client):
        """Test fallback when LLM fails"""
        request_data = {
            "search_query": "fitness equipment",
            "user_id": "test_user",
            "use_autocomplete": True,
            "use_llm": True
        }
        
        # Mock autocomplete success but LLM failure
        mock_autocomplete_data = {
            "query": "fitness equipment",
            "suggestions": ["fitness equipment", "fitness equipment for home"],
            "total_suggestions": 2,
            "processing_time": 0.3,
            "success": True
        }
        
        with patch('enhanced_topic_decomposition_service.google_autocomplete_service') as mock_autocomplete, \
             patch('enhanced_topic_decomposition_service.llm_service') as mock_llm:
            
            mock_autocomplete.get_suggestions.return_value = mock_autocomplete_data
            mock_llm.generate_subtopics.side_effect = Exception("LLM service unavailable")
            
            response = client.post("/api/enhanced-topics/decompose", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Should still succeed with autocomplete-only approach
            assert data["success"] is True
            assert len(data["subtopics"]) > 0
            assert "autocomplete" in data["enhancement_methods"]
    
    @pytest.mark.asyncio
    async def test_enhanced_decomposition_performance_requirements(self, client):
        """Test performance requirements are met"""
        request_data = {
            "search_query": "fitness equipment",
            "user_id": "test_user",
            "max_subtopics": 6
        }
        
        with patch('enhanced_topic_decomposition_service.google_autocomplete_service') as mock_autocomplete, \
             patch('enhanced_topic_decomposition_service.llm_service') as mock_llm:
            
            # Mock fast responses
            mock_autocomplete.get_suggestions.return_value = {
                "query": "fitness equipment",
                "suggestions": ["fitness equipment"],
                "total_suggestions": 1,
                "processing_time": 0.1,
                "success": True
            }
            
            mock_llm.generate_subtopics.return_value = ["fitness equipment basics"]
            
            response = client.post("/api/enhanced-topics/decompose", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Performance requirements
            assert data["processing_time"] < 2.0  # Total decomposition < 2s
            assert data["autocomplete_data"]["processing_time"] < 0.5  # Autocomplete < 500ms
    
    @pytest.mark.asyncio
    async def test_enhanced_decomposition_rate_limiting(self, client):
        """Test rate limiting integration"""
        request_data = {
            "search_query": "fitness equipment",
            "user_id": "test_user"
        }
        
        # Simulate rate limiting
        with patch('enhanced_topic_routes.rate_limiter') as mock_limiter:
            mock_limiter.is_rate_limited.return_value = True
            
            response = client.post("/api/enhanced-topics/decompose", json=request_data)
            
            assert response.status_code == 429
            data = response.json()
            assert data["success"] is False
            assert "rate limit" in data["message"].lower()
    
    @pytest.mark.asyncio
    async def test_enhanced_decomposition_caching_integration(self, client):
        """Test caching integration"""
        request_data = {
            "search_query": "fitness equipment",
            "user_id": "test_user"
        }
        
        # First request
        with patch('enhanced_topic_decomposition_service.google_autocomplete_service') as mock_autocomplete, \
             patch('enhanced_topic_decomposition_service.llm_service') as mock_llm:
            
            mock_autocomplete.get_suggestions.return_value = {
                "query": "fitness equipment",
                "suggestions": ["fitness equipment"],
                "total_suggestions": 1,
                "processing_time": 0.5,
                "success": True
            }
            
            mock_llm.generate_subtopics.return_value = ["fitness equipment basics"]
            
            response1 = client.post("/api/enhanced-topics/decompose", json=request_data)
            assert response1.status_code == 200
        
        # Second request should use cache
        with patch('enhanced_topic_decomposition_service.cache') as mock_cache:
            mock_cache.get.return_value = {
                "subtopics": [{"title": "cached result", "source": "hybrid"}],
                "autocomplete_data": {"query": "fitness equipment", "suggestions": []},
                "processing_time": 0.1
            }
            
            response2 = client.post("/api/enhanced-topics/decompose", json=request_data)
            assert response2.status_code == 200
            
            # Should be faster due to caching
            data2 = response2.json()
            assert data2["processing_time"] < 0.5  # Much faster than first request
    
    @pytest.mark.asyncio
    async def test_enhanced_decomposition_error_handling_integration(self, client):
        """Test comprehensive error handling"""
        request_data = {
            "search_query": "fitness equipment",
            "user_id": "test_user"
        }
        
        # Test complete service failure
        with patch('enhanced_topic_decomposition_service.google_autocomplete_service') as mock_autocomplete, \
             patch('enhanced_topic_decomposition_service.llm_service') as mock_llm:
            
            mock_autocomplete.get_suggestions.side_effect = Exception("Network error")
            mock_llm.generate_subtopics.side_effect = Exception("LLM error")
            
            response = client.post("/api/enhanced-topics/decompose", json=request_data)
            
            # Should still return 200 with fallback subtopics
            assert response.status_code == 200
            data = response.json()
            
            # Should have some fallback results
            assert data["success"] is True
            assert len(data["subtopics"]) > 0

