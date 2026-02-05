"""
Integration test for keyword research flow.

This test validates the complete keyword research workflow from API call to data processing.
The test must FAIL before implementation and PASS after implementation.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import json
import asyncio


class TestKeywordResearchFlow:
    """Integration test suite for keyword research workflow."""
    
    def setup_method(self):
        """Set up test client and mock data."""
        # This will be updated when the actual FastAPI app is created
        self.client = None  # Will be replaced with actual TestClient
        
        # Mock DataForSEO API responses
        self.mock_dataforseo_keywords_response = {
            "tasks": [
                {
                    "result": [
                        {
                            "keyword": "weight loss tips",
                            "search_volume": 50000,
                            "keyword_difficulty": 35,
                            "cpc": 2.50,
                            "competition": 0.65,
                            "competition_level": "MEDIUM",
                            "trends": {
                                "monthly_data": [
                                    {"month": "2024-01-01", "volume": 45000},
                                    {"month": "2024-02-01", "volume": 48000}
                                ],
                                "trend_direction": "RISING",
                                "trend_percentage": 15.5
                            },
                            "related_keywords": ["weight loss diet", "fat burning tips"],
                            "intent": "COMMERCIAL"
                        },
                        {
                            "keyword": "diet plans",
                            "search_volume": 30000,
                            "keyword_difficulty": 45,
                            "cpc": 1.80,
                            "competition": 0.55,
                            "competition_level": "MEDIUM",
                            "trends": {
                                "monthly_data": [
                                    {"month": "2024-01-01", "volume": 28000},
                                    {"month": "2024-02-01", "volume": 31000}
                                ],
                                "trend_direction": "RISING",
                                "trend_percentage": 8.2
                            },
                            "related_keywords": ["meal plans", "diet programs"],
                            "intent": "COMMERCIAL"
                        }
                    ]
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_complete_keyword_research_workflow(self):
        """Test complete keyword research workflow from API to response."""
        # Mock DataForSEO API client
        with patch('backend.src.dataforseo.api_client.DataForSEOClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock API responses
            mock_client.get_keyword_ideas.return_value = self.mock_dataforseo_keywords_response
            
            # Mock Supabase credentials
            with patch('backend.src.dataforseo.api_client.get_api_credentials') as mock_creds:
                mock_creds.return_value = {
                    "base_url": "https://api.dataforseo.com",
                    "key_value": "test_api_key"
                }
                
                # Test POST keyword research
                response = self.client.post(
                    "/api/v1/keyword-research/dataforseo",
                    json={
                        "seed_keywords": ["weight loss tips", "diet plans"],
                        "max_difficulty": 50,
                        "min_volume": 100,
                        "intent_types": ["COMMERCIAL", "TRANSACTIONAL"],
                        "max_results": 100
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert "data" in data
                assert "metadata" in data
                assert len(data["data"]) == 2
                assert data["data"][0]["keyword"] == "weight loss tips"
                assert data["metadata"]["total_keywords"] == 2
                assert data["metadata"]["filtered_keywords"] == 2
    
    @pytest.mark.asyncio
    async def test_keyword_research_filtering(self):
        """Test keyword filtering based on difficulty and volume."""
        with patch('backend.src.dataforseo.api_client.DataForSEOClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get_keyword_ideas.return_value = self.mock_dataforseo_keywords_response
            
            with patch('backend.src.dataforseo.api_client.get_api_credentials') as mock_creds:
                mock_creds.return_value = {
                    "base_url": "https://api.dataforseo.com",
                    "key_value": "test_api_key"
                }
                
                # Test with strict filtering
                response = self.client.post(
                    "/api/v1/keyword-research/dataforseo",
                    json={
                        "seed_keywords": ["weight loss tips"],
                        "max_difficulty": 30,  # Very low difficulty
                        "min_volume": 60000,   # Very high volume
                        "intent_types": ["COMMERCIAL"],
                        "max_results": 100
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                # Should filter out keywords that don't meet criteria
                assert data["metadata"]["filtered_keywords"] < data["metadata"]["total_keywords"]
    
    @pytest.mark.asyncio
    async def test_keyword_research_prioritization(self):
        """Test keyword prioritization workflow."""
        with patch('backend.src.dataforseo.api_client.DataForSEOClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get_keyword_ideas.return_value = self.mock_dataforseo_keywords_response
            
            with patch('backend.src.dataforseo.api_client.get_api_credentials') as mock_creds:
                mock_creds.return_value = {
                    "base_url": "https://api.dataforseo.com",
                    "key_value": "test_api_key"
                }
                
                # Test keyword prioritization
                response = self.client.post(
                    "/api/v1/keyword-research/dataforseo/prioritize",
                    json={
                        "keywords": [
                            {
                                "keyword": "weight loss tips",
                                "search_volume": 50000,
                                "cpc": 2.50,
                                "trends": {"trend_percentage": 15.5}
                            },
                            {
                                "keyword": "diet plans",
                                "search_volume": 30000,
                                "cpc": 1.80,
                                "trends": {"trend_percentage": 8.2}
                            }
                        ],
                        "priority_factors": {
                            "cpc_weight": 0.3,
                            "volume_weight": 0.4,
                            "trend_weight": 0.3
                        }
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert "prioritized_keywords" in data
                assert len(data["prioritized_keywords"]) == 2
                
                # Verify prioritization order
                prioritized = data["prioritized_keywords"]
                assert prioritized[0]["rank"] == 1
                assert prioritized[1]["rank"] == 2
                assert prioritized[0]["priority_score"] > prioritized[1]["priority_score"]
    
    @pytest.mark.asyncio
    async def test_keyword_research_with_caching(self):
        """Test keyword research with Redis caching."""
        # Mock Redis cache
        with patch('backend.src.dataforseo.cache_service.RedisCache') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            
            # First call - cache miss
            mock_redis_instance.get.return_value = None
            mock_redis_instance.set.return_value = True
            
            with patch('backend.src.dataforseo.api_client.DataForSEOClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                mock_client.get_keyword_ideas.return_value = self.mock_dataforseo_keywords_response
                
                with patch('backend.src.dataforseo.api_client.get_api_credentials') as mock_creds:
                    mock_creds.return_value = {
                        "base_url": "https://api.dataforseo.com",
                        "key_value": "test_api_key"
                    }
                    
                    response = self.client.post(
                        "/api/v1/keyword-research/dataforseo",
                        json={
                            "seed_keywords": ["weight loss tips"],
                            "max_difficulty": 50
                        }
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["success"] is True
                    
                    # Verify cache was set
                    mock_redis_instance.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_keyword_research_error_handling(self):
        """Test keyword research error handling and retry logic."""
        with patch('backend.src.dataforseo.api_client.DataForSEOClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock API error
            mock_client.get_keyword_ideas.side_effect = Exception("API quota exceeded")
            
            with patch('backend.src.dataforseo.api_client.get_api_credentials') as mock_creds:
                mock_creds.return_value = {
                    "base_url": "https://api.dataforseo.com",
                    "key_value": "test_api_key"
                }
                
                response = self.client.post(
                    "/api/v1/keyword-research/dataforseo",
                    json={
                        "seed_keywords": ["weight loss tips"],
                        "max_difficulty": 50
                    }
                )
                
                assert response.status_code == 500
                data = response.json()
                assert data["success"] is False
                assert data["error"] == "API_UNAVAILABLE"
    
    @pytest.mark.asyncio
    async def test_keyword_research_validation(self):
        """Test input validation for keyword research."""
        # Test missing required parameters
        response = self.client.post(
            "/api/v1/keyword-research/dataforseo",
            json={
                "max_difficulty": 50
            }
        )
        assert response.status_code == 400
        
        # Test invalid max_difficulty
        response = self.client.post(
            "/api/v1/keyword-research/dataforseo",
            json={
                "seed_keywords": ["weight loss tips"],
                "max_difficulty": 101
            }
        )
        assert response.status_code == 400
        
        # Test invalid intent_types
        response = self.client.post(
            "/api/v1/keyword-research/dataforseo",
            json={
                "seed_keywords": ["weight loss tips"],
                "intent_types": ["INVALID"]
            }
        )
        assert response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_keyword_research_performance(self):
        """Test keyword research performance requirements."""
        import time
        
        with patch('backend.src.dataforseo.api_client.DataForSEOClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get_keyword_ideas.return_value = self.mock_dataforseo_keywords_response
            
            with patch('backend.src.dataforseo.api_client.get_api_credentials') as mock_creds:
                mock_creds.return_value = {
                    "base_url": "https://api.dataforseo.com",
                    "key_value": "test_api_key"
                }
                
                start_time = time.time()
                response = self.client.post(
                    "/api/v1/keyword-research/dataforseo",
                    json={
                        "seed_keywords": ["weight loss tips"],
                        "max_difficulty": 50
                    }
                )
                end_time = time.time()
                
                assert response.status_code == 200
                # Performance requirement: < 5 seconds
                assert (end_time - start_time) < 5.0
    
    @pytest.mark.asyncio
    async def test_keyword_research_data_processing(self):
        """Test data processing and transformation."""
        with patch('backend.src.dataforseo.api_client.DataForSEOClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get_keyword_ideas.return_value = self.mock_dataforseo_keywords_response
            
            with patch('backend.src.dataforseo.api_client.get_api_credentials') as mock_creds:
                mock_creds.return_value = {
                    "base_url": "https://api.dataforseo.com",
                    "key_value": "test_api_key"
                }
                
                response = self.client.post(
                    "/api/v1/keyword-research/dataforseo",
                    json={
                        "seed_keywords": ["weight loss tips"],
                        "max_difficulty": 50
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Validate data transformation
                keyword_data = data["data"][0]
                assert "created_at" in keyword_data
                assert "updated_at" in keyword_data
                assert keyword_data["keyword"] == "weight loss tips"
                
                # Validate numeric ranges
                assert 0 <= keyword_data["keyword_difficulty"] <= 100
                assert 0 <= keyword_data["competition"] <= 1
                assert keyword_data["search_volume"] >= 0
                assert keyword_data["cpc"] >= 0
                
                # Validate enum values
                assert keyword_data["competition_level"] in ["LOW", "MEDIUM", "HIGH"]
                assert keyword_data["intent"] in ["INFORMATIONAL", "COMMERCIAL", "TRANSACTIONAL"]
