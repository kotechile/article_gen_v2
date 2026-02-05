"""
Contract test for POST /api/v1/keyword-research/dataforseo/prioritize endpoint.

This test validates the API contract for keyword prioritization.
The test must FAIL before implementation and PASS after implementation.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json


class TestKeywordResearchPrioritize:
    """Test suite for POST /api/v1/keyword-research/dataforseo/prioritize endpoint."""
    
    def setup_method(self):
        """Set up test client and mock data."""
        # This will be updated when the actual FastAPI app is created
        self.client = None  # Will be replaced with actual TestClient
        self.base_url = "/api/v1/keyword-research/dataforseo/prioritize"
        
        # Mock response data based on contract
        self.mock_prioritized_data = {
            "success": True,
            "prioritized_keywords": [
                {
                    "keyword": "weight loss tips",
                    "search_volume": 50000,
                    "cpc": 2.50,
                    "trends": {"trend_percentage": 15.5},
                    "priority_score": 85.5,
                    "rank": 1
                },
                {
                    "keyword": "diet plans",
                    "search_volume": 30000,
                    "cpc": 1.80,
                    "trends": {"trend_percentage": 8.2},
                    "priority_score": 72.3,
                    "rank": 2
                }
            ]
        }
    
    def test_prioritize_keywords_success(self):
        """Test successful keyword prioritization."""
        # This test will FAIL until the endpoint is implemented
        with patch('backend.src.api.keyword_research.get_keyword_research_service') as mock_service:
            mock_service.return_value.prioritize_keywords.return_value = self.mock_prioritized_data
            
            response = self.client.post(
                self.base_url,
                json={
                    "keywords": [
                        {
                            "keyword": "weight loss tips",
                            "search_volume": 50000,
                            "cpc": 2.50,
                            "trends": {"trend_percentage": 15.5}
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
            assert len(data["prioritized_keywords"]) == 1
            assert data["prioritized_keywords"][0]["keyword"] == "weight loss tips"
            assert data["prioritized_keywords"][0]["priority_score"] == 85.5
            assert data["prioritized_keywords"][0]["rank"] == 1
    
    def test_prioritize_keywords_missing_keywords(self):
        """Test error handling for missing keywords parameter."""
        response = self.client.post(
            self.base_url,
            json={
                "priority_factors": {
                    "cpc_weight": 0.3,
                    "volume_weight": 0.4,
                    "trend_weight": 0.3
                }
            }
        )
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        assert "message" in data
    
    def test_prioritize_keywords_empty_keywords(self):
        """Test error handling for empty keywords array."""
        response = self.client.post(
            self.base_url,
            json={
                "keywords": [],
                "priority_factors": {
                    "cpc_weight": 0.3,
                    "volume_weight": 0.4,
                    "trend_weight": 0.3
                }
            }
        )
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "error" in data
    
    def test_prioritize_keywords_missing_priority_factors(self):
        """Test error handling for missing priority_factors parameter."""
        response = self.client.post(
            self.base_url,
            json={
                "keywords": [
                    {
                        "keyword": "weight loss tips",
                        "search_volume": 50000,
                        "cpc": 2.50
                    }
                ]
            }
        )
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "error" in data
    
    def test_prioritize_keywords_invalid_priority_factors(self):
        """Test error handling for invalid priority_factors values."""
        # Invalid weight values
        response = self.client.post(
            self.base_url,
            json={
                "keywords": [
                    {
                        "keyword": "weight loss tips",
                        "search_volume": 50000,
                        "cpc": 2.50
                    }
                ],
                "priority_factors": {
                    "cpc_weight": 0.5,
                    "volume_weight": 0.5,
                    "trend_weight": 0.5  # Sum > 1
                }
            }
        )
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        
        # Negative weight values
        response = self.client.post(
            self.base_url,
            json={
                "keywords": [
                    {
                        "keyword": "weight loss tips",
                        "search_volume": 50000,
                        "cpc": 2.50
                    }
                ],
                "priority_factors": {
                    "cpc_weight": -0.1,
                    "volume_weight": 0.4,
                    "trend_weight": 0.3
                }
            }
        )
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "error" in data
    
    def test_prioritize_keywords_invalid_keyword_data(self):
        """Test error handling for invalid keyword data."""
        response = self.client.post(
            self.base_url,
            json={
                "keywords": [
                    {
                        "keyword": "",  # Empty keyword
                        "search_volume": 50000,
                        "cpc": 2.50
                    }
                ],
                "priority_factors": {
                    "cpc_weight": 0.3,
                    "volume_weight": 0.4,
                    "trend_weight": 0.3
                }
            }
        )
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "error" in data
    
    def test_prioritize_keywords_default_priority_factors(self):
        """Test default values for priority_factors parameter."""
        with patch('backend.src.api.keyword_research.get_keyword_research_service') as mock_service:
            mock_service.return_value.prioritize_keywords.return_value = self.mock_prioritized_data
            
            response = self.client.post(
                self.base_url,
                json={
                    "keywords": [
                        {
                            "keyword": "weight loss tips",
                            "search_volume": 50000,
                            "cpc": 2.50
                        }
                    ]
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "prioritized_keywords" in data
    
    def test_prioritize_keywords_rate_limit(self):
        """Test rate limit error handling."""
        with patch('backend.src.api.keyword_research.get_keyword_research_service') as mock_service:
            mock_service.return_value.prioritize_keywords.side_effect = Exception("Rate limit exceeded")
            
            response = self.client.post(
                self.base_url,
                json={
                    "keywords": [
                        {
                            "keyword": "weight loss tips",
                            "search_volume": 50000,
                            "cpc": 2.50
                        }
                    ],
                    "priority_factors": {
                        "cpc_weight": 0.3,
                        "volume_weight": 0.4,
                        "trend_weight": 0.3
                    }
                }
            )
            
            assert response.status_code == 429
            data = response.json()
            assert data["success"] is False
            assert data["error"] == "RATE_LIMIT_EXCEEDED"
    
    def test_prioritize_keywords_response_schema(self):
        """Test response schema validation."""
        with patch('backend.src.api.keyword_research.get_keyword_research_service') as mock_service:
            mock_service.return_value.prioritize_keywords.return_value = self.mock_prioritized_data
            
            response = self.client.post(
                self.base_url,
                json={
                    "keywords": [
                        {
                            "keyword": "weight loss tips",
                            "search_volume": 50000,
                            "cpc": 2.50,
                            "trends": {"trend_percentage": 15.5}
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
            
            # Validate response schema
            assert "success" in data
            assert "prioritized_keywords" in data
            
            # Validate prioritized_keywords structure
            assert len(data["prioritized_keywords"]) == 2
            for keyword in data["prioritized_keywords"]:
                assert "keyword" in keyword
                assert "priority_score" in keyword
                assert "rank" in keyword
                
                # Validate numeric values
                assert isinstance(keyword["priority_score"], (int, float))
                assert 0 <= keyword["priority_score"] <= 100
                assert isinstance(keyword["rank"], int)
                assert keyword["rank"] > 0
    
    def test_prioritize_keywords_api_unavailable(self):
        """Test API unavailable error handling."""
        with patch('backend.src.api.keyword_research.get_keyword_research_service') as mock_service:
            mock_service.return_value.prioritize_keywords.side_effect = Exception("API unavailable")
            
            response = self.client.post(
                self.base_url,
                json={
                    "keywords": [
                        {
                            "keyword": "weight loss tips",
                            "search_volume": 50000,
                            "cpc": 2.50
                        }
                    ],
                    "priority_factors": {
                        "cpc_weight": 0.3,
                        "volume_weight": 0.4,
                        "trend_weight": 0.3
                    }
                }
            )
            
            assert response.status_code == 500
            data = response.json()
            assert data["success"] is False
            assert data["error"] == "API_UNAVAILABLE"
