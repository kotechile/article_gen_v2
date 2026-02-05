"""
Contract tests for Trend Analysis API endpoints
These tests MUST fail before implementation - they define the expected API contract
"""

import pytest
import json
from fastapi.testclient import TestClient


class TestTrendAnalysisContract:
    """Test contract for trend analysis endpoints"""
    
    def setup_method(self):
        """Setup test client"""
        # This will fail until we implement the main app
        from src.main import app
        self.client = TestClient(app)
    
    def test_post_trend_analysis_contract(self):
        """Test POST /api/trends/analyze contract"""
        # This test will fail until we implement the endpoint
        payload = {
            "topics": ["home coffee roasting", "coffee equipment"],
            "affiliate_research_id": "123e4567-e89b-12d3-a456-426614174000"
        }
        
        response = self.client.post("/api/trends/analyze", json=payload)
        
        # Contract requirements
        assert response.status_code == 201
        data = response.json()
        
        # Required fields in response
        assert "id" in data
        assert "status" in data
        assert "created_at" in data
        
        # Status should be PENDING initially
        assert data["status"] == "PENDING"
        
        # ID should be UUID format
        import uuid
        assert uuid.UUID(data["id"])
    
    def test_get_trend_analysis_contract(self):
        """Test GET /api/trends/analysis/{id} contract"""
        # This test will fail until we implement the endpoint
        analysis_id = "123e4567-e89b-12d3-a456-426614174000"
        
        response = self.client.get(f"/api/trends/analysis/{analysis_id}")
        
        # Contract requirements
        assert response.status_code == 200
        data = response.json()
        
        # Required fields in response
        assert "id" in data
        assert "status" in data
        assert "opportunity_scores" in data
        assert "llm_forecast" in data
        assert "social_signals" in data
        assert "created_at" in data
        
        # Opportunity scores should be object with numeric values
        assert isinstance(data["opportunity_scores"], dict)
        for score in data["opportunity_scores"].values():
            assert isinstance(score, (int, float))
            assert 0 <= score <= 100
        
        # LLM forecast should have specific structure
        forecast = data["llm_forecast"]
        assert "forecast" in forecast
        assert isinstance(forecast["forecast"], list)
        
        # Social signals should have expected platforms
        signals = data["social_signals"]
        assert "reddit" in signals
        assert "twitter" in signals
        assert "tiktok" in signals
        assert isinstance(signals["reddit"], list)
        assert isinstance(signals["twitter"], list)
        assert isinstance(signals["tiktok"], list)
    
    def test_trend_analysis_validation_contract(self):
        """Test validation contract for trend analysis"""
        # Test missing required fields
        payload = {}
        response = self.client.post("/api/trends/analyze", json=payload)
        assert response.status_code == 422
        
        # Test empty topics array
        payload = {
            "topics": [],
            "affiliate_research_id": "123e4567-e89b-12d3-a456-426614174000"
        }
        response = self.client.post("/api/trends/analyze", json=payload)
        assert response.status_code == 422
        
        # Test too many topics
        payload = {
            "topics": [f"topic{i}" for i in range(11)],  # Max 10
            "affiliate_research_id": "123e4567-e89b-12d3-a456-426614174000"
        }
        response = self.client.post("/api/trends/analyze", json=payload)
        assert response.status_code == 422
        
        # Test invalid affiliate research ID format
        payload = {
            "topics": ["test topic"],
            "affiliate_research_id": "invalid-id"
        }
        response = self.client.post("/api/trends/analyze", json=payload)
        assert response.status_code == 422
    
    def test_trend_analysis_error_handling_contract(self):
        """Test error handling contract"""
        # Test 404 for non-existent analysis
        analysis_id = "00000000-0000-0000-0000-000000000000"
        response = self.client.get(f"/api/trends/analysis/{analysis_id}")
        assert response.status_code == 404
        
        # Test invalid UUID format
        response = self.client.get("/api/trends/analysis/invalid-id")
        assert response.status_code == 422
    
    def test_trend_analysis_authentication_contract(self):
        """Test authentication contract"""
        # Test unauthenticated request
        payload = {
            "topics": ["test topic"],
            "affiliate_research_id": "123e4567-e89b-12d3-a456-426614174000"
        }
        response = self.client.post("/api/trends/analyze", json=payload)
        # Should require authentication
        assert response.status_code == 401
    
    def test_trend_analysis_processing_states_contract(self):
        """Test processing states contract"""
        # Test that status transitions are handled correctly
        analysis_id = "123e4567-e89b-12d3-a456-426614174000"
        
        # Mock different status responses
        with patch('src.services.trend_service.TrendService.get_analysis') as mock_get:
            # Test PENDING state
            mock_get.return_value = {
                "id": analysis_id,
                "status": "PENDING",
                "opportunity_scores": {},
                "llm_forecast": {"forecast": []},
                "social_signals": {"reddit": [], "twitter": [], "tiktok": []},
                "created_at": "2025-10-02T10:00:00Z"
            }
            
            response = self.client.get(f"/api/trends/analysis/{analysis_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "PENDING"
            
            # Test PROCESSING state
            mock_get.return_value["status"] = "PROCESSING"
            response = self.client.get(f"/api/trends/analysis/{analysis_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "PROCESSING"
            
            # Test COMPLETED state
            mock_get.return_value["status"] = "COMPLETED"
            mock_get.return_value["opportunity_scores"] = {"topic1": 85.5}
            response = self.client.get(f"/api/trends/analysis/{analysis_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "COMPLETED"
            assert data["opportunity_scores"]["topic1"] == 85.5
