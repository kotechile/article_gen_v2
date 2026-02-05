"""
Contract tests for Content Generation API endpoints
These tests MUST fail before implementation - they define the expected API contract
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


class TestContentGenerationContract:
    """Test contract for content generation endpoints"""
    
    def setup_method(self):
        """Setup test client"""
        # This will fail until we implement the main app
        from src.main import app
        self.client = TestClient(app)
    
    def test_post_content_generate_contract(self):
        """Test POST /api/content/generate contract"""
        # This test will fail until we implement the endpoint
        payload = {
            "trend_analysis_id": "123e4567-e89b-12d3-a456-426614174000",
            "keyword_data_id": "123e4567-e89b-12d3-a456-426614174001",
            "content_types": ["ARTICLE", "GUIDE", "REVIEW"],
            "max_ideas": 5
        }
        
        response = self.client.post("/api/content/generate", json=payload)
        
        # Contract requirements
        assert response.status_code == 201
        data = response.json()
        
        # Required fields in response
        assert "id" in data
        assert "content_ideas" in data
        assert "created_at" in data
        
        # Content ideas should be array
        assert isinstance(data["content_ideas"], list)
        assert len(data["content_ideas"]) <= 5
        
        # Each content idea should have required fields
        for idea in data["content_ideas"]:
            assert "id" in idea
            assert "title" in idea
            assert "content_type" in idea
            assert "angle" in idea
            assert "headline_score" in idea
            assert "priority_score" in idea
            assert "outline" in idea
            assert "seo_recommendations" in idea
            
            # Validate content type
            assert idea["content_type"] in ["ARTICLE", "GUIDE", "REVIEW", "TUTORIAL", "LISTICLE"]
            
            # Validate angle
            assert idea["angle"] in ["how-to", "vs", "listicle", "pain-point", "story"]
            
            # Validate scores
            assert 0 <= idea["headline_score"] <= 100
            assert 0 <= idea["priority_score"] <= 1
        
        # ID should be UUID format
        import uuid
        assert uuid.UUID(data["id"])
    
    def test_post_software_generate_contract(self):
        """Test POST /api/software/generate contract"""
        # This test will fail until we implement the endpoint
        payload = {
            "trend_analysis_id": "123e4567-e89b-12d3-a456-426614174000",
            "keyword_data_id": "123e4567-e89b-12d3-a456-426614174001",
            "software_types": ["CALCULATOR", "ANALYZER", "GENERATOR"],
            "max_solutions": 5
        }
        
        response = self.client.post("/api/software/generate", json=payload)
        
        # Contract requirements
        assert response.status_code == 201
        data = response.json()
        
        # Required fields in response
        assert "id" in data
        assert "software_solutions" in data
        assert "created_at" in data
        
        # Software solutions should be array
        assert isinstance(data["software_solutions"], list)
        assert len(data["software_solutions"]) <= 5
        
        # Each software solution should have required fields
        for solution in data["software_solutions"]:
            assert "id" in solution
            assert "name" in solution
            assert "description" in solution
            assert "software_type" in solution
            assert "complexity_score" in solution
            assert "priority_score" in solution
            assert "target_keywords" in solution
            assert "technical_requirements" in solution
            
            # Validate software type
            assert solution["software_type"] in ["CALCULATOR", "ANALYZER", "GENERATOR", "CONVERTER", "ESTIMATOR"]
            
            # Validate complexity score
            assert 1 <= solution["complexity_score"] <= 10
            
            # Validate priority score
            assert 0 <= solution["priority_score"] <= 1
            
            # Target keywords should be array
            assert isinstance(solution["target_keywords"], list)
        
        # ID should be UUID format
        import uuid
        assert uuid.UUID(data["id"])
    
    def test_content_generation_validation_contract(self):
        """Test validation contract for content generation"""
        # Test missing required fields
        payload = {}
        response = self.client.post("/api/content/generate", json=payload)
        assert response.status_code == 422
        
        # Test invalid content types
        payload = {
            "trend_analysis_id": "123e4567-e89b-12d3-a456-426614174000",
            "keyword_data_id": "123e4567-e89b-12d3-a456-426614174001",
            "content_types": ["INVALID_TYPE"]
        }
        response = self.client.post("/api/content/generate", json=payload)
        assert response.status_code == 422
        
        # Test max ideas too high
        payload = {
            "trend_analysis_id": "123e4567-e89b-12d3-a456-426614174000",
            "keyword_data_id": "123e4567-e89b-12d3-a456-426614174001",
            "max_ideas": 25  # Max 20
        }
        response = self.client.post("/api/content/generate", json=payload)
        assert response.status_code == 422
        
        # Test invalid UUID format
        payload = {
            "trend_analysis_id": "invalid-id",
            "keyword_data_id": "123e4567-e89b-12d3-a456-426614174001"
        }
        response = self.client.post("/api/content/generate", json=payload)
        assert response.status_code == 422
    
    def test_software_generation_validation_contract(self):
        """Test validation contract for software generation"""
        # Test missing required fields
        payload = {}
        response = self.client.post("/api/software/generate", json=payload)
        assert response.status_code == 422
        
        # Test invalid software types
        payload = {
            "trend_analysis_id": "123e4567-e89b-12d3-a456-426614174000",
            "keyword_data_id": "123e4567-e89b-12d3-a456-426614174001",
            "software_types": ["INVALID_TYPE"]
        }
        response = self.client.post("/api/software/generate", json=payload)
        assert response.status_code == 422
        
        # Test max solutions too high
        payload = {
            "trend_analysis_id": "123e4567-e89b-12d3-a456-426614174000",
            "keyword_data_id": "123e4567-e89b-12d3-a456-426614174001",
            "max_solutions": 15  # Max 10
        }
        response = self.client.post("/api/software/generate", json=payload)
        assert response.status_code == 422
    
    def test_content_generation_error_handling_contract(self):
        """Test error handling contract"""
        # Test 404 for non-existent content
        content_id = "00000000-0000-0000-0000-000000000000"
        response = self.client.get(f"/api/content/{content_id}")
        assert response.status_code == 404
        
        # Test invalid UUID format
        response = self.client.get("/api/content/invalid-id")
        assert response.status_code == 422
    
    def test_content_generation_authentication_contract(self):
        """Test authentication contract"""
        # Test unauthenticated content generation request
        payload = {
            "trend_analysis_id": "123e4567-e89b-12d3-a456-426614174000",
            "keyword_data_id": "123e4567-e89b-12d3-a456-426614174001"
        }
        response = self.client.post("/api/content/generate", json=payload)
        # Should require authentication
        assert response.status_code == 401
        
        # Test unauthenticated software generation request
        response = self.client.post("/api/software/generate", json=payload)
        # Should require authentication
        assert response.status_code == 401
    
    def test_content_generation_processing_states_contract(self):
        """Test processing states contract"""
        # Test that status transitions are handled correctly
        content_id = "123e4567-e89b-12d3-a456-426614174000"
        
        # Mock different status responses
        with patch('src.services.content_service.ContentService.get_content_idea') as mock_get:
            # Test DRAFT state
            mock_get.return_value = {
                "id": content_id,
                "title": "Test Article",
                "content_type": "ARTICLE",
                "angle": "how-to",
                "headline_score": 85.5,
                "priority_score": 0.8,
                "status": "DRAFT",
                "outline": {},
                "seo_recommendations": {}
            }
            
            response = self.client.get(f"/api/content/{content_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "DRAFT"
            
            # Test SCHEDULED state
            mock_get.return_value["status"] = "SCHEDULED"
            response = self.client.get(f"/api/content/{content_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "SCHEDULED"
            
            # Test PUBLISHED state
            mock_get.return_value["status"] = "PUBLISHED"
            response = self.client.get(f"/api/content/{content_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "PUBLISHED"
