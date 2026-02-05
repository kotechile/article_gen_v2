"""
Contract test for GET /api/research-topics/{id} endpoint
This test validates the API contract for retrieving research topics
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# This test should FAIL until the endpoint is implemented
class TestResearchTopicsGet:
    """Test GET /api/research-topics/{id} endpoint contract"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        # This will fail until the endpoint is implemented
        from backend.src.api.research_topics_routes import app
        return TestClient(app)
    
    @pytest.fixture
    def sample_research_topic(self):
        """Sample research topic data"""
        return {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "user_id": "987fcdeb-51a2-43d1-9f12-345678901234",
            "title": "Sustainable Fashion Trends",
            "description": "Research on emerging sustainable fashion trends and consumer behavior",
            "status": "active",
            "created_at": "2025-01-27T10:00:00Z",
            "updated_at": "2025-01-27T10:00:00Z",
            "version": 1
        }
    
    def test_get_research_topic_success(self, client, sample_research_topic):
        """Test successful research topic retrieval"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.get_by_id') as mock_get:
            mock_get.return_value = sample_research_topic
            
            response = client.get(
                f"/api/research-topics/{topic_id}",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == topic_id
            assert data["title"] == "Sustainable Fashion Trends"
            assert data["status"] == "active"
            assert data["version"] == 1
    
    def test_get_research_topic_with_dataflow(self, client, sample_research_topic):
        """Test research topic retrieval with complete dataflow"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.get_with_dataflow') as mock_get:
            mock_dataflow = {
                "research_topic": sample_research_topic,
                "subtopics": [
                    {"name": "sustainable fashion trends", "description": "General overview"},
                    {"name": "eco-friendly materials", "description": "Sustainable materials trends"}
                ],
                "trend_analyses": [
                    {
                        "id": "789e0123-e89b-12d3-a456-426614174002",
                        "subtopic_name": "eco-friendly materials",
                        "status": "completed"
                    }
                ],
                "content_ideas": [
                    {
                        "id": "abc12345-e89b-12d3-a456-426614174003",
                        "title": "Guide to Eco-Friendly Materials",
                        "content_type": "guide",
                        "idea_type": "evergreen"
                    }
                ]
            }
            mock_get.return_value = mock_dataflow
            
            response = client.get(
                f"/api/research-topics/{topic_id}?include_dataflow=true",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "research_topic" in data
            assert "subtopics" in data
            assert "trend_analyses" in data
            assert "content_ideas" in data
            assert len(data["subtopics"]) == 2
            assert len(data["trend_analyses"]) == 1
            assert len(data["content_ideas"]) == 1
    
    def test_get_research_topic_not_found(self, client):
        """Test research topic retrieval when topic doesn't exist"""
        topic_id = "00000000-0000-0000-0000-000000000000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.get_by_id') as mock_get:
            mock_get.return_value = None
            
            response = client.get(
                f"/api/research-topics/{topic_id}",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 404
            data = response.json()
            assert "error" in data
            assert "not found" in data["message"].lower()
    
    def test_get_research_topic_unauthorized(self, client):
        """Test research topic retrieval without authentication"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        response = client.get(f"/api/research-topics/{topic_id}")
        
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert "unauthorized" in data["message"].lower()
    
    def test_get_research_topic_forbidden(self, client, sample_research_topic):
        """Test research topic retrieval for another user's topic"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.get_by_id') as mock_get:
            from backend.src.services.research_topic_service import ForbiddenError
            mock_get.side_effect = ForbiddenError("Access denied to this research topic")
            
            response = client.get(
                f"/api/research-topics/{topic_id}",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 403
            data = response.json()
            assert "error" in data
            assert "forbidden" in data["error"].lower()
    
    def test_get_research_topic_invalid_uuid(self, client):
        """Test research topic retrieval with invalid UUID"""
        invalid_id = "invalid-uuid"
        
        response = client.get(
            f"/api/research-topics/{invalid_id}",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
    
    def test_get_research_topic_server_error(self, client):
        """Test research topic retrieval with server error"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.get_by_id') as mock_get:
            mock_get.side_effect = Exception("Database connection failed")
            
            response = client.get(
                f"/api/research-topics/{topic_id}",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "message" in data
    
    def test_get_research_topic_response_schema(self, client, sample_research_topic):
        """Test that response matches expected schema"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.get_by_id') as mock_get:
            mock_get.return_value = sample_research_topic
            
            response = client.get(
                f"/api/research-topics/{topic_id}",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate required fields are present
            required_fields = ["id", "user_id", "title", "status", "created_at", "updated_at", "version"]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
            
            # Validate field types
            assert isinstance(data["id"], str)
            assert isinstance(data["user_id"], str)
            assert isinstance(data["title"], str)
            assert isinstance(data["status"], str)
            assert isinstance(data["version"], int)
            assert data["version"] > 0
