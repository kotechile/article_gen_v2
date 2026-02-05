"""
Contract test for POST /api/research-topics endpoint
This test validates the API contract for creating research topics
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# This test should FAIL until the endpoint is implemented
class TestResearchTopicsPost:
    """Test POST /api/research-topics endpoint contract"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        # This will fail until the endpoint is implemented
        from backend.src.api.research_topics_routes import app
        return TestClient(app)
    
    @pytest.fixture
    def valid_research_topic_data(self):
        """Valid research topic creation data"""
        return {
            "title": "Sustainable Fashion Trends",
            "description": "Research on emerging sustainable fashion trends and consumer behavior"
        }
    
    @pytest.fixture
    def invalid_research_topic_data(self):
        """Invalid research topic creation data"""
        return {
            "title": "",  # Empty title should fail
            "description": "A" * 5001  # Description too long should fail
        }
    
    def test_create_research_topic_success(self, client, valid_research_topic_data):
        """Test successful research topic creation"""
        with patch('backend.src.services.research_topic_service.ResearchTopicService.create') as mock_create:
            # Mock the service response
            mock_topic = {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "user_id": "987fcdeb-51a2-43d1-9f12-345678901234",
                "title": "Sustainable Fashion Trends",
                "description": "Research on emerging sustainable fashion trends and consumer behavior",
                "status": "active",
                "created_at": "2025-01-27T10:00:00Z",
                "updated_at": "2025-01-27T10:00:00Z",
                "version": 1
            }
            mock_create.return_value = mock_topic
            
            response = client.post(
                "/api/research-topics",
                json=valid_research_topic_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 201
            data = response.json()
            assert data["id"] == "123e4567-e89b-12d3-a456-426614174000"
            assert data["title"] == "Sustainable Fashion Trends"
            assert data["status"] == "active"
            assert data["version"] == 1
            assert "created_at" in data
            assert "updated_at" in data
    
    def test_create_research_topic_validation_error(self, client, invalid_research_topic_data):
        """Test research topic creation with validation errors"""
        response = client.post(
            "/api/research-topics",
            json=invalid_research_topic_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "message" in data
        assert "details" in data
    
    def test_create_research_topic_unauthorized(self, client, valid_research_topic_data):
        """Test research topic creation without authentication"""
        response = client.post(
            "/api/research-topics",
            json=valid_research_topic_data
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert "message" in data
    
    def test_create_research_topic_duplicate_title(self, client, valid_research_topic_data):
        """Test research topic creation with duplicate title"""
        with patch('backend.src.services.research_topic_service.ResearchTopicService.create') as mock_create:
            from backend.src.services.research_topic_service import DuplicateTitleError
            mock_create.side_effect = DuplicateTitleError("Research topic with this title already exists")
            
            response = client.post(
                "/api/research-topics",
                json=valid_research_topic_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 409
            data = response.json()
            assert "error" in data
            assert "duplicate" in data["error"].lower()
    
    def test_create_research_topic_server_error(self, client, valid_research_topic_data):
        """Test research topic creation with server error"""
        with patch('backend.src.services.research_topic_service.ResearchTopicService.create') as mock_create:
            mock_create.side_effect = Exception("Database connection failed")
            
            response = client.post(
                "/api/research-topics",
                json=valid_research_topic_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "message" in data
    
    def test_create_research_topic_missing_required_fields(self, client):
        """Test research topic creation with missing required fields"""
        incomplete_data = {
            "description": "Missing title"
        }
        
        response = client.post(
            "/api/research-topics",
            json=incomplete_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "title" in data.get("details", {})
    
    def test_create_research_topic_response_schema(self, client, valid_research_topic_data):
        """Test that response matches expected schema"""
        with patch('backend.src.services.research_topic_service.ResearchTopicService.create') as mock_create:
            mock_topic = {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "user_id": "987fcdeb-51a2-43d1-9f12-345678901234",
                "title": "Sustainable Fashion Trends",
                "description": "Research on emerging sustainable fashion trends and consumer behavior",
                "status": "active",
                "created_at": "2025-01-27T10:00:00Z",
                "updated_at": "2025-01-27T10:00:00Z",
                "version": 1
            }
            mock_create.return_value = mock_topic
            
            response = client.post(
                "/api/research-topics",
                json=valid_research_topic_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 201
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
