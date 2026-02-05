"""
Contract test for PUT /api/research-topics/{id} endpoint
This test validates the API contract for updating research topics
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# This test should FAIL until the endpoint is implemented
class TestResearchTopicsPut:
    """Test PUT /api/research-topics/{id} endpoint contract"""
    
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
    
    @pytest.fixture
    def valid_update_data(self):
        """Valid research topic update data"""
        return {
            "title": "Updated Sustainable Fashion Trends",
            "description": "Updated research on sustainable fashion trends",
            "status": "completed",
            "version": 1
        }
    
    @pytest.fixture
    def invalid_update_data(self):
        """Invalid research topic update data"""
        return {
            "title": "",  # Empty title should fail
            "description": "A" * 5001,  # Description too long
            "status": "invalid_status",  # Invalid status
            "version": -1  # Invalid version
        }
    
    def test_update_research_topic_success(self, client, sample_research_topic, valid_update_data):
        """Test successful research topic update"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.update') as mock_update:
            updated_topic = sample_research_topic.copy()
            updated_topic.update(valid_update_data)
            updated_topic["version"] = 2
            updated_topic["updated_at"] = "2025-01-27T11:00:00Z"
            mock_update.return_value = updated_topic
            
            response = client.put(
                f"/api/research-topics/{topic_id}",
                json=valid_update_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["title"] == "Updated Sustainable Fashion Trends"
            assert data["status"] == "completed"
            assert data["version"] == 2
            assert data["updated_at"] == "2025-01-27T11:00:00Z"
    
    def test_update_research_topic_validation_error(self, client, invalid_update_data):
        """Test research topic update with validation errors"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        response = client.put(
            f"/api/research-topics/{topic_id}",
            json=invalid_update_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "message" in data
        assert "details" in data
    
    def test_update_research_topic_not_found(self, client, valid_update_data):
        """Test research topic update when topic doesn't exist"""
        topic_id = "00000000-0000-0000-0000-000000000000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.update') as mock_update:
            mock_update.return_value = None
            
            response = client.put(
                f"/api/research-topics/{topic_id}",
                json=valid_update_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 404
            data = response.json()
            assert "error" in data
            assert "not found" in data["message"].lower()
    
    def test_update_research_topic_version_conflict(self, client, valid_update_data):
        """Test research topic update with version conflict"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.update') as mock_update:
            from backend.src.services.research_topic_service import VersionConflictError
            mock_update.side_effect = VersionConflictError("Version conflict", current_version=2, provided_version=1)
            
            response = client.put(
                f"/api/research-topics/{topic_id}",
                json=valid_update_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 409
            data = response.json()
            assert "error" in data
            assert "version_conflict" in data["error"]
            assert data["current_version"] == 2
            assert data["provided_version"] == 1
    
    def test_update_research_topic_unauthorized(self, client, valid_update_data):
        """Test research topic update without authentication"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        response = client.put(
            f"/api/research-topics/{topic_id}",
            json=valid_update_data
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert "unauthorized" in data["message"].lower()
    
    def test_update_research_topic_forbidden(self, client, valid_update_data):
        """Test research topic update for another user's topic"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.update') as mock_update:
            from backend.src.services.research_topic_service import ForbiddenError
            mock_update.side_effect = ForbiddenError("Access denied to this research topic")
            
            response = client.put(
                f"/api/research-topics/{topic_id}",
                json=valid_update_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 403
            data = response.json()
            assert "error" in data
            assert "forbidden" in data["error"].lower()
    
    def test_update_research_topic_duplicate_title(self, client, valid_update_data):
        """Test research topic update with duplicate title"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.update') as mock_update:
            from backend.src.services.research_topic_service import DuplicateTitleError
            mock_update.side_effect = DuplicateTitleError("Research topic with this title already exists")
            
            response = client.put(
                f"/api/research-topics/{topic_id}",
                json=valid_update_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 409
            data = response.json()
            assert "error" in data
            assert "duplicate" in data["error"].lower()
    
    def test_update_research_topic_invalid_uuid(self, client, valid_update_data):
        """Test research topic update with invalid UUID"""
        invalid_id = "invalid-uuid"
        
        response = client.put(
            f"/api/research-topics/{invalid_id}",
            json=valid_update_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
    
    def test_update_research_topic_server_error(self, client, valid_update_data):
        """Test research topic update with server error"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.update') as mock_update:
            mock_update.side_effect = Exception("Database connection failed")
            
            response = client.put(
                f"/api/research-topics/{topic_id}",
                json=valid_update_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "message" in data
    
    def test_update_research_topic_partial_update(self, client, sample_research_topic):
        """Test research topic partial update (only some fields)"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        partial_update = {
            "title": "Updated Title Only",
            "version": 1
        }
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.update') as mock_update:
            updated_topic = sample_research_topic.copy()
            updated_topic["title"] = "Updated Title Only"
            updated_topic["version"] = 2
            updated_topic["updated_at"] = "2025-01-27T11:00:00Z"
            mock_update.return_value = updated_topic
            
            response = client.put(
                f"/api/research-topics/{topic_id}",
                json=partial_update,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["title"] == "Updated Title Only"
            assert data["description"] == sample_research_topic["description"]  # Unchanged
            assert data["version"] == 2
    
    def test_update_research_topic_response_schema(self, client, sample_research_topic, valid_update_data):
        """Test that response matches expected schema"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.update') as mock_update:
            updated_topic = sample_research_topic.copy()
            updated_topic.update(valid_update_data)
            updated_topic["version"] = 2
            updated_topic["updated_at"] = "2025-01-27T11:00:00Z"
            mock_update.return_value = updated_topic
            
            response = client.put(
                f"/api/research-topics/{topic_id}",
                json=valid_update_data,
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
