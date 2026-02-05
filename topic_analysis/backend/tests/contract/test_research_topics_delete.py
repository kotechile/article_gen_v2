"""
Contract test for DELETE /api/research-topics/{id} endpoint
This test validates the API contract for deleting research topics
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# This test should FAIL until the endpoint is implemented
class TestResearchTopicsDelete:
    """Test DELETE /api/research-topics/{id} endpoint contract"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        # This will fail until the endpoint is implemented
        from backend.src.api.research_topics_routes import app
        return TestClient(app)
    
    def test_delete_research_topic_success(self, client):
        """Test successful research topic deletion"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.delete') as mock_delete:
            mock_delete.return_value = True
            
            response = client.delete(
                f"/api/research-topics/{topic_id}",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 204
            assert response.content == b""  # No content for successful deletion
    
    def test_delete_research_topic_not_found(self, client):
        """Test research topic deletion when topic doesn't exist"""
        topic_id = "00000000-0000-0000-0000-000000000000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.delete') as mock_delete:
            mock_delete.return_value = False
            
            response = client.delete(
                f"/api/research-topics/{topic_id}",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 404
            data = response.json()
            assert "error" in data
            assert "not found" in data["message"].lower()
    
    def test_delete_research_topic_unauthorized(self, client):
        """Test research topic deletion without authentication"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        response = client.delete(f"/api/research-topics/{topic_id}")
        
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert "unauthorized" in data["message"].lower()
    
    def test_delete_research_topic_forbidden(self, client):
        """Test research topic deletion for another user's topic"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.delete') as mock_delete:
            from backend.src.services.research_topic_service import ForbiddenError
            mock_delete.side_effect = ForbiddenError("Access denied to this research topic")
            
            response = client.delete(
                f"/api/research-topics/{topic_id}",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 403
            data = response.json()
            assert "error" in data
            assert "forbidden" in data["error"].lower()
    
    def test_delete_research_topic_invalid_uuid(self, client):
        """Test research topic deletion with invalid UUID"""
        invalid_id = "invalid-uuid"
        
        response = client.delete(
            f"/api/research-topics/{invalid_id}",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
    
    def test_delete_research_topic_server_error(self, client):
        """Test research topic deletion with server error"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.delete') as mock_delete:
            mock_delete.side_effect = Exception("Database connection failed")
            
            response = client.delete(
                f"/api/research-topics/{topic_id}",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "message" in data
    
    def test_delete_research_topic_cascade_behavior(self, client):
        """Test that deletion cascades to related data"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.delete') as mock_delete:
            # Mock that deletion affects related data
            mock_delete.return_value = True
            
            response = client.delete(
                f"/api/research-topics/{topic_id}",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 204
            # Verify that the service was called with cascade=True
            mock_delete.assert_called_once_with(topic_id, cascade=True)
    
    def test_delete_research_topic_with_dependencies(self, client):
        """Test research topic deletion when it has dependencies"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.delete') as mock_delete:
            from backend.src.services.research_topic_service import DependencyError
            mock_delete.side_effect = DependencyError("Cannot delete research topic with active subtopics")
            
            response = client.delete(
                f"/api/research-topics/{topic_id}",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 409
            data = response.json()
            assert "error" in data
            assert "dependency" in data["error"].lower()
    
    def test_delete_research_topic_confirmation_required(self, client):
        """Test research topic deletion with confirmation parameter"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.delete') as mock_delete:
            mock_delete.return_value = True
            
            response = client.delete(
                f"/api/research-topics/{topic_id}?confirm=true",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 204
            # Verify that the service was called with confirmation
            mock_delete.assert_called_once_with(topic_id, cascade=True, confirm=True)
    
    def test_delete_research_topic_without_confirmation(self, client):
        """Test research topic deletion without confirmation"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.delete') as mock_delete:
            from backend.src.services.research_topic_service import ConfirmationRequiredError
            mock_delete.side_effect = ConfirmationRequiredError("Confirmation required for deletion")
            
            response = client.delete(
                f"/api/research-topics/{topic_id}",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 400
            data = response.json()
            assert "error" in data
            assert "confirmation" in data["error"].lower()
    
    def test_delete_research_topic_soft_delete(self, client):
        """Test research topic soft deletion (archiving)"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.soft_delete') as mock_soft_delete:
            mock_soft_delete.return_value = True
            
            response = client.delete(
                f"/api/research-topics/{topic_id}?soft=true",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 204
            # Verify that soft delete was called instead of hard delete
            mock_soft_delete.assert_called_once_with(topic_id)
