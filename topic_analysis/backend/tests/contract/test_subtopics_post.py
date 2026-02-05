"""
Contract test for POST /api/research-topics/{id}/subtopics endpoint
This test validates the API contract for creating subtopics
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# This test should FAIL until the endpoint is implemented
class TestSubtopicsPost:
    """Test POST /api/research-topics/{id}/subtopics endpoint contract"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        # This will fail until the endpoint is implemented
        from backend.src.api.research_topics_routes import app
        return TestClient(app)
    
    @pytest.fixture
    def valid_subtopics_data(self):
        """Valid subtopics creation data"""
        return {
            "search_query": "sustainable fashion trends",
            "subtopics": [
                {
                    "name": "sustainable fashion trends",
                    "description": "General overview of sustainable fashion trends"
                },
                {
                    "name": "eco-friendly materials",
                    "description": "Trends in sustainable and eco-friendly fashion materials"
                },
                {
                    "name": "circular fashion",
                    "description": "Circular fashion economy and recycling trends"
                },
                {
                    "name": "sustainable fashion brands",
                    "description": "Leading sustainable fashion brands and their strategies"
                },
                {
                    "name": "consumer behavior",
                    "description": "How consumers are adopting sustainable fashion"
                }
            ]
        }
    
    @pytest.fixture
    def invalid_subtopics_data(self):
        """Invalid subtopics creation data"""
        return {
            "search_query": "",  # Empty search query should fail
            "subtopics": []  # Empty subtopics array should fail
        }
    
    def test_create_subtopics_success(self, client, valid_subtopics_data):
        """Test successful subtopics creation"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.topic_decomposition_service.TopicDecompositionService.create_subtopics') as mock_create:
            mock_subtopics = [
                {"name": "sustainable fashion trends", "description": "General overview of sustainable fashion trends"},
                {"name": "eco-friendly materials", "description": "Trends in sustainable and eco-friendly fashion materials"},
                {"name": "circular fashion", "description": "Circular fashion economy and recycling trends"},
                {"name": "sustainable fashion brands", "description": "Leading sustainable fashion brands and their strategies"},
                {"name": "consumer behavior", "description": "How consumers are adopting sustainable fashion"}
            ]
            mock_create.return_value = mock_subtopics
            
            response = client.post(
                f"/api/research-topics/{topic_id}/subtopics",
                json=valid_subtopics_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 201
            data = response.json()
            assert "data" in data
            assert len(data["data"]) == 5
            assert data["data"][0]["name"] == "sustainable fashion trends"
            assert data["data"][1]["name"] == "eco-friendly materials"
    
    def test_create_subtopics_validation_error(self, client, invalid_subtopics_data):
        """Test subtopics creation with validation errors"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        response = client.post(
            f"/api/research-topics/{topic_id}/subtopics",
            json=invalid_subtopics_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "message" in data
        assert "details" in data
    
    def test_create_subtopics_research_topic_not_found(self, client, valid_subtopics_data):
        """Test subtopics creation when research topic doesn't exist"""
        topic_id = "00000000-0000-0000-0000-000000000000"
        
        with patch('backend.src.services.topic_decomposition_service.TopicDecompositionService.create_subtopics') as mock_create:
            mock_create.return_value = None
            
            response = client.post(
                f"/api/research-topics/{topic_id}/subtopics",
                json=valid_subtopics_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 404
            data = response.json()
            assert "error" in data
            assert "not found" in data["message"].lower()
    
    def test_create_subtopics_unauthorized(self, client, valid_subtopics_data):
        """Test subtopics creation without authentication"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        response = client.post(
            f"/api/research-topics/{topic_id}/subtopics",
            json=valid_subtopics_data
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert "unauthorized" in data["message"].lower()
    
    def test_create_subtopics_forbidden(self, client, valid_subtopics_data):
        """Test subtopics creation for another user's research topic"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.topic_decomposition_service.TopicDecompositionService.create_subtopics') as mock_create:
            from backend.src.services.topic_decomposition_service import ForbiddenError
            mock_create.side_effect = ForbiddenError("Access denied to this research topic")
            
            response = client.post(
                f"/api/research-topics/{topic_id}/subtopics",
                json=valid_subtopics_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 403
            data = response.json()
            assert "error" in data
            assert "forbidden" in data["error"].lower()
    
    def test_create_subtopics_invalid_uuid(self, client, valid_subtopics_data):
        """Test subtopics creation with invalid UUID"""
        invalid_id = "invalid-uuid"
        
        response = client.post(
            f"/api/research-topics/{invalid_id}/subtopics",
            json=valid_subtopics_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
    
    def test_create_subtopics_server_error(self, client, valid_subtopics_data):
        """Test subtopics creation with server error"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.topic_decomposition_service.TopicDecompositionService.create_subtopics') as mock_create:
            mock_create.side_effect = Exception("Database connection failed")
            
            response = client.post(
                f"/api/research-topics/{topic_id}/subtopics",
                json=valid_subtopics_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "message" in data
    
    def test_create_subtopics_duplicate_decomposition(self, client, valid_subtopics_data):
        """Test subtopics creation with duplicate decomposition"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.topic_decomposition_service.TopicDecompositionService.create_subtopics') as mock_create:
            from backend.src.services.topic_decomposition_service import DuplicateDecompositionError
            mock_create.side_effect = DuplicateDecompositionError("Decomposition already exists for this research topic")
            
            response = client.post(
                f"/api/research-topics/{topic_id}/subtopics",
                json=valid_subtopics_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 409
            data = response.json()
            assert "error" in data
            assert "duplicate" in data["error"].lower()
    
    def test_create_subtopics_minimum_subtopics(self, client):
        """Test subtopics creation with minimum required subtopics"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        minimal_data = {
            "search_query": "test query",
            "subtopics": [
                {"name": "topic 1", "description": "Description 1"},
                {"name": "topic 2", "description": "Description 2"},
                {"name": "topic 3", "description": "Description 3"}
            ]
        }
        
        with patch('backend.src.services.topic_decomposition_service.TopicDecompositionService.create_subtopics') as mock_create:
            mock_create.return_value = minimal_data["subtopics"]
            
            response = client.post(
                f"/api/research-topics/{topic_id}/subtopics",
                json=minimal_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 201
            data = response.json()
            assert len(data["data"]) == 3
    
    def test_create_subtopics_original_topic_included(self, client, valid_subtopics_data):
        """Test that original topic is included as subtopic"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.topic_decomposition_service.TopicDecompositionService.create_subtopics') as mock_create:
            mock_create.return_value = valid_subtopics_data["subtopics"]
            
            response = client.post(
                f"/api/research-topics/{topic_id}/subtopics",
                json=valid_subtopics_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 201
            data = response.json()
            # Verify that the first subtopic is the original topic
            assert data["data"][0]["name"] == "sustainable fashion trends"
    
    def test_create_subtopics_response_schema(self, client, valid_subtopics_data):
        """Test that response matches expected schema"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.topic_decomposition_service.TopicDecompositionService.create_subtopics') as mock_create:
            mock_create.return_value = valid_subtopics_data["subtopics"]
            
            response = client.post(
                f"/api/research-topics/{topic_id}/subtopics",
                json=valid_subtopics_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 201
            data = response.json()
            
            # Validate response structure
            assert "data" in data
            assert isinstance(data["data"], list)
            assert len(data["data"]) > 0
            
            # Validate subtopic structure
            subtopic = data["data"][0]
            assert "name" in subtopic
            assert "description" in subtopic
            assert isinstance(subtopic["name"], str)
            assert isinstance(subtopic["description"], str)
