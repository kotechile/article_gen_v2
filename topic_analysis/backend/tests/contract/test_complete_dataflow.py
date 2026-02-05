"""
Contract test for GET /api/research-topics/{id}/complete endpoint
This test validates the API contract for retrieving complete dataflow
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# This test should FAIL until the endpoint is implemented
class TestCompleteDataflow:
    """Test GET /api/research-topics/{id}/complete endpoint contract"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        # This will fail until the endpoint is implemented
        from backend.src.api.research_topics_routes import app
        return TestClient(app)
    
    @pytest.fixture
    def sample_complete_dataflow(self):
        """Sample complete dataflow data"""
        return {
            "research_topic": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "user_id": "987fcdeb-51a2-43d1-9f12-345678901234",
                "title": "Sustainable Fashion Trends",
                "description": "Research on emerging sustainable fashion trends and consumer behavior",
                "status": "active",
                "created_at": "2025-01-27T10:00:00Z",
                "updated_at": "2025-01-27T10:00:00Z",
                "version": 1
            },
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
            ],
            "trend_analyses": [
                {
                    "id": "789e0123-e89b-12d3-a456-426614174002",
                    "subtopic_name": "eco-friendly materials",
                    "analysis_name": "Eco-Friendly Materials Trend Analysis",
                    "status": "completed",
                    "trend_data": {
                        "search_volume": 12500,
                        "trend_score": 85
                    },
                    "created_at": "2025-01-27T10:05:00Z"
                },
                {
                    "id": "789e0123-e89b-12d3-a456-426614174003",
                    "subtopic_name": "circular fashion",
                    "analysis_name": "Circular Fashion Trend Analysis",
                    "status": "completed",
                    "trend_data": {
                        "search_volume": 8900,
                        "trend_score": 72
                    },
                    "created_at": "2025-01-27T10:10:00Z"
                }
            ],
            "content_ideas": [
                {
                    "id": "abc12345-e89b-12d3-a456-426614174003",
                    "title": "The Complete Guide to Eco-Friendly Fashion Materials",
                    "content_type": "guide",
                    "idea_type": "evergreen",
                    "status": "draft",
                    "primary_keyword": "eco-friendly fashion materials",
                    "created_at": "2025-01-27T10:15:00Z"
                },
                {
                    "id": "abc12345-e89b-12d3-a456-426614174004",
                    "title": "Circular Fashion: The Future of Sustainable Style",
                    "content_type": "article",
                    "idea_type": "trending",
                    "status": "draft",
                    "primary_keyword": "circular fashion",
                    "created_at": "2025-01-27T10:20:00Z"
                }
            ]
        }
    
    def test_get_complete_dataflow_success(self, client, sample_complete_dataflow):
        """Test successful complete dataflow retrieval"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.get_complete_dataflow') as mock_get:
            mock_get.return_value = sample_complete_dataflow
            
            response = client.get(
                f"/api/research-topics/{topic_id}/complete",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "research_topic" in data
            assert "subtopics" in data
            assert "trend_analyses" in data
            assert "content_ideas" in data
            
            # Validate research topic
            assert data["research_topic"]["id"] == topic_id
            assert data["research_topic"]["title"] == "Sustainable Fashion Trends"
            
            # Validate subtopics
            assert len(data["subtopics"]) == 5
            assert data["subtopics"][0]["name"] == "sustainable fashion trends"
            
            # Validate trend analyses
            assert len(data["trend_analyses"]) == 2
            assert data["trend_analyses"][0]["subtopic_name"] == "eco-friendly materials"
            
            # Validate content ideas
            assert len(data["content_ideas"]) == 2
            assert data["content_ideas"][0]["title"] == "The Complete Guide to Eco-Friendly Fashion Materials"
    
    def test_get_complete_dataflow_not_found(self, client):
        """Test complete dataflow retrieval when research topic doesn't exist"""
        topic_id = "00000000-0000-0000-0000-000000000000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.get_complete_dataflow') as mock_get:
            mock_get.return_value = None
            
            response = client.get(
                f"/api/research-topics/{topic_id}/complete",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 404
            data = response.json()
            assert "error" in data
            assert "not found" in data["message"].lower()
    
    def test_get_complete_dataflow_unauthorized(self, client):
        """Test complete dataflow retrieval without authentication"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        response = client.get(f"/api/research-topics/{topic_id}/complete")
        
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert "unauthorized" in data["message"].lower()
    
    def test_get_complete_dataflow_forbidden(self, client, sample_complete_dataflow):
        """Test complete dataflow retrieval for another user's research topic"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.get_complete_dataflow') as mock_get:
            from backend.src.services.research_topic_service import ForbiddenError
            mock_get.side_effect = ForbiddenError("Access denied to this research topic")
            
            response = client.get(
                f"/api/research-topics/{topic_id}/complete",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 403
            data = response.json()
            assert "error" in data
            assert "forbidden" in data["error"].lower()
    
    def test_get_complete_dataflow_invalid_uuid(self, client):
        """Test complete dataflow retrieval with invalid UUID"""
        invalid_id = "invalid-uuid"
        
        response = client.get(
            f"/api/research-topics/{invalid_id}/complete",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
    
    def test_get_complete_dataflow_server_error(self, client):
        """Test complete dataflow retrieval with server error"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.get_complete_dataflow') as mock_get:
            mock_get.side_effect = Exception("Database connection failed")
            
            response = client.get(
                f"/api/research-topics/{topic_id}/complete",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "message" in data
    
    def test_get_complete_dataflow_empty_data(self, client):
        """Test complete dataflow retrieval with empty related data"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        empty_dataflow = {
            "research_topic": {
                "id": topic_id,
                "user_id": "987fcdeb-51a2-43d1-9f12-345678901234",
                "title": "New Research Topic",
                "description": "A new research topic with no data yet",
                "status": "active",
                "created_at": "2025-01-27T10:00:00Z",
                "updated_at": "2025-01-27T10:00:00Z",
                "version": 1
            },
            "subtopics": [],
            "trend_analyses": [],
            "content_ideas": []
        }
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.get_complete_dataflow') as mock_get:
            mock_get.return_value = empty_dataflow
            
            response = client.get(
                f"/api/research-topics/{topic_id}/complete",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "research_topic" in data
            assert data["subtopics"] == []
            assert data["trend_analyses"] == []
            assert data["content_ideas"] == []
    
    def test_get_complete_dataflow_response_schema(self, client, sample_complete_dataflow):
        """Test that response matches expected schema"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.get_complete_dataflow') as mock_get:
            mock_get.return_value = sample_complete_dataflow
            
            response = client.get(
                f"/api/research-topics/{topic_id}/complete",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate required top-level fields
            required_fields = ["research_topic", "subtopics", "trend_analyses", "content_ideas"]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
            
            # Validate research_topic structure
            research_topic = data["research_topic"]
            assert "id" in research_topic
            assert "title" in research_topic
            assert "status" in research_topic
            
            # Validate subtopics structure
            assert isinstance(data["subtopics"], list)
            if data["subtopics"]:
                subtopic = data["subtopics"][0]
                assert "name" in subtopic
                assert "description" in subtopic
            
            # Validate trend_analyses structure
            assert isinstance(data["trend_analyses"], list)
            if data["trend_analyses"]:
                trend_analysis = data["trend_analyses"][0]
                assert "id" in trend_analysis
                assert "subtopic_name" in trend_analysis
                assert "status" in trend_analysis
            
            # Validate content_ideas structure
            assert isinstance(data["content_ideas"], list)
            if data["content_ideas"]:
                content_idea = data["content_ideas"][0]
                assert "id" in content_idea
                assert "title" in content_idea
                assert "content_type" in content_idea
                assert "idea_type" in content_idea
    
    def test_get_complete_dataflow_performance(self, client, sample_complete_dataflow):
        """Test complete dataflow retrieval performance"""
        topic_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.get_complete_dataflow') as mock_get:
            mock_get.return_value = sample_complete_dataflow
            
            import time
            start_time = time.time()
            
            response = client.get(
                f"/api/research-topics/{topic_id}/complete",
                headers={"Authorization": "Bearer test-token"}
            )
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            assert response.status_code == 200
            # Performance should be under 200ms as per requirements
            assert response_time < 200, f"Response time {response_time}ms exceeds 200ms limit"
