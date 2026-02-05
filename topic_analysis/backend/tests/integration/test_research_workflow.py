"""
Integration test for complete research workflow
This test validates the end-to-end research workflow with dataflow persistence using Supabase Client/SDK
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import uuid
from supabase import create_client, Client

# This test should FAIL until the complete workflow is implemented
class TestResearchWorkflow:
    """Test complete research workflow integration"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        # This will fail until the endpoints are implemented
        from backend.src.api.research_topics_routes import app
        return TestClient(app)
    
    @pytest.fixture
    def test_user_id(self):
        """Test user ID"""
        return str(uuid.uuid4())
    
    @pytest.fixture
    def auth_headers(self, test_user_id):
        """Authentication headers"""
        return {"Authorization": f"Bearer test-token-{test_user_id}"}
    
    def test_complete_research_workflow(self, client, test_user_id, auth_headers):
        """Test complete research workflow from start to finish"""
        
        # Step 1: Create research topic
        research_topic_data = {
            "title": "AI in Healthcare",
            "description": "Research on artificial intelligence applications in healthcare"
        }
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.create') as mock_create_topic:
            mock_topic = {
                "id": str(uuid.uuid4()),
                "user_id": test_user_id,
                "title": "AI in Healthcare",
                "description": "Research on artificial intelligence applications in healthcare",
                "status": "active",
                "created_at": "2025-01-27T10:00:00Z",
                "updated_at": "2025-01-27T10:00:00Z",
                "version": 1
            }
            mock_create_topic.return_value = mock_topic
            
            response = client.post(
                "/api/research-topics",
                json=research_topic_data,
                headers=auth_headers
            )
            
            assert response.status_code == 201
            topic_data = response.json()
            topic_id = topic_data["id"]
            assert topic_data["title"] == "AI in Healthcare"
            assert topic_data["status"] == "active"
        
        # Step 2: Create subtopics
        subtopics_data = {
            "search_query": "AI in healthcare",
            "subtopics": [
                {
                    "name": "AI in healthcare",
                    "description": "General overview of AI applications in healthcare"
                },
                {
                    "name": "medical imaging AI",
                    "description": "AI applications in medical imaging and diagnostics"
                },
                {
                    "name": "drug discovery AI",
                    "description": "AI-powered drug discovery and development"
                },
                {
                    "name": "patient care AI",
                    "description": "AI applications in patient care and monitoring"
                },
                {
                    "name": "healthcare data AI",
                    "description": "AI for healthcare data analysis and insights"
                }
            ]
        }
        
        with patch('backend.src.services.topic_decomposition_service.TopicDecompositionService.create_subtopics') as mock_create_subtopics:
            mock_subtopics = subtopics_data["subtopics"]
            mock_create_subtopics.return_value = mock_subtopics
            
            response = client.post(
                f"/api/research-topics/{topic_id}/subtopics",
                json=subtopics_data,
                headers=auth_headers
            )
            
            assert response.status_code == 201
            subtopics_response = response.json()
            assert len(subtopics_response["data"]) == 5
            assert subtopics_response["data"][0]["name"] == "AI in healthcare"
        
        # Step 3: Perform trend analysis on subtopics
        trend_analysis_data = {
            "topic_decomposition_id": str(uuid.uuid4()),
            "subtopic_name": "medical imaging AI",
            "analysis_name": "Medical Imaging AI Trend Analysis",
            "description": "Analysis of AI trends in medical imaging",
            "keywords": ["medical imaging AI", "radiology AI", "diagnostic AI"],
            "timeframe": "12m",
            "geo": "US"
        }
        
        with patch('backend.src.services.trend_analysis_service.TrendAnalysisService.create') as mock_create_analysis:
            mock_analysis = {
                "id": str(uuid.uuid4()),
                "user_id": test_user_id,
                "topic_decomposition_id": trend_analysis_data["topic_decomposition_id"],
                "subtopic_name": "medical imaging AI",
                "analysis_name": "Medical Imaging AI Trend Analysis",
                "status": "completed",
                "trend_data": {
                    "search_volume": 15000,
                    "trend_score": 88
                },
                "created_at": "2025-01-27T10:05:00Z"
            }
            mock_create_analysis.return_value = mock_analysis
            
            response = client.post(
                "/api/trend-analyses",
                json=trend_analysis_data,
                headers=auth_headers
            )
            
            assert response.status_code == 201
            analysis_data = response.json()
            assert analysis_data["subtopic_name"] == "medical imaging AI"
            assert analysis_data["status"] == "completed"
        
        # Step 4: Generate content ideas from trend analysis
        content_idea_data = {
            "trend_analysis_id": analysis_data["id"],
            "research_topic_id": topic_id,
            "title": "The Future of Medical Imaging AI",
            "description": "Comprehensive guide to AI applications in medical imaging",
            "content_type": "guide",
            "idea_type": "evergreen",
            "primary_keyword": "medical imaging AI",
            "secondary_keywords": ["radiology AI", "diagnostic AI"],
            "target_audience": "healthcare professionals"
        }
        
        with patch('backend.src.services.content_idea_service.ContentIdeaService.create') as mock_create_content:
            mock_content = {
                "id": str(uuid.uuid4()),
                "user_id": test_user_id,
                "trend_analysis_id": analysis_data["id"],
                "research_topic_id": topic_id,
                "title": "The Future of Medical Imaging AI",
                "content_type": "guide",
                "idea_type": "evergreen",
                "status": "draft",
                "primary_keyword": "medical imaging AI",
                "created_at": "2025-01-27T10:10:00Z"
            }
            mock_create_content.return_value = mock_content
            
            response = client.post(
                "/api/content-ideas",
                json=content_idea_data,
                headers=auth_headers
            )
            
            assert response.status_code == 201
            content_data = response.json()
            assert content_data["title"] == "The Future of Medical Imaging AI"
            assert content_data["idea_type"] == "evergreen"
        
        # Step 5: Retrieve complete dataflow
        with patch('backend.src.services.research_topic_service.ResearchTopicService.get_complete_dataflow') as mock_get_dataflow:
            mock_dataflow = {
                "research_topic": mock_topic,
                "subtopics": mock_subtopics,
                "trend_analyses": [mock_analysis],
                "content_ideas": [mock_content]
            }
            mock_get_dataflow.return_value = mock_dataflow
            
            response = client.get(
                f"/api/research-topics/{topic_id}/complete",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            dataflow_data = response.json()
            
            # Validate complete dataflow structure
            assert "research_topic" in dataflow_data
            assert "subtopics" in dataflow_data
            assert "trend_analyses" in dataflow_data
            assert "content_ideas" in dataflow_data
            
            # Validate relationships
            assert dataflow_data["research_topic"]["id"] == topic_id
            assert len(dataflow_data["subtopics"]) == 5
            assert len(dataflow_data["trend_analyses"]) == 1
            assert len(dataflow_data["content_ideas"]) == 1
            
            # Validate data integrity
            trend_analysis = dataflow_data["trend_analyses"][0]
            content_idea = dataflow_data["content_ideas"][0]
            
            assert trend_analysis["subtopic_name"] in [sub["name"] for sub in dataflow_data["subtopics"]]
            assert content_idea["trend_analysis_id"] == trend_analysis["id"]
            assert content_idea["research_topic_id"] == topic_id
    
    def test_research_workflow_with_multiple_analyses(self, client, test_user_id, auth_headers):
        """Test research workflow with multiple trend analyses"""
        
        # Create research topic
        with patch('backend.src.services.research_topic_service.ResearchTopicService.create') as mock_create_topic:
            mock_topic = {
                "id": str(uuid.uuid4()),
                "user_id": test_user_id,
                "title": "Sustainable Energy",
                "description": "Research on sustainable energy solutions",
                "status": "active",
                "created_at": "2025-01-27T10:00:00Z",
                "updated_at": "2025-01-27T10:00:00Z",
                "version": 1
            }
            mock_create_topic.return_value = mock_topic
            
            response = client.post(
                "/api/research-topics",
                json={"title": "Sustainable Energy", "description": "Research on sustainable energy solutions"},
                headers=auth_headers
            )
            
            assert response.status_code == 201
            topic_id = response.json()["id"]
        
        # Create subtopics
        with patch('backend.src.services.topic_decomposition_service.TopicDecompositionService.create_subtopics') as mock_create_subtopics:
            mock_subtopics = [
                {"name": "solar energy", "description": "Solar power technologies"},
                {"name": "wind energy", "description": "Wind power systems"},
                {"name": "battery storage", "description": "Energy storage solutions"}
            ]
            mock_create_subtopics.return_value = mock_subtopics
            
            response = client.post(
                f"/api/research-topics/{topic_id}/subtopics",
                json={"search_query": "sustainable energy", "subtopics": mock_subtopics},
                headers=auth_headers
            )
            
            assert response.status_code == 201
        
        # Create multiple trend analyses
        analyses = []
        for i, subtopic in enumerate(mock_subtopics):
            with patch('backend.src.services.trend_analysis_service.TrendAnalysisService.create') as mock_create_analysis:
                mock_analysis = {
                    "id": str(uuid.uuid4()),
                    "user_id": test_user_id,
                    "subtopic_name": subtopic["name"],
                    "analysis_name": f"{subtopic['name']} Trend Analysis",
                    "status": "completed",
                    "trend_data": {"search_volume": 10000 + i * 1000, "trend_score": 80 + i * 5},
                    "created_at": f"2025-01-27T10:0{i+5}:00Z"
                }
                mock_create_analysis.return_value = mock_analysis
                analyses.append(mock_analysis)
                
                response = client.post(
                    "/api/trend-analyses",
                    json={
                        "topic_decomposition_id": str(uuid.uuid4()),
                        "subtopic_name": subtopic["name"],
                        "analysis_name": f"{subtopic['name']} Trend Analysis",
                        "keywords": [subtopic["name"]],
                        "timeframe": "12m"
                    },
                    headers=auth_headers
                )
                
                assert response.status_code == 201
        
        # Create content ideas for each analysis
        content_ideas = []
        for i, analysis in enumerate(analyses):
            with patch('backend.src.services.content_idea_service.ContentIdeaService.create') as mock_create_content:
                mock_content = {
                    "id": str(uuid.uuid4()),
                    "user_id": test_user_id,
                    "trend_analysis_id": analysis["id"],
                    "research_topic_id": topic_id,
                    "title": f"Guide to {analysis['subtopic_name']}",
                    "content_type": "guide",
                    "idea_type": "evergreen",
                    "status": "draft",
                    "primary_keyword": analysis["subtopic_name"],
                    "created_at": f"2025-01-27T10:1{i}:00Z"
                }
                mock_create_content.return_value = mock_content
                content_ideas.append(mock_content)
                
                response = client.post(
                    "/api/content-ideas",
                    json={
                        "trend_analysis_id": analysis["id"],
                        "research_topic_id": topic_id,
                        "title": f"Guide to {analysis['subtopic_name']}",
                        "content_type": "guide",
                        "idea_type": "evergreen",
                        "primary_keyword": analysis["subtopic_name"]
                    },
                    headers=auth_headers
                )
                
                assert response.status_code == 201
        
        # Retrieve complete dataflow
        with patch('backend.src.services.research_topic_service.ResearchTopicService.get_complete_dataflow') as mock_get_dataflow:
            mock_dataflow = {
                "research_topic": mock_topic,
                "subtopics": mock_subtopics,
                "trend_analyses": analyses,
                "content_ideas": content_ideas
            }
            mock_get_dataflow.return_value = mock_dataflow
            
            response = client.get(
                f"/api/research-topics/{topic_id}/complete",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            dataflow_data = response.json()
            
            # Validate multiple analyses and content ideas
            assert len(dataflow_data["trend_analyses"]) == 3
            assert len(dataflow_data["content_ideas"]) == 3
            
            # Validate relationships are maintained
            for analysis in dataflow_data["trend_analyses"]:
                assert analysis["subtopic_name"] in [sub["name"] for sub in dataflow_data["subtopics"]]
            
            for content in dataflow_data["content_ideas"]:
                assert content["research_topic_id"] == topic_id
                assert any(content["trend_analysis_id"] == analysis["id"] for analysis in dataflow_data["trend_analyses"])
    
    def test_research_workflow_error_recovery(self, client, test_user_id, auth_headers):
        """Test research workflow error recovery and partial success handling"""
        
        # Create research topic successfully
        with patch('backend.src.services.research_topic_service.ResearchTopicService.create') as mock_create_topic:
            mock_topic = {
                "id": str(uuid.uuid4()),
                "user_id": test_user_id,
                "title": "Test Topic",
                "description": "Test description",
                "status": "active",
                "created_at": "2025-01-27T10:00:00Z",
                "updated_at": "2025-01-27T10:00:00Z",
                "version": 1
            }
            mock_create_topic.return_value = mock_topic
            
            response = client.post(
                "/api/research-topics",
                json={"title": "Test Topic", "description": "Test description"},
                headers=auth_headers
            )
            
            assert response.status_code == 201
            topic_id = response.json()["id"]
        
        # Attempt to create subtopics with error
        with patch('backend.src.services.topic_decomposition_service.TopicDecompositionService.create_subtopics') as mock_create_subtopics:
            mock_create_subtopics.side_effect = Exception("Database connection failed")
            
            response = client.post(
                f"/api/research-topics/{topic_id}/subtopics",
                json={"search_query": "test", "subtopics": [{"name": "test", "description": "test"}]},
                headers=auth_headers
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
        
        # Verify research topic still exists and can be retrieved
        with patch('backend.src.services.research_topic_service.ResearchTopicService.get_by_id') as mock_get_topic:
            mock_get_topic.return_value = mock_topic
            
            response = client.get(
                f"/api/research-topics/{topic_id}",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            assert response.json()["id"] == topic_id
            assert response.json()["title"] == "Test Topic"
    
    def test_research_workflow_data_consistency(self, client, test_user_id, auth_headers):
        """Test that data remains consistent throughout the workflow"""
        
        # This test ensures that all relationships are properly maintained
        # and data integrity is preserved throughout the workflow
        
        with patch('backend.src.services.research_topic_service.ResearchTopicService.create') as mock_create_topic, \
             patch('backend.src.services.topic_decomposition_service.TopicDecompositionService.create_subtopics') as mock_create_subtopics, \
             patch('backend.src.services.trend_analysis_service.TrendAnalysisService.create') as mock_create_analysis, \
             patch('backend.src.services.content_idea_service.ContentIdeaService.create') as mock_create_content, \
             patch('backend.src.services.research_topic_service.ResearchTopicService.get_complete_dataflow') as mock_get_dataflow:
            
            # Mock all service calls
            topic_id = str(uuid.uuid4())
            decomposition_id = str(uuid.uuid4())
            analysis_id = str(uuid.uuid4())
            content_id = str(uuid.uuid4())
            
            mock_topic = {
                "id": topic_id,
                "user_id": test_user_id,
                "title": "Consistency Test",
                "description": "Test for data consistency",
                "status": "active",
                "created_at": "2025-01-27T10:00:00Z",
                "updated_at": "2025-01-27T10:00:00Z",
                "version": 1
            }
            
            mock_subtopics = [
                {"name": "test subtopic", "description": "test description"}
            ]
            
            mock_analysis = {
                "id": analysis_id,
                "user_id": test_user_id,
                "topic_decomposition_id": decomposition_id,
                "subtopic_name": "test subtopic",
                "analysis_name": "Test Analysis",
                "status": "completed",
                "trend_data": {"search_volume": 1000, "trend_score": 50},
                "created_at": "2025-01-27T10:05:00Z"
            }
            
            mock_content = {
                "id": content_id,
                "user_id": test_user_id,
                "trend_analysis_id": analysis_id,
                "research_topic_id": topic_id,
                "title": "Test Content",
                "content_type": "guide",
                "idea_type": "evergreen",
                "status": "draft",
                "primary_keyword": "test",
                "created_at": "2025-01-27T10:10:00Z"
            }
            
            mock_dataflow = {
                "research_topic": mock_topic,
                "subtopics": mock_subtopics,
                "trend_analyses": [mock_analysis],
                "content_ideas": [mock_content]
            }
            
            # Set up mocks
            mock_create_topic.return_value = mock_topic
            mock_create_subtopics.return_value = mock_subtopics
            mock_create_analysis.return_value = mock_analysis
            mock_create_content.return_value = mock_content
            mock_get_dataflow.return_value = mock_dataflow
            
            # Execute workflow
            response = client.post(
                "/api/research-topics",
                json={"title": "Consistency Test", "description": "Test for data consistency"},
                headers=auth_headers
            )
            assert response.status_code == 201
            
            response = client.post(
                f"/api/research-topics/{topic_id}/subtopics",
                json={"search_query": "test", "subtopics": mock_subtopics},
                headers=auth_headers
            )
            assert response.status_code == 201
            
            response = client.post(
                "/api/trend-analyses",
                json={
                    "topic_decomposition_id": decomposition_id,
                    "subtopic_name": "test subtopic",
                    "analysis_name": "Test Analysis",
                    "keywords": ["test"],
                    "timeframe": "12m"
                },
                headers=auth_headers
            )
            assert response.status_code == 201
            
            response = client.post(
                "/api/content-ideas",
                json={
                    "trend_analysis_id": analysis_id,
                    "research_topic_id": topic_id,
                    "title": "Test Content",
                    "content_type": "guide",
                    "idea_type": "evergreen",
                    "primary_keyword": "test"
                },
                headers=auth_headers
            )
            assert response.status_code == 201
            
            # Verify complete dataflow
            response = client.get(
                f"/api/research-topics/{topic_id}/complete",
                headers=auth_headers
            )
            assert response.status_code == 200
            
            dataflow = response.json()
            
            # Verify all relationships are correct
            assert dataflow["research_topic"]["id"] == topic_id
            assert dataflow["research_topic"]["user_id"] == test_user_id
            
            assert len(dataflow["subtopics"]) == 1
            assert dataflow["subtopics"][0]["name"] == "test subtopic"
            
            assert len(dataflow["trend_analyses"]) == 1
            assert dataflow["trend_analyses"][0]["id"] == analysis_id
            assert dataflow["trend_analyses"][0]["subtopic_name"] == "test subtopic"
            
            assert len(dataflow["content_ideas"]) == 1
            assert dataflow["content_ideas"][0]["id"] == content_id
            assert dataflow["content_ideas"][0]["research_topic_id"] == topic_id
            assert dataflow["content_ideas"][0]["trend_analysis_id"] == analysis_id
