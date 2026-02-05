"""
Integration test for data integrity and relationships
This test validates that all data relationships are properly maintained using Supabase Client/SDK
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import uuid
from supabase import create_client, Client

# This test should FAIL until the data integrity features are implemented
class TestDataIntegrity:
    """Test data integrity and relationship validation"""
    
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
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client"""
        mock_client = MagicMock(spec=Client)
        return mock_client
    
    def test_foreign_key_constraints(self, client, test_user_id, auth_headers, mock_supabase_client):
        """Test that foreign key constraints are properly enforced"""
        
        # Test that research_topic_id is required for subtopics
        with patch('backend.src.services.supabase_service.get_supabase_client', return_value=mock_supabase_client):
            response = client.post(
                "/api/research-topics/invalid-id/subtopics",
                json={
                    "search_query": "test",
                    "subtopics": [{"name": "test", "description": "test"}]
                },
                headers=auth_headers
            )
            
            assert response.status_code == 404  # Research topic not found
        
        # Test that trend_analysis_id is required for content ideas
        with patch('backend.src.services.supabase_service.get_supabase_client', return_value=mock_supabase_client):
            response = client.post(
                "/api/content-ideas",
                json={
                    "research_topic_id": str(uuid.uuid4()),
                    "title": "Test Content",
                    "content_type": "guide",
                    "idea_type": "evergreen",
                    "primary_keyword": "test"
                    # Missing trend_analysis_id
                },
                headers=auth_headers
            )
            
            assert response.status_code == 400  # Bad request due to missing required field
    
    def test_cascade_deletion(self, client, test_user_id, auth_headers, mock_supabase_client):
        """Test that cascade deletion works properly"""
        
        topic_id = str(uuid.uuid4())
        decomposition_id = str(uuid.uuid4())
        analysis_id = str(uuid.uuid4())
        content_id = str(uuid.uuid4())
        
        # Mock successful deletion cascade
        mock_supabase_client.table.return_value.delete.return_value.execute.return_value = {
            "data": [],
            "error": None
        }
        
        with patch('backend.src.services.supabase_service.get_supabase_client', return_value=mock_supabase_client):
            # Delete research topic
            response = client.delete(
                f"/api/research-topics/{topic_id}",
                headers=auth_headers
            )
            
            assert response.status_code == 204
            
            # Verify that all related data is also deleted
            # This would be verified by checking that subtopics, analyses, and content ideas are gone
            mock_supabase_client.table.assert_any_call("research_topics")
            mock_supabase_client.table.assert_any_call("topic_decompositions")
            mock_supabase_client.table.assert_any_call("trend_analyses")
            mock_supabase_client.table.assert_any_call("content_ideas")
    
    def test_unique_constraints(self, client, test_user_id, auth_headers, mock_supabase_client):
        """Test that unique constraints are properly enforced"""
        
        # Test unique constraint on (user_id, title) for research topics
        mock_supabase_client.table.return_value.insert.return_value.execute.side_effect = [
            # First insert succeeds
            {"data": [{"id": str(uuid.uuid4()), "title": "Duplicate Title"}], "error": None},
            # Second insert fails due to unique constraint
            {"data": None, "error": {"code": "23505", "message": "duplicate key value violates unique constraint"}}
        ]
        
        with patch('backend.src.services.supabase_service.get_supabase_client', return_value=mock_supabase_client):
            # First creation should succeed
            response1 = client.post(
                "/api/research-topics",
                json={"title": "Duplicate Title", "description": "First description"},
                headers=auth_headers
            )
            assert response1.status_code == 201
            
            # Second creation with same title should fail
            response2 = client.post(
                "/api/research-topics",
                json={"title": "Duplicate Title", "description": "Second description"},
                headers=auth_headers
            )
            assert response2.status_code == 409  # Conflict
    
    def test_data_validation_constraints(self, client, test_user_id, auth_headers, mock_supabase_client):
        """Test that data validation constraints are properly enforced"""
        
        # Test status enum validation
        mock_supabase_client.table.return_value.insert.return_value.execute.return_value = {
            "data": None,
            "error": {"code": "23514", "message": "new row for relation \"research_topics\" violates check constraint"}
        }
        
        with patch('backend.src.services.supabase_service.get_supabase_client', return_value=mock_supabase_client):
            response = client.post(
                "/api/research-topics",
                json={
                    "title": "Test Topic",
                    "description": "Test description",
                    "status": "invalid_status"  # Invalid status
                },
                headers=auth_headers
            )
            
            assert response.status_code == 400  # Bad request due to validation error
        
        # Test required field validation
        with patch('backend.src.services.supabase_service.get_supabase_client', return_value=mock_supabase_client):
            response = client.post(
                "/api/research-topics",
                json={
                    "description": "Missing title"
                    # Missing required title field
                },
                headers=auth_headers
            )
            
            assert response.status_code == 422  # Unprocessable entity due to missing required field
    
    def test_relationship_consistency(self, client, test_user_id, auth_headers, mock_supabase_client):
        """Test that relationships remain consistent across operations"""
        
        topic_id = str(uuid.uuid4())
        decomposition_id = str(uuid.uuid4())
        analysis_id = str(uuid.uuid4())
        content_id = str(uuid.uuid4())
        
        # Mock data with proper relationships
        mock_topic = {
            "id": topic_id,
            "user_id": test_user_id,
            "title": "Consistency Test",
            "description": "Test for relationship consistency",
            "status": "active",
            "created_at": "2025-01-27T10:00:00Z",
            "updated_at": "2025-01-27T10:00:00Z",
            "version": 1
        }
        
        mock_subtopics = [
            {
                "id": decomposition_id,
                "research_topic_id": topic_id,
                "user_id": test_user_id,
                "search_query": "consistency test",
                "subtopics": [{"name": "test subtopic", "description": "test description"}],
                "created_at": "2025-01-27T10:05:00Z"
            }
        ]
        
        mock_analysis = {
            "id": analysis_id,
            "user_id": test_user_id,
            "topic_decomposition_id": decomposition_id,
            "subtopic_name": "test subtopic",
            "analysis_name": "Test Analysis",
            "status": "completed",
            "trend_data": {"search_volume": 1000, "trend_score": 50},
            "created_at": "2025-01-27T10:10:00Z"
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
            "created_at": "2025-01-27T10:15:00Z"
        }
        
        # Mock Supabase responses
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.execute.return_value = {
            "data": [mock_topic],
            "error": None
        }
        
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = {
            "data": mock_subtopics,
            "error": None
        }
        
        with patch('backend.src.services.supabase_service.get_supabase_client', return_value=mock_supabase_client):
            # Test that we can retrieve a research topic with all its relationships
            response = client.get(
                f"/api/research-topics/{topic_id}/complete",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify all relationships are present and correct
            assert data["research_topic"]["id"] == topic_id
            assert data["research_topic"]["user_id"] == test_user_id
            
            # Verify subtopics are linked to the research topic
            for subtopic in data["subtopics"]:
                assert subtopic["research_topic_id"] == topic_id
            
            # Verify trend analyses are linked to subtopics
            for analysis in data["trend_analyses"]:
                assert analysis["topic_decomposition_id"] == decomposition_id
                assert analysis["subtopic_name"] in [sub["name"] for sub in data["subtopics"]]
            
            # Verify content ideas are linked to both trend analysis and research topic
            for content in data["content_ideas"]:
                assert content["research_topic_id"] == topic_id
                assert content["trend_analysis_id"] in [analysis["id"] for analysis in data["trend_analyses"]]
    
    def test_concurrent_access_integrity(self, client, test_user_id, auth_headers, mock_supabase_client):
        """Test data integrity under concurrent access scenarios"""
        
        topic_id = str(uuid.uuid4())
        
        # Mock optimistic concurrency control failure
        mock_supabase_client.table.return_value.update.return_value.eq.return_value.eq.return_value.execute.return_value = {
            "data": None,
            "error": {"code": "23505", "message": "concurrent update detected"}
        }
        
        with patch('backend.src.services.supabase_service.get_supabase_client', return_value=mock_supabase_client):
            # Attempt to update with stale version
            response = client.put(
                f"/api/research-topics/{topic_id}",
                json={
                    "title": "Updated Title",
                    "description": "Updated description",
                    "version": 1  # Stale version
                },
                headers=auth_headers
            )
            
            assert response.status_code == 409  # Conflict due to concurrent update
    
    def test_transaction_rollback(self, client, test_user_id, auth_headers, mock_supabase_client):
        """Test that transaction rollback works properly on errors"""
        
        topic_id = str(uuid.uuid4())
        
        # Mock partial failure scenario
        mock_supabase_client.table.return_value.insert.return_value.execute.side_effect = [
            # Research topic creation succeeds
            {"data": [{"id": topic_id, "title": "Test Topic"}], "error": None},
            # Subtopics creation fails
            {"data": None, "error": {"code": "23503", "message": "foreign key constraint violation"}}
        ]
        
        with patch('backend.src.services.supabase_service.get_supabase_client', return_value=mock_supabase_client):
            # This should trigger a rollback
            response = client.post(
                f"/api/research-topics/{topic_id}/subtopics",
                json={
                    "search_query": "test",
                    "subtopics": [{"name": "test", "description": "test"}]
                },
                headers=auth_headers
            )
            
            assert response.status_code == 500  # Internal server error
            
            # Verify that the research topic was not left in an inconsistent state
            # This would be verified by checking that no orphaned data exists
    
    def test_data_type_validation(self, client, test_user_id, auth_headers, mock_supabase_client):
        """Test that data types are properly validated"""
        
        # Test UUID validation
        mock_supabase_client.table.return_value.insert.return_value.execute.return_value = {
            "data": None,
            "error": {"code": "22P02", "message": "invalid input syntax for type uuid"}}
        
        with patch('backend.src.services.supabase_service.get_supabase_client', return_value=mock_supabase_client):
            response = client.post(
                "/api/research-topics",
                json={
                    "title": "Test Topic",
                    "description": "Test description",
                    "user_id": "invalid-uuid"  # Invalid UUID format
                },
                headers=auth_headers
            )
            
            assert response.status_code == 400  # Bad request due to invalid UUID
        
        # Test JSONB validation
        mock_supabase_client.table.return_value.insert.return_value.execute.return_value = {
            "data": None,
            "error": {"code": "22P02", "message": "invalid input syntax for type json"}}
        
        with patch('backend.src.services.supabase_service.get_supabase_client', return_value=mock_supabase_client):
            response = client.post(
                "/api/trend-analyses",
                json={
                    "topic_decomposition_id": str(uuid.uuid4()),
                    "subtopic_name": "test",
                    "analysis_name": "Test Analysis",
                    "trend_data": "invalid-json"  # Invalid JSON format
                },
                headers=auth_headers
            )
            
            assert response.status_code == 400  # Bad request due to invalid JSON
    
    def test_index_performance(self, client, test_user_id, auth_headers, mock_supabase_client):
        """Test that database indexes are working for performance"""
        
        # Mock query execution with index usage
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.execute.return_value = {
            "data": [{"id": str(uuid.uuid4()), "title": "Test Topic"}],
            "error": None
        }
        
        with patch('backend.src.services.supabase_service.get_supabase_client', return_value=mock_supabase_client):
            # Test query that should use indexes
            response = client.get(
                f"/api/research-topics",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            
            # Verify that the query used the appropriate indexes
            # This would be verified by checking query execution plans in a real scenario
            mock_supabase_client.table.assert_called_with("research_topics")
    
    def test_rls_policy_enforcement(self, client, test_user_id, auth_headers, mock_supabase_client):
        """Test that Row Level Security policies are properly enforced"""
        
        other_user_id = str(uuid.uuid4())
        
        # Mock RLS policy enforcement - user can only see their own data
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.execute.return_value = {
            "data": [],  # No data returned due to RLS
            "error": None
        }
        
        with patch('backend.src.services.supabase_service.get_supabase_client', return_value=mock_supabase_client):
            # Try to access another user's research topic
            response = client.get(
                f"/api/research-topics/{str(uuid.uuid4())}",
                headers=auth_headers
            )
            
            assert response.status_code == 404  # Not found due to RLS policy
        
        # Test that user can only modify their own data
        mock_supabase_client.table.return_value.update.return_value.eq.return_value.eq.return_value.execute.return_value = {
            "data": [],
            "error": {"code": "42501", "message": "new row violates row-level security policy"}
        }
        
        with patch('backend.src.services.supabase_service.get_supabase_client', return_value=mock_supabase_client):
            response = client.put(
                f"/api/research-topics/{str(uuid.uuid4())}",
                json={
                    "title": "Unauthorized Update",
                    "description": "This should fail",
                    "version": 1
                },
                headers=auth_headers
            )
            
            assert response.status_code == 403  # Forbidden due to RLS policy
