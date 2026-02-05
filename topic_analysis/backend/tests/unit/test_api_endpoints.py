"""
Backend Unit Tests for API Endpoints
T110: Backend unit tests for all services
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add backend src to path
backend_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(backend_src))

from main import app


class TestAffiliateAPI:
    """Test Affiliate API endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_search_programs_endpoint(self, client):
        """Test POST /api/v1/affiliate/search endpoint"""
        with patch('api.affiliate_routes.affiliate_service') as mock_service:
            mock_service.search_programs.return_value = []
            
            response = client.post(
                "/api/v1/affiliate/search",
                json={"query": "test", "category": "technology"}
            )
            
            assert response.status_code == 200
            assert response.json() == []
    
    def test_get_program_details_endpoint(self, client):
        """Test GET /api/v1/affiliate/programs/{program_id} endpoint"""
        with patch('api.affiliate_routes.affiliate_service') as mock_service:
            mock_program = {"id": "prog123", "name": "Test Program"}
            mock_service.get_program_details.return_value = mock_program
            
            response = client.get("/api/v1/affiliate/programs/prog123")
            
            assert response.status_code == 200
            assert response.json() == mock_program
    
    def test_create_research_endpoint(self, client):
        """Test POST /api/v1/affiliate/research endpoint"""
        with patch('api.affiliate_routes.affiliate_service') as mock_service:
            mock_research = {"id": "research123", "query": "test"}
            mock_service.create_research.return_value = mock_research
            
            response = client.post(
                "/api/v1/affiliate/research",
                json={"query": "test", "category": "technology"}
            )
            
            assert response.status_code == 201
            assert response.json() == mock_research


class TestTrendAPI:
    """Test Trend API endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_analyze_trends_endpoint(self, client):
        """Test POST /api/v1/trends/analyze endpoint"""
        with patch('api.trend_routes.trend_service') as mock_service:
            mock_analysis = {"id": "trend123", "keywords": ["test"]}
            mock_service.analyze_trends.return_value = mock_analysis
            
            response = client.post(
                "/api/v1/trends/analyze",
                json={"keywords": ["test"], "geo": "US"}
            )
            
            assert response.status_code == 200
            assert response.json() == mock_analysis
    
    def test_get_trend_data_endpoint(self, client):
        """Test GET /api/v1/trends/{analysis_id} endpoint"""
        with patch('api.trend_routes.trend_service') as mock_service:
            mock_trend = {"id": "trend123", "keywords": ["test"]}
            mock_service.get_trend_data.return_value = mock_trend
            
            response = client.get("/api/v1/trends/trend123")
            
            assert response.status_code == 200
            assert response.json() == mock_trend


class TestKeywordAPI:
    """Test Keyword API endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_upload_keywords_endpoint(self, client):
        """Test POST /api/v1/keywords/upload endpoint"""
        with patch('api.keyword_routes.keyword_service') as mock_service:
            mock_result = {"uploaded": 2, "keywords": []}
            mock_service.upload_keywords.return_value = mock_result
            
            response = client.post(
                "/api/v1/keywords/upload",
                json={"keywords": [{"keyword": "test", "search_volume": 1000}]}
            )
            
            assert response.status_code == 201
            assert response.json() == mock_result
    
    def test_get_keywords_endpoint(self, client):
        """Test GET /api/v1/keywords endpoint"""
        with patch('api.keyword_routes.keyword_service') as mock_service:
            mock_keywords = [{"keyword": "test", "search_volume": 1000}]
            mock_service.get_keywords.return_value = mock_keywords
            
            response = client.get("/api/v1/keywords")
            
            assert response.status_code == 200
            assert response.json() == mock_keywords


class TestContentAPI:
    """Test Content API endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_generate_content_ideas_endpoint(self, client):
        """Test POST /api/v1/content/generate endpoint"""
        with patch('api.content_routes.content_service') as mock_service:
            mock_ideas = [{"id": "idea123", "title": "Test Idea"}]
            mock_service.generate_content_ideas.return_value = mock_ideas
            
            response = client.post(
                "/api/v1/content/generate",
                json={"title": "Test Idea", "keywords": ["test"]}
            )
            
            assert response.status_code == 201
            assert response.json() == mock_ideas
    
    def test_get_content_ideas_endpoint(self, client):
        """Test GET /api/v1/content/ideas endpoint"""
        with patch('api.content_routes.content_service') as mock_service:
            mock_ideas = [{"id": "idea123", "title": "Test Idea"}]
            mock_service.get_content_ideas.return_value = mock_ideas
            
            response = client.get("/api/v1/content/ideas")
            
            assert response.status_code == 200
            assert response.json() == mock_ideas


class TestSoftwareAPI:
    """Test Software API endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_generate_software_ideas_endpoint(self, client):
        """Test POST /api/v1/software/generate endpoint"""
        with patch('api.software_routes.software_service') as mock_service:
            mock_ideas = [{"id": "soft123", "name": "Test Software"}]
            mock_service.generate_software_ideas.return_value = mock_ideas
            
            response = client.post(
                "/api/v1/software/generate",
                json={"name": "Test Software", "description": "Test Description"}
            )
            
            assert response.status_code == 201
            assert response.json() == mock_ideas
    
    def test_get_software_ideas_endpoint(self, client):
        """Test GET /api/v1/software/ideas endpoint"""
        with patch('api.software_routes.software_service') as mock_service:
            mock_ideas = [{"id": "soft123", "name": "Test Software"}]
            mock_service.get_software_ideas.return_value = mock_ideas
            
            response = client.get("/api/v1/software/ideas")
            
            assert response.status_code == 200
            assert response.json() == mock_ideas


class TestExportAPI:
    """Test Export API endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_export_data_endpoint(self, client):
        """Test POST /api/v1/export endpoint"""
        with patch('api.export_routes.export_service') as mock_service:
            mock_export = {"id": "export123", "status": "completed"}
            mock_service.export_data.return_value = mock_export
            
            response = client.post(
                "/api/v1/export",
                json={"format": "csv", "data": [{"test": "data"}]}
            )
            
            assert response.status_code == 201
            assert response.json() == mock_export
    
    def test_get_export_history_endpoint(self, client):
        """Test GET /api/v1/export/history endpoint"""
        with patch('api.export_routes.export_service') as mock_service:
            mock_history = [{"id": "export123", "status": "completed"}]
            mock_service.get_export_history.return_value = mock_history
            
            response = client.get("/api/v1/export/history")
            
            assert response.status_code == 200
            assert response.json() == mock_history


class TestCalendarAPI:
    """Test Calendar API endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_create_calendar_entry_endpoint(self, client):
        """Test POST /api/v1/calendar/entries endpoint"""
        with patch('api.calendar_routes.calendar_service') as mock_service:
            mock_entry = {"id": "entry123", "title": "Test Entry"}
            mock_service.create_calendar_entry.return_value = mock_entry
            
            response = client.post(
                "/api/v1/calendar/entries",
                json={"title": "Test Entry", "date": "2024-01-01"}
            )
            
            assert response.status_code == 201
            assert response.json() == mock_entry
    
    def test_get_calendar_entries_endpoint(self, client):
        """Test GET /api/v1/calendar/entries endpoint"""
        with patch('api.calendar_routes.calendar_service') as mock_service:
            mock_entries = [{"id": "entry123", "title": "Test Entry"}]
            mock_service.get_calendar_entries.return_value = mock_entries
            
            response = client.get("/api/v1/calendar/entries")
            
            assert response.status_code == 200
            assert response.json() == mock_entries


class TestUserAPI:
    """Test User API endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_get_user_profile_endpoint(self, client):
        """Test GET /api/v1/users/profile endpoint"""
        with patch('api.user_routes.user_service') as mock_service:
            mock_user = {"id": "user123", "email": "test@example.com"}
            mock_service.get_user.return_value = mock_user
            
            response = client.get("/api/v1/users/profile")
            
            assert response.status_code == 200
            assert response.json() == mock_user
    
    def test_update_user_profile_endpoint(self, client):
        """Test PUT /api/v1/users/profile endpoint"""
        with patch('api.user_routes.user_service') as mock_service:
            mock_user = {"id": "user123", "name": "Updated Name"}
            mock_service.update_user.return_value = mock_user
            
            response = client.put(
                "/api/v1/users/profile",
                json={"name": "Updated Name"}
            )
            
            assert response.status_code == 200
            assert response.json() == mock_user


class TestHealthAPI:
    """Test Health API endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_health_check_endpoint(self, client):
        """Test GET /health endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_health_detailed_endpoint(self, client):
        """Test GET /health/detailed endpoint"""
        with patch('api.health_routes.check_database_health') as mock_db:
            with patch('api.health_routes.check_redis_health') as mock_redis:
                mock_db.return_value = {"status": "healthy"}
                mock_redis.return_value = {"status": "healthy"}
                
                response = client.get("/health/detailed")
                
                assert response.status_code == 200
                data = response.json()
                assert "database" in data
                assert "redis" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
