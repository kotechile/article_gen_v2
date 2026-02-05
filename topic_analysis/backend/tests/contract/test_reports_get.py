"""
Contract test for GET /api/v1/reports/{report_id} endpoint
Tests the report retrieval functionality
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import uuid


class TestReportsGet:
    """Test cases for GET /api/v1/reports/{report_id} endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from main import app
        return TestClient(app)
    
    @pytest.fixture
    def valid_report_id(self):
        """Generate valid UUID for testing"""
        return str(uuid.uuid4())
    
    @pytest.fixture
    def invalid_report_id(self):
        """Invalid report ID for testing"""
        return "invalid-report-id"
    
    @pytest.fixture
    def sample_report(self):
        """Sample report data for testing"""
        return {
            "report_id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "filename": "keywords.tsv",
            "total_keywords": 50000,
            "high_opportunity_count": 12500,
            "medium_opportunity_count": 25000,
            "low_opportunity_count": 12500,
            "total_search_volume": 2500000,
            "average_difficulty": 45.2,
            "average_cpc": 2.85,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:03:00Z",
            "expires_at": "2024-04-01T00:00:00Z"
        }
    
    def test_get_report_success(self, client, valid_report_id, sample_report):
        """Test successful report retrieval"""
        with patch('src.api.reports.get_report') as mock_get_report:
            mock_get_report.return_value = sample_report
            
            response = client.get(f"/api/v1/reports/{valid_report_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["report_id"] == sample_report["report_id"]
            assert data["filename"] == sample_report["filename"]
            assert data["total_keywords"] == sample_report["total_keywords"]
    
    def test_get_report_not_found(self, client, valid_report_id):
        """Test report retrieval for non-existent report"""
        with patch('src.api.reports.get_report') as mock_get_report:
            mock_get_report.return_value = None
            
            response = client.get(f"/api/v1/reports/{valid_report_id}")
            
            assert response.status_code == 404
            data = response.json()
            assert "error" in data
            assert "Report not found" in data["error"]
    
    def test_get_report_invalid_report_id(self, client, invalid_report_id):
        """Test report retrieval with invalid report ID format"""
        response = client.get(f"/api/v1/reports/{invalid_report_id}")
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_get_report_expired(self, client, valid_report_id):
        """Test report retrieval for expired report"""
        expired_report = {
            "report_id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "filename": "keywords.tsv",
            "total_keywords": 50000,
            "high_opportunity_count": 12500,
            "medium_opportunity_count": 25000,
            "low_opportunity_count": 12500,
            "total_search_volume": 2500000,
            "average_difficulty": 45.2,
            "average_cpc": 2.85,
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:03:00Z",
            "expires_at": "2023-04-01T00:00:00Z"
        }
        
        with patch('src.api.reports.get_report') as mock_get_report:
            mock_get_report.return_value = expired_report
            
            response = client.get(f"/api/v1/reports/{valid_report_id}")
            
            assert response.status_code == 410
            data = response.json()
            assert "error" in data
            assert "Report has expired" in data["error"]
    
    def test_get_report_response_schema(self, client, valid_report_id, sample_report):
        """Test report response matches expected schema"""
        with patch('src.api.reports.get_report') as mock_get_report:
            mock_get_report.return_value = sample_report
            
            response = client.get(f"/api/v1/reports/{valid_report_id}")
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate response schema
            required_fields = [
                "report_id", "user_id", "filename", "total_keywords",
                "high_opportunity_count", "medium_opportunity_count", "low_opportunity_count",
                "total_search_volume", "average_difficulty", "average_cpc",
                "created_at", "updated_at", "expires_at"
            ]
            for field in required_fields:
                assert field in data
            
            # Validate field types
            assert isinstance(data["report_id"], str)
            assert isinstance(data["user_id"], str)
            assert isinstance(data["filename"], str)
            assert isinstance(data["total_keywords"], int)
            assert isinstance(data["high_opportunity_count"], int)
            assert isinstance(data["medium_opportunity_count"], int)
            assert isinstance(data["low_opportunity_count"], int)
            assert isinstance(data["total_search_volume"], int)
            assert isinstance(data["average_difficulty"], float)
            assert isinstance(data["average_cpc"], float)
            assert isinstance(data["created_at"], str)
            assert isinstance(data["updated_at"], str)
            assert isinstance(data["expires_at"], str)
            
            # Validate counts add up
            assert (data["high_opportunity_count"] + 
                   data["medium_opportunity_count"] + 
                   data["low_opportunity_count"]) == data["total_keywords"]
            
            # Validate UUID format for IDs
            import uuid
            try:
                uuid.UUID(data["report_id"])
                uuid.UUID(data["user_id"])
            except ValueError:
                pytest.fail("report_id or user_id is not a valid UUID")
    
    def test_get_report_with_keywords(self, client, valid_report_id):
        """Test report retrieval with keyword details"""
        report_with_keywords = {
            "report_id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "filename": "keywords.tsv",
            "total_keywords": 2,
            "high_opportunity_count": 1,
            "medium_opportunity_count": 1,
            "low_opportunity_count": 0,
            "total_search_volume": 20000,
            "average_difficulty": 40.0,
            "average_cpc": 2.5,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:03:00Z",
            "expires_at": "2024-04-01T00:00:00Z",
            "keywords": [
                {
                    "keyword": "best project management tools",
                    "search_volume": 12000,
                    "difficulty": 45,
                    "cpc": 2.50,
                    "opportunity_score": 85.5,
                    "category": "high"
                },
                {
                    "keyword": "project management basics",
                    "search_volume": 8000,
                    "difficulty": 35,
                    "cpc": 2.50,
                    "opportunity_score": 75.2,
                    "category": "medium"
                }
            ]
        }
        
        with patch('src.api.reports.get_report') as mock_get_report:
            mock_get_report.return_value = report_with_keywords
            
            response = client.get(f"/api/v1/reports/{valid_report_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert "keywords" in data
            assert len(data["keywords"]) == 2
    
    def test_get_report_database_error(self, client, valid_report_id):
        """Test report retrieval with database error"""
        with patch('src.api.reports.get_report') as mock_get_report:
            mock_get_report.side_effect = Exception("Database connection error")
            
            response = client.get(f"/api/v1/reports/{valid_report_id}")
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "Internal server error" in data["error"]
    
    def test_get_report_with_pagination(self, client, valid_report_id, sample_report):
        """Test report retrieval with pagination parameters"""
        with patch('src.api.reports.get_report') as mock_get_report:
            mock_get_report.return_value = sample_report
            
            response = client.get(
                f"/api/v1/reports/{valid_report_id}",
                params={"include_keywords": "true", "page": 1, "limit": 10}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "report_id" in data
