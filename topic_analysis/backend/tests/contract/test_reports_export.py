"""
Contract test for GET /api/v1/reports/{report_id}/export endpoint
Tests the report export functionality
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import uuid
import json


class TestReportsExport:
    """Test cases for GET /api/v1/reports/{report_id}/export endpoint"""
    
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
    def sample_export_data(self):
        """Sample export data for testing"""
        return {
            "report_id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "filename": "keywords.tsv",
            "export_format": "json",
            "export_timestamp": "2024-01-01T00:00:00Z",
            "data": {
                "summary": {
                    "total_keywords": 50000,
                    "high_opportunity_count": 12500,
                    "medium_opportunity_count": 25000,
                    "low_opportunity_count": 12500,
                    "total_search_volume": 2500000,
                    "average_difficulty": 45.2,
                    "average_cpc": 2.85
                },
                "top_opportunities": {
                    "high_opportunity_keywords": [
                        {
                            "keyword": "best project management tools",
                            "search_volume": 12000,
                            "difficulty": 45,
                            "cpc": 2.50,
                            "opportunity_score": 85.5,
                            "category": "high"
                        }
                    ]
                },
                "seo_content_ideas": [
                    {
                        "id": str(uuid.uuid4()),
                        "title": "Best Project Management Tools for Remote Teams in 2024",
                        "content_type": "list-article",
                        "primary_keywords": ["best project management tools"],
                        "secondary_keywords": ["remote work tools"],
                        "seo_optimization_score": 92,
                        "traffic_potential_score": 88,
                        "total_search_volume": 45000,
                        "average_difficulty": 45,
                        "average_cpc": 3.20,
                        "optimization_tips": [
                            "Include 'best project management tools' in your title"
                        ],
                        "content_outline": "Introduction → Top 10 Tools → Conclusion"
                    }
                ]
            }
        }
    
    def test_export_report_json_success(self, client, valid_report_id, sample_export_data):
        """Test successful JSON export of report"""
        with patch('src.api.reports.export_report') as mock_export:
            mock_export.return_value = sample_export_data
            
            response = client.get(f"/api/v1/reports/{valid_report_id}/export")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"
            data = response.json()
            assert data["report_id"] == sample_export_data["report_id"]
            assert data["export_format"] == "json"
            assert "data" in data
    
    def test_export_report_csv_success(self, client, valid_report_id):
        """Test successful CSV export of report"""
        csv_content = "keyword,search_volume,difficulty,cpc,opportunity_score,category\n"
        csv_content += "best project management tools,12000,45,2.50,85.5,high\n"
        csv_content += "project management software,8500,38,3.20,82.1,high\n"
        
        with patch('src.api.reports.export_report') as mock_export:
            mock_export.return_value = {
                "content": csv_content,
                "filename": "keywords_analysis.csv",
                "content_type": "text/csv"
            }
            
            response = client.get(
                f"/api/v1/reports/{valid_report_id}/export",
                params={"format": "csv"}
            )
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/csv"
            assert "best project management tools" in response.text
    
    def test_export_report_xlsx_success(self, client, valid_report_id):
        """Test successful XLSX export of report"""
        with patch('src.api.reports.export_report') as mock_export:
            mock_export.return_value = {
                "content": b"binary_xlsx_content",
                "filename": "keywords_analysis.xlsx",
                "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            }
            
            response = client.get(
                f"/api/v1/reports/{valid_report_id}/export",
                params={"format": "xlsx"}
            )
            
            assert response.status_code == 200
            assert "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in response.headers["content-type"]
    
    def test_export_report_not_found(self, client, valid_report_id):
        """Test export for non-existent report"""
        with patch('src.api.reports.export_report') as mock_export:
            mock_export.return_value = None
            
            response = client.get(f"/api/v1/reports/{valid_report_id}/export")
            
            assert response.status_code == 404
            data = response.json()
            assert "error" in data
            assert "Report not found" in data["error"]
    
    def test_export_report_invalid_report_id(self, client, invalid_report_id):
        """Test export with invalid report ID format"""
        response = client.get(f"/api/v1/reports/{invalid_report_id}/export")
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_export_report_invalid_format(self, client, valid_report_id):
        """Test export with invalid format parameter"""
        response = client.get(
            f"/api/v1/reports/{valid_report_id}/export",
            params={"format": "invalid_format"}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_export_report_expired(self, client, valid_report_id):
        """Test export for expired report"""
        with patch('src.api.reports.export_report') as mock_export:
            mock_export.side_effect = ValueError("Report has expired")
            
            response = client.get(f"/api/v1/reports/{valid_report_id}/export")
            
            assert response.status_code == 410
            data = response.json()
            assert "error" in data
            assert "Report has expired" in data["error"]
    
    def test_export_report_response_schema(self, client, valid_report_id, sample_export_data):
        """Test export response matches expected schema"""
        with patch('src.api.reports.export_report') as mock_export:
            mock_export.return_value = sample_export_data
            
            response = client.get(f"/api/v1/reports/{valid_report_id}/export")
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate response schema
            required_fields = [
                "report_id", "user_id", "filename", "export_format",
                "export_timestamp", "data"
            ]
            for field in required_fields:
                assert field in data
            
            # Validate field types
            assert isinstance(data["report_id"], str)
            assert isinstance(data["user_id"], str)
            assert isinstance(data["filename"], str)
            assert isinstance(data["export_format"], str)
            assert isinstance(data["export_timestamp"], str)
            assert isinstance(data["data"], dict)
            
            # Validate export format
            assert data["export_format"] in ["json", "csv", "xlsx"]
            
            # Validate data structure
            data_content = data["data"]
            assert "summary" in data_content
            assert "top_opportunities" in data_content
            assert "seo_content_ideas" in data_content
    
    def test_export_report_with_filters(self, client, valid_report_id, sample_export_data):
        """Test export with filtering parameters"""
        with patch('src.api.reports.export_report') as mock_export:
            mock_export.return_value = sample_export_data
            
            response = client.get(
                f"/api/v1/reports/{valid_report_id}/export",
                params={
                    "format": "json",
                    "category": "high",
                    "min_volume": 1000,
                    "include_keywords": "true"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
    
    def test_export_report_large_dataset(self, client, valid_report_id):
        """Test export for large dataset"""
        large_export_data = {
            "report_id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "filename": "large_keywords.tsv",
            "export_format": "json",
            "export_timestamp": "2024-01-01T00:00:00Z",
            "data": {
                "summary": {
                    "total_keywords": 100000,
                    "high_opportunity_count": 25000,
                    "medium_opportunity_count": 50000,
                    "low_opportunity_count": 25000,
                    "total_search_volume": 5000000,
                    "average_difficulty": 42.8,
                    "average_cpc": 2.95
                },
                "top_opportunities": {
                    "high_opportunity_keywords": [],
                    "quick_wins": [],
                    "high_volume_targets": []
                },
                "seo_content_ideas": []
            }
        }
        
        with patch('src.api.reports.export_report') as mock_export:
            mock_export.return_value = large_export_data
            
            response = client.get(f"/api/v1/reports/{valid_report_id}/export")
            
            assert response.status_code == 200
            data = response.json()
            assert data["data"]["summary"]["total_keywords"] == 100000
    
    def test_export_report_database_error(self, client, valid_report_id):
        """Test export with database error"""
        with patch('src.api.reports.export_report') as mock_export:
            mock_export.side_effect = Exception("Database connection error")
            
            response = client.get(f"/api/v1/reports/{valid_report_id}/export")
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "Internal server error" in data["error"]
    
    def test_export_report_download_filename(self, client, valid_report_id, sample_export_data):
        """Test export includes proper download filename"""
        with patch('src.api.reports.export_report') as mock_export:
            mock_export.return_value = sample_export_data
            
            response = client.get(f"/api/v1/reports/{valid_report_id}/export")
            
            assert response.status_code == 200
            # Check if Content-Disposition header is set for file download
            assert "Content-Disposition" in response.headers
            assert "attachment" in response.headers["Content-Disposition"]
