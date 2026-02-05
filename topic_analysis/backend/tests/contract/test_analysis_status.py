"""
Contract test for GET /api/v1/analysis/{file_id}/status endpoint
Tests the analysis status checking functionality
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import uuid


class TestAnalysisStatus:
    """Test cases for GET /api/v1/analysis/{file_id}/status endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from main import app
        return TestClient(app)
    
    @pytest.fixture
    def valid_file_id(self):
        """Generate valid UUID for testing"""
        return str(uuid.uuid4())
    
    @pytest.fixture
    def invalid_file_id(self):
        """Invalid file ID for testing"""
        return "invalid-file-id"
    
    def test_get_analysis_status_processing(self, client, valid_file_id):
        """Test status retrieval for analysis in progress"""
        with patch('src.api.analysis.get_analysis_status') as mock_get_status:
            mock_get_status.return_value = {
                "analysis_id": str(uuid.uuid4()),
                "file_id": valid_file_id,
                "status": "processing",
                "progress_percentage": 45,
                "started_at": "2024-01-01T00:00:00Z",
                "estimated_completion": "2024-01-01T00:05:00Z",
                "keywords_processed": 22500,
                "total_keywords": 50000
            }
            
            response = client.get(f"/api/v1/analysis/{valid_file_id}/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "processing"
            assert data["progress_percentage"] == 45
            assert data["keywords_processed"] == 22500
            assert data["total_keywords"] == 50000
    
    def test_get_analysis_status_completed(self, client, valid_file_id):
        """Test status retrieval for completed analysis"""
        with patch('src.api.analysis.get_analysis_status') as mock_get_status:
            mock_get_status.return_value = {
                "analysis_id": str(uuid.uuid4()),
                "file_id": valid_file_id,
                "status": "completed",
                "progress_percentage": 100,
                "started_at": "2024-01-01T00:00:00Z",
                "completed_at": "2024-01-01T00:03:00Z",
                "keywords_processed": 50000,
                "total_keywords": 50000,
                "report_id": str(uuid.uuid4())
            }
            
            response = client.get(f"/api/v1/analysis/{valid_file_id}/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["progress_percentage"] == 100
            assert data["completed_at"] is not None
            assert "report_id" in data
    
    def test_get_analysis_status_failed(self, client, valid_file_id):
        """Test status retrieval for failed analysis"""
        with patch('src.api.analysis.get_analysis_status') as mock_get_status:
            mock_get_status.return_value = {
                "analysis_id": str(uuid.uuid4()),
                "file_id": valid_file_id,
                "status": "failed",
                "progress_percentage": 0,
                "started_at": "2024-01-01T00:00:00Z",
                "failed_at": "2024-01-01T00:01:00Z",
                "error_message": "Invalid TSV format detected",
                "keywords_processed": 0,
                "total_keywords": 50000
            }
            
            response = client.get(f"/api/v1/analysis/{valid_file_id}/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "failed"
            assert data["error_message"] == "Invalid TSV format detected"
            assert data["failed_at"] is not None
    
    def test_get_analysis_status_not_found(self, client, valid_file_id):
        """Test status retrieval for non-existent analysis"""
        with patch('src.api.analysis.get_analysis_status') as mock_get_status:
            mock_get_status.return_value = None
            
            response = client.get(f"/api/v1/analysis/{valid_file_id}/status")
            
            assert response.status_code == 404
            data = response.json()
            assert "error" in data
            assert "Analysis not found" in data["error"]
    
    def test_get_analysis_status_invalid_file_id(self, client, invalid_file_id):
        """Test status retrieval with invalid file ID format"""
        response = client.get(f"/api/v1/analysis/{invalid_file_id}/status")
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_get_analysis_status_response_schema(self, client, valid_file_id):
        """Test analysis status response matches expected schema"""
        with patch('src.api.analysis.get_analysis_status') as mock_get_status:
            mock_get_status.return_value = {
                "analysis_id": str(uuid.uuid4()),
                "file_id": valid_file_id,
                "status": "processing",
                "progress_percentage": 45,
                "started_at": "2024-01-01T00:00:00Z",
                "estimated_completion": "2024-01-01T00:05:00Z",
                "keywords_processed": 22500,
                "total_keywords": 50000
            }
            
            response = client.get(f"/api/v1/analysis/{valid_file_id}/status")
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate response schema
            required_fields = [
                "analysis_id", "file_id", "status", "progress_percentage",
                "started_at", "keywords_processed", "total_keywords"
            ]
            for field in required_fields:
                assert field in data
            
            # Validate field types
            assert isinstance(data["analysis_id"], str)
            assert isinstance(data["file_id"], str)
            assert isinstance(data["status"], str)
            assert isinstance(data["progress_percentage"], int)
            assert isinstance(data["started_at"], str)
            assert isinstance(data["keywords_processed"], int)
            assert isinstance(data["total_keywords"], int)
            
            # Validate status values
            assert data["status"] in ["processing", "completed", "failed"]
            
            # Validate progress percentage range
            assert 0 <= data["progress_percentage"] <= 100
            
            # Validate UUID format for analysis_id
            import uuid
            try:
                uuid.UUID(data["analysis_id"])
            except ValueError:
                pytest.fail("analysis_id is not a valid UUID")
    
    def test_get_analysis_status_database_error(self, client, valid_file_id):
        """Test analysis status retrieval with database error"""
        with patch('src.api.analysis.get_analysis_status') as mock_get_status:
            mock_get_status.side_effect = Exception("Database connection error")
            
            response = client.get(f"/api/v1/analysis/{valid_file_id}/status")
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "Internal server error" in data["error"]
    
    def test_get_analysis_status_processing_with_eta(self, client, valid_file_id):
        """Test status retrieval for processing analysis with ETA"""
        with patch('src.api.analysis.get_analysis_status') as mock_get_status:
            mock_get_status.return_value = {
                "analysis_id": str(uuid.uuid4()),
                "file_id": valid_file_id,
                "status": "processing",
                "progress_percentage": 75,
                "started_at": "2024-01-01T00:00:00Z",
                "estimated_completion": "2024-01-01T00:02:00Z",
                "keywords_processed": 37500,
                "total_keywords": 50000,
                "processing_rate": 1250  # keywords per second
            }
            
            response = client.get(f"/api/v1/analysis/{valid_file_id}/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "processing"
            assert data["progress_percentage"] == 75
            assert "processing_rate" in data
            assert data["processing_rate"] == 1250
