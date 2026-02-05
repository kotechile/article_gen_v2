"""
Contract test for GET /api/v1/upload/{file_id}/status endpoint
Tests the file upload status checking functionality
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import uuid


class TestUploadStatus:
    """Test cases for GET /api/v1/upload/{file_id}/status endpoint"""
    
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
    
    def test_get_upload_status_success(self, client, valid_file_id):
        """Test successful status retrieval for valid file ID"""
        with patch('src.api.upload.get_file_status') as mock_get_status:
            mock_get_status.return_value = {
                "file_id": valid_file_id,
                "filename": "keywords.tsv",
                "file_size": 1024,
                "status": "pending",
                "uploaded_at": "2024-01-01T00:00:00Z",
                "processing_started_at": None,
                "processing_completed_at": None,
                "error_message": None
            }
            
            response = client.get(f"/api/v1/upload/{valid_file_id}/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["file_id"] == valid_file_id
            assert data["filename"] == "keywords.tsv"
            assert data["status"] == "pending"
            assert "uploaded_at" in data
    
    def test_get_upload_status_processing(self, client, valid_file_id):
        """Test status retrieval for file in processing state"""
        with patch('src.api.upload.get_file_status') as mock_get_status:
            mock_get_status.return_value = {
                "file_id": valid_file_id,
                "filename": "keywords.tsv",
                "file_size": 1024,
                "status": "processing",
                "uploaded_at": "2024-01-01T00:00:00Z",
                "processing_started_at": "2024-01-01T00:01:00Z",
                "processing_completed_at": None,
                "error_message": None
            }
            
            response = client.get(f"/api/v1/upload/{valid_file_id}/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "processing"
            assert data["processing_started_at"] is not None
            assert data["processing_completed_at"] is None
    
    def test_get_upload_status_completed(self, client, valid_file_id):
        """Test status retrieval for completed file"""
        with patch('src.api.upload.get_file_status') as mock_get_status:
            mock_get_status.return_value = {
                "file_id": valid_file_id,
                "filename": "keywords.tsv",
                "file_size": 1024,
                "status": "completed",
                "uploaded_at": "2024-01-01T00:00:00Z",
                "processing_started_at": "2024-01-01T00:01:00Z",
                "processing_completed_at": "2024-01-01T00:02:00Z",
                "error_message": None
            }
            
            response = client.get(f"/api/v1/upload/{valid_file_id}/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["processing_completed_at"] is not None
    
    def test_get_upload_status_failed(self, client, valid_file_id):
        """Test status retrieval for failed file"""
        with patch('src.api.upload.get_file_status') as mock_get_status:
            mock_get_status.return_value = {
                "file_id": valid_file_id,
                "filename": "keywords.tsv",
                "file_size": 1024,
                "status": "failed",
                "uploaded_at": "2024-01-01T00:00:00Z",
                "processing_started_at": "2024-01-01T00:01:00Z",
                "processing_completed_at": None,
                "error_message": "Invalid TSV format"
            }
            
            response = client.get(f"/api/v1/upload/{valid_file_id}/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "failed"
            assert data["error_message"] == "Invalid TSV format"
    
    def test_get_upload_status_not_found(self, client, valid_file_id):
        """Test status retrieval for non-existent file"""
        with patch('src.api.upload.get_file_status') as mock_get_status:
            mock_get_status.return_value = None
            
            response = client.get(f"/api/v1/upload/{valid_file_id}/status")
            
            assert response.status_code == 404
            data = response.json()
            assert "error" in data
            assert "File not found" in data["error"]
    
    def test_get_upload_status_invalid_file_id(self, client, invalid_file_id):
        """Test status retrieval with invalid file ID format"""
        response = client.get(f"/api/v1/upload/{invalid_file_id}/status")
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_get_upload_status_response_schema(self, client, valid_file_id):
        """Test upload status response matches expected schema"""
        with patch('src.api.upload.get_file_status') as mock_get_status:
            mock_get_status.return_value = {
                "file_id": valid_file_id,
                "filename": "keywords.tsv",
                "file_size": 1024,
                "status": "pending",
                "uploaded_at": "2024-01-01T00:00:00Z",
                "processing_started_at": None,
                "processing_completed_at": None,
                "error_message": None
            }
            
            response = client.get(f"/api/v1/upload/{valid_file_id}/status")
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate response schema
            required_fields = [
                "file_id", "filename", "file_size", "status", 
                "uploaded_at", "processing_started_at", 
                "processing_completed_at", "error_message"
            ]
            for field in required_fields:
                assert field in data
            
            # Validate field types
            assert isinstance(data["file_id"], str)
            assert isinstance(data["filename"], str)
            assert isinstance(data["file_size"], int)
            assert isinstance(data["status"], str)
            assert isinstance(data["uploaded_at"], str)
            
            # Validate status values
            assert data["status"] in ["pending", "processing", "completed", "failed"]
            
            # Validate timestamps format (ISO 8601)
            import datetime
            try:
                datetime.datetime.fromisoformat(data["uploaded_at"].replace('Z', '+00:00'))
            except ValueError:
                pytest.fail("uploaded_at is not a valid ISO 8601 timestamp")
    
    def test_get_upload_status_database_error(self, client, valid_file_id):
        """Test status retrieval with database error"""
        with patch('src.api.upload.get_file_status') as mock_get_status:
            mock_get_status.side_effect = Exception("Database connection error")
            
            response = client.get(f"/api/v1/upload/{valid_file_id}/status")
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "Internal server error" in data["error"]
