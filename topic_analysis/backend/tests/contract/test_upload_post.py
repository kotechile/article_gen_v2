"""
Contract test for POST /api/v1/upload endpoint
Tests the file upload functionality according to the API contract
"""

import pytest
import io
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import tempfile
import os


class TestUploadPost:
    """Test cases for POST /api/v1/upload endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from main import app
        return TestClient(app)
    
    @pytest.fixture
    def sample_tsv_content(self):
        """Sample TSV content for testing"""
        return """Keyword	Volume	Difficulty	CPC	Intents
best project management tools	12000	45	2.50	Informational,Commercial
project management software	8500	38	3.20	Commercial,Informational
how to manage projects	3200	25	1.80	Informational
agile project management	5600	42	2.90	Informational,Commercial
project planning tools	2100	28	2.10	Commercial,Informational"""
    
    @pytest.fixture
    def sample_tsv_file(self, sample_tsv_content):
        """Create temporary TSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write(sample_tsv_content)
            f.flush()
            yield f.name
        os.unlink(f.name)
    
    def test_upload_valid_tsv_file_success(self, client, sample_tsv_file):
        """Test successful upload of valid TSV file"""
        with open(sample_tsv_file, 'rb') as f:
            response = client.post(
                "/api/v1/upload",
                files={"file": ("keywords.tsv", f, "text/tab-separated-values")},
                data={"user_id": "123e4567-e89b-12d3-a456-426614174000"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data
        assert "filename" in data
        assert "file_size" in data
        assert "status" in data
        assert "message" in data
        assert data["filename"] == "keywords.tsv"
        assert data["status"] == "pending"
        assert data["message"] == "File uploaded successfully"
    
    def test_upload_invalid_file_type_fails(self, client):
        """Test upload fails with invalid file type"""
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.txt", io.BytesIO(b"test content"), "text/plain")},
            data={"user_id": "123e4567-e89b-12d3-a456-426614174000"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "Invalid file type" in data["error"]
    
    def test_upload_file_too_large_fails(self, client):
        """Test upload fails with file too large"""
        # Create a large file (simulate > 10MB)
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB
        
        response = client.post(
            "/api/v1/upload",
            files={"file": ("large.tsv", io.BytesIO(large_content), "text/tab-separated-values")},
            data={"user_id": "123e4567-e89b-12d3-a456-426614174000"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "File too large" in data["error"]
    
    def test_upload_missing_user_id_fails(self, client, sample_tsv_file):
        """Test upload fails without user_id"""
        with open(sample_tsv_file, 'rb') as f:
            response = client.post(
                "/api/v1/upload",
                files={"file": ("keywords.tsv", f, "text/tab-separated-values")}
            )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_upload_invalid_user_id_format_fails(self, client, sample_tsv_file):
        """Test upload fails with invalid user_id format"""
        with open(sample_tsv_file, 'rb') as f:
            response = client.post(
                "/api/v1/upload",
                files={"file": ("keywords.tsv", f, "text/tab-separated-values")},
                data={"user_id": "invalid-uuid"}
            )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_upload_missing_file_fails(self, client):
        """Test upload fails without file"""
        response = client.post(
            "/api/v1/upload",
            data={"user_id": "123e4567-e89b-12d3-a456-426614174000"}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_upload_empty_file_fails(self, client):
        """Test upload fails with empty file"""
        response = client.post(
            "/api/v1/upload",
            files={"file": ("empty.tsv", io.BytesIO(b""), "text/tab-separated-values")},
            data={"user_id": "123e4567-e89b-12d3-a456-426614174000"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "Empty file" in data["error"]
    
    def test_upload_response_schema(self, client, sample_tsv_file):
        """Test upload response matches expected schema"""
        with open(sample_tsv_file, 'rb') as f:
            response = client.post(
                "/api/v1/upload",
                files={"file": ("keywords.tsv", f, "text/tab-separated-values")},
                data={"user_id": "123e4567-e89b-12d3-a456-426614174000"}
            )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response schema
        required_fields = ["file_id", "filename", "file_size", "status", "message"]
        for field in required_fields:
            assert field in data
        
        # Validate field types
        assert isinstance(data["file_id"], str)
        assert isinstance(data["filename"], str)
        assert isinstance(data["file_size"], int)
        assert isinstance(data["status"], str)
        assert isinstance(data["message"], str)
        
        # Validate status values
        assert data["status"] in ["pending", "processing", "completed", "failed"]
        
        # Validate file_id is UUID format
        import uuid
        try:
            uuid.UUID(data["file_id"])
        except ValueError:
            pytest.fail("file_id is not a valid UUID")
