"""
Contract test for POST /api/v1/analysis/{file_id}/start endpoint
Tests the keyword analysis start functionality
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import uuid


class TestAnalysisStart:
    """Test cases for POST /api/v1/analysis/{file_id}/start endpoint"""
    
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
    
    @pytest.fixture
    def default_scoring_weights(self):
        """Default scoring weights for testing"""
        return {
            "search_volume": 0.4,
            "keyword_difficulty": 0.3,
            "cpc": 0.2,
            "search_intent": 0.1
        }
    
    def test_start_analysis_success(self, client, valid_file_id, default_scoring_weights):
        """Test successful analysis start with default weights"""
        with patch('src.api.analysis.start_analysis') as mock_start_analysis:
            mock_start_analysis.return_value = {
                "analysis_id": str(uuid.uuid4()),
                "file_id": valid_file_id,
                "status": "started",
                "message": "Analysis started successfully",
                "estimated_completion_time": "2024-01-01T00:05:00Z"
            }
            
            response = client.post(
                f"/api/v1/analysis/{valid_file_id}/start",
                json={"scoring_weights": default_scoring_weights}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "started"
            assert "analysis_id" in data
            assert "estimated_completion_time" in data
    
    def test_start_analysis_custom_weights(self, client, valid_file_id):
        """Test analysis start with custom scoring weights"""
        custom_weights = {
            "search_volume": 0.5,
            "keyword_difficulty": 0.2,
            "cpc": 0.2,
            "search_intent": 0.1
        }
        
        with patch('src.api.analysis.start_analysis') as mock_start_analysis:
            mock_start_analysis.return_value = {
                "analysis_id": str(uuid.uuid4()),
                "file_id": valid_file_id,
                "status": "started",
                "message": "Analysis started successfully",
                "estimated_completion_time": "2024-01-01T00:05:00Z"
            }
            
            response = client.post(
                f"/api/v1/analysis/{valid_file_id}/start",
                json={"scoring_weights": custom_weights}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "started"
    
    def test_start_analysis_invalid_file_id(self, client, invalid_file_id, default_scoring_weights):
        """Test analysis start with invalid file ID format"""
        response = client.post(
            f"/api/v1/analysis/{invalid_file_id}/start",
            json={"scoring_weights": default_scoring_weights}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_start_analysis_file_not_found(self, client, valid_file_id, default_scoring_weights):
        """Test analysis start with non-existent file"""
        with patch('src.api.analysis.start_analysis') as mock_start_analysis:
            mock_start_analysis.side_effect = FileNotFoundError("File not found")
            
            response = client.post(
                f"/api/v1/analysis/{valid_file_id}/start",
                json={"scoring_weights": default_scoring_weights}
            )
            
            assert response.status_code == 404
            data = response.json()
            assert "error" in data
            assert "File not found" in data["error"]
    
    def test_start_analysis_file_already_processing(self, client, valid_file_id, default_scoring_weights):
        """Test analysis start when file is already being processed"""
        with patch('src.api.analysis.start_analysis') as mock_start_analysis:
            mock_start_analysis.side_effect = ValueError("File is already being processed")
            
            response = client.post(
                f"/api/v1/analysis/{valid_file_id}/start",
                json={"scoring_weights": default_scoring_weights}
            )
            
            assert response.status_code == 409
            data = response.json()
            assert "error" in data
            assert "already being processed" in data["error"]
    
    def test_start_analysis_invalid_scoring_weights(self, client, valid_file_id):
        """Test analysis start with invalid scoring weights"""
        invalid_weights = {
            "search_volume": 0.6,  # Total > 1.0
            "keyword_difficulty": 0.3,
            "cpc": 0.2,
            "search_intent": 0.1
        }
        
        response = client.post(
            f"/api/v1/analysis/{valid_file_id}/start",
            json={"scoring_weights": invalid_weights}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_start_analysis_missing_scoring_weights(self, client, valid_file_id):
        """Test analysis start without scoring weights (should use defaults)"""
        with patch('src.api.analysis.start_analysis') as mock_start_analysis:
            mock_start_analysis.return_value = {
                "analysis_id": str(uuid.uuid4()),
                "file_id": valid_file_id,
                "status": "started",
                "message": "Analysis started successfully",
                "estimated_completion_time": "2024-01-01T00:05:00Z"
            }
            
            response = client.post(f"/api/v1/analysis/{valid_file_id}/start")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "started"
    
    def test_start_analysis_negative_weights(self, client, valid_file_id):
        """Test analysis start with negative scoring weights"""
        negative_weights = {
            "search_volume": -0.4,
            "keyword_difficulty": 0.3,
            "cpc": 0.2,
            "search_intent": 0.1
        }
        
        response = client.post(
            f"/api/v1/analysis/{valid_file_id}/start",
            json={"scoring_weights": negative_weights}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_start_analysis_response_schema(self, client, valid_file_id, default_scoring_weights):
        """Test analysis start response matches expected schema"""
        with patch('src.api.analysis.start_analysis') as mock_start_analysis:
            mock_start_analysis.return_value = {
                "analysis_id": str(uuid.uuid4()),
                "file_id": valid_file_id,
                "status": "started",
                "message": "Analysis started successfully",
                "estimated_completion_time": "2024-01-01T00:05:00Z"
            }
            
            response = client.post(
                f"/api/v1/analysis/{valid_file_id}/start",
                json={"scoring_weights": default_scoring_weights}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate response schema
            required_fields = [
                "analysis_id", "file_id", "status", "message", 
                "estimated_completion_time"
            ]
            for field in required_fields:
                assert field in data
            
            # Validate field types
            assert isinstance(data["analysis_id"], str)
            assert isinstance(data["file_id"], str)
            assert isinstance(data["status"], str)
            assert isinstance(data["message"], str)
            assert isinstance(data["estimated_completion_time"], str)
            
            # Validate status values
            assert data["status"] in ["started", "failed"]
            
            # Validate UUID format for analysis_id
            import uuid
            try:
                uuid.UUID(data["analysis_id"])
            except ValueError:
                pytest.fail("analysis_id is not a valid UUID")
    
    def test_start_analysis_database_error(self, client, valid_file_id, default_scoring_weights):
        """Test analysis start with database error"""
        with patch('src.api.analysis.start_analysis') as mock_start_analysis:
            mock_start_analysis.side_effect = Exception("Database connection error")
            
            response = client.post(
                f"/api/v1/analysis/{valid_file_id}/start",
                json={"scoring_weights": default_scoring_weights}
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "Internal server error" in data["error"]
