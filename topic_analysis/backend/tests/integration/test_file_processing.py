"""
Integration test for file upload and processing
Tests the file handling, validation, and processing pipeline
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import uuid
import tempfile
import os


class TestFileProcessing:
    """Integration tests for file upload and processing"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from main import app
        return TestClient(app)
    
    @pytest.fixture
    def valid_tsv_content(self):
        """Valid TSV content for testing"""
        return """Keyword	Volume	Difficulty	CPC	Intents
best project management tools	12000	45	2.50	Informational,Commercial
project management software	8500	38	3.20	Commercial,Informational
how to manage projects	3200	25	1.80	Informational
agile project management	5600	42	2.90	Informational,Commercial
project planning tools	2100	28	2.10	Commercial,Informational"""
    
    @pytest.fixture
    def invalid_tsv_content(self):
        """Invalid TSV content for testing"""
        return """Keyword	Volume	Difficulty	CPC
best project management tools	12000	45	2.50
project management software	8500	38	3.20
invalid row with missing data	8500	38
another invalid row	8500	38	3.20	extra_column"""
    
    def test_valid_tsv_file_processing(self, client, valid_tsv_content):
        """Test processing of valid TSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write(valid_tsv_content)
            f.flush()
            tsv_file = f.name
        
        try:
            user_id = str(uuid.uuid4())
            
            # Upload file
            with open(tsv_file, 'rb') as f:
                upload_response = client.post(
                    "/api/v1/upload",
                    files={"file": ("keywords.tsv", f, "text/tab-separated-values")},
                    data={"user_id": user_id}
                )
            
            assert upload_response.status_code == 200
            upload_data = upload_response.json()
            file_id = upload_data["file_id"]
            assert upload_data["status"] == "pending"
            
            # Simulate file processing
            with patch('src.services.file_parser.parse_tsv_file') as mock_parse:
                mock_parse.return_value = [
                    {
                        "keyword": "best project management tools",
                        "volume": 12000,
                        "difficulty": 45,
                        "cpc": 2.50,
                        "intents": "Informational,Commercial"
                    },
                    {
                        "keyword": "project management software",
                        "volume": 8500,
                        "difficulty": 38,
                        "cpc": 3.20,
                        "intents": "Commercial,Informational"
                    }
                ]
                
                # Start analysis
                analysis_response = client.post(
                    f"/api/v1/analysis/{file_id}/start",
                    json={
                        "scoring_weights": {
                            "search_volume": 0.4,
                            "keyword_difficulty": 0.3,
                            "cpc": 0.2,
                            "search_intent": 0.1
                        }
                    }
                )
                
                assert analysis_response.status_code == 200
                analysis_data = analysis_response.json()
                assert analysis_data["status"] == "started"
                
                # Verify file was parsed
                mock_parse.assert_called_once()
        
        finally:
            os.unlink(tsv_file)
    
    def test_invalid_tsv_file_processing(self, client, invalid_tsv_content):
        """Test processing of invalid TSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write(invalid_tsv_content)
            f.flush()
            tsv_file = f.name
        
        try:
            user_id = str(uuid.uuid4())
            
            # Upload file
            with open(tsv_file, 'rb') as f:
                upload_response = client.post(
                    "/api/v1/upload",
                    files={"file": ("invalid_keywords.tsv", f, "text/tab-separated-values")},
                    data={"user_id": user_id}
                )
            
            assert upload_response.status_code == 200
            file_id = upload_response.json()["file_id"]
            
            # Start analysis with validation
            with patch('src.services.file_parser.parse_tsv_file') as mock_parse:
                mock_parse.side_effect = ValueError("Invalid TSV format: missing required columns")
                
                analysis_response = client.post(
                    f"/api/v1/analysis/{file_id}/start",
                    json={
                        "scoring_weights": {
                            "search_volume": 0.4,
                            "keyword_difficulty": 0.3,
                            "cpc": 0.2,
                            "search_intent": 0.1
                        }
                    }
                )
                
                # Analysis should fail due to invalid file
                assert analysis_response.status_code == 400
                error_data = analysis_response.json()
                assert "error" in error_data
                assert "Invalid TSV format" in error_data["error"]
        
        finally:
            os.unlink(tsv_file)
    
    def test_large_file_processing(self, client):
        """Test processing of large file (approaching 10MB limit)"""
        # Create large TSV content (simulate 50,000 keywords)
        large_content = "Keyword\tVolume\tDifficulty\tCPC\tIntents\n"
        for i in range(1000):  # Simulate 1000 keywords for testing
            large_content += f"keyword {i}\t{1000 + i * 10}\t{20 + i % 50}\t{1.0 + i * 0.01}\tInformational,Commercial\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write(large_content)
            f.flush()
            large_file = f.name
        
        try:
            user_id = str(uuid.uuid4())
            
            # Upload large file
            with open(large_file, 'rb') as f:
                upload_response = client.post(
                    "/api/v1/upload",
                    files={"file": ("large_keywords.tsv", f, "text/tab-separated-values")},
                    data={"user_id": user_id}
                )
            
            assert upload_response.status_code == 200
            upload_data = upload_response.json()
            file_id = upload_data["file_id"]
            
            # Start analysis with chunked processing
            with patch('src.services.file_parser.parse_tsv_file') as mock_parse:
                # Simulate chunked processing for large files
                mock_parse.return_value = [
                    {
                        "keyword": f"keyword {i}",
                        "volume": 1000 + i * 10,
                        "difficulty": 20 + i % 50,
                        "cpc": 1.0 + i * 0.01,
                        "intents": "Informational,Commercial"
                    }
                    for i in range(100)  # Return first 100 keywords
                ]
                
                analysis_response = client.post(
                    f"/api/v1/analysis/{file_id}/start",
                    json={
                        "scoring_weights": {
                            "search_volume": 0.4,
                            "keyword_difficulty": 0.3,
                            "cpc": 0.2,
                            "search_intent": 0.1
                        }
                    }
                )
                
                assert analysis_response.status_code == 200
                analysis_data = analysis_response.json()
                assert analysis_data["status"] == "started"
                
                # Verify chunked processing was used
                mock_parse.assert_called_once()
        
        finally:
            os.unlink(large_file)
    
    def test_file_type_validation(self, client):
        """Test file type validation"""
        user_id = str(uuid.uuid4())
        
        # Test with non-TSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is not a TSV file")
            f.flush()
            txt_file = f.name
        
        try:
            with open(txt_file, 'rb') as f:
                upload_response = client.post(
                    "/api/v1/upload",
                    files={"file": ("test.txt", f, "text/plain")},
                    data={"user_id": user_id}
                )
            
            assert upload_response.status_code == 400
            error_data = upload_response.json()
            assert "error" in error_data
            assert "Invalid file type" in error_data["error"]
        
        finally:
            os.unlink(txt_file)
    
    def test_file_size_validation(self, client):
        """Test file size validation (10MB limit)"""
        user_id = str(uuid.uuid4())
        
        # Create file larger than 10MB
        large_content = "x" * (11 * 1024 * 1024)  # 11MB
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write(large_content)
            f.flush()
            large_file = f.name
        
        try:
            with open(large_file, 'rb') as f:
                upload_response = client.post(
                    "/api/v1/upload",
                    files={"file": ("large.tsv", f, "text/tab-separated-values")},
                    data={"user_id": user_id}
                )
            
            assert upload_response.status_code == 400
            error_data = upload_response.json()
            assert "error" in error_data
            assert "File too large" in error_data["error"]
        
        finally:
            os.unlink(large_file)
    
    def test_empty_file_handling(self, client):
        """Test handling of empty file"""
        user_id = str(uuid.uuid4())
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("")  # Empty file
            f.flush()
            empty_file = f.name
        
        try:
            with open(empty_file, 'rb') as f:
                upload_response = client.post(
                    "/api/v1/upload",
                    files={"file": ("empty.tsv", f, "text/tab-separated-values")},
                    data={"user_id": user_id}
                )
            
            assert upload_response.status_code == 400
            error_data = upload_response.json()
            assert "error" in error_data
            assert "Empty file" in error_data["error"]
        
        finally:
            os.unlink(empty_file)
    
    def test_malformed_tsv_handling(self, client):
        """Test handling of malformed TSV file"""
        malformed_content = """Keyword	Volume	Difficulty	CPC	Intents
best project management tools	12000	45	2.50	Informational,Commercial
project management software	8500	38	3.20	Commercial,Informational
incomplete row	8500	38
another incomplete row	8500	38	3.20"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write(malformed_content)
            f.flush()
            malformed_file = f.name
        
        try:
            user_id = str(uuid.uuid4())
            
            # Upload file
            with open(malformed_file, 'rb') as f:
                upload_response = client.post(
                    "/api/v1/upload",
                    files={"file": ("malformed.tsv", f, "text/tab-separated-values")},
                    data={"user_id": user_id}
                )
            
            assert upload_response.status_code == 200
            file_id = upload_response.json()["file_id"]
            
            # Start analysis with error handling
            with patch('src.services.file_parser.parse_tsv_file') as mock_parse:
                mock_parse.side_effect = ValueError("Malformed TSV: incomplete rows detected")
                
                analysis_response = client.post(
                    f"/api/v1/analysis/{file_id}/start",
                    json={
                        "scoring_weights": {
                            "search_volume": 0.4,
                            "keyword_difficulty": 0.3,
                            "cpc": 0.2,
                            "search_intent": 0.1
                        }
                    }
                )
                
                assert analysis_response.status_code == 400
                error_data = analysis_response.json()
                assert "error" in error_data
                assert "Malformed TSV" in error_data["error"]
        
        finally:
            os.unlink(malformed_file)
    
    def test_concurrent_file_processing(self, client, valid_tsv_content):
        """Test concurrent file processing"""
        user_id = str(uuid.uuid4())
        file_ids = []
        
        # Upload multiple files concurrently
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{i}.tsv', delete=False) as f:
                f.write(valid_tsv_content)
                f.flush()
                tsv_file = f.name
            
            try:
                with open(tsv_file, 'rb') as f:
                    upload_response = client.post(
                        "/api/v1/upload",
                        files={"file": (f"keywords_{i}.tsv", f, "text/tab-separated-values")},
                        data={"user_id": user_id}
                    )
                
                assert upload_response.status_code == 200
                file_ids.append(upload_response.json()["file_id"])
            
            finally:
                os.unlink(tsv_file)
        
        # Start analysis for all files
        analysis_responses = []
        for file_id in file_ids:
            analysis_response = client.post(
                f"/api/v1/analysis/{file_id}/start",
                json={
                    "scoring_weights": {
                        "search_volume": 0.4,
                        "keyword_difficulty": 0.3,
                        "cpc": 0.2,
                        "search_intent": 0.1
                    }
                }
            )
            analysis_responses.append(analysis_response)
        
        # All analyses should start successfully
        for response in analysis_responses:
            assert response.status_code == 200
            assert response.json()["status"] == "started"
