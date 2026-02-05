"""
Contract tests for Export Integration API endpoints
These tests MUST fail before implementation - they define the expected API contract
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


class TestExportIntegrationContract:
    """Test contract for export integration endpoints"""
    
    def setup_method(self):
        """Setup test client"""
        # This will fail until we implement the main app
        from src.main import app
        self.client = TestClient(app)
    
    def test_post_export_google_docs_contract(self):
        """Test POST /api/export/google-docs contract"""
        # This test will fail until we implement the endpoint
        payload = {
            "content_ids": ["123e4567-e89b-12d3-a456-426614174000"],
            "software_ids": ["123e4567-e89b-12d3-a456-426614174001"],
            "platform": "google_docs",
            "template": "default",
            "format_options": {
                "include_seo": True,
                "include_outline": True
            }
        }
        
        response = self.client.post("/api/export/google-docs", json=payload)
        
        # Contract requirements
        assert response.status_code == 201
        data = response.json()
        
        # Required fields in response
        assert "export_id" in data
        assert "platform" in data
        assert "url" in data
        assert "status" in data
        assert "created_at" in data
        
        # Platform should match
        assert data["platform"] == "google_docs"
        
        # Status should be PROCESSING initially
        assert data["status"] == "PROCESSING"
        
        # URL should be valid Google Docs URL
        assert "docs.google.com" in data["url"]
        
        # ID should be UUID format
        import uuid
        assert uuid.UUID(data["export_id"])
    
    def test_post_export_notion_contract(self):
        """Test POST /api/export/notion contract"""
        # This test will fail until we implement the endpoint
        payload = {
            "content_ids": ["123e4567-e89b-12d3-a456-426614174000"],
            "platform": "notion",
            "template": "notion-template",
            "format_options": {
                "include_seo": True,
                "include_outline": True
            }
        }
        
        response = self.client.post("/api/export/notion", json=payload)
        
        # Contract requirements
        assert response.status_code == 201
        data = response.json()
        
        # Required fields in response
        assert "export_id" in data
        assert "platform" in data
        assert "url" in data
        assert "status" in data
        assert "created_at" in data
        
        # Platform should match
        assert data["platform"] == "notion"
        
        # Status should be PROCESSING initially
        assert data["status"] == "PROCESSING"
        
        # URL should be valid Notion URL
        assert "notion.so" in data["url"]
    
    def test_post_export_wordpress_contract(self):
        """Test POST /api/export/wordpress contract"""
        # This test will fail until we implement the endpoint
        payload = {
            "content_ids": ["123e4567-e89b-12d3-a456-426614174000"],
            "platform": "wordpress",
            "template": "wordpress-template",
            "format_options": {
                "include_seo": True,
                "include_outline": True,
                "publish_status": "draft"
            }
        }
        
        response = self.client.post("/api/export/wordpress", json=payload)
        
        # Contract requirements
        assert response.status_code == 201
        data = response.json()
        
        # Required fields in response
        assert "export_id" in data
        assert "platform" in data
        assert "url" in data
        assert "status" in data
        assert "created_at" in data
        
        # Platform should match
        assert data["platform"] == "wordpress"
        
        # Status should be PROCESSING initially
        assert data["status"] == "PROCESSING"
    
    def test_export_validation_contract(self):
        """Test validation contract for export"""
        # Test missing required fields
        payload = {}
        response = self.client.post("/api/export/google-docs", json=payload)
        assert response.status_code == 422
        
        # Test invalid platform
        payload = {
            "content_ids": ["123e4567-e89b-12d3-a456-426614174000"],
            "platform": "invalid_platform"
        }
        response = self.client.post("/api/export/google-docs", json=payload)
        assert response.status_code == 422
        
        # Test empty content_ids and software_ids
        payload = {
            "content_ids": [],
            "software_ids": [],
            "platform": "google_docs"
        }
        response = self.client.post("/api/export/google-docs", json=payload)
        assert response.status_code == 422
        
        # Test invalid UUID format
        payload = {
            "content_ids": ["invalid-id"],
            "platform": "google_docs"
        }
        response = self.client.post("/api/export/google-docs", json=payload)
        assert response.status_code == 422
    
    def test_export_error_handling_contract(self):
        """Test error handling contract"""
        # Test 404 for non-existent export
        export_id = "00000000-0000-0000-0000-000000000000"
        response = self.client.get(f"/api/export/{export_id}")
        assert response.status_code == 404
        
        # Test invalid UUID format
        response = self.client.get("/api/export/invalid-id")
        assert response.status_code == 422
    
    def test_export_authentication_contract(self):
        """Test authentication contract"""
        # Test unauthenticated export request
        payload = {
            "content_ids": ["123e4567-e89b-12d3-a456-426614174000"],
            "platform": "google_docs"
        }
        response = self.client.post("/api/export/google-docs", json=payload)
        # Should require authentication
        assert response.status_code == 401
    
    def test_export_processing_states_contract(self):
        """Test processing states contract"""
        # Test that status transitions are handled correctly
        export_id = "123e4567-e89b-12d3-a456-426614174000"
        
        # Mock different status responses
        with patch('src.services.export_service.ExportService.get_export') as mock_get:
            # Test PROCESSING state
            mock_get.return_value = {
                "export_id": export_id,
                "platform": "google_docs",
                "url": "https://docs.google.com/document/d/test",
                "status": "PROCESSING",
                "created_at": "2025-10-02T10:00:00Z"
            }
            
            response = self.client.get(f"/api/export/{export_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "PROCESSING"
            
            # Test COMPLETED state
            mock_get.return_value["status"] = "COMPLETED"
            response = self.client.get(f"/api/export/{export_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "COMPLETED"
            
            # Test FAILED state
            mock_get.return_value["status"] = "FAILED"
            response = self.client.get(f"/api/export/{export_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "FAILED"
    
    def test_export_platform_specific_validation_contract(self):
        """Test platform-specific validation contract"""
        # Test Google Docs specific validation
        payload = {
            "content_ids": ["123e4567-e89b-12d3-a456-426614174000"],
            "platform": "google_docs",
            "format_options": {
                "include_seo": True,
                "include_outline": True,
                "google_docs_template": "trendtap-template"
            }
        }
        response = self.client.post("/api/export/google-docs", json=payload)
        assert response.status_code == 201
        
        # Test Notion specific validation
        payload = {
            "content_ids": ["123e4567-e89b-12d3-a456-426614174000"],
            "platform": "notion",
            "format_options": {
                "include_seo": True,
                "notion_database_id": "notion-db-id"
            }
        }
        response = self.client.post("/api/export/notion", json=payload)
        assert response.status_code == 201
        
        # Test WordPress specific validation
        payload = {
            "content_ids": ["123e4567-e89b-12d3-a456-426614174000"],
            "platform": "wordpress",
            "format_options": {
                "include_seo": True,
                "publish_status": "draft",
                "wordpress_category": "trendtap"
            }
        }
        response = self.client.post("/api/export/wordpress", json=payload)
        assert response.status_code == 201
