"""
Contract test for GET /api/v1/analysis/{file_id}/results endpoint
Tests the analysis results retrieval functionality
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import uuid


class TestAnalysisResults:
    """Test cases for GET /api/v1/analysis/{file_id}/results endpoint"""
    
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
    def sample_analysis_results(self):
        """Sample analysis results for testing"""
        return {
            "report_id": str(uuid.uuid4()),
            "file_id": str(uuid.uuid4()),
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
                ],
                "quick_wins": [
                    {
                        "keyword": "simple project management",
                        "search_volume": 800,
                        "difficulty": 20,
                        "cpc": 1.20,
                        "opportunity_score": 78.3,
                        "category": "high"
                    }
                ],
                "high_volume_targets": [
                    {
                        "keyword": "project management software",
                        "search_volume": 8500,
                        "difficulty": 38,
                        "cpc": 3.20,
                        "opportunity_score": 82.1,
                        "category": "high"
                    }
                ]
            },
            "content_recommendations": [
                {
                    "keyword": "best project management tools",
                    "content_format": "list-article",
                    "seo_score": 92
                }
            ],
            "insights": [
                "Multiple low-competition keywords available for quick wins",
                "High commercial value keywords identified for monetization"
            ],
            "next_steps": [
                "Prioritize high-opportunity keywords for immediate content creation",
                "Create pillar content around project management software comparisons"
            ],
            "seo_content_ideas": [
                {
                    "id": str(uuid.uuid4()),
                    "title": "Best Project Management Tools for Remote Teams in 2024",
                    "content_type": "list-article",
                    "primary_keywords": [
                        "best project management tools",
                        "remote team project management",
                        "project management software"
                    ],
                    "secondary_keywords": [
                        "remote work tools",
                        "team productivity"
                    ],
                    "seo_optimization_score": 92,
                    "traffic_potential_score": 88,
                    "total_search_volume": 45000,
                    "average_difficulty": 45,
                    "average_cpc": 3.20,
                    "optimization_tips": [
                        "Include 'best project management tools' in your title and first paragraph",
                        "Create comparison sections for commercial keywords"
                    ],
                    "content_outline": "Introduction → Top 10 Tools → Detailed Reviews → Comparison Table → Conclusion"
                }
            ]
        }
    
    def test_get_analysis_results_success(self, client, valid_file_id, sample_analysis_results):
        """Test successful results retrieval for completed analysis"""
        with patch('src.api.analysis.get_analysis_results') as mock_get_results:
            mock_get_results.return_value = sample_analysis_results
            
            response = client.get(f"/api/v1/analysis/{valid_file_id}/results")
            
            assert response.status_code == 200
            data = response.json()
            assert data["report_id"] is not None
            assert "summary" in data
            assert "top_opportunities" in data
            assert "seo_content_ideas" in data
    
    def test_get_analysis_results_not_completed(self, client, valid_file_id):
        """Test results retrieval for analysis not yet completed"""
        with patch('src.api.analysis.get_analysis_results') as mock_get_results:
            mock_get_results.side_effect = ValueError("Analysis not yet completed")
            
            response = client.get(f"/api/v1/analysis/{valid_file_id}/results")
            
            assert response.status_code == 409
            data = response.json()
            assert "error" in data
            assert "Analysis not yet completed" in data["error"]
    
    def test_get_analysis_results_not_found(self, client, valid_file_id):
        """Test results retrieval for non-existent analysis"""
        with patch('src.api.analysis.get_analysis_results') as mock_get_results:
            mock_get_results.return_value = None
            
            response = client.get(f"/api/v1/analysis/{valid_file_id}/results")
            
            assert response.status_code == 404
            data = response.json()
            assert "error" in data
            assert "Analysis not found" in data["error"]
    
    def test_get_analysis_results_invalid_file_id(self, client, invalid_file_id):
        """Test results retrieval with invalid file ID format"""
        response = client.get(f"/api/v1/analysis/{invalid_file_id}/results")
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_get_analysis_results_response_schema(self, client, valid_file_id, sample_analysis_results):
        """Test analysis results response matches expected schema"""
        with patch('src.api.analysis.get_analysis_results') as mock_get_results:
            mock_get_results.return_value = sample_analysis_results
            
            response = client.get(f"/api/v1/analysis/{valid_file_id}/results")
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate top-level schema
            required_fields = [
                "report_id", "file_id", "summary", "top_opportunities",
                "content_recommendations", "insights", "next_steps", "seo_content_ideas"
            ]
            for field in required_fields:
                assert field in data
            
            # Validate summary schema
            summary = data["summary"]
            summary_fields = [
                "total_keywords", "high_opportunity_count", "medium_opportunity_count",
                "low_opportunity_count", "total_search_volume", "average_difficulty", "average_cpc"
            ]
            for field in summary_fields:
                assert field in summary
            
            # Validate top_opportunities schema
            opportunities = data["top_opportunities"]
            assert "high_opportunity_keywords" in opportunities
            assert "quick_wins" in opportunities
            assert "high_volume_targets" in opportunities
            
            # Validate SEO content ideas schema
            content_ideas = data["seo_content_ideas"]
            assert isinstance(content_ideas, list)
            if content_ideas:
                idea = content_ideas[0]
                idea_fields = [
                    "id", "title", "content_type", "primary_keywords", "secondary_keywords",
                    "seo_optimization_score", "traffic_potential_score", "total_search_volume",
                    "average_difficulty", "average_cpc", "optimization_tips", "content_outline"
                ]
                for field in idea_fields:
                    assert field in idea
    
    def test_get_analysis_results_with_pagination(self, client, valid_file_id, sample_analysis_results):
        """Test results retrieval with pagination parameters"""
        with patch('src.api.analysis.get_analysis_results') as mock_get_results:
            mock_get_results.return_value = sample_analysis_results
            
            response = client.get(
                f"/api/v1/analysis/{valid_file_id}/results",
                params={"page": 1, "limit": 10}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "summary" in data
    
    def test_get_analysis_results_with_filters(self, client, valid_file_id, sample_analysis_results):
        """Test results retrieval with filtering parameters"""
        with patch('src.api.analysis.get_analysis_results') as mock_get_results:
            mock_get_results.return_value = sample_analysis_results
            
            response = client.get(
                f"/api/v1/analysis/{valid_file_id}/results",
                params={"category": "high", "min_volume": 1000}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "summary" in data
    
    def test_get_analysis_results_database_error(self, client, valid_file_id):
        """Test results retrieval with database error"""
        with patch('src.api.analysis.get_analysis_results') as mock_get_results:
            mock_get_results.side_effect = Exception("Database connection error")
            
            response = client.get(f"/api/v1/analysis/{valid_file_id}/results")
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "Internal server error" in data["error"]
    
    def test_get_analysis_results_large_dataset(self, client, valid_file_id):
        """Test results retrieval for large dataset"""
        large_results = {
            "report_id": str(uuid.uuid4()),
            "file_id": valid_file_id,
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
            "content_recommendations": [],
            "insights": ["Large dataset processed successfully"],
            "next_steps": ["Consider breaking down into smaller analysis batches"],
            "seo_content_ideas": []
        }
        
        with patch('src.api.analysis.get_analysis_results') as mock_get_results:
            mock_get_results.return_value = large_results
            
            response = client.get(f"/api/v1/analysis/{valid_file_id}/results")
            
            assert response.status_code == 200
            data = response.json()
            assert data["summary"]["total_keywords"] == 100000
