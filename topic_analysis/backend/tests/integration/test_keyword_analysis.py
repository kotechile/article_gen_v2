"""
Integration test for complete keyword analysis workflow
Tests the end-to-end process from file upload to results generation
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import uuid
import tempfile
import os


class TestKeywordAnalysisWorkflow:
    """Integration tests for complete keyword analysis workflow"""
    
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
project planning tools	2100	28	2.10	Commercial,Informational
team collaboration software	4800	35	2.75	Commercial,Informational
project tracking tools	1800	22	1.95	Informational,Commercial
workflow management	3400	40	2.60	Informational,Commercial
task management apps	2900	30	2.25	Commercial,Informational
project management best practices	1500	18	1.50	Informational"""
    
    @pytest.fixture
    def sample_tsv_file(self, sample_tsv_content):
        """Create temporary TSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write(sample_tsv_content)
            f.flush()
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def mock_database_operations(self):
        """Mock database operations for testing"""
        with patch('src.services.file_parser.save_file_metadata') as mock_save_file, \
             patch('src.services.keyword_analyzer.save_keywords') as mock_save_keywords, \
             patch('src.services.report_generator.save_report') as mock_save_report, \
             patch('src.services.content_idea_generator.save_content_ideas') as mock_save_ideas:
            
            mock_save_file.return_value = str(uuid.uuid4())
            mock_save_keywords.return_value = True
            mock_save_report.return_value = str(uuid.uuid4())
            mock_save_ideas.return_value = [str(uuid.uuid4())]
            
            yield {
                'save_file': mock_save_file,
                'save_keywords': mock_save_keywords,
                'save_report': mock_save_report,
                'save_ideas': mock_save_ideas
            }
    
    def test_complete_workflow_success(self, client, sample_tsv_file, mock_database_operations):
        """Test complete workflow from upload to results"""
        user_id = str(uuid.uuid4())
        
        # Step 1: Upload file
        with open(sample_tsv_file, 'rb') as f:
            upload_response = client.post(
                "/api/v1/upload",
                files={"file": ("keywords.tsv", f, "text/tab-separated-values")},
                data={"user_id": user_id}
            )
        
        assert upload_response.status_code == 200
        upload_data = upload_response.json()
        file_id = upload_data["file_id"]
        assert upload_data["status"] == "pending"
        
        # Step 2: Start analysis
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
        analysis_id = analysis_data["analysis_id"]
        assert analysis_data["status"] == "started"
        
        # Step 3: Check analysis status (processing)
        with patch('src.api.analysis.get_analysis_status') as mock_status:
            mock_status.return_value = {
                "analysis_id": analysis_id,
                "file_id": file_id,
                "status": "processing",
                "progress_percentage": 50,
                "started_at": "2024-01-01T00:00:00Z",
                "estimated_completion": "2024-01-01T00:05:00Z",
                "keywords_processed": 5,
                "total_keywords": 10
            }
            
            status_response = client.get(f"/api/v1/analysis/{file_id}/status")
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert status_data["status"] == "processing"
        
        # Step 4: Check analysis status (completed)
        with patch('src.api.analysis.get_analysis_status') as mock_status:
            mock_status.return_value = {
                "analysis_id": analysis_id,
                "file_id": file_id,
                "status": "completed",
                "progress_percentage": 100,
                "started_at": "2024-01-01T00:00:00Z",
                "completed_at": "2024-01-01T00:03:00Z",
                "keywords_processed": 10,
                "total_keywords": 10,
                "report_id": str(uuid.uuid4())
            }
            
            status_response = client.get(f"/api/v1/analysis/{file_id}/status")
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert status_data["status"] == "completed"
            report_id = status_data["report_id"]
        
        # Step 5: Get analysis results
        with patch('src.api.analysis.get_analysis_results') as mock_results:
            mock_results.return_value = {
                "report_id": report_id,
                "file_id": file_id,
                "summary": {
                    "total_keywords": 10,
                    "high_opportunity_count": 3,
                    "medium_opportunity_count": 4,
                    "low_opportunity_count": 3,
                    "total_search_volume": 50000,
                    "average_difficulty": 32.5,
                    "average_cpc": 2.25
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
                            "keyword": "project tracking tools",
                            "search_volume": 1800,
                            "difficulty": 22,
                            "cpc": 1.95,
                            "opportunity_score": 78.3,
                            "category": "high"
                        }
                    ],
                    "high_volume_targets": [
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
                            "project management software",
                            "team collaboration software"
                        ],
                        "secondary_keywords": [
                            "remote work tools",
                            "project tracking",
                            "workflow management"
                        ],
                        "seo_optimization_score": 92,
                        "traffic_potential_score": 88,
                        "total_search_volume": 45000,
                        "average_difficulty": 45,
                        "average_cpc": 3.20,
                        "optimization_tips": [
                            "Include 'best project management tools' in your title and first paragraph",
                            "Create comparison sections for commercial keywords like 'Asana vs Trello'",
                            "Use secondary keywords like 'remote work tools' in H2 and H3 headings"
                        ],
                        "content_outline": "Introduction → Top 10 Tools → Detailed Reviews → Comparison Table → Conclusion"
                    }
                ]
            }
            
            results_response = client.get(f"/api/v1/analysis/{file_id}/results")
            assert results_response.status_code == 200
            results_data = results_response.json()
            assert "summary" in results_data
            assert "top_opportunities" in results_data
            assert "seo_content_ideas" in results_data
            assert len(results_data["seo_content_ideas"]) > 0
        
        # Step 6: Get report details
        with patch('src.api.reports.get_report') as mock_report:
            mock_report.return_value = {
                "report_id": report_id,
                "user_id": user_id,
                "filename": "keywords.tsv",
                "total_keywords": 10,
                "high_opportunity_count": 3,
                "medium_opportunity_count": 4,
                "low_opportunity_count": 3,
                "total_search_volume": 50000,
                "average_difficulty": 32.5,
                "average_cpc": 2.25,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:03:00Z",
                "expires_at": "2024-04-01T00:00:00Z"
            }
            
            report_response = client.get(f"/api/v1/reports/{report_id}")
            assert report_response.status_code == 200
            report_data = report_response.json()
            assert report_data["total_keywords"] == 10
            assert report_data["high_opportunity_count"] == 3
        
        # Step 7: Export report
        with patch('src.api.reports.export_report') as mock_export:
            mock_export.return_value = {
                "report_id": report_id,
                "user_id": user_id,
                "filename": "keywords.tsv",
                "export_format": "json",
                "export_timestamp": "2024-01-01T00:00:00Z",
                "data": results_data
            }
            
            export_response = client.get(f"/api/v1/reports/{report_id}/export")
            assert export_response.status_code == 200
            export_data = export_response.json()
            assert export_data["export_format"] == "json"
            assert "data" in export_data
    
    def test_workflow_with_large_dataset(self, client, mock_database_operations):
        """Test workflow with large dataset (50,000 keywords)"""
        # Create large TSV content
        large_content = "Keyword\tVolume\tDifficulty\tCPC\tIntents\n"
        for i in range(100):  # Simulate 100 keywords for testing
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
            
            # Start analysis with processing simulation
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
            
        finally:
            os.unlink(large_file)
    
    def test_workflow_error_handling(self, client, sample_tsv_file):
        """Test workflow error handling and recovery"""
        user_id = str(uuid.uuid4())
        
        # Upload file
        with open(sample_tsv_file, 'rb') as f:
            upload_response = client.post(
                "/api/v1/upload",
                files={"file": ("keywords.tsv", f, "text/tab-separated-values")},
                data={"user_id": user_id}
            )
        
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]
        
        # Start analysis with database error simulation
        with patch('src.api.analysis.start_analysis') as mock_start:
            mock_start.side_effect = Exception("Database connection error")
            
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
            
            assert analysis_response.status_code == 500
            error_data = analysis_response.json()
            assert "error" in error_data
            assert "Internal server error" in error_data["error"]
    
    def test_workflow_performance_requirements(self, client, sample_tsv_file, mock_database_operations):
        """Test workflow meets performance requirements"""
        import time
        
        user_id = str(uuid.uuid4())
        
        # Upload file
        start_time = time.time()
        with open(sample_tsv_file, 'rb') as f:
            upload_response = client.post(
                "/api/v1/upload",
                files={"file": ("keywords.tsv", f, "text/tab-separated-values")},
                data={"user_id": user_id}
            )
        upload_time = time.time() - start_time
        
        assert upload_response.status_code == 200
        assert upload_time < 1.0  # Upload should complete in under 1 second
        
        file_id = upload_response.json()["file_id"]
        
        # Start analysis
        start_time = time.time()
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
        analysis_start_time = time.time() - start_time
        
        assert analysis_response.status_code == 200
        assert analysis_start_time < 0.2  # Analysis start should be under 200ms
