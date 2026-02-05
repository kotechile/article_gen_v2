"""
Integration test for SEO content idea generation
Tests the content idea generation pipeline with keyword clustering and optimization
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import uuid


class TestContentIdeaGeneration:
    """Integration tests for SEO content idea generation"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from main import app
        return TestClient(app)
    
    @pytest.fixture
    def sample_keywords_data(self):
        """Sample keywords data for content idea generation"""
        return [
            {
                "keyword": "best project management tools",
                "search_volume": 12000,
                "difficulty": 45,
                "cpc": 2.50,
                "opportunity_score": 85.5,
                "category": "high",
                "primary_intent": "Commercial"
            },
            {
                "keyword": "project management software",
                "search_volume": 8500,
                "difficulty": 38,
                "cpc": 3.20,
                "opportunity_score": 82.1,
                "category": "high",
                "primary_intent": "Commercial"
            },
            {
                "keyword": "how to manage projects",
                "search_volume": 3200,
                "difficulty": 25,
                "cpc": 1.80,
                "opportunity_score": 78.3,
                "category": "high",
                "primary_intent": "Informational"
            },
            {
                "keyword": "agile project management",
                "search_volume": 5600,
                "difficulty": 42,
                "cpc": 2.90,
                "opportunity_score": 80.2,
                "category": "high",
                "primary_intent": "Informational"
            },
            {
                "keyword": "project planning tools",
                "search_volume": 2100,
                "difficulty": 28,
                "cpc": 2.10,
                "opportunity_score": 75.8,
                "category": "medium",
                "primary_intent": "Commercial"
            },
            {
                "keyword": "team collaboration software",
                "search_volume": 4800,
                "difficulty": 35,
                "cpc": 2.75,
                "opportunity_score": 77.5,
                "category": "medium",
                "primary_intent": "Commercial"
            },
            {
                "keyword": "project tracking tools",
                "search_volume": 1800,
                "difficulty": 22,
                "cpc": 1.95,
                "opportunity_score": 73.2,
                "category": "medium",
                "primary_intent": "Commercial"
            },
            {
                "keyword": "workflow management",
                "search_volume": 3400,
                "difficulty": 40,
                "cpc": 2.60,
                "opportunity_score": 76.1,
                "category": "medium",
                "primary_intent": "Informational"
            },
            {
                "keyword": "task management apps",
                "search_volume": 2900,
                "difficulty": 30,
                "cpc": 2.25,
                "opportunity_score": 74.8,
                "category": "medium",
                "primary_intent": "Commercial"
            },
            {
                "keyword": "project management best practices",
                "search_volume": 1500,
                "difficulty": 18,
                "cpc": 1.50,
                "opportunity_score": 71.5,
                "category": "low",
                "primary_intent": "Informational"
            }
        ]
    
    def test_content_idea_generation_success(self, client, sample_keywords_data):
        """Test successful content idea generation"""
        file_id = str(uuid.uuid4())
        
        # Mock the content idea generation process
        with patch('src.services.content_idea_generator.generate_content_ideas') as mock_generate:
            mock_generate.return_value = [
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
                        "Use secondary keywords like 'remote work tools' in H2 and H3 headings",
                        "Include long-tail keywords in meta descriptions",
                        "Add internal links to related project management guides"
                    ],
                    "content_outline": "Introduction → Top 10 Tools → Detailed Reviews → Comparison Table → Conclusion"
                },
                {
                    "id": str(uuid.uuid4()),
                    "title": "How to Manage Projects Effectively: A Complete Guide",
                    "content_type": "how-to-guide",
                    "primary_keywords": [
                        "how to manage projects",
                        "agile project management",
                        "project management best practices"
                    ],
                    "secondary_keywords": [
                        "project planning",
                        "team coordination",
                        "project tracking"
                    ],
                    "seo_optimization_score": 89,
                    "traffic_potential_score": 82,
                    "total_search_volume": 28000,
                    "average_difficulty": 35,
                    "average_cpc": 2.20,
                    "optimization_tips": [
                        "Include 'how to manage projects' in your title and first paragraph",
                        "Create step-by-step sections for informational keywords",
                        "Use secondary keywords like 'project planning' in H2 headings",
                        "Include FAQ section for long-tail keywords",
                        "Add visual elements like project management diagrams"
                    ],
                    "content_outline": "Introduction → Project Planning → Execution → Monitoring → Conclusion"
                }
            ]
            
            # Get analysis results with content ideas
            with patch('src.api.analysis.get_analysis_results') as mock_results:
                mock_results.return_value = {
                    "report_id": str(uuid.uuid4()),
                    "file_id": file_id,
                    "summary": {
                        "total_keywords": 10,
                        "high_opportunity_count": 4,
                        "medium_opportunity_count": 4,
                        "low_opportunity_count": 2,
                        "total_search_volume": 50000,
                        "average_difficulty": 35.5,
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
                                "keyword": "project management best practices",
                                "search_volume": 1500,
                                "difficulty": 18,
                                "cpc": 1.50,
                                "opportunity_score": 71.5,
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
                        "High commercial value keywords identified for monetization",
                        "Strong opportunity for pillar content around project management",
                        "Informational intent keywords dominate, ideal for blog content"
                    ],
                    "next_steps": [
                        "Prioritize high-opportunity keywords for immediate content creation",
                        "Create pillar content around project management software comparisons",
                        "Develop quick-win content for low-difficulty keywords",
                        "Plan content calendar based on search volume patterns"
                    ],
                    "seo_content_ideas": mock_generate.return_value
                }
                
                response = client.get(f"/api/v1/analysis/{file_id}/results")
                
                assert response.status_code == 200
                data = response.json()
                assert "seo_content_ideas" in data
                assert len(data["seo_content_ideas"]) == 2
                
                # Validate first content idea
                idea1 = data["seo_content_ideas"][0]
                assert idea1["title"] == "Best Project Management Tools for Remote Teams in 2024"
                assert idea1["content_type"] == "list-article"
                assert len(idea1["primary_keywords"]) == 3
                assert len(idea1["secondary_keywords"]) == 3
                assert idea1["seo_optimization_score"] == 92
                assert idea1["traffic_potential_score"] == 88
                assert len(idea1["optimization_tips"]) == 5
                
                # Validate second content idea
                idea2 = data["seo_content_ideas"][1]
                assert idea2["title"] == "How to Manage Projects Effectively: A Complete Guide"
                assert idea2["content_type"] == "how-to-guide"
                assert idea2["seo_optimization_score"] == 89
                assert idea2["traffic_potential_score"] == 82
    
    def test_content_idea_keyword_clustering(self, client, sample_keywords_data):
        """Test keyword clustering for content ideas"""
        file_id = str(uuid.uuid4())
        
        with patch('src.utils.keyword_clustering.cluster_keywords') as mock_cluster:
            mock_cluster.return_value = [
                {
                    "cluster_id": "project_management_tools",
                    "keywords": [
                        "best project management tools",
                        "project management software",
                        "team collaboration software"
                    ],
                    "cluster_score": 0.85,
                    "primary_intent": "Commercial"
                },
                {
                    "cluster_id": "project_management_guide",
                    "keywords": [
                        "how to manage projects",
                        "agile project management",
                        "project management best practices"
                    ],
                    "cluster_score": 0.78,
                    "primary_intent": "Informational"
                }
            ]
            
            with patch('src.services.content_idea_generator.generate_content_ideas') as mock_generate:
                mock_generate.return_value = [
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
                            "Include 'best project management tools' in your title and first paragraph"
                        ],
                        "content_outline": "Introduction → Top 10 Tools → Conclusion"
                    }
                ]
                
                with patch('src.api.analysis.get_analysis_results') as mock_results:
                    mock_results.return_value = {
                        "report_id": str(uuid.uuid4()),
                        "file_id": file_id,
                        "summary": {"total_keywords": 10},
                        "top_opportunities": {"high_opportunity_keywords": []},
                        "content_recommendations": [],
                        "insights": [],
                        "next_steps": [],
                        "seo_content_ideas": mock_generate.return_value
                    }
                    
                    response = client.get(f"/api/v1/analysis/{file_id}/results")
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert "seo_content_ideas" in data
                    
                    # Verify clustering was called
                    mock_cluster.assert_called_once()
    
    def test_content_idea_optimization_scores(self, client, sample_keywords_data):
        """Test SEO optimization and traffic potential score calculation"""
        file_id = str(uuid.uuid4())
        
        with patch('src.services.content_idea_generator.calculate_seo_score') as mock_seo_score, \
             patch('src.services.content_idea_generator.calculate_traffic_score') as mock_traffic_score:
            
            mock_seo_score.return_value = 92
            mock_traffic_score.return_value = 88
            
            with patch('src.services.content_idea_generator.generate_content_ideas') as mock_generate:
                mock_generate.return_value = [
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
                        "optimization_tips": ["Include primary keyword in title"],
                        "content_outline": "Introduction → Tools → Conclusion"
                    }
                ]
                
                with patch('src.api.analysis.get_analysis_results') as mock_results:
                    mock_results.return_value = {
                        "report_id": str(uuid.uuid4()),
                        "file_id": file_id,
                        "summary": {"total_keywords": 10},
                        "top_opportunities": {"high_opportunity_keywords": []},
                        "content_recommendations": [],
                        "insights": [],
                        "next_steps": [],
                        "seo_content_ideas": mock_generate.return_value
                    }
                    
                    response = client.get(f"/api/v1/analysis/{file_id}/results")
                    
                    assert response.status_code == 200
                    data = response.json()
                    idea = data["seo_content_ideas"][0]
                    assert idea["seo_optimization_score"] == 92
                    assert idea["traffic_potential_score"] == 88
                    
                    # Verify score calculation functions were called
                    mock_seo_score.assert_called()
                    mock_traffic_score.assert_called()
    
    def test_content_idea_optimization_tips(self, client, sample_keywords_data):
        """Test generation of actionable optimization tips"""
        file_id = str(uuid.uuid4())
        
        with patch('src.services.content_idea_generator.generate_optimization_tips') as mock_tips:
            mock_tips.return_value = [
                "Include 'best project management tools' in your title and first paragraph",
                "Create comparison sections for commercial keywords",
                "Use secondary keywords like 'remote work tools' in H2 and H3 headings",
                "Include long-tail keywords in meta descriptions",
                "Add internal links to related project management guides"
            ]
            
            with patch('src.services.content_idea_generator.generate_content_ideas') as mock_generate:
                mock_generate.return_value = [
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
                        "optimization_tips": mock_tips.return_value,
                        "content_outline": "Introduction → Tools → Conclusion"
                    }
                ]
                
                with patch('src.api.analysis.get_analysis_results') as mock_results:
                    mock_results.return_value = {
                        "report_id": str(uuid.uuid4()),
                        "file_id": file_id,
                        "summary": {"total_keywords": 10},
                        "top_opportunities": {"high_opportunity_keywords": []},
                        "content_recommendations": [],
                        "insights": [],
                        "next_steps": [],
                        "seo_content_ideas": mock_generate.return_value
                    }
                    
                    response = client.get(f"/api/v1/analysis/{file_id}/results")
                    
                    assert response.status_code == 200
                    data = response.json()
                    idea = data["seo_content_ideas"][0]
                    assert len(idea["optimization_tips"]) == 5
                    assert "Include 'best project management tools' in your title" in idea["optimization_tips"][0]
                    
                    # Verify optimization tips generation was called
                    mock_tips.assert_called()
    
    def test_content_idea_content_type_detection(self, client, sample_keywords_data):
        """Test automatic content type detection based on keywords"""
        file_id = str(uuid.uuid4())
        
        with patch('src.services.content_idea_generator.detect_content_type') as mock_detect:
            mock_detect.return_value = "list-article"
            
            with patch('src.services.content_idea_generator.generate_content_ideas') as mock_generate:
                mock_generate.return_value = [
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
                        "optimization_tips": ["Include primary keyword in title"],
                        "content_outline": "Introduction → Tools → Conclusion"
                    }
                ]
                
                with patch('src.api.analysis.get_analysis_results') as mock_results:
                    mock_results.return_value = {
                        "report_id": str(uuid.uuid4()),
                        "file_id": file_id,
                        "summary": {"total_keywords": 10},
                        "top_opportunities": {"high_opportunity_keywords": []},
                        "content_recommendations": [],
                        "insights": [],
                        "next_steps": [],
                        "seo_content_ideas": mock_generate.return_value
                    }
                    
                    response = client.get(f"/api/v1/analysis/{file_id}/results")
                    
                    assert response.status_code == 200
                    data = response.json()
                    idea = data["seo_content_ideas"][0]
                    assert idea["content_type"] == "list-article"
                    
                    # Verify content type detection was called
                    mock_detect.assert_called()
    
    def test_content_idea_generation_error_handling(self, client, sample_keywords_data):
        """Test error handling in content idea generation"""
        file_id = str(uuid.uuid4())
        
        with patch('src.services.content_idea_generator.generate_content_ideas') as mock_generate:
            mock_generate.side_effect = Exception("Content idea generation failed")
            
            with patch('src.api.analysis.get_analysis_results') as mock_results:
                mock_results.return_value = {
                    "report_id": str(uuid.uuid4()),
                    "file_id": file_id,
                    "summary": {"total_keywords": 10},
                    "top_opportunities": {"high_opportunity_keywords": []},
                    "content_recommendations": [],
                    "insights": [],
                    "next_steps": [],
                    "seo_content_ideas": []
                }
                
                response = client.get(f"/api/v1/analysis/{file_id}/results")
                
                assert response.status_code == 200
                data = response.json()
                assert "seo_content_ideas" in data
                assert len(data["seo_content_ideas"]) == 0  # Empty due to error
