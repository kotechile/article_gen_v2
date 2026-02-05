"""
Integration test for complete 5-step user workflow
This test MUST fail before implementation - it tests the complete user workflow from start to finish
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import uuid


class TestCompleteWorkflow:
    """Integration test for complete 5-step user workflow"""
    
    def setup_method(self):
        """Setup test client and mock data"""
        # This will fail until we implement the main app
        from src.main import app
        self.client = TestClient(app)
        
        # Mock user authentication
        self.auth_headers = {"Authorization": "Bearer mock-jwt-token"}
        
        # Mock workflow data
        self.mock_workflow_data = {
            "affiliate_research": {
                "id": str(uuid.uuid4()),
                "topic": "home coffee roasting",
                "status": "COMPLETED",
                "results": [
                    {
                        "network": "ShareASale",
                        "program_name": "Coffee Equipment Pro",
                        "epc": 12.50,
                        "commission_rate": 8.5,
                        "cookie_length": 30,
                        "conversion_rate": 3.2,
                        "landing_page_compliance": True,
                        "reversal_rate": 2.1
                    }
                ],
                "selected_programs": [
                    {
                        "network": "ShareASale",
                        "program_name": "Coffee Equipment Pro",
                        "selected": True,
                        "notes": "High EPC, good conversion rate"
                    }
                ]
            },
            "trend_analysis": {
                "id": str(uuid.uuid4()),
                "topics": ["home coffee roasting"],
                "status": "COMPLETED",
                "opportunity_scores": {
                    "home coffee roasting": 85.5
                },
                "llm_forecast": {
                    "forecast": [
                        {
                            "topic": "home coffee roasting",
                            "month": "2025-11",
                            "predicted_interest": 78.2,
                            "confidence_interval": [65.1, 91.3]
                        }
                    ]
                }
            },
            "keyword_data": {
                "id": str(uuid.uuid4()),
                "status": "COMPLETED",
                "keyword_count": 50,
                "keywords": [
                    {
                        "keyword": "coffee roaster",
                        "search_volume": 1200,
                        "difficulty": 45,
                        "cpc": 2.50,
                        "priority_score": 0.85
                    }
                ]
            },
            "content_ideas": {
                "id": str(uuid.uuid4()),
                "content_ideas": [
                    {
                        "id": str(uuid.uuid4()),
                        "title": "The Complete Guide to Home Coffee Roasting",
                        "content_type": "GUIDE",
                        "angle": "how-to",
                        "headline_score": 87.5,
                        "priority_score": 0.92
                    }
                ]
            },
            "software_solutions": {
                "id": str(uuid.uuid4()),
                "software_solutions": [
                    {
                        "id": str(uuid.uuid4()),
                        "name": "Coffee Roasting Time Calculator",
                        "software_type": "CALCULATOR",
                        "complexity_score": 6,
                        "priority_score": 0.88
                    }
                ]
            },
            "calendar_entries": [
                {
                    "id": str(uuid.uuid4()),
                    "title": "Coffee Roasting Guide Article",
                    "scheduled_date": "2025-10-15T10:00:00Z",
                    "entry_type": "CONTENT",
                    "status": "SCHEDULED"
                }
            ]
        }
    
    def test_complete_5_step_workflow(self):
        """Test complete 5-step user workflow from start to finish"""
        
        # Step 0: Seed - User enters broad niche
        niche = "home coffee roasting"
        
        # Step 1: Monetisation First - Affiliate Research
        with patch('src.services.affiliate_service.AffiliateService.create_research') as mock_affiliate:
            mock_affiliate.return_value = self.mock_workflow_data["affiliate_research"]
            
            payload = {
                "topic": niche,
                "search_query": "coffee roasting equipment affiliate programs"
            }
            
            response = self.client.post(
                "/api/affiliate/research",
                json=payload,
                headers=self.auth_headers
            )
            
            assert response.status_code == 201
            affiliate_data = response.json()
            affiliate_id = affiliate_data["id"]
            assert affiliate_data["status"] == "PENDING"
        
        # Step 2: Trend Validation - Hybrid Forecast Engine
        with patch('src.services.trend_service.TrendService.create_analysis') as mock_trend:
            mock_trend.return_value = self.mock_workflow_data["trend_analysis"]
            
            payload = {
                "topics": [niche],
                "affiliate_research_id": affiliate_id
            }
            
            response = self.client.post(
                "/api/trends/analyze",
                json=payload,
                headers=self.auth_headers
            )
            
            assert response.status_code == 201
            trend_data = response.json()
            trend_id = trend_data["id"]
            assert trend_data["status"] == "PENDING"
        
        # Step 3: Idea Burst - Content & Software Generation
        with patch('src.services.keyword_service.KeywordService.upload_csv') as mock_keyword:
            mock_keyword.return_value = self.mock_workflow_data["keyword_data"]
            
            # Upload keyword data
            csv_content = "keyword,search_volume,difficulty,cpc\ncoffee roaster,1200,45,2.50"
            files = {"file": ("keywords.csv", csv_content, "text/csv")}
            data = {"trend_analysis_id": trend_id}
            
            response = self.client.post(
                "/api/keywords/upload",
                files=files,
                data=data,
                headers=self.auth_headers
            )
            
            assert response.status_code == 201
            keyword_data = response.json()
            keyword_id = keyword_data["id"]
            assert keyword_data["status"] == "PROCESSING"
        
        # Generate content ideas
        with patch('src.services.content_service.ContentService.generate_ideas') as mock_content:
            mock_content.return_value = self.mock_workflow_data["content_ideas"]
            
            payload = {
                "trend_analysis_id": trend_id,
                "keyword_data_id": keyword_id,
                "content_types": ["GUIDE", "ARTICLE"],
                "max_ideas": 5
            }
            
            response = self.client.post(
                "/api/content/generate",
                json=payload,
                headers=self.auth_headers
            )
            
            assert response.status_code == 201
            content_data = response.json()
            content_id = content_data["id"]
            assert len(content_data["content_ideas"]) == 1
        
        # Generate software solutions
        with patch('src.services.software_service.SoftwareService.generate_solutions') as mock_software:
            mock_software.return_value = self.mock_workflow_data["software_solutions"]
            
            payload = {
                "trend_analysis_id": trend_id,
                "keyword_data_id": keyword_id,
                "software_types": ["CALCULATOR", "ANALYZER"],
                "max_solutions": 5
            }
            
            response = self.client.post(
                "/api/software/generate",
                json=payload,
                headers=self.auth_headers
            )
            
            assert response.status_code == 201
            software_data = response.json()
            software_id = software_data["id"]
            assert len(software_data["software_solutions"]) == 1
        
        # Step 4: Keyword Armoury - Schedule Content
        with patch('src.services.calendar_service.CalendarService.schedule_entry') as mock_calendar:
            mock_calendar.return_value = self.mock_workflow_data["calendar_entries"][0]
            
            payload = {
                "title": "Coffee Roasting Guide Article",
                "description": "Complete guide to home coffee roasting",
                "scheduled_date": "2025-10-15T10:00:00Z",
                "entry_type": "CONTENT",
                "content_idea_id": content_data["content_ideas"][0]["id"],
                "platform": "wordpress",
                "notes": "Focus on affiliate links for coffee equipment",
                "priority": 3
            }
            
            response = self.client.post(
                "/api/calendar/schedule",
                json=payload,
                headers=self.auth_headers
            )
            
            assert response.status_code == 201
            calendar_data = response.json()
            assert calendar_data["status"] == "SCHEDULED"
            assert calendar_data["entry_type"] == "CONTENT"
        
        # Step 5: Export - One-click Export
        with patch('src.services.export_service.ExportService.export_to_google_docs') as mock_export:
            export_data = {
                "export_id": str(uuid.uuid4()),
                "platform": "google_docs",
                "url": "https://docs.google.com/document/d/1ABC123DEF456GHI789JKL",
                "status": "COMPLETED",
                "content_exported": 1,
                "software_exported": 0
            }
            mock_export.return_value = export_data
            
            payload = {
                "content_ids": [content_data["content_ideas"][0]["id"]],
                "platform": "google_docs",
                "template": "trendtap-template",
                "format_options": {
                    "include_seo": True,
                    "include_outline": True,
                    "include_affiliate_links": True
                }
            }
            
            response = self.client.post(
                "/api/export/google-docs",
                json=payload,
                headers=self.auth_headers
            )
            
            assert response.status_code == 201
            export_result = response.json()
            assert export_result["status"] == "COMPLETED"
            assert export_result["platform"] == "google_docs"
            assert "docs.google.com" in export_result["url"]
    
    def test_workflow_error_handling(self):
        """Test error handling throughout the complete workflow"""
        
        # Test affiliate research failure
        with patch('src.services.affiliate_service.AffiliateService.create_research') as mock_affiliate:
            mock_affiliate.side_effect = Exception("Affiliate API unavailable")
            
            payload = {
                "topic": "test topic",
                "search_query": "test query"
            }
            
            response = self.client.post(
                "/api/affiliate/research",
                json=payload,
                headers=self.auth_headers
            )
            
            # Should handle API failure gracefully
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
        
        # Test trend analysis failure
        with patch('src.services.trend_service.TrendService.create_analysis') as mock_trend:
            mock_trend.side_effect = Exception("Trend API unavailable")
            
            payload = {
                "topics": ["test topic"],
                "affiliate_research_id": str(uuid.uuid4())
            }
            
            response = self.client.post(
                "/api/trends/analyze",
                json=payload,
                headers=self.auth_headers
            )
            
            # Should handle API failure gracefully
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
        
        # Test content generation failure
        with patch('src.services.content_service.ContentService.generate_ideas') as mock_content:
            mock_content.side_effect = Exception("Content generation service unavailable")
            
            payload = {
                "trend_analysis_id": str(uuid.uuid4()),
                "keyword_data_id": str(uuid.uuid4()),
                "content_types": ["GUIDE"],
                "max_ideas": 5
            }
            
            response = self.client.post(
                "/api/content/generate",
                json=payload,
                headers=self.auth_headers
            )
            
            # Should handle service failure gracefully
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
    
    def test_workflow_performance_requirements(self):
        """Test performance requirements for complete workflow"""
        
        # Test that complete workflow completes within time limit
        with patch('src.services.affiliate_service.AffiliateService.create_research') as mock_affiliate, \
             patch('src.services.trend_service.TrendService.create_analysis') as mock_trend, \
             patch('src.services.keyword_service.KeywordService.upload_csv') as mock_keyword, \
             patch('src.services.content_service.ContentService.generate_ideas') as mock_content, \
             patch('src.services.software_service.SoftwareService.generate_solutions') as mock_software, \
             patch('src.services.calendar_service.CalendarService.schedule_entry') as mock_calendar, \
             patch('src.services.export_service.ExportService.export_to_google_docs') as mock_export:
            
            # Setup mocks
            mock_affiliate.return_value = self.mock_workflow_data["affiliate_research"]
            mock_trend.return_value = self.mock_workflow_data["trend_analysis"]
            mock_keyword.return_value = self.mock_workflow_data["keyword_data"]
            mock_content.return_value = self.mock_workflow_data["content_ideas"]
            mock_software.return_value = self.mock_workflow_data["software_solutions"]
            mock_calendar.return_value = self.mock_workflow_data["calendar_entries"][0]
            mock_export.return_value = {
                "export_id": str(uuid.uuid4()),
                "platform": "google_docs",
                "url": "https://docs.google.com/document/d/test",
                "status": "COMPLETED"
            }
            
            import time
            start_time = time.time()
            
            # Execute complete workflow
            # Step 1: Affiliate Research
            payload = {"topic": "test topic", "search_query": "test query"}
            response = self.client.post("/api/affiliate/research", json=payload, headers=self.auth_headers)
            assert response.status_code == 201
            affiliate_id = response.json()["id"]
            
            # Step 2: Trend Analysis
            payload = {"topics": ["test topic"], "affiliate_research_id": affiliate_id}
            response = self.client.post("/api/trends/analyze", json=payload, headers=self.auth_headers)
            assert response.status_code == 201
            trend_id = response.json()["id"]
            
            # Step 3: Keyword Upload
            csv_content = "keyword,search_volume,difficulty,cpc\ncoffee,100,50,2.0"
            files = {"file": ("keywords.csv", csv_content, "text/csv")}
            data = {"trend_analysis_id": trend_id}
            response = self.client.post("/api/keywords/upload", files=files, data=data, headers=self.auth_headers)
            assert response.status_code == 201
            keyword_id = response.json()["id"]
            
            # Step 3: Content Generation
            payload = {"trend_analysis_id": trend_id, "keyword_data_id": keyword_id, "content_types": ["GUIDE"], "max_ideas": 5}
            response = self.client.post("/api/content/generate", json=payload, headers=self.auth_headers)
            assert response.status_code == 201
            content_id = response.json()["id"]
            
            # Step 3: Software Generation
            payload = {"trend_analysis_id": trend_id, "keyword_data_id": keyword_id, "software_types": ["CALCULATOR"], "max_solutions": 5}
            response = self.client.post("/api/software/generate", json=payload, headers=self.auth_headers)
            assert response.status_code == 201
            software_id = response.json()["id"]
            
            # Step 4: Calendar Scheduling
            payload = {"title": "Test Entry", "scheduled_date": "2025-10-15T10:00:00Z", "entry_type": "CONTENT", "content_idea_id": str(uuid.uuid4())}
            response = self.client.post("/api/calendar/schedule", json=payload, headers=self.auth_headers)
            assert response.status_code == 201
            
            # Step 5: Export
            payload = {"content_ids": [str(uuid.uuid4())], "platform": "google_docs"}
            response = self.client.post("/api/export/google-docs", json=payload, headers=self.auth_headers)
            assert response.status_code == 201
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should complete within 15 minutes (as per requirements)
            assert total_time < 900.0
    
    def test_workflow_data_consistency(self):
        """Test data consistency throughout the complete workflow"""
        
        # Test that data flows correctly between steps
        with patch('src.services.affiliate_service.AffiliateService.create_research') as mock_affiliate, \
             patch('src.services.trend_service.TrendService.create_analysis') as mock_trend, \
             patch('src.services.keyword_service.KeywordService.upload_csv') as mock_keyword, \
             patch('src.services.content_service.ContentService.generate_ideas') as mock_content:
            
            # Setup mocks with consistent data
            affiliate_data = self.mock_workflow_data["affiliate_research"]
            trend_data = self.mock_workflow_data["trend_analysis"]
            keyword_data = self.mock_workflow_data["keyword_data"]
            content_data = self.mock_workflow_data["content_ideas"]
            
            mock_affiliate.return_value = affiliate_data
            mock_trend.return_value = trend_data
            mock_keyword.return_value = keyword_data
            mock_content.return_value = content_data
            
            # Execute workflow steps
            # Step 1: Affiliate Research
            payload = {"topic": "home coffee roasting", "search_query": "coffee roasting equipment affiliate programs"}
            response = self.client.post("/api/affiliate/research", json=payload, headers=self.auth_headers)
            assert response.status_code == 201
            affiliate_id = response.json()["id"]
            
            # Step 2: Trend Analysis (should use affiliate research data)
            payload = {"topics": ["home coffee roasting"], "affiliate_research_id": affiliate_id}
            response = self.client.post("/api/trends/analyze", json=payload, headers=self.auth_headers)
            assert response.status_code == 201
            trend_id = response.json()["id"]
            
            # Step 3: Keyword Upload (should use trend analysis data)
            csv_content = "keyword,search_volume,difficulty,cpc\ncoffee roaster,1200,45,2.50"
            files = {"file": ("keywords.csv", csv_content, "text/csv")}
            data = {"trend_analysis_id": trend_id}
            response = self.client.post("/api/keywords/upload", files=files, data=data, headers=self.auth_headers)
            assert response.status_code == 201
            keyword_id = response.json()["id"]
            
            # Step 3: Content Generation (should use all previous data)
            payload = {"trend_analysis_id": trend_id, "keyword_data_id": keyword_id, "content_types": ["GUIDE"], "max_ideas": 5}
            response = self.client.post("/api/content/generate", json=payload, headers=self.auth_headers)
            assert response.status_code == 201
            
            # Validate that content ideas are relevant to the original topic
            content_ideas = response.json()["content_ideas"]
            for idea in content_ideas:
                assert "coffee" in idea["title"].lower() or "roasting" in idea["title"].lower()
    
    def test_workflow_concurrent_users(self):
        """Test handling of concurrent users in complete workflow"""
        
        # Test multiple users executing workflow simultaneously
        with patch('src.services.affiliate_service.AffiliateService.create_research') as mock_affiliate, \
             patch('src.services.trend_service.TrendService.create_analysis') as mock_trend, \
             patch('src.services.keyword_service.KeywordService.upload_csv') as mock_keyword, \
             patch('src.services.content_service.ContentService.generate_ideas') as mock_content:
            
            # Setup mocks
            mock_affiliate.return_value = self.mock_workflow_data["affiliate_research"]
            mock_trend.return_value = self.mock_workflow_data["trend_analysis"]
            mock_keyword.return_value = self.mock_workflow_data["keyword_data"]
            mock_content.return_value = self.mock_workflow_data["content_ideas"]
            
            # Simulate multiple users
            user_topics = ["coffee roasting", "coffee brewing", "coffee equipment"]
            
            for topic in user_topics:
                # Each user executes affiliate research
                payload = {"topic": topic, "search_query": f"{topic} affiliate programs"}
                response = self.client.post("/api/affiliate/research", json=payload, headers=self.auth_headers)
                assert response.status_code == 201
                
                # Each user executes trend analysis
                payload = {"topics": [topic], "affiliate_research_id": str(uuid.uuid4())}
                response = self.client.post("/api/trends/analyze", json=payload, headers=self.auth_headers)
                assert response.status_code == 201
                
                # Each user uploads keywords
                csv_content = f"keyword,search_volume,difficulty,cpc\n{topic},100,50,2.0"
                files = {"file": ("keywords.csv", csv_content, "text/csv")}
                data = {"trend_analysis_id": str(uuid.uuid4())}
                response = self.client.post("/api/keywords/upload", files=files, data=data, headers=self.auth_headers)
                assert response.status_code == 201
                
                # Each user generates content
                payload = {"trend_analysis_id": str(uuid.uuid4()), "keyword_data_id": str(uuid.uuid4()), "content_types": ["GUIDE"], "max_ideas": 5}
                response = self.client.post("/api/content/generate", json=payload, headers=self.auth_headers)
                assert response.status_code == 201
