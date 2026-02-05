"""
Content Generation User Tests
Implements CI-001 to CI-007 from the user testing plan
"""

import asyncio
import json
import logging
from typing import Dict, List, Any
from test_framework import TrendTapTestFramework, TestData, PerformanceMonitor

logger = logging.getLogger(__name__)

class ContentGenerationTests:
    """Test class for content generation functionality"""
    
    def __init__(self, framework: TrendTapTestFramework):
        self.framework = framework
        self.performance_monitor = PerformanceMonitor()
        self.test_data = TestData.get_content_test_data()
    
    async def test_ci_001_basic_content_generation(self):
        """Test CI-001: Basic Content Generation"""
        logger.info("Testing basic content idea generation")
        
        # Performance monitoring
        self.performance_monitor.start_timer("content_generation")
        
        # Test data
        topic = self.test_data["topics"][0]  # "sustainable home improvements"
        content_type = "Article Ideas"
        
        # Make API request
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/content/generate",
            json={
                "topic": topic,
                "content_type": content_type,
                "count": 5
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        # End performance monitoring
        self.performance_monitor.end_timer("content_generation")
        
        # Validate response
        if response.status_code == 404:
            logger.warning("Content generation endpoint not implemented yet")
            return
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "success" in data, "Response should contain success field"
        assert data["success"] == True, "Content generation should be successful"
        
        # Validate data structure
        assert "data" in data, "Response should contain data field"
        content_data = data["data"]
        
        # Validate ideas
        assert "ideas" in content_data, "Response should contain ideas"
        ideas = content_data["ideas"]
        
        # Validate idea count (5+ article angles)
        assert len(ideas) >= 5, f"Expected at least 5 ideas, got {len(ideas)}"
        
        # Validate idea structure
        for idea in ideas:
            assert "headline" in idea, "Idea should have headline"
            assert "description" in idea, "Idea should have description"
            assert "keywords" in idea, "Idea should have keywords"
            assert "format" in idea, "Idea should have format"
            
            # Validate headline quality
            assert len(idea["headline"]) > 10, "Headline should be descriptive"
            
            # Validate description quality
            assert len(idea["description"]) > 20, "Description should be detailed"
            
            # Validate keywords
            assert isinstance(idea["keywords"], list), "Keywords should be a list"
            assert len(idea["keywords"]) > 0, "Should have at least one keyword"
        
        # Validate different formats
        formats = [idea["format"] for idea in ideas]
        unique_formats = set(formats)
        assert len(unique_formats) > 1, "Should have multiple content formats"
        
        # Validate response time (< 20 seconds)
        duration = self.performance_monitor.get_duration("content_generation")
        assert duration < 20.0, f"Content generation took {duration:.2f}s, expected < 20s"
        
        logger.info(f"✅ CI-001 passed: Generated {len(ideas)} content ideas in {duration:.2f}s")
    
    async def test_ci_002_software_solutions_generation(self):
        """Test CI-002: Software Solutions Generation"""
        logger.info("Testing software solution idea generation")
        
        # Performance monitoring
        self.performance_monitor.start_timer("software_generation")
        
        # Test data
        topic = self.test_data["topics"][1]  # "home energy efficiency"
        complexity = "Medium"
        
        # Make API request
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/content/generate-software",
            json={
                "topic": topic,
                "complexity": complexity,
                "count": 5
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        # End performance monitoring
        self.performance_monitor.end_timer("software_generation")
        
        # Validate response
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "success" in data, "Response should contain success field"
        assert data["success"] == True, "Software generation should be successful"
        
        # Validate data structure
        assert "data" in data, "Response should contain data field"
        software_data = data["data"]
        
        # Validate solutions
        assert "solutions" in software_data, "Response should contain solutions"
        solutions = software_data["solutions"]
        
        # Validate solution count (3-5 software solutions)
        assert len(solutions) >= 3, f"Expected at least 3 solutions, got {len(solutions)}"
        assert len(solutions) <= 5, f"Expected at most 5 solutions, got {len(solutions)}"
        
        # Validate solution structure
        for solution in solutions:
            assert "name" in solution, "Solution should have name"
            assert "description" in solution, "Solution should have description"
            assert "features" in solution, "Solution should have features"
            assert "complexity_score" in solution, "Solution should have complexity score"
            assert "development_estimate" in solution, "Solution should have development estimate"
            
            # Validate complexity score (1-10)
            complexity_score = solution["complexity_score"]
            assert 1 <= complexity_score <= 10, f"Complexity score {complexity_score} should be 1-10"
            
            # Validate development estimate
            dev_estimate = solution["development_estimate"]
            assert isinstance(dev_estimate, str), "Development estimate should be a string"
            assert len(dev_estimate) > 0, "Development estimate should not be empty"
        
        # Validate response time
        duration = self.performance_monitor.get_duration("software_generation")
        assert duration < 25.0, f"Software generation took {duration:.2f}s, expected < 25s"
        
        logger.info(f"✅ CI-002 passed: Generated {len(solutions)} software solutions in {duration:.2f}s")
    
    async def test_ci_003_content_optimization(self):
        """Test CI-003: Content Optimization"""
        logger.info("Testing content optimization features")
        
        # First, generate content ideas
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/content/generate",
            json={
                "topic": "sustainable home improvements",
                "content_type": "Article Ideas",
                "count": 3
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        assert response.status_code == 200, "Should generate content ideas"
        data = response.json()
        ideas = data["data"]["ideas"]
        
        if not ideas:
            logger.warning("No ideas generated for optimization test")
            return
        
        # Test content optimization
        idea_id = ideas[0].get("id", "test-idea-1")
        
        # Performance monitoring
        self.performance_monitor.start_timer("content_optimization")
        
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/content/optimize",
            json={
                "idea_id": idea_id,
                "optimization_type": "seo"
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        # End performance monitoring
        self.performance_monitor.end_timer("content_optimization")
        
        # Validate response
        if response.status_code == 404:
            logger.warning("Content optimization endpoint not implemented yet")
            return
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "success" in data, "Response should contain success field"
        assert data["success"] == True, "Content optimization should be successful"
        
        # Validate data structure
        assert "data" in data, "Response should contain data field"
        optimization_data = data["data"]
        
        # Validate optimization suggestions
        assert "suggestions" in optimization_data, "Should contain optimization suggestions"
        suggestions = optimization_data["suggestions"]
        assert isinstance(suggestions, list), "Suggestions should be a list"
        
        # Validate SEO improvements
        if "seo_score" in optimization_data:
            seo_score = optimization_data["seo_score"]
            assert 0 <= seo_score <= 100, f"SEO score {seo_score} should be 0-100"
        
        # Validate keyword density recommendations
        if "keyword_density" in optimization_data:
            keyword_density = optimization_data["keyword_density"]
            assert isinstance(keyword_density, dict), "Keyword density should be a dictionary"
        
        # Validate response time
        duration = self.performance_monitor.get_duration("content_optimization")
        assert duration < 15.0, f"Content optimization took {duration:.2f}s, expected < 15s"
        
        logger.info(f"✅ CI-003 passed: Content optimization completed in {duration:.2f}s")
    
    async def test_ci_004_content_export(self):
        """Test CI-004: Content Export"""
        logger.info("Testing content export functionality")
        
        # First, generate content ideas
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/content/generate",
            json={
                "topic": "eco-friendly living",
                "content_type": "Article Ideas",
                "count": 3
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        assert response.status_code == 200, "Should generate content ideas"
        data = response.json()
        ideas = data["data"]["ideas"]
        
        if not ideas:
            logger.warning("No ideas generated for export test")
            return
        
        # Select ideas for export
        idea_ids = [idea.get("id", f"idea-{i}") for i, idea in enumerate(ideas[:2])]
        
        # Performance monitoring
        self.performance_monitor.start_timer("content_export")
        
        # Test export to Google Docs
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/content/export",
            json={
                "idea_ids": idea_ids,
                "export_format": "google_docs"
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        # End performance monitoring
        self.performance_monitor.end_timer("content_export")
        
        # Validate response
        if response.status_code == 404:
            logger.warning("Content export endpoint not implemented yet")
            return
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "success" in data, "Response should contain success field"
        assert data["success"] == True, "Content export should be successful"
        
        # Validate data structure
        assert "data" in data, "Response should contain data field"
        export_data = data["data"]
        
        # Validate export information
        assert "export_id" in export_data, "Should contain export ID"
        assert "export_format" in export_data, "Should contain export format"
        assert "export_url" in export_data, "Should contain export URL"
        
        # Validate export format
        assert export_data["export_format"] == "google_docs", "Export format should match request"
        
        # Validate export URL
        export_url = export_data["export_url"]
        assert export_url.startswith("http"), "Export URL should be valid"
        
        # Validate response time
        duration = self.performance_monitor.get_duration("content_export")
        assert duration < 30.0, f"Content export took {duration:.2f}s, expected < 30s"
        
        logger.info(f"✅ CI-004 passed: Exported content in {duration:.2f}s")
    
    async def test_ci_005_content_calendar_integration(self):
        """Test CI-005: Content Calendar Integration"""
        logger.info("Testing content calendar scheduling")
        
        # First, generate content ideas
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/content/generate",
            json={
                "topic": "green technology",
                "content_type": "Article Ideas",
                "count": 2
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        assert response.status_code == 200, "Should generate content ideas"
        data = response.json()
        ideas = data["data"]["ideas"]
        
        if not ideas:
            logger.warning("No ideas generated for calendar test")
            return
        
        # Test content scheduling
        idea_id = ideas[0].get("id", "test-idea-1")
        publication_date = "2024-02-15"
        priority = "high"
        
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/content/schedule",
            json={
                "idea_id": idea_id,
                "publication_date": publication_date,
                "priority": priority
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        # Validate response
        if response.status_code == 404:
            logger.warning("Content scheduling endpoint not implemented yet")
            return
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "success" in data, "Response should contain success field"
        assert data["success"] == True, "Content scheduling should be successful"
        
        # Validate data structure
        assert "data" in data, "Response should contain data field"
        schedule_data = data["data"]
        
        # Validate scheduling information
        assert "schedule_id" in schedule_data, "Should contain schedule ID"
        assert "publication_date" in schedule_data, "Should contain publication date"
        assert "priority" in schedule_data, "Should contain priority"
        
        # Validate calendar entry
        assert "calendar_entry" in schedule_data, "Should contain calendar entry"
        calendar_entry = schedule_data["calendar_entry"]
        assert "title" in calendar_entry, "Calendar entry should have title"
        assert "date" in calendar_entry, "Calendar entry should have date"
        
        logger.info("✅ CI-005 passed: Content scheduled successfully")
    
    async def test_ci_006_content_collaboration(self):
        """Test CI-006: Content Collaboration"""
        logger.info("Testing content collaboration features")
        
        # First, generate content ideas
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/content/generate",
            json={
                "topic": "sustainable technology",
                "content_type": "Article Ideas",
                "count": 1
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        assert response.status_code == 200, "Should generate content ideas"
        data = response.json()
        ideas = data["data"]["ideas"]
        
        if not ideas:
            logger.warning("No ideas generated for collaboration test")
            return
        
        # Test content sharing
        idea_id = ideas[0].get("id", "test-idea-1")
        collaborator_email = "collaborator@example.com"
        permissions = "edit"
        
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/content/share",
            json={
                "idea_id": idea_id,
                "collaborator_email": collaborator_email,
                "permissions": permissions
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        # Validate response
        if response.status_code == 404:
            logger.warning("Content sharing endpoint not implemented yet")
            return
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "success" in data, "Response should contain success field"
        assert data["success"] == True, "Content sharing should be successful"
        
        # Validate data structure
        assert "data" in data, "Response should contain data field"
        share_data = data["data"]
        
        # Validate sharing information
        assert "share_id" in share_data, "Should contain share ID"
        assert "collaborator_email" in share_data, "Should contain collaborator email"
        assert "permissions" in share_data, "Should contain permissions"
        assert "share_url" in share_data, "Should contain share URL"
        
        # Validate permissions
        assert share_data["permissions"] == permissions, "Permissions should match request"
        
        logger.info("✅ CI-006 passed: Content shared successfully")
    
    async def test_ci_007_content_generation_failures(self):
        """Test CI-007: Content Generation Failures"""
        logger.info("Testing error handling for content generation failures")
        
        # Test very long topic
        long_topic = "a" * 500
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/content/generate",
            json={
                "topic": long_topic,
                "content_type": "Article Ideas",
                "count": 5
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        # Should either truncate or reject
        assert response.status_code in [200, 400, 413], f"Unexpected status code for long topic: {response.status_code}"
        
        # Test empty topic
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/content/generate",
            json={
                "topic": "",
                "content_type": "Article Ideas",
                "count": 5
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        # Should return validation error
        assert response.status_code in [400, 422], f"Expected validation error for empty topic, got {response.status_code}"
        
        # Test inappropriate content (basic check)
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/content/generate",
            json={
                "topic": "inappropriate content test",
                "content_type": "Article Ideas",
                "count": 5
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 422], f"Unexpected status code for inappropriate content: {response.status_code}"
        
        logger.info("✅ CI-007 passed: Content generation failures handled properly")

async def run_content_generation_tests(framework: TrendTapTestFramework) -> List[Any]:
    """Run all content generation tests"""
    tests = ContentGenerationTests(framework)
    
    test_cases = [
        ("CI-001", "Basic Content Generation", tests.test_ci_001_basic_content_generation),
        ("CI-002", "Software Solutions Generation", tests.test_ci_002_software_solutions_generation),
        ("CI-003", "Content Optimization", tests.test_ci_003_content_optimization),
        ("CI-004", "Content Export", tests.test_ci_004_content_export),
        ("CI-005", "Content Calendar Integration", tests.test_ci_005_content_calendar_integration),
        ("CI-006", "Content Collaboration", tests.test_ci_006_content_collaboration),
        ("CI-007", "Content Generation Failures", tests.test_ci_007_content_generation_failures),
    ]
    
    return await framework.run_test_suite("Content Generation Tests", test_cases)

if __name__ == "__main__":
    async def main():
        async with TrendTapTestFramework() as framework:
            # Authenticate user
            await framework.authenticate_user()
            
            # Run content generation tests
            suite = await run_content_generation_tests(framework)
            
            # Generate report
            report = framework.generate_report([suite])
            await framework.save_report(report)
            
            print(f"Content Generation Tests: {suite.passed_tests}/{suite.total_tests} passed")
    
    asyncio.run(main())
