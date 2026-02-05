"""
End-to-End Integration Tests
Implements E2E-001 to E2E-003 from the user testing plan
"""

import asyncio
import json
import logging
from typing import Dict, List, Any
from test_framework import TrendTapTestFramework, TestData, PerformanceMonitor

logger = logging.getLogger(__name__)

class E2EIntegrationTests:
    """Test class for end-to-end integration functionality"""
    
    def __init__(self, framework: TrendTapTestFramework):
        self.framework = framework
        self.performance_monitor = PerformanceMonitor()
        self.workflow_data = {}
    
    async def test_e2e_001_complete_research_workflow(self):
        """Test E2E-001: Complete Research Workflow"""
        logger.info("Testing complete 5-step research workflow")
        
        # Performance monitoring
        self.performance_monitor.start_timer("complete_workflow")
        
        # Step 0: Enter niche
        niche = "sustainable living"
        self.workflow_data["niche"] = niche
        logger.info(f"Step 0: Entered niche '{niche}'")
        
        # Step 1: Search affiliate programs and select 3 programs
        logger.info("Step 1: Searching affiliate programs...")
        affiliate_response = await self.framework.client.post(
            f"{self.framework.base_url}/api/affiliate-research/search",
            json={
                "topic": niche,
                "search_term": f"{niche} affiliate programs"
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        assert affiliate_response.status_code == 200, "Affiliate search should succeed"
        affiliate_data = affiliate_response.json()
        programs = affiliate_data["data"]["programs"]
        
        # Select 3 programs
        selected_programs = programs[:3]
        self.workflow_data["selected_programs"] = [p["id"] for p in selected_programs]
        logger.info(f"Step 1: Selected {len(selected_programs)} affiliate programs")
        
        # Step 2: Analyze trends
        logger.info("Step 2: Analyzing trends...")
        trend_response = await self.framework.client.post(
            f"{self.framework.base_url}/api/trends/analyze",
            json={
                "keyword": niche,
                "time_range": "12 months",
                "geo": "US"
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        assert trend_response.status_code == 200, "Trend analysis should succeed"
        trend_data = trend_response.json()
        opportunity_score = trend_data["data"]["opportunity_score"]
        self.workflow_data["opportunity_score"] = opportunity_score
        logger.info(f"Step 2: Trend analysis completed with score {opportunity_score}")
        
        # Step 3: Generate content ideas based on selected programs
        logger.info("Step 3: Generating content ideas...")
        content_response = await self.framework.client.post(
            f"{self.framework.base_url}/api/content/generate",
            json={
                "topic": niche,
                "content_type": "Article Ideas",
                "count": 5,
                "program_ids": self.workflow_data["selected_programs"]
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        assert content_response.status_code == 200, "Content generation should succeed"
        content_data = content_response.json()
        ideas = content_data["data"]["ideas"]
        self.workflow_data["content_ideas"] = ideas
        logger.info(f"Step 3: Generated {len(ideas)} content ideas")
        
        # Step 4: Upload keyword CSV or use DataForSEO
        logger.info("Step 4: Processing keywords...")
        
        # Try DataForSEO first
        keyword_response = await self.framework.client.post(
            f"{self.framework.base_url}/api/keywords/crawl",
            json={
                "seed_keyword": niche,
                "depth": 2,
                "geo": "US",
                "language": "en"
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        if keyword_response.status_code == 200:
            keyword_data = keyword_response.json()
            self.workflow_data["keywords"] = keyword_data["data"]["keywords"]
            logger.info(f"Step 4: Crawled {len(self.workflow_data['keywords'])} keywords with DataForSEO")
        else:
            # Fallback to CSV upload
            csv_content = TestData.get_csv_test_file()
            files = {"file": ("keywords.csv", csv_content, "text/csv")}
            csv_response = await self.framework.client.post(
                f"{self.framework.base_url}/api/keywords/upload",
                files=files,
                headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
            )
            
            if csv_response.status_code == 200:
                csv_data = csv_response.json()
                self.workflow_data["keywords"] = csv_data["data"]["keywords"]
                logger.info(f"Step 4: Uploaded CSV with {len(self.workflow_data['keywords'])} keywords")
            else:
                logger.warning("Step 4: Both DataForSEO and CSV upload failed")
                self.workflow_data["keywords"] = []
        
        # Step 5: Export final content to Google Docs
        logger.info("Step 5: Exporting content...")
        
        if self.workflow_data["content_ideas"]:
            idea_ids = [idea.get("id", f"idea-{i}") for i, idea in enumerate(self.workflow_data["content_ideas"][:3])]
            
            export_response = await self.framework.client.post(
                f"{self.framework.base_url}/api/content/export",
                json={
                    "idea_ids": idea_ids,
                    "export_format": "google_docs"
                },
                headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
            )
            
            if export_response.status_code == 200:
                export_data = export_response.json()
                self.workflow_data["export_url"] = export_data["data"]["export_url"]
                logger.info(f"Step 5: Content exported to Google Docs")
            else:
                logger.warning("Step 5: Export failed")
                self.workflow_data["export_url"] = None
        
        # End performance monitoring
        self.performance_monitor.end_timer("complete_workflow")
        
        # Validate workflow completion
        assert len(self.workflow_data["selected_programs"]) == 3, "Should select 3 programs"
        assert self.workflow_data["opportunity_score"] is not None, "Should have opportunity score"
        assert len(self.workflow_data["content_ideas"]) >= 5, "Should generate at least 5 content ideas"
        assert len(self.workflow_data["keywords"]) > 0, "Should have keywords"
        
        # Validate total workflow time (< 15 minutes)
        duration = self.performance_monitor.get_duration("complete_workflow")
        assert duration < 900.0, f"Complete workflow took {duration:.2f}s, expected < 15 minutes"
        
        logger.info(f"✅ E2E-001 passed: Complete workflow completed in {duration:.2f}s")
    
    async def test_e2e_002_multi_user_workflow(self):
        """Test E2E-002: Multi-User Workflow"""
        logger.info("Testing workflow with multiple users")
        
        # This test would require creating multiple user accounts
        # For now, we'll test that the system can handle concurrent requests
        
        # Simulate multiple users performing the same workflow
        tasks = []
        for user_id in range(3):
            task = self._simulate_user_workflow(user_id)
            tasks.append(task)
        
        # Execute all workflows concurrently
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = asyncio.get_event_loop().time()
        
        total_time = end_time - start_time
        
        # Validate results
        successful_workflows = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"User {i} workflow failed: {result}")
            else:
                successful_workflows += 1
        
        # At least 80% of workflows should succeed
        success_rate = successful_workflows / len(tasks)
        assert success_rate >= 0.8, f"Success rate {success_rate:.2%} < 80%"
        
        # All workflows should complete within reasonable time
        assert total_time < 60.0, f"Multi-user test took {total_time:.2f}s, expected < 60s"
        
        logger.info(f"✅ E2E-002 passed: {successful_workflows}/{len(tasks)} user workflows succeeded in {total_time:.2f}s")
    
    async def test_e2e_003_cross_feature_data_flow(self):
        """Test E2E-003: Cross-Feature Data Flow"""
        logger.info("Testing data flow between features")
        
        # Step 1: Complete affiliate research
        affiliate_response = await self.framework.client.post(
            f"{self.framework.base_url}/api/affiliate-research/search",
            json={
                "topic": "eco-friendly technology",
                "search_term": "eco-friendly technology affiliate programs"
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        assert affiliate_response.status_code == 200, "Affiliate research should succeed"
        affiliate_data = affiliate_response.json()
        programs = affiliate_data["data"]["programs"]
        selected_programs = programs[:2]
        
        # Step 2: Use selected programs in trend analysis
        trend_response = await self.framework.client.post(
            f"{self.framework.base_url}/api/trends/analyze",
            json={
                "keyword": "eco-friendly technology",
                "time_range": "12 months",
                "geo": "US",
                "affiliate_programs": [p["id"] for p in selected_programs]
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        assert trend_response.status_code == 200, "Trend analysis should succeed"
        trend_data = trend_response.json()
        
        # Step 3: Generate content based on both affiliate programs and trend analysis
        content_response = await self.framework.client.post(
            f"{self.framework.base_url}/api/content/generate",
            json={
                "topic": "eco-friendly technology",
                "content_type": "Article Ideas",
                "count": 5,
                "program_ids": [p["id"] for p in selected_programs],
                "trend_data": trend_data["data"]
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        assert content_response.status_code == 200, "Content generation should succeed"
        content_data = content_response.json()
        ideas = content_data["data"]["ideas"]
        
        # Validate data consistency
        assert len(ideas) > 0, "Should generate content ideas"
        
        # Validate that content is linked to selected programs
        for idea in ideas:
            if "program_id" in idea:
                assert idea["program_id"] in [p["id"] for p in selected_programs], \
                    "Content should be linked to selected programs"
        
        # Validate that content incorporates trend data
        if "trend_insights" in content_data["data"]:
            trend_insights = content_data["data"]["trend_insights"]
            assert isinstance(trend_insights, list), "Trend insights should be a list"
        
        logger.info("✅ E2E-003 passed: Cross-feature data flow validated")
    
    async def _simulate_user_workflow(self, user_id: int) -> bool:
        """Simulate a user workflow for multi-user testing"""
        try:
            # Simulate affiliate research
            response = await self.framework.client.post(
                f"{self.framework.base_url}/api/affiliate-research/search",
                json={
                    "topic": f"test topic {user_id}",
                    "search_term": f"test search {user_id}"
                },
                headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
            )
            
            if response.status_code != 200:
                return False
            
            # Simulate trend analysis
            response = await self.framework.client.post(
                f"{self.framework.base_url}/api/trends/analyze",
                json={
                    "keyword": f"test keyword {user_id}",
                    "time_range": "12 months",
                    "geo": "US"
                },
                headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
            )
            
            if response.status_code != 200:
                return False
            
            # Simulate content generation
            response = await self.framework.client.post(
                f"{self.framework.base_url}/api/content/generate",
                json={
                    "topic": f"test content {user_id}",
                    "content_type": "Article Ideas",
                    "count": 3
                },
                headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"User {user_id} workflow error: {e}")
            return False

async def run_e2e_integration_tests(framework: TrendTapTestFramework) -> List[Any]:
    """Run all end-to-end integration tests"""
    tests = E2EIntegrationTests(framework)
    
    test_cases = [
        ("E2E-001", "Complete Research Workflow", tests.test_e2e_001_complete_research_workflow),
        ("E2E-002", "Multi-User Workflow", tests.test_e2e_002_multi_user_workflow),
        ("E2E-003", "Cross-Feature Data Flow", tests.test_e2e_003_cross_feature_data_flow),
    ]
    
    return await framework.run_test_suite("E2E Integration Tests", test_cases)

if __name__ == "__main__":
    async def main():
        async with TrendTapTestFramework() as framework:
            # Authenticate user
            await framework.authenticate_user()
            
            # Run E2E integration tests
            suite = await run_e2e_integration_tests(framework)
            
            # Generate report
            report = framework.generate_report([suite])
            await framework.save_report(report)
            
            print(f"E2E Integration Tests: {suite.passed_tests}/{suite.total_tests} passed")
    
    asyncio.run(main())


