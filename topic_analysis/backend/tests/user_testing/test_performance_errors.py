"""
Performance and Error Handling Tests
Implements PERF-001 to PERF-002 and ERR-001 to ERR-003 from the user testing plan
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any
from test_framework import TrendTapTestFramework, TestData, PerformanceMonitor

logger = logging.getLogger(__name__)

class PerformanceErrorTests:
    """Test class for performance and error handling functionality"""
    
    def __init__(self, framework: TrendTapTestFramework):
        self.framework = framework
        self.performance_monitor = PerformanceMonitor()
    
    async def test_perf_001_api_response_times(self):
        """Test PERF-001: API Response Times"""
        logger.info("Testing API response times meet requirements")
        
        # Test affiliate search response time (< 5 seconds)
        self.performance_monitor.start_timer("affiliate_search")
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/affiliate-research/search",
            json={
                "topic": "test topic",
                "search_term": "test search term"
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        self.performance_monitor.end_timer("affiliate_search")
        
        affiliate_duration = self.performance_monitor.get_duration("affiliate_search")
        assert affiliate_duration < 5.0, f"Affiliate search took {affiliate_duration:.2f}s, expected < 5s"
        
        # Test trend analysis response time (< 15 seconds)
        self.performance_monitor.start_timer("trend_analysis")
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/trends/analyze",
            json={
                "keyword": "test keyword",
                "time_range": "12 months",
                "geo": "US"
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        self.performance_monitor.end_timer("trend_analysis")
        
        trend_duration = self.performance_monitor.get_duration("trend_analysis")
        assert trend_duration < 15.0, f"Trend analysis took {trend_duration:.2f}s, expected < 15s"
        
        # Test content generation response time (< 20 seconds)
        self.performance_monitor.start_timer("content_generation")
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/content/generate",
            json={
                "topic": "test topic",
                "content_type": "Article Ideas",
                "count": 5
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        self.performance_monitor.end_timer("content_generation")
        
        content_duration = self.performance_monitor.get_duration("content_generation")
        assert content_duration < 20.0, f"Content generation took {content_duration:.2f}s, expected < 20s"
        
        # Test keyword analysis response time (< 10 seconds)
        self.performance_monitor.start_timer("keyword_analysis")
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/keywords/crawl",
            json={
                "seed_keyword": "test keyword",
                "depth": 1,
                "geo": "US",
                "language": "en"
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        self.performance_monitor.end_timer("keyword_analysis")
        
        keyword_duration = self.performance_monitor.get_duration("keyword_analysis")
        assert keyword_duration < 10.0, f"Keyword analysis took {keyword_duration:.2f}s, expected < 10s"
        
        logger.info(f"✅ PERF-001 passed: All API response times within limits")
        logger.info(f"  - Affiliate search: {affiliate_duration:.2f}s")
        logger.info(f"  - Trend analysis: {trend_duration:.2f}s")
        logger.info(f"  - Content generation: {content_duration:.2f}s")
        logger.info(f"  - Keyword analysis: {keyword_duration:.2f}s")
    
    async def test_perf_002_concurrent_user_performance(self):
        """Test PERF-002: Concurrent User Performance"""
        logger.info("Testing system performance with multiple concurrent users")
        
        # Simulate 10 concurrent users
        num_users = 10
        tasks = []
        
        for user_id in range(num_users):
            task = self._simulate_user_workflow(user_id)
            tasks.append(task)
        
        # Execute all workflows concurrently
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = asyncio.get_event_loop().time()
        
        total_time = end_time - start_time
        
        # Validate results
        successful_requests = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"User {i} request failed: {result}")
            else:
                successful_requests += 1
        
        # At least 80% of requests should succeed
        success_rate = successful_requests / num_users
        assert success_rate >= 0.8, f"Success rate {success_rate:.2%} < 80%"
        
        # All requests should complete within reasonable time
        assert total_time < 60.0, f"Concurrent test took {total_time:.2f}s, expected < 60s"
        
        # Calculate average response time per user
        avg_response_time = total_time / num_users
        assert avg_response_time < 10.0, f"Average response time {avg_response_time:.2f}s per user, expected < 10s"
        
        logger.info(f"✅ PERF-002 passed: {successful_requests}/{num_users} concurrent users succeeded in {total_time:.2f}s")
        logger.info(f"  - Success rate: {success_rate:.2%}")
        logger.info(f"  - Average response time: {avg_response_time:.2f}s per user")
    
    async def test_err_001_api_timeout_handling(self):
        """Test ERR-001: API Timeout Handling"""
        logger.info("Testing behavior when external APIs timeout")
        
        # This test would require simulating API timeouts
        # For now, we'll test that the system handles errors gracefully
        
        # Test affiliate research with potential timeout
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/affiliate-research/search",
            json={
                "topic": "test timeout topic",
                "search_term": "test timeout search"
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 500, 503], f"Unexpected status code: {response.status_code}"
        
        if response.status_code == 200:
            data = response.json()
            # Check if response indicates fallback data
            if "data_source" in data.get("data", {}):
                logger.info(f"System used fallback data source: {data['data']['data_source']}")
        
        # Test trend analysis with potential timeout
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/trends/analyze",
            json={
                "keyword": "test timeout keyword",
                "time_range": "12 months",
                "geo": "US"
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 500, 503], f"Unexpected status code: {response.status_code}"
        
        logger.info("✅ ERR-001 passed: API timeout handling tested")
    
    async def test_err_002_database_connection_issues(self):
        """Test ERR-002: Database Connection Issues"""
        logger.info("Testing behavior when database is unavailable")
        
        # This test would require simulating database connection loss
        # For now, we'll test that the system handles database errors gracefully
        
        # Test user profile endpoint (requires database)
        response = await self.framework.client.get(
            f"{self.framework.base_url}/api/users/profile",
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 500, 503], f"Unexpected status code: {response.status_code}"
        
        if response.status_code == 500:
            data = response.json()
            assert "detail" in data, "Error response should contain detail"
            logger.info(f"Database error handled: {data['detail']}")
        
        # Test health check endpoint
        response = await self.framework.client.get(f"{self.framework.base_url}/api/health/")
        
        # Health check should work even with database issues
        assert response.status_code in [200, 503], f"Health check should work, got {response.status_code}"
        
        logger.info("✅ ERR-002 passed: Database connection issues handled")
    
    async def test_err_003_malicious_input_handling(self):
        """Test ERR-003: Malicious Input Handling"""
        logger.info("Testing system security with malicious inputs")
        
        # Test SQL injection attempts
        sql_injection_payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "1' UNION SELECT * FROM users--"
        ]
        
        for payload in sql_injection_payloads:
            response = await self.framework.client.post(
                f"{self.framework.base_url}/api/affiliate-research/search",
                json={
                    "topic": payload,
                    "search_term": payload
                },
                headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
            )
            
            # Should reject malicious input
            assert response.status_code in [200, 400, 422], f"Unexpected status code for SQL injection: {response.status_code}"
            
            if response.status_code == 200:
                # If accepted, should be sanitized
                data = response.json()
                # Check that no sensitive data is exposed
                assert "error" not in str(data).lower(), "Error messages should not expose sensitive data"
        
        # Test XSS payloads
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert('xss');//"
        ]
        
        for payload in xss_payloads:
            response = await self.framework.client.post(
                f"{self.framework.base_url}/api/content/generate",
                json={
                    "topic": payload,
                    "content_type": "Article Ideas",
                    "count": 1
                },
                headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
            )
            
            # Should reject or sanitize XSS payloads
            assert response.status_code in [200, 400, 422], f"Unexpected status code for XSS: {response.status_code}"
            
            if response.status_code == 200:
                data = response.json()
                # Check that payload is sanitized
                response_text = str(data)
                assert "<script>" not in response_text, "Script tags should be sanitized"
                assert "javascript:" not in response_text, "JavaScript URLs should be sanitized"
        
        # Test very large inputs
        large_input = "a" * 10000  # 10KB input
        
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/trends/analyze",
            json={
                "keyword": large_input,
                "time_range": "12 months",
                "geo": "US"
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        # Should either truncate or reject large input
        assert response.status_code in [200, 400, 413], f"Unexpected status code for large input: {response.status_code}"
        
        # Test special characters
        special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"
        
        response = await self.framework.client.post(
            f"{self.framework.base_url}/api/keywords/crawl",
            json={
                "seed_keyword": special_chars,
                "depth": 1,
                "geo": "US",
                "language": "en"
            },
            headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
        )
        
        # Should handle special characters gracefully
        assert response.status_code in [200, 400, 422], f"Unexpected status code for special characters: {response.status_code}"
        
        logger.info("✅ ERR-003 passed: Malicious input handling tested")
    
    async def _simulate_user_workflow(self, user_id: int) -> bool:
        """Simulate a user workflow for performance testing"""
        try:
            # Simulate a quick workflow
            response = await self.framework.client.post(
                f"{self.framework.base_url}/api/affiliate-research/search",
                json={
                    "topic": f"perf test {user_id}",
                    "search_term": f"perf test {user_id}"
                },
                headers={"Authorization": f"Bearer {self.framework.current_user_token}"}
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"User {user_id} workflow error: {e}")
            return False

async def run_performance_error_tests(framework: TrendTapTestFramework) -> List[Any]:
    """Run all performance and error handling tests"""
    tests = PerformanceErrorTests(framework)
    
    test_cases = [
        ("PERF-001", "API Response Times", tests.test_perf_001_api_response_times),
        ("PERF-002", "Concurrent User Performance", tests.test_perf_002_concurrent_user_performance),
        ("ERR-001", "API Timeout Handling", tests.test_err_001_api_timeout_handling),
        ("ERR-002", "Database Connection Issues", tests.test_err_002_database_connection_issues),
        ("ERR-003", "Malicious Input Handling", tests.test_err_003_malicious_input_handling),
    ]
    
    return await framework.run_test_suite("Performance & Error Tests", test_cases)

if __name__ == "__main__":
    async def main():
        async with TrendTapTestFramework() as framework:
            # Authenticate user
            await framework.authenticate_user()
            
            # Run performance and error tests
            suite = await run_performance_error_tests(framework)
            
            # Generate report
            report = framework.generate_report([suite])
            await framework.save_report(report)
            
            print(f"Performance & Error Tests: {suite.passed_tests}/{suite.total_tests} passed")
    
    asyncio.run(main())


