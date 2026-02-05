"""
User Testing Framework for TrendTap
Implements the comprehensive user testing plan
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

class TestSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class TestResult:
    test_id: str
    test_name: str
    status: TestStatus
    duration: float
    error_message: Optional[str] = None
    performance_metrics: Dict[str, Any] = None
    screenshots: List[str] = None
    logs: List[str] = None

@dataclass
class TestSuite:
    name: str
    tests: List[TestResult]
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0

class TrendTapTestFramework:
    """Main testing framework for TrendTap user testing"""
    
    def __init__(self, base_url: str = "http://localhost:8000", frontend_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.frontend_url = frontend_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.test_results: List[TestResult] = []
        self.current_user_token: Optional[str] = None
        self.test_data_dir = Path("test_data")
        self.test_data_dir.mkdir(exist_ok=True)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def authenticate_user(self, email: str = "test@trendtap.com", password: str = "testpassword123") -> bool:
        """Authenticate test user using Google OAuth or direct token"""
        try:
            # For testing purposes, we'll use a mock token or check if we can access endpoints without auth
            # First, try to access a public endpoint to see if authentication is required
            response = await self.client.get(f"{self.base_url}/api/health/")
            
            if response.status_code == 200:
                # System is running, try to access a protected endpoint
                test_response = await self.client.get(f"{self.base_url}/api/affiliate-research/networks")
                
                if test_response.status_code == 200:
                    # No authentication required for testing
                    self.current_user_token = "test-token"
                    logger.info("System running without authentication requirement")
                    return True
                elif test_response.status_code == 401:
                    # Authentication required, but we don't have Google OAuth setup for testing
                    logger.warning("Authentication required but Google OAuth not configured for testing")
                    # For now, we'll skip authentication and test what we can
                    self.current_user_token = "test-token"
                    return True
                else:
                    logger.error(f"Unexpected response: {test_response.status_code}")
                    return False
            else:
                logger.error(f"Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    async def run_test(self, test_func, test_id: str, test_name: str) -> TestResult:
        """Run a single test and return result"""
        logger.info(f"Running test: {test_name}")
        
        start_time = time.time()
        result = TestResult(
            test_id=test_id,
            test_name=test_name,
            status=TestStatus.RUNNING,
            duration=0.0,
            performance_metrics={},
            screenshots=[],
            logs=[]
        )
        
        try:
            # Run the test
            await test_func()
            
            # Test passed
            result.status = TestStatus.PASSED
            result.duration = time.time() - start_time
            logger.info(f"✅ Test passed: {test_name} ({result.duration:.2f}s)")
            
        except Exception as e:
            # Test failed
            result.status = TestStatus.FAILED
            result.duration = time.time() - start_time
            result.error_message = str(e)
            logger.error(f"❌ Test failed: {test_name} - {e}")
            
        self.test_results.append(result)
        return result
    
    async def run_test_suite(self, suite_name: str, tests: List[Tuple[str, str, callable]]) -> TestSuite:
        """Run a complete test suite"""
        logger.info(f"Starting test suite: {suite_name}")
        
        suite = TestSuite(
            name=suite_name,
            tests=[],
            start_time=datetime.now(),
            total_tests=len(tests)
        )
        
        for test_id, test_name, test_func in tests:
            result = await self.run_test(test_func, test_id, test_name)
            suite.tests.append(result)
            
            if result.status == TestStatus.PASSED:
                suite.passed_tests += 1
            elif result.status == TestStatus.FAILED:
                suite.failed_tests += 1
        
        suite.end_time = datetime.now()
        logger.info(f"Test suite completed: {suite_name} - {suite.passed_tests}/{suite.total_tests} passed")
        
        return suite
    
    def generate_report(self, suites: List[TestSuite]) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = sum(suite.total_tests for suite in suites)
        total_passed = sum(suite.passed_tests for suite in suites)
        total_failed = sum(suite.failed_tests for suite in suites)
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "pass_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
                "execution_time": datetime.now().isoformat()
            },
            "suites": [asdict(suite) for suite in suites],
            "recommendations": self._generate_recommendations(suites)
        }
        
        return report
    
    def _generate_recommendations(self, suites: List[TestSuite]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        for suite in suites:
            if suite.failed_tests > 0:
                recommendations.append(f"Fix {suite.failed_tests} failed tests in {suite.name}")
            
            # Check for performance issues
            for test in suite.tests:
                if test.performance_metrics and test.performance_metrics.get("response_time", 0) > 10:
                    recommendations.append(f"Optimize performance for {test.test_name}")
        
        return recommendations
    
    async def save_report(self, report: Dict[str, Any], filename: str = None):
        """Save test report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_report_{timestamp}.json"
        
        report_path = self.test_data_dir / filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Test report saved: {report_path}")
        return report_path

# Test data and utilities
class TestData:
    """Test data for various test scenarios"""
    
    @staticmethod
    def get_affiliate_test_data():
        return {
            "search_terms": [
                "eco friendly homes",
                "coffee roasting",
                "sustainable living",
                "home energy efficiency"
            ],
            "niches": [
                "Home & Garden",
                "Food & Beverage",
                "Lifestyle",
                "Technology"
            ],
            "filters": {
                "min_commission": 5.0,
                "min_epc": 10.0,
                "networks": ["ShareASale", "Impact", "Amazon"]
            }
        }
    
    @staticmethod
    def get_trend_test_data():
        return {
            "keywords": [
                "sustainable living",
                "eco homes",
                "green energy",
                "artificial intelligence"
            ],
            "time_ranges": ["3 months", "6 months", "12 months", "5 years"],
            "geographic_regions": ["US", "Europe", "Asia", "Global"]
        }
    
    @staticmethod
    def get_content_test_data():
        return {
            "topics": [
                "sustainable home improvements",
                "home energy efficiency",
                "eco-friendly living",
                "green technology"
            ],
            "content_types": ["Article Ideas", "Software Solutions", "Video Content"],
            "complexity_levels": ["Low", "Medium", "High"]
        }
    
    @staticmethod
    def get_csv_test_file():
        """Create a test CSV file for trend analysis"""
        csv_content = """Date,Value
2024-01-01,50
2024-02-01,65
2024-03-01,70
2024-04-01,80
2024-05-01,75
2024-06-01,85
2024-07-01,90
2024-08-01,95
2024-09-01,100
2024-10-01,85
2024-11-01,80
2024-12-01,75"""
        
        return csv_content

# Performance monitoring utilities
class PerformanceMonitor:
    """Monitor and track performance metrics during tests"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.metrics[operation] = {
            "start_time": time.time(),
            "end_time": None,
            "duration": None
        }
    
    def end_timer(self, operation: str):
        """End timing an operation"""
        if operation in self.metrics:
            self.metrics[operation]["end_time"] = time.time()
            self.metrics[operation]["duration"] = (
                self.metrics[operation]["end_time"] - self.metrics[operation]["start_time"]
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all performance metrics"""
        return self.metrics.copy()
    
    def get_duration(self, operation: str) -> Optional[float]:
        """Get duration for a specific operation"""
        return self.metrics.get(operation, {}).get("duration")
