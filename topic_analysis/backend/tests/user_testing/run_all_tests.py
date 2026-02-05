"""
Main Test Runner for TrendTap User Testing Plan
Executes all test suites and generates comprehensive reports
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from test_framework import TrendTapTestFramework
from test_affiliate_research import run_affiliate_research_tests
from test_trend_analysis import run_trend_analysis_tests
from test_content_generation import run_content_generation_tests
from test_e2e_integration import run_e2e_integration_tests
from test_performance_errors import run_performance_error_tests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('user_testing.log')
    ]
)
logger = logging.getLogger(__name__)

class TrendTapTestRunner:
    """Main test runner for TrendTap user testing"""
    
    def __init__(self, base_url: str = "http://localhost:8000", frontend_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.frontend_url = frontend_url
        self.test_results = []
        self.start_time = None
        self.end_time = None
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and generate comprehensive report"""
        logger.info("🚀 Starting TrendTap User Testing Plan Execution")
        self.start_time = datetime.now()
        
        async with TrendTapTestFramework(self.base_url, self.frontend_url) as framework:
            # Authenticate user
            logger.info("🔐 Authenticating test user...")
            auth_success = await framework.authenticate_user()
            
            if not auth_success:
                logger.error("❌ Authentication failed. Please check credentials and system status.")
                return self._generate_error_report("Authentication failed")
            
            logger.info("✅ User authenticated successfully")
            
            # Run all test suites
            test_suites = []
            
            try:
                # Phase 1: Core Functionality Tests
                logger.info("📋 Phase 1: Running Core Functionality Tests...")
                
                logger.info("  🔍 Running Affiliate Research Tests...")
                affiliate_suite = await run_affiliate_research_tests(framework)
                test_suites.append(affiliate_suite)
                
                logger.info("  📈 Running Trend Analysis Tests...")
                trend_suite = await run_trend_analysis_tests(framework)
                test_suites.append(trend_suite)
                
                logger.info("  💡 Running Content Generation Tests...")
                content_suite = await run_content_generation_tests(framework)
                test_suites.append(content_suite)
                
                # Phase 2: Integration Tests
                logger.info("📋 Phase 2: Running Integration Tests...")
                
                logger.info("  🔄 Running E2E Integration Tests...")
                e2e_suite = await run_e2e_integration_tests(framework)
                test_suites.append(e2e_suite)
                
                # Phase 3: Performance & Error Tests
                logger.info("📋 Phase 3: Running Performance & Error Tests...")
                
                logger.info("  ⚡ Running Performance & Error Tests...")
                perf_suite = await run_performance_error_tests(framework)
                test_suites.append(perf_suite)
                
            except Exception as e:
                logger.error(f"❌ Test execution failed: {e}")
                return self._generate_error_report(f"Test execution failed: {e}")
            
            # Generate comprehensive report
            logger.info("📊 Generating comprehensive test report...")
            report = framework.generate_report(test_suites)
            
            # Add execution metadata
            self.end_time = datetime.now()
            report["execution_metadata"] = {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "total_duration": (self.end_time - self.start_time).total_seconds(),
                "base_url": self.base_url,
                "frontend_url": self.frontend_url
            }
            
            # Save report
            await framework.save_report(report, f"trendtap_user_testing_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json")
            
            # Print summary
            self._print_test_summary(report)
            
            return report
    
    def _generate_error_report(self, error_message: str) -> Dict[str, Any]:
        """Generate error report when tests fail to run"""
        return {
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0,
                "execution_time": datetime.now().isoformat(),
                "error": error_message
            },
            "suites": [],
            "recommendations": [f"Fix critical error: {error_message}"],
            "execution_metadata": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": datetime.now().isoformat(),
                "total_duration": 0,
                "base_url": self.base_url,
                "frontend_url": self.frontend_url,
                "error": error_message
            }
        }
    
    def _print_test_summary(self, report: Dict[str, Any]):
        """Print test execution summary"""
        summary = report["summary"]
        
        print("\n" + "="*80)
        print("🎯 TRENDTAP USER TESTING PLAN - EXECUTION SUMMARY")
        print("="*80)
        
        print(f"📊 Overall Results:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Pass Rate: {summary['pass_rate']:.1f}%")
        
        if "execution_metadata" in report:
            metadata = report["execution_metadata"]
            print(f"   Duration: {metadata['total_duration']:.1f} seconds")
            print(f"   Start Time: {metadata['start_time']}")
            print(f"   End Time: {metadata['end_time']}")
        
        print(f"\n📋 Test Suite Results:")
        for suite in report["suites"]:
            status_icon = "✅" if suite["failed_tests"] == 0 else "❌"
            print(f"   {status_icon} {suite['name']}: {suite['passed_tests']}/{suite['total_tests']} passed")
        
        if summary["pass_rate"] >= 95:
            print(f"\n🎉 EXCELLENT! Pass rate {summary['pass_rate']:.1f}% meets quality standards")
        elif summary["pass_rate"] >= 80:
            print(f"\n✅ GOOD! Pass rate {summary['pass_rate']:.1f}% is acceptable")
        else:
            print(f"\n⚠️  ATTENTION! Pass rate {summary['pass_rate']:.1f}% needs improvement")
        
        if report["recommendations"]:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"   {i}. {rec}")
        
        print("="*80)
        print("📄 Detailed report saved to test_data/ directory")
        print("="*80)

async def main():
    """Main entry point for test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run TrendTap User Testing Plan")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Backend API base URL")
    parser.add_argument("--frontend-url", default="http://localhost:3000", help="Frontend URL")
    parser.add_argument("--suite", choices=["all", "affiliate", "trend", "content", "e2e", "performance"], 
                       default="all", help="Test suite to run")
    
    args = parser.parse_args()
    
    runner = TrendTapTestRunner(args.base_url, args.frontend_url)
    
    if args.suite == "all":
        report = await runner.run_all_tests()
    else:
        # Run specific test suite
        async with TrendTapTestFramework(args.base_url, args.frontend_url) as framework:
            auth_success = await framework.authenticate_user()
            if not auth_success:
                print("❌ Authentication failed")
                return
            
            if args.suite == "affiliate":
                suite = await run_affiliate_research_tests(framework)
            elif args.suite == "trend":
                suite = await run_trend_analysis_tests(framework)
            elif args.suite == "content":
                suite = await run_content_generation_tests(framework)
            elif args.suite == "e2e":
                suite = await run_e2e_integration_tests(framework)
            elif args.suite == "performance":
                suite = await run_performance_error_tests(framework)
            
            report = framework.generate_report([suite])
            await framework.save_report(report)
            
            print(f"{args.suite.title()} Tests: {suite.passed_tests}/{suite.total_tests} passed")
    
    # Exit with appropriate code
    if report["summary"]["pass_rate"] >= 80:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == "__main__":
    asyncio.run(main())


