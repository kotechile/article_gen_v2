#!/usr/bin/env python3
"""
Quick Test Script for TrendTap User Testing
Runs a subset of critical tests to verify system functionality
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from test_framework import TrendTapTestFramework
from test_affiliate_research import AffiliateResearchTests
from test_trend_analysis import TrendAnalysisTests
from test_content_generation import ContentGenerationTests

async def quick_test():
    """Run a quick test to verify system functionality"""
    print("🚀 Running Quick TrendTap System Test...")
    
    async with TrendTapTestFramework() as framework:
        # Test authentication
        print("🔐 Testing authentication...")
        auth_success = await framework.authenticate_user()
        if not auth_success:
            print("❌ Authentication failed - please check if the system is running")
            return False
        print("✅ Authentication successful")
        
        # Test basic affiliate research
        print("🔍 Testing affiliate research...")
        try:
            affiliate_tests = AffiliateResearchTests(framework)
            await affiliate_tests.test_ar_001_basic_affiliate_search()
            print("✅ Affiliate research working")
        except Exception as e:
            print(f"❌ Affiliate research failed: {e}")
            return False
        
        # Test basic trend analysis
        print("📈 Testing trend analysis...")
        try:
            trend_tests = TrendAnalysisTests(framework)
            await trend_tests.test_ta_001_basic_trend_analysis()
            print("✅ Trend analysis working")
        except Exception as e:
            print(f"❌ Trend analysis failed: {e}")
            return False
        
        # Test basic content generation
        print("💡 Testing content generation...")
        try:
            content_tests = ContentGenerationTests(framework)
            await content_tests.test_ci_001_basic_content_generation()
            print("✅ Content generation working")
        except Exception as e:
            print(f"❌ Content generation failed: {e}")
            return False
        
        print("🎉 All core systems are working!")
        return True

if __name__ == "__main__":
    success = asyncio.run(quick_test())
    sys.exit(0 if success else 1)
