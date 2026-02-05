#!/usr/bin/env python3
"""
Direct test of DataForSEO API to debug the issue.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def test_direct_dataforseo():
    """Test DataForSEO API directly"""
    
    # Import the API client
    from dataforseo.api_integration import DataForSEOAPIClient
    
    logger.info("🚀 Testing DataForSEO API directly")
    
    # Test keywords
    keywords = ["Market Analysis", "Consumer Behavior", "Competitive Landscape"]
    location = "United States"
    time_range = "12m"
    
    logger.info(f"📊 Keywords: {keywords}")
    logger.info(f"📍 Location: {location}")
    logger.info(f"⏰ Time range: {time_range}")
    
    try:
        # Initialize client
        logger.info("🌐 Initializing DataForSEO API client...")
        client = DataForSEOAPIClient()
        await client.initialize()
        logger.info("✅ Client initialized")
        
        # Get trend data
        logger.info("🔄 Calling get_trend_data...")
        trend_data = await client.get_trend_data(keywords, location, time_range)
        logger.info(f"✅ get_trend_data completed")
        
        # Log results
        logger.info(f"📊 Trend data received: {len(trend_data)} items")
        logger.info(f"📊 Trend data type: {type(trend_data)}")
        
        if trend_data:
            logger.info("✅ Success! Got trend data:")
            for i, item in enumerate(trend_data):
                logger.info(f"📊 Item {i}: keyword='{item.keyword}', time_series_length={len(item.time_series) if item.time_series else 0}")
        else:
            logger.warning("⚠️ No trend data received")
        
        return trend_data
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        return None

if __name__ == "__main__":
    asyncio.run(test_direct_dataforseo())
