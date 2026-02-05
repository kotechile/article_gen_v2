
import asyncio
import sys
import os
import logging
from pprint import pprint

# Add backend to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock database dependency for standalone running
from unittest.mock import MagicMock, patch
sys.modules['src.core.database'] = MagicMock()
sys.modules['src.core.redis'] = MagicMock()
sys.modules['src.core.config'] = MagicMock()
sys.modules['aiohttp'] = MagicMock()
sys.modules['structlog'] = MagicMock()
sys.modules['..core.database'] = MagicMock()
sys.modules['..core.redis'] = MagicMock()
sys.modules['..core.config'] = MagicMock()

# Now import the service
from src.services.trend_service import TrendService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_components_individually():
    print("=== Testing Trend Service Components ===\n")
    service = TrendService()
    topics = ["Remote Work Tools"]

    # 1. Google Trends
    print("--- 1. Testing Google Trends Component ---")
    try:
        # We call the internal method directly to isolate it
        trends_data = await service._get_google_trends_data(topics)
        print("✅ Google Trends Data Retrieved:")
        pprint(trends_data)
    except Exception as e:
        print(f"❌ Google Trends Failed: {e}")
    print("\n")

    # 2. LLM Forecasting
    print("--- 2. Testing LLM Forecasting Component ---")
    try:
        # We pass the trends data we just got (or mock it) into the LLM component
        llm_forecast = await service._generate_llm_forecast(topics, trends_data, affiliate_data=None)
        print("✅ LLM Forecast Generated:")
        pprint(llm_forecast)
    except Exception as e:
        print(f"❌ LLM Forecasting Failed: {e}")
    print("\n")

    # 3. Social Signals
    print("--- 3. Testing Social Signals Component ---")
    try:
        social_signals = await service._get_social_signals(topics)
        print("✅ Social Signals Retrieved:")
        pprint(social_signals)
    except Exception as e:
        print(f"❌ Social Signals Failed: {e}")
    print("\n")

if __name__ == "__main__":
    asyncio.run(test_components_individually())
