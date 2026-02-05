
import asyncio
import sys
import os
import logging
from pprint import pprint
from unittest.mock import MagicMock, AsyncMock, patch

# Add backend to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock database and external dependencies
sys.modules['src.core.database'] = MagicMock()
sys.modules['src.core.redis'] = MagicMock()
sys.modules['src.core.config'] = MagicMock()
sys.modules['httpx'] = MagicMock()
sys.modules['pydantic_settings'] = MagicMock()
sys.modules['pydantic'] = MagicMock()
sys.modules['aiohttp'] = MagicMock()
sys.modules['structlog'] = MagicMock()
sys.modules['supabase'] = MagicMock()
sys.modules['sqlalchemy'] = MagicMock()
sys.modules['sqlalchemy.orm'] = MagicMock()
sys.modules['sqlalchemy.ext.declarative'] = MagicMock()
sys.modules['..core.database'] = MagicMock()
sys.modules['..core.redis'] = MagicMock()
sys.modules['..core.config'] = MagicMock()

from src.services.trend_service import TrendService
from src.services.dataforseo_service import DataForSEOService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_dataforseo_trend_integration():
    print("=== Testing Trend Service with DataForSEO ===\n")
    
    # Configure cache mock to return None so we don't skip logic
    from src.core.redis import cache
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()

    service = TrendService()
    topics = ["Remote Work Tools"]

    # Mock DataForSEOService.get_search_volume_history to simulate real API response
    mock_history = {
        "Remote Work Tools": {
            "search_volume": 5000,
            "monthly_searches": [
                {"year": 2024, "month": 1, "search_volume": 4000},
                {"year": 2024, "month": 2, "search_volume": 4200},
                {"year": 2024, "month": 3, "search_volume": 5000},
            ],
            "competition": 0.45,
            "cpc": 2.5
        }
    }

    # Mock DataForSEOService.get_google_trends_explore
    mock_explore = {
        "Remote Work Tools": [
            {"date": "2024-01-01", "value": 70},
            {"date": "2024-02-01", "value": 75},
            {"date": "2024-03-01", "value": 100},
        ]
    }

    with patch('src.services.trend_service.DataForSEOService', autospec=True) as MockDF:
        mock_df_instance = MockDF.return_value
        mock_df_instance.get_search_volume_history = AsyncMock(return_value=mock_history)
        mock_df_instance.get_google_trends_explore = AsyncMock(return_value=mock_explore)
        mock_df_instance.close = AsyncMock()
        
        print("--- 1. Testing _get_dataforseo_trends rich merging ---")
        trends_data = await service._get_dataforseo_trends(topics)
        
        if trends_data:
            print("✅ DataForSEO Rich Trends Mapped Successfully:")
            pprint(trends_data)
            
            # Verify merging logic
            assert trends_data["source"] == "dataforseo_rich"
            assert trends_data["search_volume"] == 5000
            # From Explore data
            assert trends_data["historical"][0]["interest"] == 70
            assert trends_data["historical"][2]["interest"] == 100
            # From Search Volume data
            assert trends_data["historical"][0]["absolute_volume"] == 4000
            assert trends_data["historical"][2]["absolute_volume"] == 5000
        else:
            print("❌ DataForSEO Trends Mapping Failed")

        print("\n--- 2. Testing _get_google_trends_data prioritization ---")
        # Ensure it picks DataForSeo over Mock or Google Trends
        final_trends = await service._get_google_trends_data(topics)
        print("✅ Final Trend Source used:", final_trends.get("source"))
        assert final_trends.get("source") == "dataforseo_rich"

if __name__ == "__main__":
    asyncio.run(test_dataforseo_trend_integration())
