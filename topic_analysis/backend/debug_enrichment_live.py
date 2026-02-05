import asyncio
import logging
import sys
import os
from uuid import uuid4

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Config logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_enrichment():
    try:
        from src.services.trend_service import TrendService
        from src.services.enhanced_affiliate_research_service import EnhancedAffiliateResearchService
        
        print("\n--- 1. Testing Trend Service (DataForSEO) ---")
        trend_service = TrendService()
        keyword = "coffee grinder"
        print(f"Fetching trends for: {keyword}")
        
        try:
            trend_data = await trend_service._get_google_trends_data([keyword])
            print(f"Trend Data Result Type: {type(trend_data)}")
            if trend_data:
                print(f"Trend Source: {trend_data.get('source')}")
                print(f"Historical Points: {len(trend_data.get('historical', []))}")
                print(f"First Point: {trend_data.get('historical', [])[0] if trend_data.get('historical') else 'None'}")
            else:
                print("❌ Trend Data is None")
        except Exception as e:
            print(f"❌ Trend Service Failed: {e}")

        print("\n--- 2. Testing Affiliate Service (LLM/LinkUp) ---")
        affiliate_service = EnhancedAffiliateResearchService()
        user_id = "test-user-id" # We might need a real UUID if the service validates it strict, but string is usually fine in python
        
        print(f"Discovering offers for: {keyword}")
        try:
            discovery = await affiliate_service.intelligent_offer_discovery(
                search_terms=[keyword],
                user_id=str(uuid4()),
                research_scope="comprehensive",
                max_offers=5
            )
            print(f"Discovery Result Keys: {list(discovery.keys())}")
            print(f"Discovered Programs: {discovery.get('discovered_programs')}")
            print(f"Recommended Offers: {len(discovery.get('recommended_offers', []))}")
            if discovery.get('recommended_offers'):
                print(f"Sample Offer: {discovery['recommended_offers'][0].get('program_name')}")
        except Exception as e:
            print(f"❌ Affiliate Service Failed: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"Global Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_enrichment())
