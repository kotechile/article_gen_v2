import asyncio
import sys
import os
import logging

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_dataforseo():
    print("\n--- Testing DataForSEO Service ---")
    try:
        from src.services.trend_service import TrendService
        service = TrendService()
        
        topic = "coffee grinder"
        print(f"Fetching trends for: {topic}")
        
        # This calls DataForSEO internally
        data = await service._get_dataforseo_trends([topic])
        
        if data:
            print("✅ SUCCESS: DataForSEO returned data")
            print(f"Keys: {data.keys()}")
            if "historical" in data:
                print(f"Historical points: {len(data['historical'])}")
        else:
            print("❌ FAILURE: DataForSEO returned None")
            
    except Exception as e:
        print(f"❌ EXCEPTION in DataForSEO check: {e}")
        import traceback
        traceback.print_exc()

async def debug_affiliate():
    print("\n--- Testing Affiliate Research Service ---")
    try:
        from src.services.enhanced_affiliate_research_service import EnhancedAffiliateResearchService
        service = EnhancedAffiliateResearchService()
        
        topic = "coffee grinder"
        user_id = "test-user-debug" # Mock user ID
        
        print(f"Discovering offers for: {topic}")
        
        # We need to mock the methods that write to DB if we don't have a real user/session
        # But let's see if we can just run it and catch where it might fail on DB constraints
        # Or better, just call the internal discovery method if possible, 
        # but `intelligent_offer_discovery` is the public API.
        
        # To avoid DB constraints with invalid user_id, we might hit issues. 
        # But let's try. If it fails on DB, at least we know the API part started.
        
        data = await service.intelligent_offer_discovery(
            search_terms=[topic],
            user_id=user_id,
            research_scope="quick",
            max_offers=3
        )
        
        print(f"✅ Affiliate Discovery Result: {data.keys()}")
        print(f"Discovered items: {data.get('discovered_programs', 'N/A')}")
        
    except Exception as e:
        print(f"❌ EXCEPTION in Affiliate check: {e}")
        import traceback
        traceback.print_exc()

async def main():
    # Load env vars if needed
    from dotenv import load_dotenv
    load_dotenv()
    
    await debug_dataforseo()
    await debug_affiliate()

if __name__ == "__main__":
    asyncio.run(main())
