
import asyncio
import logging
import sys
import os

# Ensure backend directory is in path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from src.services.enhanced_affiliate_research_service import EnhancedAffiliateResearchService
from src.integrations.linkup_api import LinkUpAPI

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

async def debug_offer_flow():
    topic = "Home and Garden" # Default or override
    
    print(f"\n--- 1. Testing LinkUp Search for '{topic}' ---")
    linkup = LinkUpAPI()
    results = await linkup.search_offers(topic, limit=5)
    
    print(f"\n[LinkUp Raw Results Count]: {len(results)}")
    for i, res in enumerate(results):
        print(f"[{i+1}] {res.get('name')} - {res.get('url')}")
        print(f"    Desc: {res.get('description')[:100]}...")

    print(f"\n--- 2. Testing Enhanced Service (LLM Processing) ---")
    service = EnhancedAffiliateResearchService()
    
    # We want to see the RAW LLM response.
    # The service logs warnings if parsing fails, but we want to see the SUCCESS or FAILURE content.
    # We will invoke the method and catch the result.
    
    # To see the prompt/response, we'd ideally enable debug logging for the service
    logging.getLogger('src.services.enhanced_affiliate_research_service').setLevel(logging.DEBUG)
    
    try:
        # Mock user_id
        user_id = "f248b7ed-b8df-4464-8544-8304d7ae4c30" 
        
        print("\n[Executing intelligent_offer_discovery]...")
        result = await service.intelligent_offer_discovery(
            search_terms=[topic],
            user_id=user_id,
            ignore_cache=True
        )
        
        print("\n[Discovery Result]:")
        print(f"Session ID: {result.get('session_id')}")
        print(f"Programs Found: {len(result.get('recommended_offers', []))}")
        
        for prog in result.get('recommended_offers', []):
            print(f"- {prog.get('name')} ({prog.get('commission')})")

    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_offer_flow())
