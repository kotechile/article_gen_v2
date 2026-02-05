import asyncio
import os
import sys
from dotenv import load_dotenv

# Add src to python path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Load environment variables
load_dotenv()

async def repro():
    try:
        from src.services.affiliate_research_service import AffiliateResearchService
        from src.core.api_key_manager import api_key_manager
        
        print("--- API Key Verification ---")
        openai_key = api_key_manager.get_openai_key()
        linkup_key = api_key_manager.get_linkup_key()
        
        print(f"OpenAI API Key found: {openai_key is not None}")
        if openai_key:
            print(f"OpenAI API Key starts with: {openai_key[:8]}...")
            
        print(f"LinkUp API Key found: {linkup_key is not None}")
        if linkup_key:
            print(f"LinkUp API Key starts with: {linkup_key[:8]}...")
            
        service = AffiliateResearchService()
        
        topics = [
            "The pursuit of financial independence",
            "Backyard astronomy and telescopes",
            "Sustainable outdoor camping gear"
        ]
        
        print("\n--- Starting Search Tests ---")
        for topic in topics:
            print(f"\nSearching for: '{topic}'")
            # We bypass cache for testing
            result = await service.search_affiliate_programs(
                search_term=topic,
                user_id=None # Don't save to DB
            )
            
            programs = result.get('programs', [])
            print(f"Found {len(programs)} programs")
            
            for i, p in enumerate(programs[:3]):
                print(f"  {i+1}. {p.get('name')} (Source: {p.get('source')})")
                
            if result.get('analysis'):
                print(f"  Analysis category: {result['analysis'].get('category')}")
                
    except Exception as e:
        print(f"Error during reproduction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(repro())
