
import asyncio
import os
import sys
import logging
import json
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Load env
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_volume")

from src.integrations.dataforseo import dataforseo_api

async def test_volume():
    # Test with a known high-volume keyword
    seeds = ["credit card"]
    logger.info(f"Testing Standard API for seeds: {seeds}")
    
    try:
        # We need to manually invoke the steps to see the raw response if possible, 
        # or rely on the fact that existing code returns parsed dicts.
        # But if parsed dicts have 0, we need to inspect the raw response.
        # DataForSEOAPI class doesn't expose raw response easily.
        # I will copy the 'requests' logic here to spy on it.
        
        # Or simpler: I will add a spy log inside dataforseo.py temporarily? No, tool overhead.
        # I will use the debug script to call the method and print the result.
        # IF the result is 0, I can't see why without raw.
        # So I will reimplement the raw call here.
        
        import httpx
        import base64
        
        # Get Auth
        auth = dataforseo_api.auth_header
        headers = {"Authorization": auth, "Content-Type": "application/json"}
        
        # 1. Post Task
        url = "https://api.dataforseo.com/v3/keywords_data/google_ads/keywords_for_keywords/task_post"
        payload = [{
            "keywords": seeds,
            "language_code": "en",
            "location_code": 2840,
            "order_by": ["keyword_info.search_volume,desc"]
        }]
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=headers)
            print("Post Resp:", resp.json())
            task_id = resp.json()['tasks'][0]['id']
            
            # 2. Poll
            print(f"Polling Task {task_id}...")
            for _ in range(20):
                await asyncio.sleep(2)
                get_url = f"https://api.dataforseo.com/v3/keywords_data/google_ads/keywords_for_keywords/task_get/{task_id}"
                resp = await client.get(get_url, headers=headers)
                data = resp.json()
                status = data['tasks'][0]['status_message']
                print(f"Status: {status}")
                if status == "Ok.":
                     # PRINT RAW RESULT ITEM
                     result_item = data['tasks'][0]['result'][0]
                     print("RAW RESULT TYPE:", type(result_item))
                     print("RAW ITEM EXAMPLE:", json.dumps(result_item, indent=2)[:500]) 
                     # Check the first keyword item
                     if result_item.get('items'):
                         print("FIRST KEYWORD ITEM:", json.dumps(result_item['items'][0], indent=2))
                     else:
                         print("NO ITEMS FOUND IN RESULT")
                     break
    except Exception as e:
        logger.error(f"Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_volume())
