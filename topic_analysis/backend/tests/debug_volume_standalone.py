
import asyncio
import os
import json
import base64
import requests
from dotenv import load_dotenv

# Load Supabase Env
env_path = os.path.join(os.path.dirname(__file__), '../.env')
load_dotenv(env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

def get_auth_header():
    # Fetch key from Supabase directly
    url = f"{SUPABASE_URL}/rest/v1/api_keys?select=*&provider=eq.dataforseo"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    resp = requests.get(url, headers=headers)
    data = resp.json()
    if data:
        val = data[0]['key_value']
        if val.startswith("Basic "): return val
        return f"Basic {val}"
    return None

async def test_volume():
    auth = get_auth_header()
    if not auth:
        print("Failed to get auth")
        return

    print(f"Auth header: {auth[:20]}...")
    
    seeds = ["credit card"]
    
    import httpx
    headers = {"Authorization": auth, "Content-Type": "application/json"}
    url = "https://api.dataforseo.com/v3/keywords_data/google_ads/search_volume/task_post"
    
    payload = [{
        "keywords": seeds,
        "language_code": "en",
        "location_code": 2840
    }]
    
    async with httpx.AsyncClient() as client:
        print(f"Posting to {url}...")
        resp = await client.post(url, json=payload, headers=headers)
        print("Post Resp:", resp.status_code)
        
        if resp.status_code != 200:
            print(resp.text)
            return

        data = resp.json()
        if 'tasks' not in data:
            print("No tasks returned")
            return
            
        task_id = data['tasks'][0]['id']
        print(f"Task ID: {task_id}")
        
        for _ in range(20):
            print("Polling...")
            await asyncio.sleep(2)
            get_url = f"https://api.dataforseo.com/v3/keywords_data/google_ads/search_volume/task_get/{task_id}"
            resp = await client.get(get_url, headers=headers)
            data = resp.json()
            status = data['tasks'][0]['status_message']
            print(f"Status: {status}")
            
            if status == "Ok.":
                 result_item = data['tasks'][0]['result'][0]
                 print("RAW RESULT TYPE:", type(result_item))
                 # For search_volume, result is list of KeywordInfo?
                 if 'items' not in result_item:
                      print("RESULT KEYS:", result_item.keys())
                      # Likely flat list or 'keywords' key?
                      # Check example result
                 else:
                      print("FIRST ITEM:", json.dumps(result_item['items'][0], indent=2))
                 break

if __name__ == "__main__":
    asyncio.run(test_volume())
