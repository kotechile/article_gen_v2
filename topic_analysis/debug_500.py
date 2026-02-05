import httpx
import asyncio

async def test_generate():
    topic_id = "dbffc2cd-9b02-4737-a178-a20917ed92f5"
    url = f"http://localhost:8000/api/research-topics/{topic_id}/subtopics/generate"
    headers = {
        "Content-Type": "application/json"
    }
    
    # We need a user_id query param or header? 
    # Based on logs, user_id is passed via Supabase/Auth header usually, 
    # but the logs show: Generating subtopics for ... (User: f248b7ed-b8df-4464-8544-8304d7ae4c30)
    # The API might fallback to a test user if not provided in dev, or we might get 401.
    # But the 500 happens *after* success, so auth is likely fine or bypassed.
    # The API client adds "X-API-Key" and "Authorization".
    
    # Let's try to hit it without auth first (dev mode usually permissive) 
    # or mimicking the logs.
    
    print(f"POST {url}")
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(url, headers=headers)
            print(f"Status: {resp.status_code}")
            print(f"Response: {resp.text}")
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_generate())
