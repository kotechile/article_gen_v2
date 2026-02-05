
import asyncio
import httpx
import json
import os
import sys
from typing import Dict, Any

# Test Config
BASE_URL = "http://localhost:8000"
TEST_EMAIL = "test_kw_flow@example.com"
TEST_PASSWORD = "password123"

async def main():
    print("🚀 Starting Keyword Flow Verification...")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. Authenticate (Login or Signup)
        token = await get_auth_token(client)
        if not token:
            print("❌ Authentication failed due to missing credentials or server error.")
            # Try a fake token if dev mode allows, or exit
            # Assuming backend requires real token.
            print("Assuming existing user or manual token required if this fails.")
            # For now, let's try to signup/login flow if we can.
            # If standard auth flow is complex, we might need a test user seed.
            return

        headers = {"Authorization": f"Bearer {token}"}
        
        # 2. Create Research Topic
        print("\n📝 Creating Research Topic 'Remote Work Tools'...")
        topic_payload = {
            "title": "Remote Work Tools",
            "description": "Tools for remote workers",
            "search_volume": 1000,
            "keyword_difficulty": 30
        }
        resp = await client.post(f"{BASE_URL}/api/research-topics/", json=topic_payload, headers=headers)
        if resp.status_code != 201:
            print(f"❌ Failed to create topic: {resp.text}")
            return
        topic = resp.json()
        topic_id = topic["id"]
        print(f"✅ Topic Created: {topic_id}")
        
        # 3. Generate Subtopics (Triggers Expansion)
        print("\n🔄 Generating Subtopics (and triggering keyword expansion)...")
        # Note: This might take time due to DataForSEO calls
        resp = await client.post(f"{BASE_URL}/api/research-topics/{topic_id}/subtopics/generate", headers=headers, timeout=120.0)
        
        if resp.status_code != 200:
            print(f"❌ Failed to generate subtopics: {resp.text}")
            return
        
        subtopics_data = resp.json()
        subtopics = subtopics_data.get("items", [])
        print(f"✅ Generated {len(subtopics)} subtopics.")
        
        if not subtopics:
            print("⚠️ No subtopics generated. Aborting.")
            return

        # 4. Check for Keywords (wait a bit if async side-effect is slow, but our logic awaits it)
        print("\n🔍 Checking for expanded keywords...")
        total_keywords = 0
        for sub in subtopics[:3]: # Check first 3
            sub_id = sub["id"]
            resp = await client.get(f"{BASE_URL}/api/research-topics/{topic_id}/subtopics/{sub_id}/keywords", headers=headers)
            if resp.status_code == 200:
                kws = resp.json()
                count = len(kws)
                print(f"   Subtopic '{sub['name']}': {count} keywords")
                total_keywords += count
            else:
                print(f"   ❌ Failed to get keywords for subtopic {sub_id}: {resp.status_code}")
        
        if total_keywords == 0:
            print("⚠️ No keywords found. Expansion might have failed or no profitable keywords found.")
            # We can try manual expansion trigger on one subtopics
            print("   Attempting manual expansion trigger...")
            first_sub_id = subtopics[0]["id"]
            resp = await client.post(f"{BASE_URL}/api/research-topics/{topic_id}/subtopics/{first_sub_id}/keywords/expand", headers=headers)
            print(f"   Manual trigger result: {resp.json()}")
        else:
            print(f"✅ Found total {total_keywords} keywords across checked subtopics.")

        # 5. Cluster Keywords
        print("\n🧩 Clustering Keywords and Generating Content Topics...")
        resp = await client.post(f"{BASE_URL}/api/research-topics/{topic_id}/keywords/cluster", headers=headers)
        if resp.status_code == 200:
            result = resp.json()
            print(f"✅ Clustering Result: {result}")
        else:
            print(f"❌ Clustering failed: {resp.text}")
            
        # 6. Check Content Topics
        print("\n📑 Fetching Content Topics...")
        resp = await client.get(f"{BASE_URL}/api/research-topics/{topic_id}/content-topics", headers=headers)
        if resp.status_code == 200:
            topics = resp.json()
            print(f"✅ Found {len(topics)} Content Topics:")
            for t in topics[:5]:
                print(f"   - {t['title']} (Score: {t.get('estimated_profitability_score')})")
        else:
            print(f"❌ Failed to fetch content topics: {resp.text}")

        # Cleanup (Optional)
        # await client.delete(f"{BASE_URL}/api/research-topics/{topic_id}", headers=headers)

async def get_auth_token(client):
    # Try login first
    try:
        data = {"username": TEST_EMAIL, "password": TEST_PASSWORD}
        # Assuming standard OAuth2 form or JSON login
        # Adapting to likely auth endpoint. Checking user's project structure would confirm.
        # usually /api/auth/login or /auth/token
        # Try JSON login to /api/auth/login (common in this project stack?)
        # Or checking main.py... it imports health_routes, etc.
        # Let's try to assume we can signup or login.
        
        # Attempt login
        resp = await client.post(f"{BASE_URL}/api/auth/login", json={"email": TEST_EMAIL, "password": TEST_PASSWORD})
        if resp.status_code == 200:
            return resp.json().get("access_token")
            
        # Attempt signup if login fails
        print("   Login failed, trying signup...")
        resp = await client.post(f"{BASE_URL}/api/auth/signup", json={"email": TEST_EMAIL, "password": TEST_PASSWORD, "full_name": "Test User"})
        if resp.status_code == 200 or resp.status_code == 201:
            # Login again
            resp = await client.post(f"{BASE_URL}/api/auth/login", json={"email": TEST_EMAIL, "password": TEST_PASSWORD})
            if resp.status_code == 200:
                return resp.json().get("access_token")
    except Exception as e:
        print(f"Auth error: {e}")
    return None

if __name__ == "__main__":
    asyncio.run(main())
