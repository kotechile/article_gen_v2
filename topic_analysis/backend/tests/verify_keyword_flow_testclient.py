
import os
import sys
from uuid import UUID, uuid4
from fastapi.testclient import TestClient

# Add project root to path so we can import 'backend' or 'src'
current_dir = os.getcwd()
sys.path.append(current_dir)

# Import app and router
try:
    from src.main import app
    from src.api import research_topics_routes
except ImportError:
    # If running from root/backend
    sys.path.append(os.path.join(current_dir, "backend"))
    from backend.src.main import app
    from backend.src.api import research_topics_routes

# Initialize TestClient
client = TestClient(app, base_url="http://localhost")

# Mock User ID - Fetch a real one if possible
# Mock User ID - Fetch a real one if possible
try:
    from src.services.supabase_service import SupabaseService
except ImportError:
    from backend.src.services.supabase_service import SupabaseService

def get_real_user_id():
    try:
        svc = SupabaseService()
        # Query existing research topics to find a user
        # Note: This is hacky. Better to use admin auth if available.
        # But valid user_id is needed.
        # Let's try to list users from 'auth.users' if we have service role? 
        # Supabase-py client usually has auth.admin
        
        # Try retrieving mapped users from public schema if any?
        # Or just research_topics.
        # We need async call? SupabaseService is async for some methods? 
        # Wait, SupabaseService wrappes async/sync? It uses `self.client.table(...)`.
        # `execute_query` is async.
        # We need a synchronous way or run async loop.
        # Since this is a script, we can run async loop to get ID.
        pass
    except:
        pass
    return UUID("00000000-0000-0000-0000-000000000001")

# Hardcoded valid user from logs or previous knowledge?
# The logs showed a connection to `https://sbcontent.giniloh.com`
# I don't know any user.

# Alternative: Use "verify_keyword_flow.py" pattern but fix the HOST header issue?
# If I fix HOST header in `verify_keyword_flow.py`, maybe signup works?
# The "Authentication failed" in `verify_keyword_flow.py` step 561 didn't show 400. It showed generic failure.
# But Step 584 (TestClient) showed 400.

# Let's try to fix HOST in `verify_keyword_flow.py`? 
# No, `httpx` (external) sends correct host header usually.
# The `verify_keyword_flow.py` failed because it couldn't get a token.

# Back to TestClient. I need a user ID.
# I'll just use a UUID that I HOPE exists. 
# Or I can try to insert a fake user into `auth.users`? No, no access.
# I will try to fetch the first user from `research_topics` via raw SQL or query.
import asyncio

async def fetch_user_id():
    svc = SupabaseService()
    # Try to get one topic
    try:
        resp = await svc.get_by_filters("research_topics", {}, limit=1)
        if resp:
            return UUID(resp[0]["user_id"])
    except:
        print("Failed to fetch existing user.")
    return None

# We can run this async function
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
REAL_USER_ID = loop.run_until_complete(fetch_user_id())
if not REAL_USER_ID:
    print("⚠️ No existing user found. Test likely to fail FK.")
    REAL_USER_ID = UUID("00000000-0000-0000-0000-000000000001")
else:
    print(f"✅ Found existing user ID: {REAL_USER_ID}")

def mock_get_user_id():
    return REAL_USER_ID

# Override dependency
app.dependency_overrides[research_topics_routes.get_user_id] = mock_get_user_id

def verify_flow():
    print("🚀 Starting Keyword Flow Verification (TestClient)...")
    
    # 1. Create Topic
    print("\n📝 Creating Research Topic 'Remote Work Tools'...")
    topic_payload = {
        "title": "Remote Work Tools",
        "description": "Tools for remote workers",
        "search_volume": 1000,
        "keyword_difficulty": 30
    }
    
    resp = client.post("/api/research-topics/", json=topic_payload)
    if resp.status_code == 500:
        print(f"❌ Server Error: {resp.text}")
        if "foreign key constraint" in resp.text.lower():
             print("⚠️ FK Constraint on auth.users failed. We need a real user ID.")
        return
        
    if resp.status_code != 201:
        print(f"❌ Failed to create topic: {resp.status_code} {resp.text}")
        return

    topic = resp.json()
    topic_id = topic["id"]
    print(f"✅ Topic Created: {topic_id}")
    
    # 2. Generate Subtopics (Triggers Expansion)
    print("\n🔄 Generating Subtopics...")
    resp = client.post(f"/api/research-topics/{topic_id}/subtopics/generate")
    if resp.status_code != 200:
        print(f"❌ Failed to generate subtopics: {resp.status_code} {resp.text}")
        return
        
    subtopics_data = resp.json()
    subtopics = subtopics_data.get("items", [])
    print(f"✅ Generated {len(subtopics)} subtopics.")
    
    if not subtopics:
        return

    # 3. Check Keywords
    print("\n🔍 Checking for expanded keywords...")
    total_keywords = 0
    for sub in subtopics[:3]:
        sub_id = sub["id"]
        resp = client.get(f"/api/research-topics/{topic_id}/subtopics/{sub_id}/keywords")
        if resp.status_code == 200:
            kws = resp.json()
            print(f"   Subtopic '{sub['name']}': {len(kws)} keywords")
            total_keywords += len(kws)
        else:
            print(f"   ❌ Failed: {resp.status_code}")
            
    # 4. Cluster
    print("\n🧩 Clustering Keywords...")
    resp = client.post(f"/api/research-topics/{topic_id}/keywords/cluster")
    if resp.status_code == 200:
        print(f"✅ Result: {resp.json()}")
    else:
        print(f"❌ Failed: {resp.status_code} {resp.text}")
        
    # 5. Content Topics
    print("\n📑 Fetching Content Topics...")
    resp = client.get(f"/api/research-topics/{topic_id}/content-topics")
    if resp.status_code == 200:
        topics = resp.json()
        print(f"✅ Found {len(topics)} Content Topics.")
    else:
        print(f"❌ Failed: {resp.status_code}")

if __name__ == "__main__":
    verify_flow()
