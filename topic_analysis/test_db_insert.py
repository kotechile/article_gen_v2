
import asyncio
import os
import sys
import uuid
from datetime import datetime

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from dotenv import load_dotenv

# Load env vars from backend/.env
env_path = os.path.join(os.path.dirname(__file__), 'backend', '.env')
load_dotenv(env_path)

from src.core.supabase_singleton import get_supabase_client

async def test_insert():
    print("--- Testing Insert into content_ideas ---")
    supabase = get_supabase_client()
    
    # Generate dummy data
    user_id = str(uuid.uuid4()) # Use a random UUID for test, or user might need real one if RLS checks auth (but we are service role usually?)
    # Wait, usually backend uses service_role key. If using anon key + RLS, we need a real user.
    # Assuming backend utilizes service_role or we have permissive RLS.
    
    # payload data matches the schema we just migrated to
    data = {
        "title": "Test Idea Persistence",
        "description": "This is a test row to verify database writes.",
        "content_type": "blog",
        "topic_id": "test-topic-123", # Text
        "research_id": "test-topic-123", # Legacy Text
        "subtopic": "Test Subtopic",
        "user_id": user_id, 
        "keywords": ["test", "persistence"],
        "status": "draft",
        "created_at": datetime.utcnow().isoformat()
    }
    
    print(f"Attempting to insert: {data}")
    
    try:
        response = supabase.table('content_ideas').insert(data).execute()
        print("\nSUCCESS! Insert Result:")
        print(response.data)
    except Exception as e:
        print("\nFAILURE! Insert Error:")
        print(e)
        if hasattr(e, 'message'):
            print(f"Message: {e.message}")
        if hasattr(e, 'details'):
             print(f"Details: {e.details}")
        if hasattr(e, 'hint'):
             print(f"Hint: {e.hint}")

if __name__ == "__main__":
    asyncio.run(test_insert())
