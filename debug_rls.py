
import os
import asyncio
from supabase import create_client
from dotenv import load_dotenv

# Load env directly to be sure
load_dotenv(dotenv_path="/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/.env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

print(f"URL: {SUPABASE_URL}")
print(f"Service Key (first 10): {SUPABASE_SERVICE_KEY[:10] if SUPABASE_SERVICE_KEY else 'None'}")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("Missing credentials!")
    exit(1)

# Initialize client with verify=False to match app behavior
import httpx
original_init = httpx.Client.__init__
def new_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_init(self, *args, **kwargs)
httpx.Client.__init__ = new_init

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def check_rls_status():
    print("\n--- Checking RLS Status ---")
    try:
        # We can't query pg_class directly via PostgREST easily unless we have a rpc or special access
        # But we can try to select from the table.
        # However, checking policies via SQL injection methods on `execute()` is not standard.
        # We'll try to use a direct SQL query if possible (not possible via standard client unless SQL editor or RPC).
        # We'll check if we can insert.
        print("Attempting to insert a test record with Service Role...")
        data = {
            "title": "RLS Test Topic",
            "description": "Output of debug_rls.py",
            "status": "active",
            "user_id": "f248b7ed-b8df-4464-8544-8304d7ae4c30" # The user id obtained from logs
        }
        response = supabase.table('research_topics').insert(data).execute()
        print("✅ Insert SUCCESS!")
        print(response.data)
        
        # Cleanup
        if response.data:
            rec_id = response.data[0]['id']
            supabase.table('research_topics').delete().eq('id', rec_id).execute()
            print("✅ Cleanup SUCCESS")

    except Exception as e:
        print(f"❌ Insert FAILED: {e}")

def list_policies():
    # If we can't run SQL, this might be hard. But normally service role works.
    # If insert failed, it confirms Service Role is NOT bypassing RLS or key is wrong.
    pass

if __name__ == "__main__":
    check_rls_status()
