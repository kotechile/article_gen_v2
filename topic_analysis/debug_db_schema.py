
import asyncio
import os
import sys

# Add backend directory to sys.path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from dotenv import load_dotenv

# Load env vars from backend/.env
env_path = os.path.join(os.path.dirname(__file__), 'backend', '.env')
load_dotenv(env_path)

from src.core.supabase_singleton import get_supabase_client

async def inspect_schema():
    print("Connecting to Supabase...")
    supabase = get_supabase_client()
    
    print("\n--- Inspecting 'content_ideas' table ---")
    try:
        # Fetch 1 row to see columns
        response = supabase.table('content_ideas').select('*').limit(1).execute()
        if response.data:
            print("Row found! Columns:")
            for key in response.data[0].keys():
                print(f" - {key}")
        else:
            print("Table empty. Trying to insert a dummy row to check for 'title' column error explicitly...")
            # We can't really "check error" easily with supabase-py client except by catching exception
            try:
                dummy = {"research_id": "dummy", "title": "test"}
                supabase.table('content_ideas').insert(dummy).execute()
            except Exception as e:
                print(f"Insert failed as expected (or not): {str(e)}")
                if "column" in str(e).lower():
                    print("CONFIRMED: Column issue detected.")

    except Exception as e:
        print(f"Error fetching: {str(e)}")

if __name__ == "__main__":
    asyncio.run(inspect_schema())
