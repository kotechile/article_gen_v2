
import os
import asyncio
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not url or not key:
    print("Error: Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
    exit(1)

supabase: Client = create_client(url, key)

try:
    # Try to select one row to see columns
    response = supabase.table("llm_providers").select("*").limit(1).execute()
    if response.data:
        print("Table 'llm_providers' exists. Columns and first row:")
        print(response.data[0].keys())
        print(response.data[0])
    else:
        print("Table 'llm_providers' exists but is empty. Trying to list columns another way is tricky via client, but assuming it exists.")
        # Try to insert a dummy to see error or just assume it's empty
        
except Exception as e:
    print(f"Error accessing 'llm_providers': {e}")
