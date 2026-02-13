import os
from supabase import create_client, Client
from dotenv import load_dotenv
import json

load_dotenv()

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")

if not url or not key:
    print("Error: Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")
    exit(1)

supabase: Client = create_client(url, key)

try:
    print("Fetching one row from llm_providers...")
    response = supabase.table("llm_providers").select("*").limit(1).execute()
    
    if response.data:
        row = response.data[0]
        print("Columns found in llm_providers:")
        print(json.dumps(list(row.keys()), indent=2))
        print("\nSample Data:")
        print(json.dumps(row, indent=2, default=str)) # default=str for datetime
    else:
        print("Table 'wordPress_details' is empty. Cannot determine columns via select(*).")
        
        # Try to insert a dummy row to see if 'categories' is accepted? 
        # No, that's dangerous.
        
except Exception as e:
    print(f"Error: {e}")
