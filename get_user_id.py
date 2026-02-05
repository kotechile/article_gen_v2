
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_SERVICE_KEY")

if not url or not key:
    print("Error: Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")
    exit(1)

supabase: Client = create_client(url, key)

# Get the first user who has wordpress details
try:
    response = supabase.table("wordPress_details").select("user_id").limit(1).execute()
    if response.data:
        print(f"USER_ID_FOUND: {response.data[0]['user_id']}")
    else:
        print("No users with wordpress details found")
except Exception as e:
    print(f"Error: {e}")
