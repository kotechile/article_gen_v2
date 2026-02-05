import os
from supabase import create_client, Client

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_KEY")

if not url or not key:
    print("Missing env vars")
    exit(1)

supabase: Client = create_client(url, key)

print("--- wordPress_details columns ---")
try:
    # Fetch one row to see keys, as we can't easily desc table via client
    res = supabase.table("wordPress_details").select("*").limit(1).execute()
    if res.data:
        print(res.data[0].keys())
    else:
        print("No data in wordPress_details")
except Exception as e:
    print(e)
    
print("\n--- wordpress_imported_posts columns ---")
try:
    res = supabase.table("wordpress_imported_posts").select("*").limit(1).execute()
    if res.data:
        print(res.data[0].keys())
    else:
         print("No data in wordpress_imported_posts")

except Exception as e:
    print(e)
