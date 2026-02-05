
import os
import sys
from pprint import pprint

# Add root directory to path
sys.path.append(os.getcwd())

from dotenv import load_dotenv
load_dotenv()

from supabase_client import get_supabase_client

def inspect_titles():
    supabase = get_supabase_client()
    if not supabase:
        print("Failed to initialize Supabase client")
        return

    try:
        response = supabase.table('research_topics').select("*").limit(1).execute()
        # print("Response:", response)
        if response.data:
            print("Record found:")
            pprint(response.data[0])
        else:
            print("No records found in research_topics table.")
    except Exception as e:
        print(f"Error querying research_topics: {e}")

if __name__ == "__main__":
    inspect_titles()
