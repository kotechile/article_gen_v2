
import os
import sys
import logging
from dotenv import load_dotenv

env_path = os.path.join(os.getcwd(), '.env')
print(f"Checking for .env at: {env_path}")
print(f"Exists: {os.path.exists(env_path)}")

loaded = load_dotenv(env_path)
print(f"Dotenv loaded: {loaded}")

if os.environ.get('SUPABASE_URL'):
    print("SUPABASE_URL is set.")
else:
    print("SUPABASE_URL is NOT set.")

if os.environ.get('SUPABASE_SERVICE_KEY'):
    print("SUPABASE_SERVICE_KEY is set (GOOD).")
else:
    print("SUPABASE_SERVICE_KEY is NOT set (BAD).")

try:
    from supabase_client import get_llm_api_key, get_supabase_client
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

# Configure logging to see output
logging.basicConfig(level=logging.INFO)

def test_fetch_key():
    print("Testing get_llm_api_key...")
    

    # Check if we have Supabase credentials
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY') or os.environ.get('SUPABASE_ANON_KEY')
    
    if not url or not key:
        print("Error: SUPABASE_URL or (SUPABASE_KEY/SUPABASE_ANON_KEY) not set in environment")
        return

    # Try to fetch key for the model seen in logs
    provider = 'google' # or 'gemini'
    model = 'gemini-3-flash-preview'
    
    print(f"Fetching key for {provider}/{model}...")
    key = get_llm_api_key(provider, model)
    
    if key:
        print(f"Success! Found key: {key[:5]}...{key[-5:]}")
    else:
        print("Failed: No key found in DB.")
        
        # Try listing providers to see what is available
        client = get_supabase_client()
        if client:
             print("Listing available LLM providers in DB:")
             resp = client.table('llm_providers').select('*').execute()
             for item in resp.data:
                 print(f"- {item.get('provider_type')}/{item.get('model_name')}")

if __name__ == "__main__":
    test_fetch_key()
