
import os
import json
from supabase_client import get_supabase_client
from dotenv import load_dotenv

load_dotenv()

def analyze_llm_data():
    print("--- Analyzing LLM Data ---")
    sb = get_supabase_client()
    
    # 1. Inspect llm_providers
    print("\nFetching 'llm_providers'...")
    providers = sb.from_('llm_providers').select('*').execute()
    
    if not providers.data:
        print("No providers found.")
    else:
        # Print column names from first record
        columns = list(providers.data[0].keys())
        print(f"Columns found in llm_providers: {columns}")
        
        # Check for specific variations
        target_cols = ['api_key', 'api_key_id', 'api_keys_id']
        for col in target_cols:
             if col in columns:
                 print(f"✅ Found column: {col}")
             else:
                 print(f"❌ Did not find column: {col}")


    # 2. Inspect api_keys
    print("\nFetching 'api_keys'...")
    keys = sb.from_('api_keys').select('*').execute()
    
    if not keys.data:
        print("No API keys found.")
    else:
        columns = list(keys.data[0].keys())
        print(f"Columns: {columns}")
        
        print(f"\nFound {len(keys.data)} keys:")
        for k in keys.data:
            print(f"- Provider: {k.get('provider')} (ID: {k.get('id')})")
            print(f"  Name: {k.get('name')}")
            val = k.get('key_value')
            masked = f"{val[:5]}..." if val else "None"
            print(f"  Value: {masked}")

if __name__ == "__main__":
    analyze_llm_data()
