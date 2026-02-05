
import os
from supabase_client import get_supabase_client
from dotenv import load_dotenv

load_dotenv()

def check_llm_setup():
    print("Checking LLM Setup...")
    sb = get_supabase_client()
    
    # Check Providers
    print("\n1. Checking llm_providers for default...")
    response = sb.from_('llm_providers').select('*').eq('is_default', True).execute()
    
    if not response.data:
        print("❌ No default provider found in 'llm_providers' (is_default=True).")
        # List all providers
        all_providers = sb.from_('llm_providers').select('*').execute()
        print(f"   Found {len(all_providers.data)} total providers:")
        for p in all_providers.data:
            print(f"   - {p.get('provider_name')} (default: {p.get('is_default')})")
        return

    provider = response.data[0]
    print(f"✅ Found default provider: {provider.get('provider_name')} (ID: {provider.get('id')})")
    print(f"   Model Name Field: {provider.get('model_named') or provider.get('model_name')}")
    
    # Check API Key
    key_id = provider.get('api_keys_id')
    print(f"\n2. Checking API Key (Linked ID: {key_id})...")
    
    if not key_id:
        print("❌ Default provider has no 'api_keys_id' linked.")
        return
        
    key_response = sb.from_('api_keys').select('*').eq('id', key_id).execute()
    if not key_response.data:
         print(f"❌ API Key record with ID {key_id} not found in 'api_keys' table.")
         return
         
    key_rec = key_response.data[0]
    key_val = key_rec.get('key_value')
    if not key_val:
        print("❌ API Key value is empty/null.")
    else:
        print(f"✅ API Key found (starts with: {key_val[:5]}...)")

if __name__ == "__main__":
    check_llm_setup()
