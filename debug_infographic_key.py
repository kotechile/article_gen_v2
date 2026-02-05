
import os
from dotenv import load_dotenv
from supabase_client import get_supabase_client

load_dotenv()

def debug_llm_config():
    sb = get_supabase_client()
    if not sb:
        print("Failed to get Supabase client")
        return

    print("--- Debugging LLM Configuration ---")
    
    # 1. Fetch default providers
    print("\n--- Testing .single().execute() ---")
    try:
        single_provider = sb.from_('llm_providers').select('*').eq('is_default', True).single().execute()
        print(f"Data type with .single(): {type(single_provider.data)}")
        print(f"Data with .single(): {single_provider.data}")
    except Exception as e:
        print(f"Error with .single(): {e}")

    providers = sb.from_('llm_providers').select('*').eq('is_default', True).execute()
    print(f"Number of default providers found: {len(providers.data)}")
    
    for p in providers.data:
        print(f"\nProvider Record: {p}")
        print(f"Provider: {p.get('name')} (ID: {p.get('id')})")
        print(f"Provider Name field: {p.get('provider_name')}")
        print(f"Provider field: {p.get('provider')}")
        print(f"Model Name field: {p.get('model_name')}")
        print(f"API Keys ID field: {p.get('api_keys_id')}")
        
        key_id = p.get('api_keys_id')
        if key_id:
            key_data = sb.from_('api_keys').select('*').eq('id', key_id).execute()
            if key_data.data:
                k = key_data.data[0]
                print(f"✅ Key found for ID {key_id}")
                print(f"   Key Provider: {k.get('provider')}")
                val = k.get('key_value')
                masked = f"{val[:5]}..." if val else "None"
                print(f"   Key Value: {masked}")
            else:
                print(f"❌ No key found in api_keys for ID {key_id}")
        else:
            print("❌ No api_keys_id linked to this provider")

if __name__ == "__main__":
    debug_llm_config()
