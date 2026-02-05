
import sys
import os
import asyncio
import json
import urllib.request
import urllib.parse
from unittest.mock import MagicMock

# Ensure we can import backend src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.config import settings

class SimpleSupabaseClient:
    def __init__(self, url, key):
        self.url = url
        self.key = key
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }

    def table(self, name):
        return SimpleQueryBuilder(self.url, self.headers, name)

class SimpleQueryBuilder:
    def __init__(self, base_url, headers, table):
        self.base_url = f"{base_url}/rest/v1/{table}"
        self.headers = headers
        self.params = {}

    def select(self, cols):
        self.params['select'] = cols
        return self

    def eq(self, col, val):
        self.params[col] = f"eq.{val}"
        return self

    def execute(self):
        query_string = urllib.parse.urlencode(self.params)
        full_url = f"{self.base_url}?{query_string}"
        req = urllib.request.Request(full_url, headers=self.headers)
        try:
            with urllib.request.urlopen(req) as response:
                if 200 <= response.status < 300:
                    return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            print(f"Error: {e}")
        return []

def check_keys():
    url = settings.supabase_url
    key = settings.supabase_service_role_key
    
    if not url or not key:
        print("Error: Supabase credentials missing from settings.")
        return

    client = SimpleSupabaseClient(url, key)
    
    print("Querying api_keys table...")
    # Try generic select first
    results = client.table("api_keys").select("*").execute()
    
    found = False
    for row in results:
        print(f"Found Key: Provider={row.get('provider')}, Name={row.get('name')}")
        if str(row.get('provider')).lower() == 'dataforseo' or 'dataforseo' in str(row.get('name')).lower():
            found = True
            print(">>> FOUND DATAFORSEO KEY! <<<")
            print(f"Login (from key_value or other col?): {row.get('key_value')}")
            # Check if there is a separate password or login column. 
            # Usually DataForSEO needs login:password. 
            # Maybe key_value holds "login:password" string?
            
    if not found:
        print("No DataForSEO key found in api_keys table.")

if __name__ == "__main__":
    check_keys()
