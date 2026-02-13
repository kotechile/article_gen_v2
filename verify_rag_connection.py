#!/usr/bin/env python3
"""
Script to verify connectivity to the Cloud RAG system using standard library.
"""

import os
import sys
import json
import urllib.request
import urllib.error

def load_env_manual():
    """Load .env file manually without python-dotenv dependency"""
    env_path = os.path.join(os.getcwd(), '.env')
    if not os.path.exists(env_path):
        print("⚠️ .env file not found")
        return
        
    print(f"Loading env from {env_path}")
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Simple parsing of KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                
                os.environ[key] = value

def verify_connection():
    load_env_manual()
    
    rag_url = os.environ.get('RAG_API_URL')
    if not rag_url:
        print("❌ Error: RAG_API_URL is not set in environment variables.")
        print("Please set it in your .env file.")
        return False
        
    print(f"Testing connection to RAG Base URL: {rag_url}")
    
    # Construct Health Check URL
    base_url = rag_url.rstrip('/')
    health_url = f"{base_url}/health"
    
    print(f"Target Health Endpoint: {health_url}")
    
    try:
        # Create request with timeout
        req = urllib.request.Request(health_url)
        with urllib.request.urlopen(req, timeout=10) as response:
            status = response.getcode()
            body = response.read().decode('utf-8')
            
            if status == 200:
                print("✅ Connection Successful!")
                print(f"Status Code: {status}")
                print(f"Response: {body}")
                return True
            else:
                print(f"❌ Connection Failed with Status Code: {status}")
                return False
                
    except urllib.error.URLError as e:
        print(f"❌ Connection Error: {e.reason}")
        print("Check if the VPS IP/Domain is correct and the service is running.")
        return False
    except TimeoutError:
        print("❌ Timeout Error: Connection timed out.")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False

if __name__ == "__main__":
    success = verify_connection()
    sys.exit(0 if success else 1)
