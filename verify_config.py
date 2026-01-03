#!/usr/bin/env python3
"""
Configuration verification script.

This script checks if all required environment variables are set correctly
and tests the Supabase connection.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_env_var(name: str, required: bool = False) -> tuple[bool, str]:
    """Check if an environment variable is set."""
    value = os.environ.get(name)
    if value:
        # Mask sensitive values
        if 'KEY' in name or 'SECRET' in name or 'PASSWORD' in name:
            masked = value[:10] + '...' if len(value) > 10 else '***'
            return True, f"✓ {name} is set ({masked})"
        else:
            return True, f"✓ {name} is set ({value})"
    else:
        if required:
            return False, f"✗ {name} is REQUIRED but not set"
        else:
            return False, f"⚠ {name} is not set (optional)"

def test_supabase_connection():
    """Test Supabase connection and Linkup API key retrieval."""
    try:
        from supabase_client import get_linkup_api_key, get_supabase_client
        
        print("\n" + "="*60)
        print("Testing Supabase Connection")
        print("="*60)
        
        # Check if Supabase client can be created
        client = get_supabase_client()
        if client:
            print("✓ Supabase client created successfully")
            
            # Try to fetch Linkup API key
            linkup_key = get_linkup_api_key()
            if linkup_key:
                masked_key = linkup_key[:10] + '...' if len(linkup_key) > 10 else '***'
                print(f"✓ Linkup API key retrieved from Supabase ({masked_key})")
                return True
            else:
                print("✗ Linkup API key not found in Supabase api_keys table")
                print("  Make sure you have a record with provider='linkup' in the api_keys table")
                return False
        else:
            print("✗ Failed to create Supabase client")
            print("  Check SUPABASE_URL and SUPABASE_KEY in your .env file")
            return False
            
    except ImportError as e:
        print(f"✗ Failed to import supabase_client: {e}")
        print("  Make sure you've installed the supabase package: pip install supabase")
        return False
    except Exception as e:
        print(f"✗ Error testing Supabase connection: {e}")
        return False

def main():
    """Main verification function."""
    print("="*60)
    print("Content Generator V2 - Configuration Verification")
    print("="*60)
    
    # Required variables
    print("\nRequired Configuration:")
    print("-" * 60)
    required_vars = [
        'SUPABASE_URL',
        'SUPABASE_KEY',  # or SUPABASE_ANON_KEY
    ]
    
    all_required_set = True
    for var in required_vars:
        is_set, message = check_env_var(var, required=True)
        print(message)
        if not is_set:
            all_required_set = False
    
    # Check for alternative SUPABASE_ANON_KEY
    if not os.environ.get('SUPABASE_KEY'):
        is_set, message = check_env_var('SUPABASE_ANON_KEY', required=True)
        print(message)
        if is_set:
            all_required_set = True
    
    # Optional but recommended variables
    print("\nOptional Configuration (with fallbacks):")
    print("-" * 60)
    optional_vars = [
        'LINKUP_API_KEY',  # Fallback if not in Supabase
        'RAG_ENDPOINT',
        'RAG_COLLECTION',
        'CELERY_BROKER_URL',
        'CELERY_RESULT_BACKEND',
    ]
    
    for var in optional_vars:
        is_set, message = check_env_var(var, required=False)
        print(message)
    
    # Test Supabase connection
    if all_required_set:
        success = test_supabase_connection()
        
        print("\n" + "="*60)
        if success:
            print("✓ Configuration looks good!")
            print("="*60)
            return 0
        else:
            print("✗ Configuration has issues - see above")
            print("="*60)
            return 1
    else:
        print("\n" + "="*60)
        print("✗ Missing required configuration variables")
        print("="*60)
        return 1

if __name__ == '__main__':
    sys.exit(main())

