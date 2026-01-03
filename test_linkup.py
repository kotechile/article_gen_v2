#!/usr/bin/env python3
"""
Test script to verify Linkup integration.

This script checks:
1. Supabase connection
2. Linkup API key retrieval from Supabase
3. Linkup client initialization
4. Linkup API connection (optional test search)
"""

import os
import sys

# Try to load dotenv if available, but don't fail if it's not
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, assume environment variables are already set
    pass

def test_supabase_connection():
    """Test Supabase connection."""
    print("="*60)
    print("1. Testing Supabase Connection")
    print("="*60)
    
    # Check environment variables
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_KEY') or os.environ.get('SUPABASE_ANON_KEY')
    
    if not supabase_url:
        print("✗ SUPABASE_URL not found in environment")
        return False
    else:
        print(f"✓ SUPABASE_URL is set: {supabase_url[:30]}...")
    
    if not supabase_key:
        print("✗ SUPABASE_KEY or SUPABASE_ANON_KEY not found in environment")
        return False
    else:
        print(f"✓ SUPABASE_KEY is set: {supabase_key[:20]}...")
    
    # Try to create Supabase client
    try:
        from supabase_client import get_supabase_client
        client = get_supabase_client()
        
        if client:
            print("✓ Supabase client created successfully")
            return True
        else:
            print("✗ Failed to create Supabase client")
            return False
    except Exception as e:
        print(f"✗ Error creating Supabase client: {e}")
        return False

def test_linkup_key_retrieval():
    """Test retrieving Linkup API key from Supabase."""
    print("\n" + "="*60)
    print("2. Testing Linkup API Key Retrieval from Supabase")
    print("="*60)
    
    try:
        from supabase_client import get_linkup_api_key
        
        linkup_key = get_linkup_api_key()
        
        if linkup_key:
            masked_key = linkup_key[:15] + '...' if len(linkup_key) > 15 else '***'
            print(f"✓ Linkup API key retrieved successfully: {masked_key}")
            print(f"  Key length: {len(linkup_key)} characters")
            return True, linkup_key
        else:
            print("✗ Linkup API key not found in Supabase")
            print("  Make sure you have a record in api_keys table:")
            print("    INSERT INTO api_keys (provider, key_value) VALUES ('linkup', 'your-key');")
            return False, None
            
    except Exception as e:
        print(f"✗ Error retrieving Linkup API key: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_linkup_client_initialization(api_key):
    """Test Linkup client initialization."""
    print("\n" + "="*60)
    print("3. Testing Linkup Client Initialization")
    print("="*60)
    
    if not api_key:
        print("✗ Cannot test - no API key available")
        return False
    
    try:
        from linkup_client import create_linkup_client
        
        client = create_linkup_client(api_key=api_key, cache_enabled=False)
        
        if client:
            print("✓ Linkup client created successfully")
            return True, client
        else:
            print("✗ Failed to create Linkup client")
            return False, None
            
    except Exception as e:
        print(f"✗ Error creating Linkup client: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_linkup_search(client):
    """Test Linkup search functionality."""
    print("\n" + "="*60)
    print("4. Testing Linkup Search (Optional)")
    print("="*60)
    
    if not client:
        print("⚠ Skipping search test - no client available")
        return False
    
    try:
        from linkup_client import SearchQuery
        
        # Simple test query
        test_query = SearchQuery(
            query="artificial intelligence",
            max_results=3,
            depth="standard"
        )
        
        print(f"  Testing search with query: '{test_query.query}'")
        print("  This may take a few seconds...")
        
        response = client.search(test_query)
        
        if response.success:
            print(f"✓ Search successful!")
            print(f"  Results returned: {len(response.results)}")
            if response.results:
                print(f"  First result: {response.results[0].url[:60]}...")
            return True
        else:
            print(f"✗ Search failed: {response.error}")
            return False
            
    except Exception as e:
        print(f"✗ Error during search test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Linkup Integration Test")
    print("="*60)
    print()
    
    results = {}
    
    # Test 1: Supabase connection
    results['supabase'] = test_supabase_connection()
    
    # Test 2: Linkup key retrieval
    key_retrieved, linkup_key = test_linkup_key_retrieval()
    results['key_retrieval'] = key_retrieved
    
    # Test 3: Linkup client initialization
    if key_retrieved:
        client_ok, client = test_linkup_client_initialization(linkup_key)
        results['client_init'] = client_ok
        
        # Test 4: Optional search test
        if client_ok:
            print("\nDo you want to test Linkup search? (This will make an API call)")
            print("Press Enter to skip, or type 'yes' to test:")
            try:
                user_input = input().strip().lower()
                if user_input == 'yes':
                    results['search'] = test_linkup_search(client)
                else:
                    print("  Skipping search test")
                    results['search'] = None
            except (EOFError, KeyboardInterrupt):
                print("  Skipping search test")
                results['search'] = None
    else:
        results['client_init'] = False
        results['search'] = None
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    print(f"Supabase Connection:     {'✓ PASS' if results['supabase'] else '✗ FAIL'}")
    print(f"Linkup Key Retrieval:    {'✓ PASS' if results['key_retrieval'] else '✗ FAIL'}")
    print(f"Linkup Client Init:      {'✓ PASS' if results['client_init'] else '✗ FAIL'}")
    if results['search'] is not None:
        print(f"Linkup Search Test:      {'✓ PASS' if results['search'] else '✗ FAIL'}")
    else:
        print(f"Linkup Search Test:      ⚠ SKIPPED")
    
    print("="*60)
    
    # Overall status
    critical_tests = [results['supabase'], results['key_retrieval'], results['client_init']]
    if all(critical_tests):
        print("\n✓ Linkup integration is working correctly!")
        if results['search']:
            print("✓ Search functionality is also working!")
        return 0
    else:
        print("\n✗ Linkup integration has issues - check the errors above")
        return 1

if __name__ == '__main__':
    sys.exit(main())

