#!/usr/bin/env python3
"""
API Key Verification Script

This script verifies that API keys can be correctly fetched from Supabase
using the new helper functions, ensuring the refactoring was successful.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def verify_api_keys():
    """Verify API key retrieval from Supabase."""
    print("="*60)
    print("API Key Verification")
    print("="*60)

    try:
        from supabase_client import get_supabase_client, get_llm_api_key, get_linkup_api_key
        
        # 1. Test Supabase Connection
        print("\n1. Testing Supabase Connection...")
        client = get_supabase_client()
        if not client:
            print("✗ Failed to initialize Supabase client")
            return False
        print("✓ Supabase client initialized")
        
        # 2. Test Linkup API Key
        print("\n2. Testing Linkup API Key...")
        linkup_key = get_linkup_api_key()
        if linkup_key:
            print(f"✓ Linkup API key found: {linkup_key[:10]}...")
        else:
            print("✗ Linkup API key NOT found in Supabase")
            
        # 3. Test LLM API Keys (OpenAI)
        print("\n3. Testing OpenAI API Key (gpt-4)...")
        openai_key = get_llm_api_key('openai', 'gpt-4')
        if openai_key:
            print(f"✓ OpenAI key found for gpt-4: {openai_key[:10]}...")
        else:
            print("⚠ OpenAI key for gpt-4 not found (might not be in DB)")
            # Try generic looking
            openai_key = get_llm_api_key('openai', 'gpt-3.5-turbo')
            if openai_key:
                print(f"✓ OpenAI key found for gpt-3.5-turbo: {openai_key[:10]}...")
        
        # 4. Test LLM API Keys (Gemini)
        print("\n4. Testing Gemini API Key (gemini-2.5-flash)...")
        gemini_key = get_llm_api_key('gemini', 'gemini-2.5-flash')
        if gemini_key:
            print(f"✓ Gemini key found for gemini-2.5-flash: {gemini_key[:10]}...")
        else:
             print("⚠ Gemini key for gemini-2.5-flash not found")
             # Try generic
             gemini_key = get_llm_api_key('google', 'gemini-1.5-pro')
             if gemini_key:
                 print(f"✓ Gemini key found for gemini-1.5-pro: {gemini_key[:10]}...")

        # 5. Test Anthropic API Key
        print("\n5. Testing Anthropic API Key (claude-3-opus)...")
        anthropic_key = get_llm_api_key('anthropic', 'claude-3-opus')
        if anthropic_key:
             print(f"✓ Anthropic key found for claude-3-opus: {anthropic_key[:10]}...")
        else:
             print("⚠ Anthropic key for claude-3-opus not found")

        return True

    except Exception as e:
        print(f"\n✗ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    verify_api_keys()
