"""
Supabase client utility for fetching API keys and configuration.

This module provides functions to interact with Supabase database
to retrieve API keys and other configuration stored in the database.
"""

import os
import logging
from typing import Optional
from supabase import create_client, Client

logger = logging.getLogger(__name__)

# Cache for Supabase client
_supabase_client: Optional[Client] = None


def get_supabase_client() -> Optional[Client]:
    """
    Get or create Supabase client instance.
    
    Returns:
        Supabase client instance or None if credentials are not configured
    """
    global _supabase_client
    
    if _supabase_client is not None:
        return _supabase_client
    
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_KEY') or os.environ.get('SUPABASE_ANON_KEY')
    
    if not supabase_url or not supabase_key:
        logger.warning("Supabase credentials not found (SUPABASE_URL and SUPABASE_KEY required)")
        return None
    
    try:
        _supabase_client = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized successfully")
        return _supabase_client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {str(e)}")
        return None


def get_api_key_from_supabase(provider: str) -> Optional[str]:
    """
    Fetch API key from Supabase api_keys table.
    
    Args:
        provider: Provider name (e.g., 'linkup', 'openai', etc.)
        
    Returns:
        API key value or None if not found or error occurred
    """
    try:
        client = get_supabase_client()
        if not client:
            logger.warning(f"Cannot fetch {provider} API key: Supabase client not available")
            return None
        
        # Query Supabase for the API key
        response = client.table('api_keys').select('key_value').eq('provider', provider).execute()
        
        if response.data and len(response.data) > 0:
            api_key = response.data[0].get('key_value')
            if api_key:
                logger.info(f"Successfully fetched {provider} API key from Supabase")
                return api_key
            else:
                logger.warning(f"{provider} API key found in Supabase but key_value is empty")
        else:
            logger.warning(f"{provider} API key not found in Supabase api_keys table")
        
        return None
        
    except Exception as e:
        logger.error(f"Error fetching {provider} API key from Supabase: {str(e)}")
        return None


def get_linkup_api_key() -> Optional[str]:
    """
    Get Linkup API key from Supabase api_keys table.
    
    All API keys are stored in Supabase, not in environment variables.
    Only Supabase credentials (SUPABASE_URL, SUPABASE_KEY) should be in .env.
    
    Returns:
        Linkup API key or None if not found
    """
    api_key = get_api_key_from_supabase('linkup')
    if not api_key:
        logger.warning("Linkup API key not found in Supabase api_keys table (provider='linkup')")
    return api_key


def get_api_key(provider: str) -> Optional[str]:
    """
    Generic function to get any API key from Supabase.
    
    All API keys are stored in Supabase api_keys table, not in environment variables.
    Only Supabase credentials (SUPABASE_URL, SUPABASE_KEY) should be in .env.
    
    Args:
        provider: Provider name (e.g., 'linkup', 'openai', 'gemini', 'anthropic', etc.)
        
    Returns:
        API key value or None if not found
    """
    return get_api_key_from_supabase(provider)

