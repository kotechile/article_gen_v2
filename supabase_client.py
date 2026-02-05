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
    
    # Prioritize Service Key (Bypass RLS) -> Standard Key -> Anon Key
    service_key = os.environ.get('SUPABASE_SERVICE_KEY')
    anon_key = os.environ.get('SUPABASE_ANON_KEY')
    common_key = os.environ.get('SUPABASE_KEY')
    
    # Use the most privileged key available
    supabase_key = service_key or common_key or anon_key
    
    auth_type = "Service Role (Bypass RLS)" if service_key else "Standard/Anon (RLS Restricted)"
    
    if not supabase_url or not supabase_key:
        logger.warning("Supabase credentials not found (SUPABASE_URL and SUPABASE_KEY/SUPABASE_SERVICE_KEY required)")
        return None
    
    try:
        # Verify=False patch for self-hosted Supabase with self-signed certs
        import httpx
        original_init = httpx.Client.__init__
        
        def new_init(self, *args, **kwargs):
            kwargs['verify'] = False
            original_init(self, *args, **kwargs)
            
        httpx.Client.__init__ = new_init
        
        _supabase_client = create_client(supabase_url, supabase_key)
        logger.info(f"Supabase client initialized successfully using {auth_type}")
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
        
        # Fallback: Try case-insensitive search
        logger.info(f"Exact match failed for {provider}, trying case-insensitive search")
        response = client.table('api_keys').select('key_value').ilike('provider', provider).execute()
        
        if response.data and len(response.data) > 0:
            api_key = response.data[0].get('key_value')
            if api_key:
                logger.info(f"Successfully fetched {provider} API key via case-insensitive search")
                return api_key
        
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


def get_llm_api_key(provider: str, model: str) -> Optional[str]:
    """
    Fetch API key for a specific LLM provider and model.
    
    Logic:
    1. Query 'llm_providers' with provider_type and model_name to get api_key (UUID).
    2. Query 'api_keys' with the UUID to get key_value.
    
    Args:
        provider: Provider type (e.g., 'google', 'openai')
        model: Model name (e.g., 'gemini-1.5-pro', 'gpt-4')
        
    Returns:
        API key value or None if not found
    """
    try:
        client = get_supabase_client()
        if not client:
            logger.warning(f"Cannot fetch API key for {provider}/{model}: Supabase client not available")
            return None
        
        # 1. Find the provider record
        # Note: provider_type in DB might match 'provider' arg, valid check required
        # Some mappings might be needed if frontend sends 'openai' but DB has 'OpenAI'
        
        # Try exact match first
        provider_query = client.table('llm_providers').select('api_keys_id').eq('model_name', model)
        # Optional: also filter by provider if duplication exists across providers (rare for models)
        if provider:
            provider_query = provider_query.eq('provider', provider)
            
        provider_resp = provider_query.execute()
        
        if not provider_resp.data or len(provider_resp.data) == 0:
            logger.warning(f"LLM Provider record not found for {provider}/{model}")
            # Fallback: Try identifying by model name only if provider didn't match
            if provider:
                fallback_resp = client.table('llm_providers').select('api_keys_id').eq('model_name', model).execute()
                if fallback_resp.data and len(fallback_resp.data) > 0:
                    provider_resp = fallback_resp
                    logger.info(f"Found LLM record by model name '{model}' ignoring provider '{provider}'")
                else:
                    return None
            else:
                return None
                
        api_key_id = provider_resp.data[0].get('api_keys_id')
        if not api_key_id:
            logger.warning(f"LLM Provider record found for {model} but has no linked api_key")
            return None
            
        # 2. Get the actual key
        key_resp = client.table('api_keys').select('key_value').eq('id', api_key_id).execute()
        
        if key_resp.data and len(key_resp.data) > 0:
            key_value = key_resp.data[0].get('key_value')
            if key_value:
                return key_value
                
        logger.warning(f"API Key linked to {model} (ID: {api_key_id}) not found or empty")
        return None
        
    except Exception as e:
        logger.error(f"Error fetching LLM API key for {provider}/{model}: {str(e)}")
        return None
