"""
LLM Provider Configuration Manager
Loads LLM provider configurations from Supabase database
"""

import os
from typing import Dict, Optional, Any, List
import structlog
from datetime import datetime

logger = structlog.get_logger()

class LLMProviderConfig:
    """Manages LLM provider configurations from Supabase database"""
    
    def __init__(self):
        self.supabase_client = None
        self._providers_cache: Dict[str, Any] = {}
        self._cache_loaded = False
        self._initialize_supabase_client()
    
    def _initialize_supabase_client(self):
        """Initialize Supabase client"""
        try:
            from .supabase_singleton import get_supabase_client
            self.supabase_client = get_supabase_client()
            logger.info("LLM Provider Config initialized with Supabase")
        except Exception as e:
            logger.warning("Failed to initialize Supabase client", error=str(e))
            self.supabase_client = None
    
    def _load_providers_from_supabase(self):
        """Load LLM providers from Supabase database"""
        if self._cache_loaded or not self.supabase_client:
            return
        
        try:
            # Get active providers ordered by priority (highest first)
            response = self.supabase_client.table('llm_providers').select('*').eq('is_active', True).order('priority', desc=True).execute()
            
            if response.data:
                self._providers_cache = {provider['provider_type']: provider for provider in response.data}
                logger.info("Loaded LLM providers from Supabase", count=len(self._providers_cache))
            else:
                logger.warning("No active LLM providers found in Supabase")
                
        except Exception as e:
            logger.error("Failed to load LLM providers from Supabase", error=str(e))
        
        self._cache_loaded = True
    
    def get_default_provider(self) -> Optional[Dict[str, Any]]:
        """Get the default LLM provider configuration"""
        self._load_providers_from_supabase()
        
        # Find provider with is_default=True
        for provider in self._providers_cache.values():
            if provider.get('is_default', False):
                return provider
        
        # If no default found, return the highest priority provider
        if self._providers_cache:
            return max(self._providers_cache.values(), key=lambda p: p.get('priority', 0))
        
        return None
    
    def get_provider_by_type(self, provider_type: str) -> Optional[Dict[str, Any]]:
        """Get provider configuration by type"""
        self._load_providers_from_supabase()
        return self._providers_cache.get(provider_type)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider types"""
        self._load_providers_from_supabase()
        return list(self._providers_cache.keys())
    
    def get_provider_api_key(self, provider_type: str) -> Optional[str]:
        """Get API key for a specific provider type"""
        if not self.supabase_client:
            return None
        
        try:
            # Get the provider config to find the API key environment variable name
            provider = self.get_provider_by_type(provider_type)
            if not provider:
                return None
            
            api_key_env_var = provider.get('api_key_env_var')
            if not api_key_env_var:
                return None
            
            # Get the API key from the api_keys table
            response = self.supabase_client.table('api_keys').select('key_value').eq('key_name', api_key_env_var.lower()).eq('is_active', True).execute()
            
            if response.data:
                return response.data[0]['key_value']
            
        except Exception as e:
            logger.error(f"Failed to get API key for provider {provider_type}", error=str(e))
        
        return None

# Global instance
llm_provider_config = LLMProviderConfig()
