"""
API Key Manager for Supabase-based key storage
Loads API keys from Supabase database using the same client as the rest of the application
"""

import os
from typing import Dict, Optional, Any
import structlog

logger = structlog.get_logger()

class APIKeyManager:
    """Manages API keys stored in Supabase database"""
    
    def __init__(self):
        self._keys_cache: Dict[str, str] = {}
        self._cache_loaded = False
        self.supabase_client = None
        self._initialize_supabase_client()
    
    def _initialize_supabase_client(self):
        """Initialize Supabase client using the same pattern as the rest of the application"""
        try:
            # Import the global Supabase client from the singleton
            from .supabase_singleton import get_supabase_client
            self.supabase_client = get_supabase_client()
            logger.info("API Key Manager initialized with Supabase")
        except Exception as e:
            logger.warning("Failed to initialize Supabase client, falling back to environment variables", error=str(e))
            self.supabase_client = None
    
    def get_key(self, key_name: str, fallback_env_var: Optional[str] = None) -> Optional[str]:
        """
        Get an API key by name from Supabase or fallback to environment variable
        
        Args:
            key_name: Name of the key in the database (e.g., 'openai_api_key')
            fallback_env_var: Environment variable to fallback to (e.g., 'OPENAI_API_KEY')
            
        Returns:
            API key value or None if not found
        """
        # Try to get from cache first
        if key_name in self._keys_cache:
            return self._keys_cache[key_name]
        
        # Load all keys if cache not loaded
        if not self._cache_loaded:
            self._load_all_keys()
        
        # Return from cache if available
        if key_name in self._keys_cache:
            return self._keys_cache[key_name]
        
        # Fallback to environment variable
        if fallback_env_var:
            env_value = os.getenv(fallback_env_var)
            if env_value:
                logger.info(f"Using environment variable fallback for {key_name}")
                return env_value
        
        logger.warning(f"API key not found: {key_name}")
        return None
    
    def _load_all_keys(self) -> None:
        """Load all active API keys from Supabase into cache"""
        if not self.supabase_client:
            logger.warning("Supabase client not available, skipping key loading")
            return
        
        try:
            # Query all active API keys
            response = self.supabase_client.table('api_keys').select('key_name, key_value').eq('is_active', True).execute()
            
            if response.data:
                for key_data in response.data:
                    self._keys_cache[key_data['key_name']] = key_data['key_value']
                
                logger.info(f"Loaded {len(self._keys_cache)} API keys from Supabase")
                self._cache_loaded = True
            else:
                logger.warning("No API keys found in Supabase database")
                
        except Exception as e:
            logger.error("Failed to load API keys from Supabase", error=str(e))
            self._cache_loaded = True  # Prevent repeated attempts
    
    def refresh_keys(self) -> None:
        """Refresh the API keys cache from Supabase"""
        self._keys_cache.clear()
        self._cache_loaded = False
        self._load_all_keys()
    
    def get_openai_key(self) -> Optional[str]:
        """Get OpenAI API key"""
        return self.get_key('openai_api_key', 'OPENAI_API_KEY')
    
    def get_linkup_key(self) -> Optional[str]:
        """Get LinkUp.so API key"""
        key = self.get_key('linkup_api_key', 'LINKUP_API_KEY')
        if not key:
            key = self.get_key('linkup_key')
        return key
    
    def get_anthropic_key(self) -> Optional[str]:
        """Get Anthropic API key"""
        return self.get_key('anthropic_api_key', 'ANTHROPIC_API_KEY')
    
    def get_google_ai_key(self) -> Optional[str]:
        """Get Google AI API key"""
        return self.get_key('google_ai_api_key', 'GOOGLE_AI_API_KEY')
    
    def get_deepseek_key(self) -> Optional[str]:
        """Get DeepSeek API key"""
        return self.get_key('deepseek_api_key', 'DEEPSEEK_API_KEY')
    
    def get_key_by_provider(self, provider: str) -> Optional[str]:
        """Get API key by provider name"""
        if not self.supabase_client:
            return None
        
        try:
            response = self.supabase_client.table('api_keys').select('key_value').eq('provider', provider).eq('is_active', True).limit(1).execute()
            
            if response.data:
                return response.data[0]['key_value']
            return None
            
        except Exception as e:
            logger.error(f"Failed to get key for provider {provider}", error=str(e))
            return None

# Global instance
api_key_manager = APIKeyManager()

