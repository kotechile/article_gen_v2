import logging
from typing import Optional, Any
from src.database.supabase_client import get_supabase_client
from .llm_provider import LLMResponse
from .providers import get_provider_class

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service to manage LLM interactions, handling provider selection and configuration 
    dynamically from the database.
    """
    
    def __init__(self):
        self.supabase = get_supabase_client()
        self.provider_cache = {}

    async def get_provider(self, provider_name: Optional[str] = None):
        """
        Get an initialized LLM provider instance.
        
        Args:
            provider_name: Optional name of the provider to use.
                           If None, uses the default provider from DB.
                           
        Returns:
            An instance of a subclass of LLMProvider
        """
        # 0. Check Cache
        cache_key = provider_name or "default"
        if cache_key in self.provider_cache:
            return self.provider_cache[cache_key]

        try:
            # 1. Query for the provider configuration
            # Fetch provider first (no join to avoid relationship errors)
            query = self.supabase.table("llm_providers").select("*")
            
            if provider_name:
                # Case-insensitive search on 'provider' or 'name'
                result = query.eq("provider", provider_name).eq("is_active", True).execute()
                
                # If no match on provider type, try specific model name
                if not result.data:
                     # Create fresh query as the builder is stateful
                     result = self.supabase.table("llm_providers").select("*").eq("name", provider_name).eq("is_active", True).execute()

                if not result.data:
                    raise ValueError(f"LLM Provider '{provider_name}' not found or not active.")
                
                provider_config = result.data[0]
            else:
                # Fetch default provider
                result = query.eq("is_default", True).eq("is_active", True).execute()
                
                if not result.data:
                    # Fallback: get the first active provider if no default is set
                    result = query.eq("is_active", True).limit(1).execute()
                    
                    if not result.data:
                        raise ValueError("No active LLM providers configured in the database.")
                
                provider_config = result.data[0]

            # 2. Fetch API Key
            # We need to manually fetch the api key because join might fail if FK is missing
            api_key_value = None
            base_url_value = provider_config.get("base_url")

            # Check for direct relationship columns
            api_key_id = provider_config.get("api_key_id") or provider_config.get("api_keys_id")
            
            if api_key_id:
                key_result = self.supabase.table("api_keys").select("*").eq("id", api_key_id).execute()
                if key_result.data:
                    api_key_data = key_result.data[0]
                    api_key_value = api_key_data.get("key_value")
                    # If provider base_url is undetermined, use key's base_url (optional fallback)
                    if not base_url_value:
                        base_url_value = api_key_data.get("base_url")
            
            if not api_key_value:
                # Could log warning, but for now we error if no key is found
                 raise ValueError(f"No API key found for provider {provider_config.get('name')} (ID: {api_key_id})")

            model_name = provider_config.get("model_name")
            
            # 3. Instantiate Provider
            provider_type = provider_config.get("provider")
            ProviderClass = get_provider_class(provider_type)
            
            instance = ProviderClass(
                api_key=api_key_value,
                model_name=model_name,
                base_url=base_url_value
            )
            
            # 4. Store in Cache
            self.provider_cache[cache_key] = instance
            return instance
            
        except Exception as e:
            logger.error(f"Error getting LLM provider: {e}")
            raise

    async def generate_text(self, prompt: str, provider: Optional[str] = None, **kwargs) -> LLMResponse:
        """
        Generate text using the specified or default LLM provider.
        
        Args:
            prompt: User prompt
            provider: Optional provider name
            **kwargs: Overrides for generation config (temperature, max_tokens)
        """
        try:
            llm_instance = await self.get_provider(provider)
            
            # TODO: Merge kwargs with DB defaults if needed (e.g., temperature from DB)
            # Currently we pass kwargs directly, allowing caller to override
            
            response = await llm_instance.generate(prompt, **kwargs)
            return response
            
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            raise

    async def generate_json(self, prompt: str, provider: Optional[str] = None, **kwargs) -> Any:
        """
        Generate a JSON response.
        Wraps generate_text and parses the output.
        """
        import json
        import re
        
        # Enforce JSON instruction if not present (optional, but good practice)
        if "json" not in prompt.lower():
            prompt += "\n\nPlease output valid JSON."
            
        try:
            response = await self.generate_text(prompt, provider, **kwargs)
            content = response.content.strip()
            
            # Simple cleanup to find JSON block if surrounded by markdown
            match = re.search(r'```json(.*?)```', content, re.DOTALL)
            if match:
                content = match.group(1).strip()
            
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Raw content: {response.content}")
            raise
        except Exception as e:
            logger.error(f"LLM JSON Generation failed: {e}")
            raise

# Singleton instance
llm_service = LLMService()
