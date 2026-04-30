import logging
from typing import Optional, Any
from src.core.supabase_singleton import get_supabase_client
from .llm_provider import LLMResponse
from .providers import get_provider_class
from supabase_client import resolve_llm_provider

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service to manage LLM interactions, handling provider selection and configuration 
    dynamically from the database.
    """
    
    def __init__(self):
        self.supabase = get_supabase_client()
        self.provider_cache = {}

    def _query_llm_providers(self, filters: list[tuple[str, Any]], require_active: bool = True):
        """
        Query llm_providers with backward-compatible handling when `is_active` does not exist.
        """
        query = self.supabase.table("llm_providers").select("*")
        for field, value in filters:
            query = query.eq(field, value)

        if require_active:
            query = query.eq("is_active", True)

        try:
            return query.execute()
        except Exception as exc:
            message = str(exc)
            if require_active and "llm_providers.is_active" in message and "does not exist" in message:
                logger.warning("llm_providers.is_active column missing; retrying provider query without active filter")
                fallback_query = self.supabase.table("llm_providers").select("*")
                for field, value in filters:
                    fallback_query = fallback_query.eq(field, value)
                return fallback_query.execute()
            raise

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
            # 1. Query for provider configuration (schema-compatible)
            if provider_name:
                # Primary lookup by provider type
                result = self._query_llm_providers([("provider", provider_name)], require_active=True)
                
                # If no match on provider type, try specific model name
                if not result.data:
                    try:
                        result = self._query_llm_providers([("name", provider_name)], require_active=True)
                    except Exception as name_lookup_error:
                        if "llm_providers.name" in str(name_lookup_error) and "does not exist" in str(name_lookup_error):
                            logger.warning("llm_providers.name column missing; skipping name-based provider lookup")
                            result = type("Obj", (), {"data": []})()
                        else:
                            raise

                if not result.data:
                    raise ValueError(f"LLM Provider '{provider_name}' not found or not active.")
                
                provider_config = result.data[0]
            else:
                # Fetch default provider
                result = self._query_llm_providers([("is_default", True)], require_active=True)
                
                if not result.data:
                    # Fallback: get the first active provider if no default is set
                    try:
                        result = self.supabase.table("llm_providers").select("*").eq("is_active", True).limit(1).execute()
                    except Exception as active_lookup_error:
                        if "llm_providers.is_active" in str(active_lookup_error) and "does not exist" in str(active_lookup_error):
                            result = self.supabase.table("llm_providers").select("*").limit(1).execute()
                        else:
                            raise
                    
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
                 raise ValueError(f"No API key found for provider {provider_config.get('provider')} (ID: {api_key_id})")

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

    async def get_provider_for_role(
        self,
        task_role: str,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Resolve provider/model/API key from llm_used_for/llm_providers for a task role.
        Falls back according to resolver policy (role -> research -> default).
        """
        resolved = resolve_llm_provider(task_role=task_role, provider=provider_name, model=model_name)
        provider = str(resolved.get("provider") or "").strip().lower()
        model = str(resolved.get("model") or "").strip()
        api_key = str(resolved.get("api_key") or "").strip()
        base_url = resolved.get("base_url")
        logger.info(
            "Resolved LLM role=%s source=%s provider=%s model=%s has_api_key=%s",
            task_role,
            resolved.get("source"),
            provider,
            model,
            bool(api_key),
        )

        if not provider or not model:
            raise ValueError(f"No LLM provider/model resolved for role '{task_role}'")
        if not api_key:
            raise ValueError(
                f"No API key resolved for role '{task_role}' provider='{provider}' model='{model}'"
            )

        cache_key = f"role:{task_role}:{provider}:{model}:{base_url or ''}"
        if cache_key in self.provider_cache:
            return self.provider_cache[cache_key]

        ProviderClass = get_provider_class(provider)
        instance = ProviderClass(api_key=api_key, model_name=model, base_url=base_url)
        self.provider_cache[cache_key] = instance
        return instance

    async def generate_text(
        self,
        prompt: str,
        provider: Optional[str] = None,
        task_role: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text using the specified or default LLM provider.
        
        Args:
            prompt: User prompt
            provider: Optional provider name
            **kwargs: Overrides for generation config (temperature, max_tokens)
        """
        try:
            if task_role:
                llm_instance = await self.get_provider_for_role(
                    task_role=task_role,
                    provider_name=provider,
                    model_name=model,
                )
            else:
                llm_instance = await self.get_provider(provider)
            
            # TODO: Merge kwargs with DB defaults if needed (e.g., temperature from DB)
            # Currently we pass kwargs directly, allowing caller to override
            
            response = await llm_instance.generate(prompt, **kwargs)
            return response
            
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            raise

    async def generate_json(
        self,
        prompt: str,
        provider: Optional[str] = None,
        task_role: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Any:
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
            response = await self.generate_text(
                prompt=prompt,
                provider=provider,
                task_role=task_role,
                model=model,
                **kwargs,
            )
            content = response.content.strip()

            # Strategy 1: complete ```json ... ``` block
            match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
            if match:
                content = match.group(1).strip()
            else:
                # Strategy 2: opening fence with no closing (truncated response) — take everything after ```json
                match = re.search(r'```(?:json)?\s*(.*)', content, re.DOTALL)
                if match:
                    content = match.group(1).strip()
                else:
                    # Strategy 3: find first { or [ and extract from there
                    json_start = re.search(r'[{\[]', content)
                    if json_start:
                        content = content[json_start.start():]

            # If content is empty after all strategies, raise a clear error
            if not content or not content.strip():
                raise ValueError("LLM returned empty content after JSON extraction attempts")

            # Try to parse the JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Strategy 4: Try to fix common truncation issues
                # If content ends abruptly, try to close open structures
                fixed_content = self._attempt_to_fix_truncated_json(content)
                if fixed_content:
                    try:
                        return json.loads(fixed_content)
                    except json.JSONDecodeError:
                        pass
                raise

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Raw content: {response.content}")
            raise
        except Exception as e:
            logger.error(f"LLM JSON Generation failed: {e}")
            raise

    def _attempt_to_fix_truncated_json(self, content: str) -> Optional[str]:
        """
        Attempt to fix truncated JSON by closing open braces/brackets.
        Returns fixed content or None if cannot fix.
        """
        import json

        # Try progressively closing open structures
        attempts = [
            content,  # Original
            content + '"}',  # Missing closing quote and brace (for "key": "value)
            content + '}',   # Missing closing brace
            content + ']}',  # Missing closing bracket and brace
            content + '}}', # Missing two closing braces
            content + ']}]}', # Common for nested arrays
        ]

        # Also try removing trailing commas before closing
        # Find the last complete key-value pair or array element
        lines = content.split('\n')
        for i in range(len(lines), 0, -1):
            partial = '\n'.join(lines[:i])
            # Remove trailing comma if present
            partial = partial.rstrip().rstrip(',').rstrip()
            attempts.append(partial + '}')
            attempts.append(partial + ']}')
            attempts.append(partial + '}}')

        # Try each attempt
        for attempt in attempts:
            try:
                json.loads(attempt)
                return attempt
            except json.JSONDecodeError:
                continue

        return None

# Singleton instance
llm_service = LLMService()
