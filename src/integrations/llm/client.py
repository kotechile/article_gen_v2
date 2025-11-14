"""
Main LLM client for Content Generator V2.

This module provides a unified interface for interacting with
various Large Language Model providers through LiteLLM.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import time

from ...core.models.llm import LLMConfig, LLMResponse, LLMError, LLMBatchRequest, LLMBatchResponse, LLMModel
from .litellm_client import LiteLLMClient
from .retry_handler import RetryHandler
from .rate_limiter import RateLimiter


logger = logging.getLogger(__name__)


class LLMClient:
    """
    Main LLM client that provides a unified interface for LLM operations.
    
    This client handles:
    - Multiple LLM providers through LiteLLM
    - Retry logic and error handling
    - Rate limiting
    - Batch processing
    - Response caching
    """
    
    def __init__(
        self,
        default_provider: str = "openai",
        default_model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limit_per_minute: int = 60,
        enable_caching: bool = True,
        cache_ttl: int = 3600
    ):
        """
        Initialize LLM client.
        
        Args:
            default_provider: Default LLM provider
            default_model: Default model name
            api_key: API key for LLM provider
            base_url: Base URL for LLM API
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            rate_limit_per_minute: Rate limit per minute
            enable_caching: Enable response caching
            cache_ttl: Cache TTL in seconds
        """
        self.default_provider = default_provider
        self.default_model = default_model
        self.api_key = api_key
        self.base_url = base_url
        
        # Initialize components
        self.litellm_client = LiteLLMClient(
            api_key=api_key,
            base_url=base_url
        )
        
        self.retry_handler = RetryHandler(
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        self.rate_limiter = RateLimiter(
            requests_per_minute=rate_limit_per_minute
        )
        
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, LLMResponse] = {}
        
        logger.info(f"LLMClient initialized with provider: {default_provider}, model: {default_model}")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            provider: LLM provider
            model: Model name
            api_key: API key
            temperature: Temperature for generation
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with generated content
            
        Raises:
            LLMError: If generation fails
        """
        # Use defaults if not provided
        provider = provider or self.default_provider
        model = model or self.default_model
        api_key = api_key or self.api_key
        
        # Create LLM config
        config = LLMConfig(
            model=LLMModel(
                provider=provider,
                model_name=model,
                api_key=api_key,
                base_url=self.base_url,
                temperature=temperature or 0.7,
                max_tokens=max_tokens
            ),
            system_prompt=system_prompt,
            user_prompt=prompt,
            **kwargs
        )
        
        # Check cache if enabled
        if self.enable_caching:
            cache_key = self._get_cache_key(config)
            if cache_key in self._cache:
                cached_response = self._cache[cache_key]
                if self._is_cache_valid(cached_response):
                    logger.debug(f"Returning cached response for key: {cache_key}")
                    return cached_response
        
        # Apply rate limiting
        await self.rate_limiter.wait_if_needed()
        
        # Generate with retry logic
        start_time = time.time()
        
        try:
            response = await self.retry_handler.execute_with_retry(
                self._generate_single,
                config
            )
            
            response.response_time = time.time() - start_time
            
            # Cache response if enabled
            if self.enable_caching:
                cache_key = self._get_cache_key(config)
                self._cache[cache_key] = response
                logger.debug(f"Cached response for key: {cache_key}")
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise LLMError(
                message=f"LLM generation failed: {str(e)}",
                provider=provider,
                model=model,
                retryable=True
            )
    
    async def generate_batch(
        self,
        requests: List[LLMConfig],
        max_concurrent: int = 10
    ) -> LLMBatchResponse:
        """
        Generate text for multiple requests in batch.
        
        Args:
            requests: List of LLM configs
            max_concurrent: Maximum concurrent requests
            
        Returns:
            LLMBatchResponse with all responses
        """
        batch_id = f"batch_{int(time.time())}"
        start_time = time.time()
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_request(config: LLMConfig) -> Union[LLMResponse, LLMError]:
            async with semaphore:
                try:
                    return await self.generate(
                        prompt=config.user_prompt,
                        system_prompt=config.system_prompt,
                        provider=config.model.provider,
                        model=config.model.model_name,
                        api_key=config.model.api_key,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens
                    )
                except Exception as e:
                    return LLMError(
                        message=str(e),
                        provider=config.model.provider,
                        model=config.model.model_name
                    )
        
        # Process all requests concurrently
        results = await asyncio.gather(
            *[process_request(req) for req in requests],
            return_exceptions=True
        )
        
        # Separate responses and errors
        responses = []
        errors = []
        
        for result in results:
            if isinstance(result, LLMResponse):
                responses.append(result)
            elif isinstance(result, LLMError):
                errors.append(result)
            else:
                errors.append(LLMError(
                    message=f"Unexpected error: {str(result)}"
                ))
        
        # Calculate statistics
        total_time = time.time() - start_time
        avg_response_time = sum(r.response_time for r in responses) / len(responses) if responses else 0
        
        return LLMBatchResponse(
            batch_id=batch_id,
            responses=responses,
            errors=errors,
            total_requests=len(requests),
            successful_requests=len(responses),
            failed_requests=len(errors),
            total_time=total_time,
            average_response_time=avg_response_time,
            started_at=datetime.fromtimestamp(start_time),
            completed_at=datetime.utcnow()
        )
    
    async def _generate_single(self, config: LLMConfig) -> LLMResponse:
        """Generate single response using LiteLLM."""
        return await self.litellm_client.generate(config)
    
    def _get_cache_key(self, config: LLMConfig) -> str:
        """Generate cache key for config."""
        import hashlib
        
        key_data = {
            "provider": config.model.provider,
            "model": config.model.model_name,
            "system_prompt": config.system_prompt,
            "user_prompt": config.user_prompt,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        }
        
        key_string = str(sorted(key_data.items()))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, response: LLMResponse) -> bool:
        """Check if cached response is still valid."""
        if not self.enable_caching:
            return False
        
        age = (datetime.utcnow() - response.created_at).total_seconds()
        return age < self.cache_ttl
    
    def clear_cache(self):
        """Clear response cache."""
        self._cache.clear()
        logger.info("LLM response cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_enabled": self.enable_caching,
            "cache_size": len(self._cache),
            "cache_ttl": self.cache_ttl
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Test with a simple prompt
            response = await self.generate(
                prompt="Hello, this is a health check.",
                max_tokens=10
            )
            
            return {
                "status": "healthy",
                "provider": self.default_provider,
                "model": self.default_model,
                "response_time": response.response_time,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
