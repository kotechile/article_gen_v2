"""
LiteLLM client implementation.

This module provides the core LiteLLM integration for interacting
with various LLM providers through a unified interface.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import time

import litellm
from litellm import completion, acompletion
from litellm.exceptions import (
    AuthenticationError,
    RateLimitError,
    APIError,
    Timeout,
    ServiceUnavailableError
)

from ...core.models.llm import LLMConfig, LLMResponse, LLMError, LLMProvider, LLMModel


logger = logging.getLogger(__name__)


class LiteLLMClient:
    """
    LiteLLM client for unified LLM provider access.
    
    This client provides a consistent interface for interacting with
    various LLM providers through LiteLLM.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3
    ):
        """
        Initialize LiteLLM client.
        
        Args:
            api_key: API key for LLM provider
            base_url: Base URL for LLM API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Configure LiteLLM
        if api_key:
            litellm.api_key = api_key
        if base_url:
            litellm.api_base = base_url
        
        logger.info(f"LiteLLMClient initialized with timeout: {timeout}s")
    
    async def generate(self, config: LLMConfig) -> LLMResponse:
        """
        Generate text using LiteLLM.
        
        Args:
            config: LLM configuration
            
        Returns:
            LLMResponse with generated content
            
        Raises:
            LLMError: If generation fails
        """
        try:
            # Prepare messages
            messages = []
            
            if config.system_prompt:
                messages.append({
                    "role": "system",
                    "content": config.system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": config.user_prompt
            })
            
            # Prepare model string
            model_string = f"{config.model.provider}/{config.model.model_name}"
            
            # Prepare parameters
            params = {
                "model": model_string,
                "messages": messages,
                "temperature": config.temperature or config.model.temperature,
                "max_tokens": config.max_tokens or config.model.max_tokens,
                "top_p": config.top_p or config.model.top_p,
                "frequency_penalty": config.model.frequency_penalty,
                "presence_penalty": config.model.presence_penalty,
                "timeout": self.timeout
            }
            
            # Add stop sequences if provided
            if config.model.stop_sequences:
                params["stop"] = config.model.stop_sequences
            
            # Add API key if provided
            if config.model.api_key:
                params["api_key"] = config.model.api_key
            
            # Add base URL if provided
            if config.model.base_url:
                params["api_base"] = config.model.base_url
            
            # Generate response
            start_time = time.time()
            
            response = await acompletion(**params)
            
            response_time = time.time() - start_time
            
            # Extract response data
            choice = response.choices[0]
            content = choice.message.content
            finish_reason = choice.finish_reason
            
            # Extract usage statistics
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0
            
            return LLMResponse(
                content=content,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model=config.model.model_name,
                provider=config.model.provider,
                request_id=config.request_id,
                response_time=response_time,
                created_at=datetime.utcnow()
            )
            
        except AuthenticationError as e:
            logger.error(f"Authentication error: {str(e)}")
            raise LLMError(
                message=f"Authentication failed: {str(e)}",
                provider=config.model.provider,
                model=config.model.model_name,
                retryable=False
            )
        
        except RateLimitError as e:
            logger.error(f"Rate limit error: {str(e)}")
            raise LLMError(
                message=f"Rate limit exceeded: {str(e)}",
                provider=config.model.provider,
                model=config.model.model_name,
                retryable=True
            )
        
        except Timeout as e:
            logger.error(f"Timeout error: {str(e)}")
            raise LLMError(
                message=f"Request timeout: {str(e)}",
                provider=config.model.provider,
                model=config.model.model_name,
                retryable=True
            )
        
        except ServiceUnavailableError as e:
            logger.error(f"Service unavailable: {str(e)}")
            raise LLMError(
                message=f"Service unavailable: {str(e)}",
                provider=config.model.provider,
                model=config.model.model_name,
                retryable=True
            )
        
        except APIError as e:
            logger.error(f"API error: {str(e)}")
            raise LLMError(
                message=f"API error: {str(e)}",
                provider=config.model.provider,
                model=config.model.model_name,
                retryable=True
            )
        
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise LLMError(
                message=f"Unexpected error: {str(e)}",
                provider=config.model.provider,
                model=config.model.model_name,
                retryable=True
            )
    
    async def generate_batch(
        self,
        configs: List[LLMConfig],
        max_concurrent: int = 10
    ) -> List[LLMResponse]:
        """
        Generate text for multiple configs concurrently.
        
        Args:
            configs: List of LLM configurations
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of LLMResponse objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single(config: LLMConfig) -> LLMResponse:
            async with semaphore:
                return await self.generate(config)
        
        # Execute all requests concurrently
        tasks = [generate_single(config) for config in configs]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Batch generation failed for config {i}: {str(response)}")
                # Create error response
                error_response = LLMResponse(
                    content="",
                    finish_reason="error",
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    model=configs[i].model.model_name,
                    provider=configs[i].model.provider,
                    response_time=0.0,
                    created_at=datetime.utcnow()
                )
                results.append(error_response)
            else:
                results.append(response)
        
        return results
    
    def get_supported_providers(self) -> List[str]:
        """Get list of supported providers."""
        return [
            "openai",
            "anthropic", 
            "google",
            "deepseek",
            "moonshot",
            "cohere",
            "mistral",
            "ollama"
        ]
    
    def get_supported_models(self, provider: str) -> List[str]:
        """Get list of supported models for a provider."""
        model_map = {
            "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o"],
            "anthropic": ["claude-3-5-sonnet", "claude-3-opus", "claude-3-haiku"],
            "google": ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"],
            "deepseek": ["deepseek-chat", "deepseek-coder"],
            "moonshot": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
            "cohere": ["command", "command-light"],
            "mistral": ["mistral-large", "mistral-medium", "mistral-small"],
            "ollama": ["llama2", "codellama", "mistral"]
        }
        
        return model_map.get(provider, [])
    
    async def health_check(self, provider: str, model: str) -> Dict[str, Any]:
        """
        Perform health check for a specific provider/model.
        
        Args:
            provider: LLM provider
            model: Model name
            
        Returns:
            Health check result
        """
        try:
            # Test with a simple prompt
            config = LLMConfig(
                model=LLMModel(
                    provider=provider,
                    model_name=model,
                    api_key=self.api_key,
                    base_url=self.base_url
                ),
                user_prompt="Hello, this is a health check.",
                max_tokens=10
            )
            
            response = await self.generate(config)
            
            return {
                "status": "healthy",
                "provider": provider,
                "model": model,
                "response_time": response.response_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": provider,
                "model": model,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
