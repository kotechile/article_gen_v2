"""
Enhanced LiteLLM client for Content Generator V2.

This module provides a robust LLM client with retry logic, fallbacks,
and support for multiple providers including GPT-5 and Gemini-2.5.
"""

import os
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import litellm
from litellm import completion, acompletion
from litellm.exceptions import (
    RateLimitError, 
    APIError, 
    Timeout, 
    APIConnectionError,
    ServiceUnavailableError
)

# Configure logging
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    MISTRAL = "mistral"
    KIMI = "kimi"

class LLMModel(Enum):
    """Supported LLM models."""
    # OpenAI Models
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_5 = "gpt-5"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    
    # Gemini Models
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_2_5 = "gemini-2.5"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    
    # Anthropic Models
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    
    # Kimi Models
    KIMI_K2_MOONSHINE = "kimi-k2-moonshine"

@dataclass
class LLMConfig:
    """Configuration for LLM requests."""
    provider: str
    model: str
    api_key: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None  # For GPT-5
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_models: Optional[List[str]] = None
    # Verbalized sampling configuration
    use_verbalized_sampling: bool = True
    verbalized_k: int = 5
    verbalized_tau: float = 0.10
    verbalized_temperature: float = 0.9
    verbalized_seed: Optional[int] = None

@dataclass
class LLMResponse:
    """Response from LLM request."""
    content: str
    model: str
    provider: str
    usage: Dict[str, Any]
    cost: float
    response_time: float
    retry_count: int = 0

class LLMClient:
    """
    Enhanced LLM client with retry logic, fallbacks, and error handling.
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM client.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Set up fallback models if not provided
        if not self.config.fallback_models:
            self.config.fallback_models = self._get_default_fallbacks()
        
        # Configure LiteLLM
        self._configure_litellm()
    
    def _configure_litellm(self):
        """Configure LiteLLM settings."""
        # Set API keys
        if self.config.provider == LLMProvider.OPENAI.value:
            os.environ["OPENAI_API_KEY"] = self.config.api_key
        elif self.config.provider == LLMProvider.GEMINI.value:
            os.environ["GEMINI_API_KEY"] = self.config.api_key
        elif self.config.provider == LLMProvider.ANTHROPIC.value:
            os.environ["ANTHROPIC_API_KEY"] = self.config.api_key
        elif self.config.provider == LLMProvider.COHERE.value:
            os.environ["COHERE_API_KEY"] = self.config.api_key
        elif self.config.provider == LLMProvider.MISTRAL.value:
            os.environ["MISTRAL_API_KEY"] = self.config.api_key
        elif self.config.provider == LLMProvider.KIMI.value or self.config.provider == "moonshot":
            os.environ["MOONSHOT_API_KEY"] = self.config.api_key
        
        # Configure LiteLLM settings
        litellm.drop_params = True  # Drop unsupported parameters
        litellm.set_verbose = False  # Set to True for debugging
    
    def _get_default_fallbacks(self) -> List[str]:
        """Get default fallback models based on provider."""
        fallbacks = {
            LLMProvider.OPENAI.value: [
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo"
            ],
            LLMProvider.GEMINI.value: [
                "gemini-1.5-pro",
                "gemini-1.5-flash"
            ],
            LLMProvider.ANTHROPIC.value: [
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ],
            LLMProvider.KIMI.value: [
                "kimi-k2-0711-preview",
                "kimi-k2-moonshine"
            ],
            "moonshot": [
                "kimi-k2-0711-preview"
            ]
        }
        return fallbacks.get(self.config.provider, [])
    
    def _get_model_name(self, model: str) -> str:
        """Get full model name with provider prefix."""
        if "/" in model:
            return model
        return f"{self.config.provider}/{model}"
    
    def _handle_llm_error(self, error: Exception, attempt: int) -> bool:
        """
        Handle LLM errors and determine if retry should be attempted.
        
        Args:
            error: The exception that occurred
            attempt: Current attempt number (0-based)
            
        Returns:
            True if retry should be attempted, False otherwise
        """
        if attempt >= self.config.max_retries:
            return False
        
        # Retry on these errors
        retryable_errors = (
            RateLimitError,
            Timeout,
            APIConnectionError,
            ServiceUnavailableError
        )
        
        if isinstance(error, retryable_errors):
            self.logger.warning(f"Retryable error on attempt {attempt + 1}: {error}")
            return True
        
        # Don't retry on these errors
        if isinstance(error, APIError):
            self.logger.error(f"API error (not retryable): {error}")
            return False
        
        # Default to retry for unknown errors
        self.logger.warning(f"Unknown error on attempt {attempt + 1}: {error}")
        return True
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        return self.config.retry_delay * (2 ** attempt)
    
    def _get_request_params(self, model: str) -> Dict[str, Any]:
        """Get request parameters for the model."""
        params = {
            "model": self._get_model_name(model),
            "temperature": self.config.temperature,
            "timeout": self.config.timeout
        }
        
        # Pass API key directly to ensure it's used (more reliable than env vars)
        if self.config.provider == LLMProvider.GEMINI.value:
            params["api_key"] = self.config.api_key
        elif self.config.provider == LLMProvider.OPENAI.value:
            params["api_key"] = self.config.api_key
        elif self.config.provider == LLMProvider.ANTHROPIC.value:
            params["api_key"] = self.config.api_key
        
        # Add max_tokens or max_completion_tokens based on model
        if model in ["gpt-5"] and self.config.max_completion_tokens:
            params["max_completion_tokens"] = self.config.max_completion_tokens
        elif self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens
        
        return params
    
    async def generate_async(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate text asynchronously with retry logic and fallbacks.
        
        Args:
            messages: List of message dictionaries
            model: Optional model override
            
        Returns:
            LLMResponse object
        """
        model = model or self.config.model
        models_to_try = [model] + (self.config.fallback_models or [])
        
        last_error = None
        
        for model_attempt, current_model in enumerate(models_to_try):
            for retry_attempt in range(self.config.max_retries + 1):
                start_time = time.time()
                
                try:
                    self.logger.info(f"Attempting request with {current_model} (attempt {retry_attempt + 1})")
                    
                    # Ensure API key is set in environment before each call (for concurrent tasks)
                    self._configure_litellm()
                    
                    params = self._get_request_params(current_model)
                    # For Gemini, LiteLLM requires api_key in a specific format
                    if self.config.provider == LLMProvider.GEMINI.value:
                        # LiteLLM for Gemini uses api_key parameter or GEMINI_API_KEY env var
                        # Pass it explicitly to avoid race conditions with concurrent tasks
                        params.pop('api_key', None)  # Remove if added, we'll use env var
                        # Ensure env var is set just before the call
                        os.environ["GEMINI_API_KEY"] = self.config.api_key
                        # Log API key info for debugging (first 10 chars only for security)
                        self.logger.info(f"Using Gemini API key: {self.config.api_key[:10]}... (length: {len(self.config.api_key)})")
                    
                    response = await acompletion(
                        messages=messages,
                        **params
                    )
                    
                    response_time = time.time() - start_time
                    
                    # Extract response data
                    content = response.choices[0].message.content
                    usage = response.usage.dict() if response.usage else {}
                    cost = getattr(response, '_hidden_params', {}).get('cost', 0.0)
                    
                    self.logger.info(f"Successfully generated response with {current_model} in {response_time:.2f}s")
                    
                    return LLMResponse(
                        content=content,
                        model=current_model,
                        provider=self.config.provider,
                        usage=usage,
                        cost=cost,
                        response_time=response_time,
                        retry_count=retry_attempt
                    )
                
                except Exception as error:
                    last_error = error
                    response_time = time.time() - start_time
                    
                    self.logger.error(f"Error with {current_model} (attempt {retry_attempt + 1}): {error}")
                    
                    # Check if we should retry
                    if not self._handle_llm_error(error, retry_attempt):
                        break
                    
                    # Wait before retry
                    if retry_attempt < self.config.max_retries:
                        delay = self._calculate_retry_delay(retry_attempt)
                        self.logger.info(f"Waiting {delay:.2f}s before retry...")
                        await asyncio.sleep(delay)
            
            # If we've exhausted retries for this model, try the next fallback
            if model_attempt < len(models_to_try) - 1:
                self.logger.warning(f"Switching to fallback model: {models_to_try[model_attempt + 1]}")
        
        # If all models and retries failed
        raise Exception(f"All models failed. Last error: {last_error}")
    
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate text synchronously with retry logic and fallbacks.
        
        Args:
            messages: List of message dictionaries
            model: Optional model override
            
        Returns:
            LLMResponse object
        """
        model = model or self.config.model
        models_to_try = [model] + (self.config.fallback_models or [])
        
        last_error = None
        
        for model_attempt, current_model in enumerate(models_to_try):
            for retry_attempt in range(self.config.max_retries + 1):
                start_time = time.time()
                
                try:
                    self.logger.info(f"Attempting request with {current_model} (attempt {retry_attempt + 1})")
                    
                    # Ensure API key is set in environment before each call (for concurrent tasks)
                    self._configure_litellm()
                    
                    params = self._get_request_params(current_model)
                    # For Gemini, LiteLLM requires api_key in a specific format
                    if self.config.provider == LLMProvider.GEMINI.value:
                        # LiteLLM for Gemini uses api_key parameter or GEMINI_API_KEY env var
                        # Pass it explicitly to avoid race conditions with concurrent tasks
                        params.pop('api_key', None)  # Remove if added, we'll use env var
                        # Ensure env var is set just before the call
                        os.environ["GEMINI_API_KEY"] = self.config.api_key
                        # Log API key info for debugging (first 10 chars only for security)
                        self.logger.info(f"Using Gemini API key: {self.config.api_key[:10]}... (length: {len(self.config.api_key)})")
                    
                    response = completion(
                        messages=messages,
                        **params
                    )
                    
                    response_time = time.time() - start_time
                    
                    # Extract response data
                    content = response.choices[0].message.content
                    usage = response.usage.dict() if response.usage else {}
                    cost = getattr(response, '_hidden_params', {}).get('cost', 0.0)
                    
                    self.logger.info(f"Successfully generated response with {current_model} in {response_time:.2f}s")
                    
                    return LLMResponse(
                        content=content,
                        model=current_model,
                        provider=self.config.provider,
                        usage=usage,
                        cost=cost,
                        response_time=response_time,
                        retry_count=retry_attempt
                    )
                
                except Exception as error:
                    last_error = error
                    response_time = time.time() - start_time
                    
                    self.logger.error(f"Error with {current_model} (attempt {retry_attempt + 1}): {error}")
                    
                    # Check if we should retry
                    if not self._handle_llm_error(error, retry_attempt):
                        break
                    
                    # Wait before retry
                    if retry_attempt < self.config.max_retries:
                        delay = self._calculate_retry_delay(retry_attempt)
                        self.logger.info(f"Waiting {delay:.2f}s before retry...")
                        time.sleep(delay)
            
            # If we've exhausted retries for this model, try the next fallback
            if model_attempt < len(models_to_try) - 1:
                self.logger.warning(f"Switching to fallback model: {models_to_try[model_attempt + 1]}")
        
        # If all models and retries failed
        raise Exception(f"All models failed. Last error: {last_error}")
    
    def get_cost_estimate(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> float:
        """
        Get cost estimate for a request.
        
        Args:
            messages: List of message dictionaries
            model: Optional model override
            
        Returns:
            Estimated cost in USD
        """
        try:
            model = model or self.config.model
            model_name = self._get_model_name(model)
            
            # Use LiteLLM's cost calculation
            cost = litellm.completion_cost(
                model=model_name,
                messages=messages,
                completion_tokens=1000  # Estimate
            )
            
            return cost
        except Exception as e:
            self.logger.warning(f"Could not calculate cost estimate: {e}")
            return 0.0
    
    def get_available_models(self) -> List[str]:
        """Get list of available models for the current provider."""
        models = {
            LLMProvider.OPENAI.value: [
                "gpt-5", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"
            ],
            LLMProvider.GEMINI.value: [
                "gemini-2.5", "gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"
            ],
            LLMProvider.ANTHROPIC.value: [
                "claude-3-5-sonnet-20241022", "claude-3-opus-20240229",
                "claude-3-sonnet-20240229", "claude-3-haiku-20240307"
            ],
            LLMProvider.KIMI.value: [
                "kimi-k2-0711-preview",
                "kimi-k2-moonshine"
            ],
            "moonshot": [
                "kimi-k2-0711-preview"
            ]
        }
        return models.get(self.config.provider, [])
    
    def get_verbalized_sampling_config(self) -> Dict[str, Any]:
        """Get verbalized sampling configuration."""
        return {
            "enabled": self.config.use_verbalized_sampling,
            "k": self.config.verbalized_k,
            "tau": self.config.verbalized_tau,
            "temperature": self.config.verbalized_temperature,
            "seed": self.config.verbalized_seed
        }

# Factory function for creating LLM clients
def create_llm_client(
    provider: str,
    model: str,
    api_key: str,
    **kwargs
) -> LLMClient:
    """
    Create an LLM client with the specified configuration.
    
    Args:
        provider: LLM provider (openai, gemini, anthropic, etc.)
        model: Model name
        api_key: API key for the provider
        **kwargs: Additional configuration options
        
    Returns:
        Configured LLMClient instance
    """
    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        **kwargs
    )
    
    return LLMClient(config)

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    client = create_llm_client(
        provider="openai",
        model="gpt-4",
        api_key="your-api-key-here",
        temperature=0.7,
        max_tokens=1000,
        max_retries=3
    )
    
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    try:
        response = client.generate(messages)
        print(f"Response: {response.content}")
        print(f"Model: {response.model}")
        print(f"Cost: ${response.cost:.4f}")
        print(f"Response time: {response.response_time:.2f}s")
    except Exception as e:
        print(f"Error: {e}")
