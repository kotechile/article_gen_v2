"""
Direct LLM provider client for Content Generator V2.
Uses native SDKs instead of LiteLLM for better reliability and control.

Supports:
- OpenAI (openai SDK)
- Gemini (google-generativeai SDK)
- Anthropic (anthropic SDK)
- DeepSeek (OpenAI-compatible API)
- Moonshot/Kimi (OpenAI-compatible API)
"""

import os
import time
import logging
import signal
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    MOONSHOT = "moonshot"
    KIMI = "kimi"
    COHERE = "cohere"
    MISTRAL = "mistral"

@dataclass
class LLMConfig:
    """Configuration for LLM requests."""
    provider: str
    model: str
    api_key: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None  # For GPT-5 models
    timeout: int = 120  # Increased default timeout to 2 minutes
    max_retries: int = 3
    retry_delay: float = 1.0

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
    Direct LLM client using native provider SDKs.
    No LiteLLM dependency - direct integration for better control.
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM client.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize provider-specific client
        self._client = self._init_provider_client()
    
    def _init_provider_client(self):
        """Initialize provider-specific client."""
        if self.config.provider == LLMProvider.OPENAI.value:
            try:
                import openai
                return openai.OpenAI(api_key=self.config.api_key, timeout=self.config.timeout)
            except ImportError:
                raise ImportError("openai package required for OpenAI provider. Install with: pip install openai")
        
        elif self.config.provider == LLMProvider.GEMINI.value:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config.api_key)
                # Store genai module for later use, not the model instance
                return genai
            except ImportError:
                raise ImportError("google-generativeai package required for Gemini provider. Install with: pip install google-generativeai")
        
        elif self.config.provider == LLMProvider.ANTHROPIC.value:
            try:
                import anthropic
                return anthropic.Anthropic(api_key=self.config.api_key, timeout=self.config.timeout)
            except ImportError:
                raise ImportError("anthropic package required for Anthropic provider. Install with: pip install anthropic")
        
        elif self.config.provider == LLMProvider.DEEPSEEK.value:
            try:
                import openai
                # DeepSeek uses OpenAI-compatible API with custom base URL
                return openai.OpenAI(
                    api_key=self.config.api_key,
                    base_url="https://api.deepseek.com",
                    timeout=self.config.timeout
                )
            except ImportError:
                raise ImportError("openai package required for DeepSeek provider. Install with: pip install openai")
        
        elif self.config.provider == LLMProvider.MOONSHOT.value or self.config.provider == LLMProvider.KIMI.value:
            try:
                import openai
                # Moonshot/Kimi uses OpenAI-compatible API with custom base URL
                return openai.OpenAI(
                    api_key=self.config.api_key,
                    base_url="https://api.moonshot.cn/v1",
                    timeout=self.config.timeout
                )
            except ImportError:
                raise ImportError("openai package required for Moonshot/Kimi provider. Install with: pip install openai")
        
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate text synchronously with retry logic.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Optional model override
            
        Returns:
            LLMResponse object
        """
        model = model or self.config.model
        start_time = time.time()
        last_error = None
        
        for retry_attempt in range(self.config.max_retries + 1):
            try:
                self.logger.info(f"Generating with {self.config.provider}/{model} (attempt {retry_attempt + 1})")
                
                if self.config.provider == LLMProvider.OPENAI.value:
                    response = self._generate_openai(messages, model)
                elif self.config.provider == LLMProvider.GEMINI.value:
                    response = self._generate_gemini(messages, model)
                elif self.config.provider == LLMProvider.ANTHROPIC.value:
                    response = self._generate_anthropic(messages, model)
                elif self.config.provider == LLMProvider.DEEPSEEK.value:
                    # DeepSeek uses OpenAI-compatible API
                    response = self._generate_openai(messages, model)
                elif self.config.provider == LLMProvider.MOONSHOT.value or self.config.provider == LLMProvider.KIMI.value:
                    # Moonshot/Kimi uses OpenAI-compatible API
                    response = self._generate_openai(messages, model)
                else:
                    raise ValueError(f"Unsupported provider: {self.config.provider}")
                
                response_time = time.time() - start_time
                self.logger.info(f"Successfully generated response in {response_time:.2f}s")
                
                return response
                
            except Exception as error:
                last_error = error
                error_msg = str(error)
                # Log full error details for debugging
                self.logger.error(f"Error on attempt {retry_attempt + 1}: {error_msg}")
                if hasattr(error, '__cause__') and error.__cause__:
                    self.logger.error(f"  Cause: {str(error.__cause__)}")
                if hasattr(error, 'args') and error.args:
                    self.logger.error(f"  Error args: {error.args}")
                
                if retry_attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** retry_attempt)
                    self.logger.info(f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"All retries failed. Last error: {error_msg}")
                    raise Exception(f"Failed after {self.config.max_retries + 1} attempts. Last error: {error_msg}")
        
        raise Exception(f"Generation failed. Last error: {last_error}")
    
    def _generate_openai(self, messages: List[Dict[str, str]], model: str) -> LLMResponse:
        """Generate using OpenAI SDK."""
        start_time = time.time()
        
        # Convert messages format
        openai_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]
        
        # GPT-5 or higher models must not have temperature parameter
        # Handles: gpt-5, gpt-5-mini, gpt-5-turbo, gpt-6, etc.
        model_lower = model.lower()
        is_gpt5_or_higher = (
            "gpt-5" in model_lower or 
            "gpt-6" in model_lower or 
            "gpt-7" in model_lower or
            model_lower.startswith("gpt-5") or
            model_lower.startswith("gpt-6") or
            model_lower.startswith("gpt-7")
        )
        
        if is_gpt5_or_higher:
            self.logger.info(f"GPT-5+ model detected ({model}) - excluding temperature parameter")
        
        # Build request parameters
        request_params = {
            "model": model,
            "messages": openai_messages,
        }
        
        # Only add temperature if not GPT-5 or higher
        if not is_gpt5_or_higher:
            request_params["temperature"] = self.config.temperature
        
        # For GPT-5 models, use max_completion_tokens instead of max_tokens
        if is_gpt5_or_higher:
            if self.config.max_completion_tokens:
                request_params["max_completion_tokens"] = self.config.max_completion_tokens
                self.logger.info(f"Using max_completion_tokens={self.config.max_completion_tokens} for GPT-5 model")
            elif self.config.max_tokens:
                # Fallback to max_tokens if max_completion_tokens not specified
                request_params["max_completion_tokens"] = self.config.max_tokens
                self.logger.info(f"Using max_completion_tokens={self.config.max_tokens} (from max_tokens) for GPT-5 model")
        else:
            # For non-GPT-5 models, use max_tokens
            if self.config.max_tokens:
                request_params["max_tokens"] = self.config.max_tokens
        
        # Log request parameters (excluding sensitive data)
        safe_params = {k: v for k, v in request_params.items() if k != "messages"}
        self.logger.debug(f"Request parameters: {safe_params}")
        
        response = self._client.chat.completions.create(**request_params)
        
        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        # Estimate cost based on provider
        if self.config.provider == LLMProvider.DEEPSEEK.value:
            cost = self._estimate_cost_deepseek(model, usage["total_tokens"])
        elif self.config.provider == LLMProvider.MOONSHOT.value or self.config.provider == LLMProvider.KIMI.value:
            cost = self._estimate_cost_moonshot(model, usage["total_tokens"])
        else:
            cost = self._estimate_cost_openai(model, usage["total_tokens"])
        
        return LLMResponse(
            content=content,
            model=model,
            provider=self.config.provider,
            usage=usage,
            cost=cost,
            response_time=time.time() - start_time
        )
    
    def _generate_gemini(self, messages: List[Dict[str, str]], model: str) -> LLMResponse:
        """Generate using Google Generative AI SDK."""
        start_time = time.time()
        
        # Get genai module from _client
        genai = self._client
        
        # Ensure API key is configured (reconfigure in case it was lost)
        genai.configure(api_key=self.config.api_key)
        
        # Create model instance with the specified model and timeout
        # Note: Google Generative AI SDK doesn't support timeout in the model constructor,
        # so we'll need to handle timeout at the request level or use signal/timeout wrapper
        model_instance = genai.GenerativeModel(model)
        
        # Build conversation history
        chat_history = []
        system_instruction = None
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_instruction = content
            elif role == "user":
                chat_history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                chat_history.append({"role": "model", "parts": [content]})
        
        # Gemini 2.5 or higher models must not have temperature parameter
        # Handles: gemini-2.5, gemini-2.5-mini, gemini-2.5-flash, gemini-3, gemini-3-mini, etc.
        model_lower = model.lower()
        is_gemini_25_or_higher = (
            "gemini-2.5" in model_lower or
            "gemini-2.5" in model or
            "gemini-3" in model_lower or
            "gemini-4" in model_lower or
            "gemini-5" in model_lower
        )
        
        if is_gemini_25_or_higher:
            self.logger.info(f"Gemini 2.5+ model detected ({model}) - excluding temperature parameter")
        
        # Use timeout wrapper for Gemini API calls since SDK doesn't support timeout
        def _call_with_timeout(func, *args, **kwargs):
            """Call function with timeout using signal alarm (Unix only) or threading (for non-main threads)."""
            import threading
            
            # Check if we're in the main thread - signals only work in main thread
            is_main_thread = threading.current_thread() is threading.main_thread()
            
            # Use signal-based timeout only if:
            # 1. We're in the main thread
            # 2. SIGALRM is available (Unix/macOS)
            if is_main_thread and hasattr(signal, 'SIGALRM'):
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Gemini API call timed out after {self.config.timeout} seconds")
                
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.config.timeout)
                try:
                    result = func(*args, **kwargs)
                    signal.alarm(0)  # Cancel alarm
                    signal.signal(signal.SIGALRM, old_handler)  # Restore handler
                    return result
                except Exception as e:
                    signal.alarm(0)  # Cancel alarm
                    signal.signal(signal.SIGALRM, old_handler)  # Restore handler
                    raise
            
            # Use threading timeout for:
            # - Non-main threads (e.g., Celery workers)
            # - Windows (doesn't support SIGALRM)
            result_container = [None]
            exception_container = [None]
            
            def target():
                try:
                    result_container[0] = func(*args, **kwargs)
                except Exception as e:
                    exception_container[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=self.config.timeout)
            
            if thread.is_alive():
                raise TimeoutError(f"Gemini API call timed out after {self.config.timeout} seconds")
            
            if exception_container[0]:
                raise exception_container[0]
            
            return result_container[0]
        
        # Start chat session if we have history, otherwise simple generation
        if len(chat_history) > 1:
            # Use chat session for multi-turn conversations
            # Build generation config without temperature for Gemini 2.5+
            generation_config = {}
            if not is_gemini_25_or_higher:
                generation_config["temperature"] = self.config.temperature
            if self.config.max_tokens:
                generation_config["max_output_tokens"] = self.config.max_tokens
            
            chat = model_instance.start_chat(history=chat_history[:-1])
            response = _call_with_timeout(
                chat.send_message,
                chat_history[-1]["parts"][0],
                generation_config=generation_config if generation_config else None
            )
        else:
            # Simple generation for single message
            prompt = chat_history[0]["parts"][0] if chat_history else messages[-1]["content"]
            if system_instruction:
                prompt = f"{system_instruction}\n\n{prompt}"
            
            # Build generation config without temperature for Gemini 2.5+
            generation_config = {}
            if not is_gemini_25_or_higher:
                generation_config["temperature"] = self.config.temperature
            if self.config.max_tokens:
                generation_config["max_output_tokens"] = self.config.max_tokens
            
            response = _call_with_timeout(
                model_instance.generate_content,
                prompt,
                generation_config=generation_config if generation_config else None
            )
        
        content = response.text
        usage = {
            "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
            "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
            "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0) if hasattr(response, 'usage_metadata') else 0
        }
        
        cost = self._estimate_cost_gemini(model, usage.get("total_tokens", 0))
        
        return LLMResponse(
            content=content,
            model=model,
            provider=self.config.provider,
            usage=usage,
            cost=cost,
            response_time=time.time() - start_time
        )
    
    def _generate_anthropic(self, messages: List[Dict[str, str]], model: str) -> LLMResponse:
        """Generate using Anthropic SDK."""
        start_time = time.time()
        
        # Anthropic uses system and messages separately
        system_message = None
        conversation_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                # Anthropic uses 'user' and 'assistant' roles
                role = msg["role"] if msg["role"] in ["user", "assistant"] else "user"
                conversation_messages.append({
                    "role": role,
                    "content": msg["content"]
                })
        
        kwargs = {
            "model": model,
            "messages": conversation_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens or 4096
        }
        
        if system_message:
            kwargs["system"] = system_message
        
        response = self._client.messages.create(**kwargs)
        
        content = response.content[0].text
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }
        
        cost = self._estimate_cost_anthropic(model, usage["total_tokens"])
        
        return LLMResponse(
            content=content,
            model=model,
            provider=self.config.provider,
            usage=usage,
            cost=cost,
            response_time=time.time() - start_time
        )
    
    def _estimate_cost_openai(self, model: str, tokens: int) -> float:
        """Estimate cost for OpenAI models."""
        # Simplified cost estimation (per 1M tokens)
        costs = {
            "gpt-4": 30.0,
            "gpt-4-turbo": 10.0,
            "gpt-3.5-turbo": 1.5,
        }
        base_cost = costs.get(model, 10.0)
        return (tokens / 1_000_000) * base_cost
    
    def _estimate_cost_gemini(self, model: str, tokens: int) -> float:
        """Estimate cost for Gemini models."""
        costs = {
            "gemini-2.5-flash": 0.075,
            "gemini-2.5": 1.25,
            "gemini-1.5-pro": 1.25,
            "gemini-1.5-flash": 0.075,
        }
        base_cost = costs.get(model, 0.5)
        return (tokens / 1_000_000) * base_cost
    
    def _estimate_cost_anthropic(self, model: str, tokens: int) -> float:
        """Estimate cost for Anthropic models."""
        costs = {
            "claude-3-5-sonnet-20241022": 3.0,
            "claude-3-opus-20240229": 15.0,
            "claude-3-sonnet-20240229": 3.0,
            "claude-3-haiku-20240307": 0.25,
        }
        base_cost = costs.get(model, 3.0)
        return (tokens / 1_000_000) * base_cost
    
    def _estimate_cost_deepseek(self, model: str, tokens: int) -> float:
        """Estimate cost for DeepSeek models."""
        # DeepSeek pricing (per 1M tokens)
        costs = {
            "deepseek-chat": 0.14,  # $0.14 per 1M input tokens, $0.28 per 1M output tokens
            "deepseek-coder": 0.14,
            "deepseek-reasoner": 0.55,
        }
        # Use average of input/output pricing for simplicity
        base_cost = costs.get(model, 0.21)  # Default to average
        return (tokens / 1_000_000) * base_cost
    
    def _estimate_cost_moonshot(self, model: str, tokens: int) -> float:
        """Estimate cost for Moonshot/Kimi models."""
        # Moonshot pricing (per 1M tokens) - approximate
        costs = {
            "moonshot-v1-8k": 0.012,
            "moonshot-v1-32k": 0.024,
            "moonshot-v1-128k": 0.06,
            "kimi-k2-0711-preview": 0.012,
            "kimi-k2-moonshine": 0.012,
        }
        base_cost = costs.get(model, 0.012)  # Default to cheapest tier
        return (tokens / 1_000_000) * base_cost


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
        provider: LLM provider (openai, gemini, anthropic)
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

