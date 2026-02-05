from typing import Dict, Any, Optional
import aiohttp
import json
import logging
from .llm_provider import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

class GeminiProvider(LLMProvider):
    """Google Gemini Provider"""
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 2048),
                "topP": kwargs.get("top_p", 0.95),
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"Gemini API Error: {response.status} - {text}")
                    raise Exception(f"Gemini API Error: {response.status}")
                
                data = await response.json()
                
                try:
                    content = data["candidates"][0]["content"]["parts"][0]["text"]
                    usage = data.get("usageMetadata", {})
                    # Map usage keys if needed, Gemini uses {promptTokenCount, candidatesTokenCount, totalTokenCount}
                    mapped_usage = {
                        "prompt_tokens": usage.get("promptTokenCount", 0),
                        "completion_tokens": usage.get("candidatesTokenCount", 0),
                        "total_tokens": usage.get("totalTokenCount", 0)
                    }
                    
                    return LLMResponse(
                        content=content,
                        raw_response=data,
                        usage=mapped_usage,
                        model_name=self.model_name,
                        provider="gemini"
                    )
                except (KeyError, IndexError) as e:
                     print(f"DEBUG: Gemini Response Data: {json.dumps(data, indent=2)}")
                     logger.error(f"Gemini Response Parsing Error: {e} - Data: {data}")
                     raise Exception("Failed to parse Gemini response")

class OpenAIProvider(LLMProvider):
    """OpenAI Provider"""
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        base = self.base_url or "https://api.openai.com/v1"
        if "/chat/completions" not in base:
             url = f"{base.rstrip('/')}/chat/completions"
        else:
             url = base
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "top_p": kwargs.get("top_p", 1.0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
            "presence_penalty": kwargs.get("presence_penalty", 0),
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                 if response.status != 200:
                    text = await response.text()
                    logger.error(f"OpenAI API Error: {response.status} - {text}")
                    raise Exception(f"OpenAI API Error: {response.status}")

                 data = await response.json()
                 
                 content = data["choices"][0]["message"]["content"]
                 usage = data.get("usage", {})
                 
                 return LLMResponse(
                    content=content,
                    raw_response=data,
                    usage={
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)
                    },
                    model_name=self.model_name,
                    provider="openai"
                )

class AnthropicProvider(LLMProvider):
    """Anthropic Claude Provider"""

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        url = self.base_url or "https://api.anthropic.com/v1/messages"
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7),
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"Anthropic API Error: {response.status} - {text}")
                    raise Exception(f"Anthropic API Error: {response.status}")
                
                data = await response.json()
                
                content = data["content"][0]["text"]
                usage = data.get("usage", {})

                return LLMResponse(
                    content=content,
                    raw_response=data,
                    usage={
                        "prompt_tokens": usage.get("input_tokens", 0),
                        "completion_tokens": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    },
                    model_name=self.model_name,
                    provider="anthropic"
                )

class DeepSeekProvider(OpenAIProvider):
    """DeepSeek Provider (OpenAI Compatible)"""
    def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
        super().__init__(api_key, model_name, base_url or "https://api.deepseek.com")
        
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        response = await super().generate(prompt, **kwargs)
        response.provider = "deepseek"
        return response

class KimiProvider(OpenAIProvider):
    """Kimi (Moonshot) Provider (OpenAI Compatible)"""
    def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
        super().__init__(api_key, model_name, base_url or "https://api.moonshot.cn/v1")

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        response = await super().generate(prompt, **kwargs)
        response.provider = "kimi"
        return response

def get_provider_class(provider_name: str):
    """Factory to get provider class based on name"""
    normalized_name = provider_name.lower()
    if "google" in normalized_name or "gemini" in normalized_name:
        return GeminiProvider
    elif "openai" in normalized_name:
        return OpenAIProvider
    elif "anthropic" in normalized_name or "claude" in normalized_name:
        return AnthropicProvider
    elif "deepseek" in normalized_name:
        return DeepSeekProvider
    elif "kimi" in normalized_name or "moonshot" in normalized_name:
        return KimiProvider
    else:
        # Default to OpenAI compatible for unknown providers, assuming base_url is set
        return OpenAIProvider
