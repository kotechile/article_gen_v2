"""
LLM Provider Configuration
Manages different LLM providers and their settings
"""
import os
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE_AI = "google_ai"
    LOCAL = "local"

@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: LLMProvider
    api_key: str
    model: str
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30
    max_retries: int = 3
    base_url: Optional[str] = None

class LLMConfigManager:
    """Manages LLM provider configurations"""
    
    def __init__(self):
        self.configs: Dict[LLMProvider, LLMConfig] = {}
        self.default_provider: Optional[LLMProvider] = None
        self._load_configurations()
    
    def _load_configurations(self):
        """Load LLM configurations from Supabase API keys or environment variables"""
        from .api_key_manager import api_key_manager
        
        # OpenAI Configuration
        openai_key = api_key_manager.get_openai_key()
        if openai_key:
            self.configs[LLMProvider.OPENAI] = LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key=openai_key,
                model=os.getenv("OPENAI_MODEL", "gpt-4"),
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
                base_url=os.getenv("OPENAI_BASE_URL")
            )
            logger.info("OpenAI configuration loaded")
        
        # Anthropic Configuration
        anthropic_key = api_key_manager.get_anthropic_key()
        if anthropic_key:
            self.configs[LLMProvider.ANTHROPIC] = LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                api_key=anthropic_key,
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "1000")),
                temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7")),
                base_url=os.getenv("ANTHROPIC_BASE_URL")
            )
            logger.info("Anthropic configuration loaded")
        
        # Google AI Configuration
        google_key = api_key_manager.get_google_ai_key()
        if google_key:
            self.configs[LLMProvider.GOOGLE_AI] = LLMConfig(
                provider=LLMProvider.GOOGLE_AI,
                api_key=google_key,
                model=os.getenv("GOOGLE_AI_MODEL", "gemini-pro"),
                max_tokens=int(os.getenv("GOOGLE_AI_MAX_TOKENS", "1000")),
                temperature=float(os.getenv("GOOGLE_AI_TEMPERATURE", "0.7")),
                base_url=os.getenv("GOOGLE_AI_BASE_URL")
            )
            logger.info("Google AI configuration loaded")
        
        # Set default provider
        default_provider_name = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
        try:
            self.default_provider = LLMProvider(default_provider_name)
            if self.default_provider not in self.configs:
                logger.warning(f"Default LLM provider {default_provider_name} not configured")
                self.default_provider = None
        except ValueError:
            logger.warning(f"Invalid default LLM provider: {default_provider_name}")
            self.default_provider = None
        
        # If no default provider is set, use the first available one
        if not self.default_provider and self.configs:
            self.default_provider = list(self.configs.keys())[0]
            logger.info(f"Using first available provider as default: {self.default_provider.value}")
    
    def get_config(self, provider: Optional[LLMProvider] = None) -> Optional[LLMConfig]:
        """Get configuration for a specific provider"""
        if provider is None:
            provider = self.default_provider
        
        if provider is None:
            logger.error("No LLM provider configured")
            return None
        
        config = self.configs.get(provider)
        if config is None:
            logger.error(f"Provider {provider.value} not configured")
            return None
        
        return config
    
    def get_available_providers(self) -> list[LLMProvider]:
        """Get list of available providers"""
        return list(self.configs.keys())
    
    def is_provider_available(self, provider: LLMProvider) -> bool:
        """Check if a provider is available"""
        return provider in self.configs
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about all configured providers"""
        info = {
            "default_provider": self.default_provider.value if self.default_provider else None,
            "available_providers": [p.value for p in self.get_available_providers()],
            "configurations": {}
        }
        
        for provider, config in self.configs.items():
            info["configurations"][provider.value] = {
                "model": config.model,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "timeout": config.timeout,
                "max_retries": config.max_retries
            }
        
        return info

# Global LLM configuration manager
llm_config_manager = LLMConfigManager()

def get_llm_config(provider: Optional[LLMProvider] = None) -> Optional[LLMConfig]:
    """Get LLM configuration for a provider"""
    return llm_config_manager.get_config(provider)

def get_available_llm_providers() -> list[LLMProvider]:
    """Get available LLM providers"""
    return llm_config_manager.get_available_providers()

def is_llm_provider_available(provider: LLMProvider) -> bool:
    """Check if an LLM provider is available"""
    return llm_config_manager.is_provider_available(provider)

def get_llm_provider_info() -> Dict[str, Any]:
    """Get LLM provider information"""
    return llm_config_manager.get_provider_info()
