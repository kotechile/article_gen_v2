from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

class LLMResponse(BaseModel):
    """Standardized response from LLM providers"""
    content: str
    raw_response: Any = None
    usage: Optional[Dict[str, int]] = None  # {prompt_tokens, completion_tokens, total_tokens}
    model_name: str
    provider: str

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific arguments (temperature, max_tokens, etc.)
            
        Returns:
            LLMResponse object
        """
        pass
