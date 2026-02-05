"""
LLM Providers Integration
Integrates with OpenAI, Anthropic, and Google AI for content generation and analysis
"""

import httpx
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
from ..core.config import settings
from ..core.api_key_manager import api_key_manager

logger = logging.getLogger(__name__)

class LLMProvider:
    """Base class for LLM provider integrations"""
    
    def __init__(self, provider_name: str, api_key: str, base_url: str):
        self.provider_name = provider_name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = 60.0
    
    async def generate_content(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate content using the LLM"""
        raise NotImplementedError("Subclasses must implement generate_content")
    
    async def analyze_trends(
        self,
        trend_data: Dict[str, Any],
        context: str = ""
    ) -> Dict[str, Any]:
        """Analyze trend data and provide insights"""
        raise NotImplementedError("Subclasses must implement analyze_trends")
    
    async def generate_headlines(
        self,
        topic: str,
        count: int = 5,
        style: str = "engaging"
    ) -> List[Dict[str, Any]]:
        """Generate headlines for a topic"""
        raise NotImplementedError("Subclasses must implement generate_headlines")

class OpenAIProvider(LLMProvider):
    """OpenAI GPT integration"""
    
    def __init__(self):
        super().__init__(
            "OpenAI",
            api_key_manager.get_openai_key(),
            "https://api.openai.com/v1"
        )
        self.model = "gpt-4"
    
    async def generate_content(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate content using OpenAI GPT"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/chat/completions"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    **kwargs
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                return {
                    "content": data["choices"][0]["message"]["content"],
                    "usage": data.get("usage", {}),
                    "model": data.get("model", self.model),
                    "provider": self.provider_name,
                    "created_at": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {"error": str(e), "provider": self.provider_name}
    
    async def analyze_trends(
        self,
        trend_data: Dict[str, Any],
        context: str = ""
    ) -> Dict[str, Any]:
        """Analyze trend data using OpenAI"""
        prompt = f"""
        Analyze the following trend data and provide insights:
        
        Trend Data: {trend_data}
        Context: {context}
        
        Please provide:
        1. Key insights about the trend
        2. Potential opportunities
        3. Risk factors
        4. Recommended actions
        5. Market implications
        
        Format your response as a structured analysis.
        """
        
        result = await self.generate_content(prompt, max_tokens=1500, temperature=0.3)
        
        if "error" in result:
            return result
        
        return {
            "analysis": result["content"],
            "trend_data": trend_data,
            "context": context,
            "provider": self.provider_name,
            "created_at": datetime.utcnow().isoformat()
        }
    
    async def generate_headlines(
        self,
        topic: str,
        count: int = 5,
        style: str = "engaging"
    ) -> List[Dict[str, Any]]:
        """Generate headlines using OpenAI"""
        prompt = f"""
        Generate {count} {style} headlines for the topic: {topic}
        
        Requirements:
        - Each headline should be compelling and click-worthy
        - Include different angles (how-to, listicle, question, etc.)
        - Keep headlines between 50-70 characters
        - Make them SEO-friendly
        
        Format as a numbered list.
        """
        
        result = await self.generate_content(prompt, max_tokens=500, temperature=0.8)
        
        if "error" in result:
            return [{"error": result["error"]}]
        
        # Parse headlines from response
        headlines = []
        lines = result["content"].split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering and clean up
                headline = line.split('.', 1)[-1].strip()
                if headline:
                    headlines.append({
                        "headline": headline,
                        "character_count": len(headline),
                        "style": style,
                        "topic": topic,
                        "provider": self.provider_name,
                        "created_at": datetime.utcnow().isoformat()
                    })
        
        return headlines[:count]

class AnthropicProvider(LLMProvider):
    """Anthropic Claude integration"""
    
    def __init__(self):
        super().__init__(
            "Anthropic",
            api_key_manager.get_anthropic_key(),
            "https://api.anthropic.com/v1"
        )
        self.model = "claude-3-sonnet-20240229"
    
    async def generate_content(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate content using Anthropic Claude"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/messages"
                
                headers = {
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                }
                
                payload = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    **kwargs
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                return {
                    "content": data["content"][0]["text"],
                    "usage": data.get("usage", {}),
                    "model": data.get("model", self.model),
                    "provider": self.provider_name,
                    "created_at": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return {"error": str(e), "provider": self.provider_name}
    
    async def analyze_trends(
        self,
        trend_data: Dict[str, Any],
        context: str = ""
    ) -> Dict[str, Any]:
        """Analyze trend data using Anthropic Claude"""
        prompt = f"""
        As an expert trend analyst, analyze this data and provide actionable insights:
        
        Trend Data: {trend_data}
        Context: {context}
        
        Provide a comprehensive analysis including:
        1. Trend interpretation and significance
        2. Market opportunities and threats
        3. Strategic recommendations
        4. Implementation timeline
        5. Success metrics to track
        
        Be specific and actionable in your recommendations.
        """
        
        result = await self.generate_content(prompt, max_tokens=2000, temperature=0.2)
        
        if "error" in result:
            return result
        
        return {
            "analysis": result["content"],
            "trend_data": trend_data,
            "context": context,
            "provider": self.provider_name,
            "created_at": datetime.utcnow().isoformat()
        }
    
    async def generate_headlines(
        self,
        topic: str,
        count: int = 5,
        style: str = "engaging"
    ) -> List[Dict[str, Any]]:
        """Generate headlines using Anthropic Claude"""
        prompt = f"""
        Create {count} {style} headlines for: {topic}
        
        Guidelines:
        - Make each headline unique and compelling
        - Use power words and emotional triggers
        - Ensure headlines are 50-70 characters
        - Include variety in formats (questions, statements, lists)
        - Make them shareable on social media
        
        Present as a clean numbered list.
        """
        
        result = await self.generate_content(prompt, max_tokens=600, temperature=0.7)
        
        if "error" in result:
            return [{"error": result["error"]}]
        
        # Parse headlines from response
        headlines = []
        lines = result["content"].split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                headline = line.split('.', 1)[-1].strip()
                if headline:
                    headlines.append({
                        "headline": headline,
                        "character_count": len(headline),
                        "style": style,
                        "topic": topic,
                        "provider": self.provider_name,
                        "created_at": datetime.utcnow().isoformat()
                    })
        
        return headlines[:count]

class GoogleAIProvider(LLMProvider):
    """Google AI Gemini integration"""
    
    def __init__(self):
        super().__init__(
            "Google AI",
            api_key_manager.get_google_ai_key(),
            "https://generativelanguage.googleapis.com/v1beta"
        )
        self.model = "gemini-pro"
    
    async def generate_content(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate content using Google AI Gemini"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/models/{self.model}:generateContent"
                
                params = {"key": self.api_key}
                
                payload = {
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }],
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature": temperature,
                        **kwargs
                    }
                }
                
                response = await client.post(url, json=payload, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                return {
                    "content": data["candidates"][0]["content"]["parts"][0]["text"],
                    "usage": data.get("usageMetadata", {}),
                    "model": self.model,
                    "provider": self.provider_name,
                    "created_at": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Google AI API error: {e}")
            return {"error": str(e), "provider": self.provider_name}
    
    async def analyze_trends(
        self,
        trend_data: Dict[str, Any],
        context: str = ""
    ) -> Dict[str, Any]:
        """Analyze trend data using Google AI"""
        prompt = f"""
        Analyze this trend data and provide strategic insights:
        
        Data: {trend_data}
        Business Context: {context}
        
        Deliver:
        1. Executive summary of the trend
        2. Key drivers and implications
        3. Competitive landscape analysis
        4. Growth opportunities
        5. Risk assessment and mitigation
        6. Next steps and recommendations
        
        Focus on actionable insights for business growth.
        """
        
        result = await self.generate_content(prompt, max_tokens=1800, temperature=0.3)
        
        if "error" in result:
            return result
        
        return {
            "analysis": result["content"],
            "trend_data": trend_data,
            "context": context,
            "provider": self.provider_name,
            "created_at": datetime.utcnow().isoformat()
        }
    
    async def generate_headlines(
        self,
        topic: str,
        count: int = 5,
        style: str = "engaging"
    ) -> List[Dict[str, Any]]:
        """Generate headlines using Google AI"""
        prompt = f"""
        Generate {count} {style} headlines for: {topic}
        
        Requirements:
        - Each headline should be unique and attention-grabbing
        - Optimize for both search engines and social sharing
        - Use numbers, questions, and emotional triggers
        - Keep length between 50-70 characters
        - Include different formats (how-to, listicle, comparison, etc.)
        
        Format as a numbered list.
        """
        
        result = await self.generate_content(prompt, max_tokens=500, temperature=0.8)
        
        if "error" in result:
            return [{"error": result["error"]}]
        
        # Parse headlines from response
        headlines = []
        lines = result["content"].split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                headline = line.split('.', 1)[-1].strip()
                if headline:
                    headlines.append({
                        "headline": headline,
                        "character_count": len(headline),
                        "style": style,
                        "topic": topic,
                        "provider": self.provider_name,
                        "created_at": datetime.utcnow().isoformat()
                    })
        
        return headlines[:count]

class DeepSeekProvider(LLMProvider):
    """DeepSeek API integration"""
    
    def __init__(self):
        super().__init__(
            "DeepSeek",
            api_key_manager.get_deepseek_key(),
            "https://api.deepseek.com/v1"
        )
        self.model = "deepseek-chat"
    
    async def generate_content(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate content using DeepSeek API"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/chat/completions"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    **kwargs
                }
                
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                return {
                    "content": data["choices"][0]["message"]["content"],
                    "usage": data.get("usage", {}),
                    "model": self.model,
                    "provider": self.provider_name,
                    "created_at": datetime.utcnow().isoformat()
                }
                
        except httpx.HTTPStatusError as e:
            logger.error(f"DeepSeek API error: {e.response.status_code} - {e.response.text}")
            return {
                "error": f"DeepSeek API error: {e.response.status_code}",
                "provider": self.provider_name,
                "created_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"DeepSeek request failed: {str(e)}")
            return {
                "error": f"DeepSeek request failed: {str(e)}",
                "provider": self.provider_name,
                "created_at": datetime.utcnow().isoformat()
            }
    
    async def analyze_trends(
        self,
        trend_data: Dict[str, Any],
        context: str = ""
    ) -> Dict[str, Any]:
        """Analyze trend data using DeepSeek"""
        prompt = f"""
        Analyze the following trend data and provide insights:
        
        Context: {context}
        
        Trend Data: {trend_data}
        
        Please provide:
        1. Key insights and patterns
        2. Potential opportunities
        3. Risks or concerns
        4. Recommendations
        """
        
        result = await self.generate_content(prompt, max_tokens=800, temperature=0.3)
        
        if "error" not in result:
            return {
                "analysis": result["content"],
                "provider": self.provider_name,
                "created_at": datetime.utcnow().isoformat()
            }
        else:
            return result
    
    async def generate_headlines(
        self,
        topic: str,
        count: int = 5,
        style: str = "engaging"
    ) -> List[Dict[str, Any]]:
        """Generate headlines using DeepSeek"""
        prompt = f"""
        Generate {count} {style} headlines for the topic: {topic}
        
        Each headline should be:
        - Compelling and attention-grabbing
        - SEO-friendly
        - Under 60 characters
        - Relevant to the topic
        
        Format as a numbered list.
        """
        
        result = await self.generate_content(prompt, max_tokens=300, temperature=0.8)
        
        if "error" in result:
            return [{"error": result["error"]}]
        
        # Parse headlines from the response
        content = result["content"]
        headlines = []
        
        for line in content.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering/bullets
                headline = line.split('.', 1)[-1].strip()
                if headline:
                    headlines.append({
                        "headline": headline,
                        "provider": self.provider_name,
                        "created_at": datetime.utcnow().isoformat()
                    })
        
        return headlines[:count]

class LLMProvidersManager:
    """Manages all LLM provider integrations"""
    
    def __init__(self):
        self.providers = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "google_ai": GoogleAIProvider(),
            "deepseek": DeepSeekProvider()
        }
    
    async def generate_content_multi_provider(
        self,
        prompt: str,
        providers: Optional[List[str]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate content using multiple providers and compare results"""
        if providers is None:
            providers = list(self.providers.keys())
        
        tasks = []
        for provider_name in providers:
            if provider_name in self.providers:
                task = self.providers[provider_name].generate_content(
                    prompt, max_tokens, temperature
                )
                tasks.append((provider_name, task))
        
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        response = {
            "prompt": prompt,
            "results": {},
            "best_result": None,
            "created_at": datetime.utcnow().isoformat()
        }
        
        best_score = 0
        for i, (provider_name, result) in enumerate(zip([name for name, _ in tasks], results)):
            if isinstance(result, Exception):
                response["results"][provider_name] = {"error": str(result)}
            else:
                response["results"][provider_name] = result
                
                # Simple scoring based on content length and no errors
                if "error" not in result:
                    score = len(result.get("content", ""))
                    if score > best_score:
                        best_score = score
                        response["best_result"] = result
        
        return response
    
    async def analyze_trends_multi_provider(
        self,
        trend_data: Dict[str, Any],
        context: str = "",
        providers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze trends using multiple providers"""
        if providers is None:
            providers = list(self.providers.keys())
        
        tasks = []
        for provider_name in providers:
            if provider_name in self.providers:
                task = self.providers[provider_name].analyze_trends(trend_data, context)
                tasks.append((provider_name, task))
        
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        response = {
            "trend_data": trend_data,
            "context": context,
            "analyses": {},
            "consensus": "",
            "created_at": datetime.utcnow().isoformat()
        }
        
        valid_analyses = []
        for i, (provider_name, result) in enumerate(zip([name for name, _ in tasks], results)):
            if isinstance(result, Exception):
                response["analyses"][provider_name] = {"error": str(result)}
            else:
                response["analyses"][provider_name] = result
                if "error" not in result:
                    valid_analyses.append(result.get("analysis", ""))
        
        # Generate consensus if we have multiple valid analyses
        if len(valid_analyses) > 1:
            consensus_prompt = f"""
            Based on these trend analyses, provide a consensus summary:
            
            {chr(10).join(f"Analysis {i+1}: {analysis}" for i, analysis in enumerate(valid_analyses))}
            
            Identify:
            1. Common themes and insights
            2. Conflicting viewpoints
            3. Overall recommendation
            """
            
            consensus_result = await self.providers["openai"].generate_content(
                consensus_prompt, max_tokens=800, temperature=0.3
            )
            
            if "error" not in consensus_result:
                response["consensus"] = consensus_result["content"]
        
        return response
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers"""
        return list(self.providers.keys())
    
    async def test_providers(self) -> Dict[str, Dict[str, Any]]:
        """Test all providers with a simple prompt"""
        test_prompt = "Generate a one-sentence summary of artificial intelligence trends in 2024."
        results = {}
        
        for provider_name, provider in self.providers.items():
            try:
                result = await provider.generate_content(test_prompt, max_tokens=100)
                results[provider_name] = {
                    "status": "success" if "error" not in result else "error",
                    "response_time": "N/A",  # Could add timing
                    "result": result
                }
            except Exception as e:
                results[provider_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results

# Global instance
llm_providers_manager = LLMProvidersManager()

# Convenience functions
async def generate_content(
    prompt: str,
    provider: str = "openai",
    max_tokens: int = 1000,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Generate content using a specific provider"""
    if provider in llm_providers_manager.providers:
        return await llm_providers_manager.providers[provider].generate_content(
            prompt, max_tokens, temperature
        )
    else:
        return {"error": f"Unknown provider: {provider}"}

async def analyze_trends(
    trend_data: Dict[str, Any],
    provider: str = "openai",
    context: str = ""
) -> Dict[str, Any]:
    """Analyze trends using a specific provider"""
    if provider in llm_providers_manager.providers:
        return await llm_providers_manager.providers[provider].analyze_trends(
            trend_data, context
        )
    else:
        return {"error": f"Unknown provider: {provider}"}

async def generate_headlines(
    topic: str,
    provider: str = "openai",
    count: int = 5,
    style: str = "engaging"
) -> List[Dict[str, Any]]:
    """Generate headlines using a specific provider"""
    if provider in llm_providers_manager.providers:
        return await llm_providers_manager.providers[provider].generate_headlines(
            topic, count, style
        )
    else:
        return [{"error": f"Unknown provider: {provider}"}]

def get_available_providers() -> List[str]:
    """Get list of available LLM providers"""
    return llm_providers_manager.get_available_providers()

async def test_all_providers() -> Dict[str, Dict[str, Any]]:
    """Test all LLM providers"""
    return await llm_providers_manager.test_providers()
