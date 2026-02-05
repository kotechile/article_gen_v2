from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from typing import List, Optional, Dict, Any, Tuple
import openai
import anthropic
import google.generativeai as genai
import requests
import json
import time
import logging
from datetime import datetime, timedelta
from enum import Enum

from ..models.llm_config import LLMProvider, LLMConfiguration, LLMUsageLog, LLMProviderTest
from ..core.config import settings

logger = logging.getLogger(__name__)

class LLMProviderType(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    CUSTOM = "custom"

class LLMService:
    def __init__(self, db: Session):
        self.db = db
        self._provider_cache = {}
        self._config_cache = None

    async def analyze_topic_with_llm(
        self, 
        topic: str, 
        provider_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a topic using the specified or default LLM provider
        """
        start_time = time.time()
        
        try:
            # Get provider configuration
            if provider_id:
                provider = self.get_provider_by_id(provider_id)
            else:
                provider = self.get_default_provider()
            
            if not provider:
                raise Exception("No LLM provider available")
            
            # Check rate limits
            if not self._check_rate_limits(provider, user_id):
                raise Exception("Rate limit exceeded")
            
            # Check cost limits
            if not self._check_cost_limits():
                raise Exception("Daily cost limit exceeded")
            
            # Call the appropriate LLM
            result = await self._call_llm_provider(provider, topic)
            
            # Log usage
            self._log_usage(provider.id, user_id, topic, start_time, result)
            
            return result
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            # Try fallback provider if available
            if provider_id:  # Only try fallback if we weren't already using default
                return await self.analyze_topic_with_llm(topic, None, user_id)
            raise

    async def _call_llm_provider(self, provider: LLMProvider, topic: str) -> Dict[str, Any]:
        """Call the specific LLM provider"""
        provider_type = LLMProviderType(provider.provider_type)
        
        if provider_type == LLMProviderType.OPENAI:
            return await self._call_openai(provider, topic)
        elif provider_type == LLMProviderType.ANTHROPIC:
            return await self._call_anthropic(provider, topic)
        elif provider_type == LLMProviderType.GOOGLE:
            return await self._call_google(provider, topic)
        elif provider_type == LLMProviderType.LOCAL:
            return await self._call_local(provider, topic)
        elif provider_type == LLMProviderType.CUSTOM:
            return await self._call_custom(provider, topic)
        else:
            raise Exception(f"Unsupported provider type: {provider.provider_type}")

    async def _call_openai(self, provider: LLMProvider, topic: str) -> Dict[str, Any]:
        """Call OpenAI API"""
        try:
            # Get API key from environment
            api_key = getattr(settings, provider.api_key_env_var, None)
            if not api_key:
                raise Exception(f"API key not found for {provider.name}")
            
            openai.api_key = api_key
            
            # Prepare the prompt
            prompt = self._create_analysis_prompt(topic)
            
            # Make the API call
            response = openai.ChatCompletion.create(
                model=provider.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert affiliate marketing analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=provider.max_tokens,
                temperature=provider.temperature,
                top_p=provider.top_p,
                frequency_penalty=provider.frequency_penalty,
                presence_penalty=provider.presence_penalty
            )
            
            # Parse response
            content = response.choices[0].message.content
            analysis = json.loads(content)
            
            # Update provider stats
            self._update_provider_stats(provider, True, response.usage.total_tokens)
            
            return analysis
            
        except Exception as e:
            self._update_provider_stats(provider, False, 0)
            raise Exception(f"OpenAI API error: {str(e)}")

    async def _call_anthropic(self, provider: LLMProvider, topic: str) -> Dict[str, Any]:
        """Call Anthropic Claude API"""
        try:
            api_key = getattr(settings, provider.api_key_env_var, None)
            if not api_key:
                raise Exception(f"API key not found for {provider.name}")
            
            client = anthropic.Anthropic(api_key=api_key)
            
            prompt = self._create_analysis_prompt(topic)
            
            response = client.messages.create(
                model=provider.model_name,
                max_tokens=provider.max_tokens,
                temperature=provider.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            analysis = json.loads(content)
            
            self._update_provider_stats(provider, True, response.usage.input_tokens + response.usage.output_tokens)
            
            return analysis
            
        except Exception as e:
            self._update_provider_stats(provider, False, 0)
            raise Exception(f"Anthropic API error: {str(e)}")

    async def _call_google(self, provider: LLMProvider, topic: str) -> Dict[str, Any]:
        """Call Google Gemini API"""
        try:
            api_key = getattr(settings, provider.api_key_env_var, None)
            if not api_key:
                raise Exception(f"API key not found for {provider.name}")
            
            genai.configure(api_key=api_key)
            
            # Handle different Gemini model versions
            model_name = provider.model_name
            if model_name.startswith('gemini-2.5'):
                # Use the latest Gemini API for 2.5 models
                model = genai.GenerativeModel(model_name)
            else:
                # Use standard Gemini API for older models
                model = genai.GenerativeModel(model_name)
            
            prompt = self._create_analysis_prompt(topic)
            
            # Configure generation parameters based on model capabilities
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=provider.max_tokens,
                temperature=provider.temperature,
                top_p=provider.top_p
            )
            
            # Add safety settings for newer models
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            content = response.text
            analysis = json.loads(content)
            
            # Estimate tokens (Google doesn't provide exact count)
            estimated_tokens = len(content.split()) * 1.3
            self._update_provider_stats(provider, True, int(estimated_tokens))
            
            return analysis
            
        except Exception as e:
            self._update_provider_stats(provider, False, 0)
            raise Exception(f"Google API error: {str(e)}")

    async def _call_local(self, provider: LLMProvider, topic: str) -> Dict[str, Any]:
        """Call local LLM (Ollama, etc.)"""
        try:
            base_url = provider.base_url or "http://localhost:11434"
            
            prompt = self._create_analysis_prompt(topic)
            
            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": provider.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": provider.temperature,
                        "top_p": provider.top_p,
                        "num_predict": provider.max_tokens
                    }
                },
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"Local LLM error: {response.text}")
            
            result = response.json()
            content = result["response"]
            analysis = json.loads(content)
            
            # Estimate tokens
            estimated_tokens = len(content.split()) * 1.3
            self._update_provider_stats(provider, True, int(estimated_tokens))
            
            return analysis
            
        except Exception as e:
            self._update_provider_stats(provider, False, 0)
            raise Exception(f"Local LLM error: {str(e)}")

    async def _call_custom(self, provider: LLMProvider, topic: str) -> Dict[str, Any]:
        """Call custom LLM API"""
        try:
            base_url = provider.base_url
            api_key = getattr(settings, provider.api_key_env_var, None)
            
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            prompt = self._create_analysis_prompt(topic)
            
            # Use custom configuration
            custom_config = provider.custom_config or {}
            payload = {
                "model": provider.model_name,
                "prompt": prompt,
                "max_tokens": provider.max_tokens,
                "temperature": provider.temperature,
                **custom_config
            }
            
            response = requests.post(
                base_url,
                json=payload,
                headers=headers,
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"Custom API error: {response.text}")
            
            result = response.json()
            analysis = result.get("analysis", result)
            
            # Estimate tokens
            estimated_tokens = len(str(analysis).split()) * 1.3
            self._update_provider_stats(provider, True, int(estimated_tokens))
            
            return analysis
            
        except Exception as e:
            self._update_provider_stats(provider, False, 0)
            raise Exception(f"Custom API error: {str(e)}")

    def _create_analysis_prompt(self, topic: str) -> str:
        """Create standardized prompt for topic analysis"""
        return f"""
        Analyze the topic "{topic}" for affiliate marketing opportunities.
        
        Provide a JSON response with this exact structure:
        {{
            "related_areas": [
                {{"area": "Area Name", "description": "Brief description", "relevance_score": 0.9}}
            ],
            "affiliate_programs": [
                {{"name": "Program Name", "commission": "5-15%", "category": "Category", "difficulty": "Easy", "description": "Description"}}
            ]
        }}
        
        Requirements:
        - Provide 8-10 related areas
        - Provide 6-8 affiliate programs
        - Use real, existing affiliate programs
        - Include realistic commission rates
        - Focus on programs accessible to content creators
        - Ensure JSON is valid and properly formatted
        """

    def get_provider_by_id(self, provider_id: str) -> Optional[LLMProvider]:
        """Get LLM provider by ID"""
        return self.db.query(LLMProvider).filter(
            LLMProvider.id == provider_id,
            LLMProvider.is_active == True
        ).first()

    def get_default_provider(self) -> Optional[LLMProvider]:
        """Get the default LLM provider"""
        return self.db.query(LLMProvider).filter(
            LLMProvider.is_default == True,
            LLMProvider.is_active == True
        ).first()

    def get_all_providers(self) -> List[LLMProvider]:
        """Get all active LLM providers"""
        return self.db.query(LLMProvider).filter(
            LLMProvider.is_active == True
        ).order_by(LLMProvider.priority.desc()).all()

    def create_provider(self, provider_data: Dict[str, Any]) -> LLMProvider:
        """Create a new LLM provider"""
        provider = LLMProvider(**provider_data)
        self.db.add(provider)
        self.db.commit()
        return provider

    def update_provider(self, provider_id: str, updates: Dict[str, Any]) -> bool:
        """Update an LLM provider"""
        try:
            provider = self.get_provider_by_id(provider_id)
            if not provider:
                return False
            
            for key, value in updates.items():
                setattr(provider, key, value)
            
            provider.updated_at = datetime.utcnow()
            self.db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update provider {provider_id}: {str(e)}")
            self.db.rollback()
            return False

    def set_default_provider(self, provider_id: str) -> bool:
        """Set a provider as the default"""
        try:
            # Remove default from all providers
            self.db.query(LLMProvider).update({"is_default": False})
            
            # Set new default
            provider = self.get_provider_by_id(provider_id)
            if provider:
                provider.is_default = True
                self.db.commit()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to set default provider {provider_id}: {str(e)}")
            self.db.rollback()
            return False

    async def test_provider(self, provider_id: str, test_topic: str = "best wireless headphones") -> Dict[str, Any]:
        """Test an LLM provider with a sample topic"""
        try:
            provider = self.get_provider_by_id(provider_id)
            if not provider:
                return {"success": False, "error": "Provider not found"}
            
            start_time = time.time()
            result = await self._call_llm_provider(provider, test_topic)
            response_time = int((time.time() - start_time) * 1000)
            
            # Save test result
            test_result = LLMProviderTest(
                provider_id=provider.id,
                test_topic=test_topic,
                success=True,
                response_time_ms=response_time,
                quality_score=0.8,  # Could be calculated based on response quality
                generated_related_areas=result.get("related_areas", []),
                generated_affiliate_programs=result.get("affiliate_programs", [])
            )
            
            self.db.add(test_result)
            self.db.commit()
            
            return {
                "success": True,
                "response_time_ms": response_time,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Provider test failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_usage_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get LLM usage statistics"""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        stats = self.db.query(
            LLMProvider.name,
            func.count(LLMUsageLog.id).label('total_requests'),
            func.sum(LLMUsageLog.total_tokens).label('total_tokens'),
            func.sum(LLMUsageLog.cost).label('total_cost'),
            func.avg(LLMUsageLog.response_time_ms).label('avg_response_time')
        ).join(
            LLMUsageLog, LLMUsageLog.provider_id == LLMProvider.id
        ).filter(
            LLMUsageLog.request_timestamp >= since_date
        ).group_by(LLMProvider.name).all()
        
        return {
            "period_days": days,
            "providers": [
                {
                    "name": stat.name,
                    "total_requests": stat.total_requests,
                    "total_tokens": stat.total_tokens,
                    "total_cost": float(stat.total_cost or 0),
                    "avg_response_time_ms": float(stat.avg_response_time or 0)
                }
                for stat in stats
            ]
        }

    def _check_rate_limits(self, provider: LLMProvider, user_id: Optional[str]) -> bool:
        """Check if rate limits are exceeded"""
        # Implement rate limiting logic
        return True  # Simplified for now

    def _check_cost_limits(self) -> bool:
        """Check if daily cost limits are exceeded"""
        # Implement cost limiting logic
        return True  # Simplified for now

    def _update_provider_stats(self, provider: LLMProvider, success: bool, tokens: int):
        """Update provider statistics"""
        try:
            provider.total_requests += 1
            if success:
                provider.successful_requests += 1
                provider.total_tokens_used += tokens
                provider.total_cost += (tokens / 1000) * provider.cost_per_1k_tokens
            else:
                provider.failed_requests += 1
            
            provider.last_used = datetime.utcnow()
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to update provider stats: {str(e)}")

    def _log_usage(self, provider_id: str, user_id: Optional[str], topic: str, start_time: float, result: Dict[str, Any]):
        """Log LLM usage for analytics"""
        try:
            response_time = int((time.time() - start_time) * 1000)
            
            usage_log = LLMUsageLog(
                provider_id=provider_id,
                user_id=user_id,
                topic=topic,
                response_time_ms=response_time,
                total_tokens=len(str(result).split()) * 1.3,  # Estimate
                success=True
            )
            
            self.db.add(usage_log)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to log usage: {str(e)}")
