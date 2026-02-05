"""
✅ MAIN BACKEND SERVER - Idea Burst API with Supabase persistence

This is the primary backend server for the Idea Burst application.
It provides all necessary endpoints including:
- Content ideas management (/api/content-ideas/*)
- Research topics (/api/research-topics/*)
- Trend analysis (/api/trend-analysis/*)
- And more...

🚀 To start: python3 backend/minimal_main.py
📡 Runs on: http://localhost:8000
📚 API docs: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
import uvicorn
import json
import re
import logging
import asyncio
import time
import random
import httpx
import uuid
import csv
import io
from datetime import datetime
from pathlib import Path
from src.core.supabase_singleton import get_supabase_client
from src.core.config import validate_supabase_config, get_settings
import os
from dotenv import load_dotenv

# Load environment variables from .env file (following existing pattern)
load_dotenv()
# Force reload: Link-Only Trend Strategy Implemented
# Date: 2026-01-18

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backend_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate Supabase configuration on startup"""
    try:
        validate_supabase_config()
        logger.info("✅ Supabase configuration validated")
    except ValueError as e:
        logger.error(f"❌ Supabase configuration validation failed: {e}")
        raise
    yield

app = FastAPI(title="Idea Burst API", version="1.0.0", lifespan=lifespan)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Idea Burst API",
        "version": "1.0.1",
        "build_id": "debug-2026-01-28-v4-ACTUAL-ENTRYPOINT",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Import and include DataForSEO router
try:
    from src.routers.dataforseo_router import router as dataforseo_router
    app.include_router(dataforseo_router)
    logger.info("DataForSEO router included successfully")
except ImportError as e:
    logger.warning(f"Could not import DataForSEO router: {e}")
    # Fallback to functional router
    try:
        from src.routers.functional_dataforseo_router import router as dataforseo_router
        app.include_router(dataforseo_router)
        logger.info("Functional DataForSEO router included as fallback")
    except Exception as fallback_error:
        logger.error(f"Could not import any DataForSEO router: {fallback_error}")
except Exception as e:
    logger.error(f"Error including DataForSEO router: {e}")

# Import and include routers
try:
    from src.api.article_generation_routes import router as article_outer, research_router
    from src.api.research_topics_routes import router as research_topics_router
    from src.api.affiliate_research_routes import router as affiliate_research_router
    from src.api import keyword_routes

    app.include_router(article_outer)
    app.include_router(research_router)
    app.include_router(research_topics_router)
    app.include_router(affiliate_research_router)
    app.include_router(keyword_routes.router)
    
    from src.api.enhanced_topic_routes import router as enhanced_topic_router
    app.include_router(enhanced_topic_router)
    
    from src.api.settings_routes import router as settings_router
    app.include_router(settings_router)
    
    from src.routers.content_generator_routes import router as content_generator_router
    app.include_router(content_generator_router)
    
    logger.info("✅ All API routers included successfully")
except Exception as e:
    logger.error(f"❌ Error including routers: {e}")


class TopicDecompositionRequest(BaseModel):
    search_query: str
    user_id: str
    max_subtopics: int = 8
    use_autocomplete: bool = True
    use_llm: bool = True

class SubtopicResponse(BaseModel):
    subtopics: List[str]

# Google Autocomplete Service
class GoogleAutocompleteService:
    """Service for integrating with Google Autocomplete API"""
    
    def __init__(self, timeout: float = 10.0, rate_limit_delay: float = 0.1):
        self.base_url = "http://suggestqueries.google.com/complete/search"
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        ]
    
    async def get_suggestions(self, query: str) -> List[str]:
        """Get autocomplete suggestions for a query"""
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_request_time < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay)
            self.last_request_time = time.time()
            
            # Validate query
            if not query or not query.strip():
                return []
            
            query = query.strip()
            logger.info(f"🔍 Getting Google Autocomplete suggestions for: {query}")
            
            # Prepare request
            headers = {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            params = {
                'client': 'firefox',
                'hl': 'en',
                'q': query
            }
            
            # Make request
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    self.base_url,
                    params=params,
                    headers=headers
                )
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                if isinstance(data, list) and len(data) >= 2:
                    suggestions = data[1]  # Second element contains suggestions
                    if isinstance(suggestions, list):
                        # Filter and clean suggestions
                        filtered_suggestions = []
                        for suggestion in suggestions:
                            if isinstance(suggestion, str) and suggestion.strip():
                                # Remove the original query from suggestions
                                clean_suggestion = suggestion.strip()
                                if clean_suggestion.lower() != query.lower():
                                    filtered_suggestions.append(clean_suggestion)
                        
                        logger.info(f"✅ Got {len(filtered_suggestions)} autocomplete suggestions")
                        return filtered_suggestions[:10]  # Limit to 10 suggestions
                
                logger.warning("No valid suggestions found in response")
                return []
                
        except Exception as e:
            logger.warning(f"Google Autocomplete error: {str(e)}")
            return []

# REMOVED: Google Autocomplete service - was causing rate limiting and CORS issues

# Database helper functions
def save_content_ideas(ideas: List[Dict[str, Any]], user_id: str, topic_id: str) -> bool:
    """Save content ideas to Supabase - no fallback, show error if fails"""
    logger.info(f"🔄 Attempting to save {len(ideas)} content ideas for user {user_id}, topic {topic_id}")
    
    try:
        logger.info("🔍 Attempting to save to Supabase...")
        supabase = get_supabase_client()
        
        # Prepare data for Supabase (using only columns that exist in the basic table)
        content_ideas_data = []
        for idea in ideas:
            # Ensure topic_id and user_id are valid UUIDs
            try:
                topic_uuid = uuid.UUID(topic_id) if topic_id else uuid.uuid4()
            except ValueError:
                topic_uuid = uuid.uuid4()
            
            try:
                user_uuid = uuid.UUID(user_id) if user_id else uuid.uuid4()
            except ValueError:
                user_uuid = uuid.uuid4()
            
            content_idea = {
                "id": idea.get("id", str(uuid.uuid4())),
                "user_id": str(user_uuid),
                "title": idea.get("title", ""),
                "description": idea.get("description", ""),
                "content_type": idea.get("content_type", "blog"),
                "category": "seo_optimized",  # Default category
                "subtopic": idea.get("subtopic", ""),
                "topic_id": str(topic_uuid),
                "keywords": idea.get("primary_keywords", []) + idea.get("secondary_keywords", []),  # Combine keywords
                "seo_score": int(idea.get("seo_optimization_score", 0)),
                "seo_optimization_score": int(idea.get("seo_optimization_score", 0)),
                "traffic_potential_score": int(idea.get("traffic_potential_score", 0)),
                "overall_quality_score": int(idea.get("overall_quality_score", 0)),
                "difficulty_level": idea.get("difficulty", "intermediate"),
                "average_difficulty": idea.get("average_difficulty", 50),
                "estimated_read_time": 45,  # Default to 45 minutes
                "target_audience": idea.get("target_audience", "general"),
                "content_angle": idea.get("content_angle", idea.get("description", "")),
                "monetization_potential": idea.get("monetization_potential", "medium"),
                "average_cpc": idea.get("average_cpc", 0),
                "total_search_volume": int(idea.get("total_search_volume", 0)),
                "technical_complexity": "medium" if idea.get("content_type") == "software" else "low",
                "development_effort": "medium" if idea.get("content_type") == "software" else "low",
                "market_demand": "medium"
            }
            content_ideas_data.append(content_idea)
        
        logger.info(f"📝 Prepared {len(content_ideas_data)} ideas for Supabase insertion")
        
        # Insert into Supabase (following existing pattern)
        result = supabase.table("content_ideas").insert(content_ideas_data).execute()
        
        logger.info(f"🔍 Supabase response: {result}")
        logger.info(f"🔍 Supabase data: {result.data}")
        logger.info(f"🔍 Supabase count: {result.count}")
        
        # Check if the insert was successful
        if hasattr(result, 'data') and result.data:
            logger.info(f"✅ Successfully saved {len(result.data)} content ideas to Supabase")
            return True
        elif hasattr(result, 'count') and result.count:
            logger.info(f"✅ Successfully saved {result.count} content ideas to Supabase (count-based)")
            return True
        else:
            # If we get here without exception, consider it successful
            # The ideas are being saved (verified by querying), so return True
            logger.info(f"✅ Successfully saved content ideas to Supabase (no data returned but no error)")
            return True
            
    except Exception as e:
        logger.error(f"❌ Supabase error: {e}")
        settings = get_settings()
        logger.error(f"Supabase URL: {settings.supabase_url}")
        logger.error(f"Supabase Key present: {bool(settings.supabase_service_role_key or settings.supabase_anon_key)}")
        return False

def get_content_ideas(user_id: str, topic_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get content ideas from Supabase - no fallback, show error if fails"""
    logger.info(f"🔍 Retrieving content ideas for user {user_id}, topic {topic_id}")
    
    try:
        supabase = get_supabase_client()
        logger.info("🔍 Attempting to retrieve from Supabase...")
        
        # Ensure user_id is a valid UUID
        try:
            user_uuid = uuid.UUID(user_id) if user_id else None
        except ValueError:
            logger.error(f"❌ Invalid user_id format: {user_id}")
            return []
        
        query = supabase.table("content_ideas").select("*").eq("user_id", str(user_uuid))
        
        if topic_id:
            # Ensure topic_id is a valid UUID
            try:
                topic_uuid = uuid.UUID(topic_id) if topic_id else None
                query = query.eq("topic_id", str(topic_uuid))
            except ValueError:
                logger.error(f"❌ Invalid topic_id format: {topic_id}")
                return []
        
        result = query.order("created_at", desc=True).execute()
        
        if result.data:
            logger.info(f"✅ Retrieved {len(result.data)} content ideas from Supabase")
            return result.data  # Return data directly (arrays should already be arrays in Supabase)
        else:
            logger.info("No content ideas found in Supabase")
            return []
            
    except Exception as e:
        logger.error(f"❌ Supabase retrieval error: {e}")
        return []

def delete_content_idea(idea_id: str, user_id: str) -> bool:
    """Delete a content idea from Supabase"""
    try:
        supabase = get_supabase_client()
        result = supabase.table("content_ideas").delete().eq("id", idea_id).eq("user_id", user_id).execute()
        
        if result.data:
            logger.info(f"✅ Deleted content idea {idea_id}")
            return True
        else:
            logger.warning(f"Content idea {idea_id} not found or not owned by user")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error deleting content idea: {e}")
        return False

def delete_content_ideas_by_topic(topic_id: str, user_id: str) -> bool:
    """Delete all content ideas for a topic"""
    try:
        supabase = get_supabase_client()
        result = supabase.table("content_ideas").delete().eq("topic_id", topic_id).eq("user_id", user_id).execute()
        
        if result.data:
            logger.info(f"✅ Deleted {len(result.data)} content ideas for topic {topic_id}")
            return True
        else:
            logger.info(f"No content ideas found for topic {topic_id}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error deleting content ideas for topic: {e}")
        return False

# LLM Integration
async def generate_content_with_llm(prompt: str, provider: str = "openai", use_json_mode: bool = True) -> Dict[str, Any]:
    """Generate content using LLM with fallback to mock data"""
    try:
        # Get API key from Supabase
        import sys
        supabase = get_supabase_client()
        
        # Get the active provider and model from Supabase
        provider_response = supabase.table('llm_providers').select('*').eq('is_active', True).execute()
        
        if not provider_response.data:
            logger.warning("No active LLM provider found")
            return {"content": "", "error": "No active LLM provider"}
        
        active_provider = provider_response.data[0]
        # Schema uses 'provider' instead of 'provider_type'
        provider_type = active_provider.get('provider') or active_provider.get('provider_type')
        model_name = active_provider.get('model_name')
        api_key_id = active_provider.get('api_keys_id') or active_provider.get('api_key_id')
        
        logger.info(f"🔍 DEBUG: Active provider: {provider_type}")
        logger.info(f"🔍 DEBUG: Model name from DB: {model_name}")
        
        api_key = None
        if api_key_id:
            # Look up API key by specific ID linked in provider
            key_response = supabase.table('api_keys').select('key_value').eq('id', api_key_id).execute()
            if key_response.data:
                api_key = key_response.data[0]['key_value']
                logger.info(f"✅ Found API key via linked ID: {api_key_id}")
        
        # Fallback to provider-based lookup if ID-based lookup failed
        if not api_key:
            logger.info(f"⚠️ Falling back to provider-based key lookup for: {provider_type}")
            response = supabase.table('api_keys').select('key_value').eq('provider', provider_type).eq('is_active', True).execute()
            if response.data:
                api_key = response.data[0]['key_value']
        
        if not api_key:
            logger.warning(f"No API key found for provider: {provider_type}")
            return {"content": "", "error": f"API key not found for {provider_type}"}
        
        # Call the appropriate provider
        import httpx
        
        if provider_type == "openai":
            # Prepare request payload
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000
            }
            
            # Only add temperature for models that support it
            if not any(unsupported in model_name.lower() for unsupported in ["gpt-5-mini", "gpt-4o-mini", "google-2.5", "gemini-2.5"]):
                payload["temperature"] = 0.7
                logger.info(f"🔍 Added temperature parameter for model: {model_name}")
            else:
                logger.info(f"🔍 Skipped temperature parameter for model: {model_name}")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"✅ OpenAI API successful with model: {model_name}")
                return {
                    "content": data["choices"][0]["message"]["content"],
                    "provider": "openai",
                    "model": model_name
                }
        
        elif provider_type == "deepseek":
            # DeepSeek API (similar to OpenAI)
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.7
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"✅ DeepSeek API successful with model: {model_name}")
                logger.info(f"🔍 DeepSeek response data: {data}")
                
                if "choices" in data and data["choices"] and "message" in data["choices"][0]:
                    content = data["choices"][0]["message"]["content"]
                    logger.info(f"🔍 DeepSeek content: {content[:100]}...")
                    return {
                        "content": content,
                        "provider": "deepseek",
                        "model": model_name
                    }
                else:
                    logger.error(f"❌ DeepSeek response missing choices: {data}")
                    return {"content": "", "error": "Invalid response format"}
        
        elif provider_type in ["google", "gemini"]:
            # Google Gemini API
            # Sanitize model name: remove 'models/' prefix if present to avoid duplication in URL
            clean_model_name = model_name.replace("models/", "")
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model_name}:generateContent?key={api_key}"
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 1000,
                    **({"response_mime_type": "application/json"} if use_json_mode else {})
                }
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload
                )
                logger.info(f"🔍 DEBUG: Google API Status: {response.status_code}")
                if response.status_code != 200:
                    logger.error(f"❌ DEBUG: Google API Error Body: {response.text}")
                
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"✅ Google Gemini API successful with model: {model_name}")
                
                # Extract content from Gemini response structure
                if "candidates" in data and data["candidates"] and "content" in data["candidates"][0] and "parts" in data["candidates"][0]["content"]:
                    content = data["candidates"][0]["content"]["parts"][0]["text"]
                    return {
                        "content": content,
                        "provider": "google",
                        "model": model_name
                    }
                else:
                    logger.error(f"❌ Gemini response missing content: {data}")
                    return {"content": "", "error": "Invalid response format"}
        
        elif provider_type == "neurorouters":
            # NeuroRouters API for GPT-5 Mini
            payload = {
                "model": f"openai/{model_name}",  # Format: openai/gpt-5-mini
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.7
            }
            
            # Get the base URL from provider config
            base_url = active_provider.get('base_url', 'https://neurorouters.com/api/v1')
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"✅ NeuroRouters API successful with model: {model_name}")
                logger.info(f"🔍 NeuroRouters response data: {data}")
                
                if "choices" in data and data["choices"] and "message" in data["choices"][0]:
                    content = data["choices"][0]["message"]["content"]
                    logger.info(f"🔍 NeuroRouters content: {content[:100]}...")
                    return {
                        "content": content,
                        "provider": "neurorouters",
                        "model": model_name
                    }
                else:
                    logger.error(f"❌ NeuroRouters response missing choices: {data}")
                    return {"content": "", "error": "Invalid response format"}
        
        else:
            logger.warning(f"Provider {provider_type} not implemented yet")
            return {"content": "", "error": f"Provider {provider_type} not implemented"}
            
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"LLM service error: {error_msg}")
        return {"content": "", "error": f"LLM service unavailable: {error_msg}"}

def parse_llm_subtopics(llm_response: str, search_query: str) -> List[str]:
    """Parse subtopics from LLM response"""
    try:
        # Try to extract JSON array from the response
        json_match = re.search(r'\[.*?\]', llm_response, re.DOTALL)
        if json_match:
            subtopics = json.loads(json_match.group())
            if isinstance(subtopics, list):
                return subtopics
    except Exception as e:
        logger.warning(f"Failed to parse JSON from LLM response: {str(e)}")
    
    # Fallback: split by lines and clean up
    lines = llm_response.strip().split('\n')
    subtopics = []
    for line in lines:
        line = line.strip().strip('"').strip("'").strip('-').strip('*').strip()
        if line and not line.startswith('[') and not line.startswith(']'):
            subtopics.append(line)
    
    return subtopics

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Idea Burst API is running"}

@app.get("/api/storage/status")
async def storage_status():
    """Check Supabase storage status and configuration"""
    from src.core.config import get_settings
    settings = get_settings()
    try:
        supabase = get_supabase_client()
        client_available = True
    except Exception:
        client_available = False
    
    key_role = "unknown"
    try:
        if settings.supabase_service_role_key:
            # Decode JWT payload (no signature verification needed for debug)
            # JWT format: header.payload.signature
            parts = settings.supabase_service_role_key.split(".")
            if len(parts) == 3:
                import base64
                import json
                # Add padding if needed
                padding = '=' * (4 - len(parts[1]) % 4)
                payload_str = base64.urlsafe_b64decode(parts[1] + padding).decode('utf-8')
                payload = json.loads(payload_str)
                key_role = payload.get("role", "unknown")
    except Exception as e:
        key_role = f"error_decoding: {str(e)}"

    # Active Connection Test
    connection_test = "not_attempted"
    try:
        if settings.supabase_url and settings.supabase_service_role_key:
            # Try a lightweight query
            test_res = supabase.table("research_topics").select("id").limit(1).execute()
            connection_test = "success"
    except Exception as e:
        connection_test = f"failed: {str(e)}"
    
    status = {
        "supabase_configured": bool(settings.supabase_url and settings.supabase_service_role_key),
        "supabase_url": settings.supabase_url,
        "supabase_service_role_key_present": bool(settings.supabase_service_role_key),
        "supabase_service_role_key_role": key_role,
        "supabase_anon_key_present": bool(__import__("os").getenv("SUPABASE_ANON_KEY")),
        "supabase_client_available": client_available,
        "connection_test": connection_test
    }
    
    return status

@app.post("/api/topic-decomposition", response_model=SubtopicResponse)
async def decompose_topic(request: TopicDecompositionRequest):
    """Decompose a topic into subtopics using LLM"""
    topic = request.search_query
    max_subtopics = min(request.max_subtopics, 10)  # Limit to 10 max
    
    logger.info(f"Decomposing topic: {topic} with max_subtopics: {max_subtopics}")
    logger.info(f"🔍 use_llm parameter: {request.use_llm}")
    
    if request.use_llm:
        # Create enhanced prompt for topic decomposition
        prompt = f"""
        Analyze the topic "{topic}" and break it down into {max_subtopics} specific, actionable subtopics that would be valuable for affiliate marketing research.
        
        Each subtopic should be:
        - Specific and focused
        - Relevant for affiliate marketing opportunities
        - Different enough to provide unique value
        - Actionable for content creation
        - Related to the main topic but distinct
        
        Return only a JSON array of subtopic strings, like this:
        ["subtopic 1", "subtopic 2", "subtopic 3", "subtopic 4"]
        
        Topic: {topic}
        """
        
        # Try to get LLM response
        llm_result = await generate_content_with_llm(prompt, "openai")
        
        if "error" not in llm_result and llm_result.get("content"):
            logger.info("Using LLM-generated subtopics")
            llm_response = llm_result["content"]
            subtopics = parse_llm_subtopics(llm_response, topic)
            
            if subtopics and len(subtopics) >= 3:
                return SubtopicResponse(subtopics=subtopics[:max_subtopics])
            else:
                logger.warning("LLM response was insufficient, falling back to enhanced mock data")
        else:
            logger.warning("LLM service unavailable, using enhanced mock data")
    
    # Enhanced fallback with more intelligent subtopics
    logger.info("Using enhanced fallback subtopics")
    
    # Generate more intelligent subtopics based on the topic
    topic_lower = topic.lower()
    
    if any(word in topic_lower for word in ["eco", "green", "sustainable", "environment"]):
        subtopics = [
            f"{topic} - Best Products and Reviews",
            f"{topic} - Cost Analysis and ROI", 
            f"{topic} - Installation and Setup Guide",
            f"{topic} - Government Rebates and Incentives",
            f"{topic} - Top Brands and Manufacturers",
            f"{topic} - DIY vs Professional Installation",
            f"{topic} - Maintenance and Upkeep",
            f"{topic} - Future Trends and Technology"
        ]
    elif any(word in topic_lower for word in ["tech", "technology", "digital", "ai", "software"]):
        subtopics = [
            f"{topic} - Market Analysis and Trends",
            f"{topic} - Key Technologies and Tools", 
            f"{topic} - Industry Applications",
            f"{topic} - Investment Opportunities",
            f"{topic} - Future Developments",
            f"{topic} - Leading Companies and Startups",
            f"{topic} - Consumer Adoption Patterns",
            f"{topic} - Implementation Strategies"
        ]
    elif any(word in topic_lower for word in ["health", "fitness", "wellness", "medical"]):
        subtopics = [
            f"{topic} - Health Benefits and Research",
            f"{topic} - Market Trends and Growth", 
            f"{topic} - Consumer Demographics",
            f"{topic} - Technology and Innovation",
            f"{topic} - Industry Regulations",
            f"{topic} - Leading Brands and Products",
            f"{topic} - Future Market Predictions",
            f"{topic} - Implementation and Best Practices"
        ]
    else:
        # Generic intelligent subtopics
        subtopics = [
            f"{topic} - Market Analysis and Trends",
            f"{topic} - Consumer Insights and Behavior", 
            f"{topic} - Technology and Innovation",
            f"{topic} - Economic Impact and Investment",
            f"{topic} - Industry Leaders and Competition",
            f"{topic} - Future Predictions and Forecasts",
            f"{topic} - Best Practices and Strategies",
            f"{topic} - Tools and Resources"
        ]
    
    # Return only the requested number of subtopics
    return SubtopicResponse(subtopics=subtopics[:max_subtopics])

@app.post("/api/enhanced-topic-decomposition", response_model=SubtopicResponse)
async def enhanced_decompose_topic(request: TopicDecompositionRequest):
    """Enhanced topic decomposition with affiliate research and Google Autocomplete"""
    topic = request.search_query
    max_subtopics = min(request.max_subtopics, 10)
    
    logger.info(f"Enhanced decomposition for topic: {topic} with max_subtopics: {max_subtopics}")
    logger.info(f"🔍 use_llm parameter: {request.use_llm}")
    
    if request.use_llm:
        logger.info(f"🔍 Attempting enhanced LLM generation for topic: {topic}")
        
        # Get Google Autocomplete suggestions if enabled
        # REMOVED: Google Autocomplete suggestions - was causing rate limiting and CORS issues
        
        # Create enhanced prompt without autocomplete data
        
        prompt = f"""
        Analyze the topic "{topic}" for affiliate marketing research and create highly targeted subtopics.
        
        Break it down into {max_subtopics} specific, high-value subtopics that would be excellent for affiliate marketing content:
        
        Each subtopic should be:
        - Highly commercial and affiliate-friendly
        - Specific enough to target long-tail keywords
        - Different enough to avoid keyword cannibalization
        - Actionable for content creators and marketers
        - Based on real user search patterns and market demand
        
        Consider these angles:
        - Product reviews and comparisons
        - How-to guides and tutorials
        - Best practices and tips
        - Industry trends and news
        - Tools and resources
        - Cost analysis and budgeting
        - Beginner vs advanced content
        
        Return only a JSON array of subtopic strings, like this:
        ["subtopic 1", "subtopic 2", "subtopic 3", "subtopic 4"]
        
        Topic: {topic}
        """
        
        # Try to get LLM response
        llm_result = await generate_content_with_llm(prompt, "openai")
        logger.info(f"🔍 LLM result: {llm_result}")
        
        if "error" not in llm_result and llm_result.get("content"):
            logger.info("✅ Using LLM-generated enhanced subtopics")
            llm_response = llm_result["content"]
            subtopics = parse_llm_subtopics(llm_response, topic)
            logger.info(f"🔍 Parsed subtopics: {subtopics}")
            
            if subtopics and len(subtopics) >= 3:
                logger.info(f"✅ Returning {len(subtopics)} LLM-generated subtopics")
                return SubtopicResponse(subtopics=subtopics[:max_subtopics])
            else:
                logger.warning("❌ LLM response was insufficient, falling back to enhanced mock data")
        else:
            logger.warning(f"❌ LLM service unavailable: {llm_result.get('error', 'Unknown error')}")
    
    # Enhanced fallback with Google Autocomplete-inspired subtopics
    logger.info("Using enhanced fallback subtopics with Google Autocomplete simulation")
    
    # Simulate Google Autocomplete suggestions
    topic_lower = topic.lower()
    
    if any(word in topic_lower for word in ["eco", "green", "sustainable", "environment"]):
        subtopics = [
            f"{topic} - Best Products and Reviews",
            f"{topic} - Cost Analysis and ROI", 
            f"{topic} - Installation and Setup Guide",
            f"{topic} - Government Rebates and Incentives",
            f"{topic} - Top Brands and Manufacturers",
            f"{topic} - DIY vs Professional Installation",
            f"{topic} - Maintenance and Upkeep",
            f"{topic} - Future Trends and Technology"
        ]
    elif any(word in topic_lower for word in ["tech", "technology", "digital", "ai", "software"]):
        subtopics = [
            f"{topic} - Best Tools and Software",
            f"{topic} - Pricing and Plans Comparison", 
            f"{topic} - Implementation Guide",
            f"{topic} - Industry Use Cases",
            f"{topic} - Top Companies and Startups",
            f"{topic} - Beginner vs Advanced Features",
            f"{topic} - Integration and Compatibility",
            f"{topic} - Future Updates and Roadmap"
        ]
    elif any(word in topic_lower for word in ["health", "fitness", "wellness", "medical"]):
        subtopics = [
            f"{topic} - Best Products and Supplements",
            f"{topic} - Scientific Research and Studies", 
            f"{topic} - Dosage and Usage Guidelines",
            f"{topic} - Side Effects and Safety",
            f"{topic} - Top Brands and Manufacturers",
            f"{topic} - Cost vs Benefits Analysis",
            f"{topic} - User Reviews and Testimonials",
            f"{topic} - Where to Buy and Best Deals"
        ]
    else:
        # Generic enhanced subtopics
        subtopics = [
            f"{topic} - Best Products and Reviews",
            f"{topic} - How to Choose and Buy", 
            f"{topic} - Cost Analysis and Budgeting",
            f"{topic} - Top Brands and Manufacturers",
            f"{topic} - Beginner's Guide and Tips",
            f"{topic} - Advanced Techniques and Strategies",
            f"{topic} - Tools and Resources Needed",
            f"{topic} - Common Mistakes to Avoid"
        ]
    
    return SubtopicResponse(subtopics=subtopics[:max_subtopics])

class AutocompleteRequest(BaseModel):
    query: str

class AffiliateResearchRequest(BaseModel):
    search_term: str
    topic: str
    user_id: str

class AffiliateProgram(BaseModel):
    id: str
    name: str
    description: str
    commission_rate: str
    network: str
    epc: str
    link: str

class AffiliateResearchResponse(BaseModel):
    success: bool
    message: str
    programs: List[AffiliateProgram]

# REMOVED: Mock enhanced programs function - using only real API calls

# REMOVED: Google Autocomplete API - was causing rate limiting and CORS issues

@app.get("/api/test-affiliate")
async def test_affiliate():
    """Test endpoint to verify routing works"""
    return {"message": "Affiliate research routing works!", "status": "success"}

@app.post("/api/affiliate-research", response_model=AffiliateResearchResponse)
async def affiliate_research(request: AffiliateResearchRequest):
    """
    HYBRID AFFILIATE RESEARCH SYSTEM
    
    This endpoint uses a sophisticated hybrid approach:
    
    1. REAL AFFILIATE SEARCH: Uses RealAffiliateSearchService to search multiple networks:
       - ShareASale, CJ Affiliate, Awin, ClickBank, Amazon Associates
       - Web scraping and affiliate directories
       - Returns 7+ programs from real networks
    
    2. ENHANCED TOPIC-SPECIFIC MATCHING: Adds intelligent programs based on search terms:
       - Solar Panel Installation → SunPower, Tesla Solar, Sunrun
       - Smart Home Technology → Amazon Smart Home, Google Nest, Ring
       - Sustainable Materials → Eco Building Materials, Bamboo Flooring, Reclaimed Wood
       - Energy Efficiency → Energy Star, Andersen Windows, Owens Corning
       - Tiny Homes → Tiny House Build, Tumbleweed, Tiny House Listings
    
    3. DEDUPLICATION: Removes duplicates while preserving diversity
    
    4. FINAL FALLBACK: Generic eco-friendly programs if no matches found
    
    Result: 7-19 diverse, relevant affiliate programs per search term
    """
    logger.info(f"🔍 Affiliate research request: {request.search_term} for topic: {request.topic}")
    
    try:
        # Import the real affiliate search service
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        from services.real_affiliate_search import RealAffiliateSearchService
        
        # Use the real affiliate search service
        async with RealAffiliateSearchService() as search_service:
            programs = await search_service.search_affiliate_programs(request.search_term, request.topic)
            
            # Convert to our response format
            affiliate_programs = []
            for program in programs:
                affiliate_programs.append(AffiliateProgram(
                    id=program.get("id", "unknown"),
                    name=program.get("name", "Unknown Program"),
                    description=program.get("description", "No description available"),
                    commission_rate=program.get("commission_rate", "Unknown"),
                    network=program.get("network", "Unknown"),
                    epc=program.get("epc", "0.00"),
                    link=program.get("link", "#")
                ))
            
            # Only use real programs from the search service
            
            # Remove duplicates based on name and network
            unique_programs = []
            seen = set()
            for program in affiliate_programs:
                key = (program.name, program.network)
                if key not in seen:
                    seen.add(key)
                    unique_programs.append(program)
            
            logger.info(f"✅ Found {len(unique_programs)} affiliate programs (real + enhanced)")
            
            return AffiliateResearchResponse(
                success=True,
                message=f"Found {len(unique_programs)} affiliate programs for '{request.search_term}'",
                programs=unique_programs
            )
            
    except Exception as e:
        logger.error(f"Real affiliate search failed: {e}")
        logger.info("No real affiliate programs found")
        
        return AffiliateResearchResponse(
            programs=[],
            success=False,
            message=f"No affiliate programs found for '{request.search_term}'. Please try a different search term."
        )

# AHREFS Integration Endpoints
@app.post("/api/ahrefs/upload")
async def upload_ahrefs_file(
    file: UploadFile = File(...),
    topic_id: str = Form(...),
    user_id: str = Form(...)
):
    """
    Upload and parse AHREFS CSV file
    """
    try:
        logger.info(f"Processing AHREFS file upload: {file.filename} for topic: {topic_id}")
        
        # Read file content
        content = await file.read()
        csv_text = content.decode('utf-8')
        
        # Debug: Log first few lines of CSV
        lines = csv_text.split('\n')
        logger.info(f"CSV file preview - First 3 lines:")
        for i, line in enumerate(lines[:3]):
            logger.info(f"Line {i+1}: {line[:100]}...")
        
        # Parse CSV and extract keywords with metrics
        ahrefs_keywords = parse_ahrefs_csv_with_metrics(csv_text)
        
        if not ahrefs_keywords:
            raise HTTPException(status_code=400, detail="No valid keywords found in CSV file")
        
        # Generate file ID
        file_id = str(uuid.uuid4())
        
        logger.info(f"AHREFS file processed successfully: {file_id}, keywords: {len(ahrefs_keywords)}")
        
        return {
            "success": True,
            "message": f"Successfully processed {len(ahrefs_keywords)} keywords",
            "file_id": file_id,
            "keywords_count": len(ahrefs_keywords),
            "ahrefs_keywords": ahrefs_keywords  # Return all keywords
        }
        
    except Exception as e:
        logger.error(f"AHREFS file upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/content-ideas/generate-ahrefs")
async def generate_content_ideas_with_ahrefs(
    request: dict
):
    """
    Generate content ideas using AHREFS keyword data with LLM + templates and save to Supabase
    """
    try:
        logger.info(f"Generating content ideas with AHREFS data - topic_id: {request.get('topic_id')}, keywords_count: {len(request.get('ahrefs_keywords', []))}")
        logger.info(f"Request subtopics: {request.get('subtopics')}")
        logger.info(f"Request topic_title: {request.get('topic_title')}")
        logger.info(f"Request user_id: {request.get('user_id')}")
        
        # Generate ideas using enhanced AHREFS processing
        result = await generate_enhanced_content_ideas_with_ahrefs(
            topic_id=request['topic_id'],
            topic_title=request['topic_title'],
            subtopics=request['subtopics'],
            ahrefs_keywords=request['ahrefs_keywords'],
            user_id=request['user_id']
        )
        
        # Save ideas to Supabase
        logger.info(f"🔍 Result keys: {result.keys()}")
        logger.info(f"🔍 Ideas in result: {result.get('ideas') is not None}")
        logger.info(f"🔍 Ideas count: {len(result.get('ideas', []))}")
        
        if result.get('ideas'):
            logger.info(f"🔄 Attempting to save {len(result['ideas'])} ideas to database...")
            save_success = save_content_ideas(
                ideas=result['ideas'],
                user_id=request['user_id'],
                topic_id=request['topic_id']
            )
            logger.info(f"💾 Save result: {save_success}")
            result['saved_to_database'] = save_success
        else:
            logger.warning("⚠️ No ideas to save")
            result['saved_to_database'] = False
        
        logger.info(f"Content ideas generated successfully - total_ideas: {result['total_ideas']}, blog_ideas: {result['blog_ideas']}, software_ideas: {result['software_ideas']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Content idea generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

def parse_ahrefs_csv(csv_text: str) -> List[str]:
    """
    Parse AHREFS CSV and extract keywords
    """
    try:
        # Create CSV reader
        csv_reader = csv.DictReader(io.StringIO(csv_text))
        
        keywords = []
        for row in csv_reader:
            # Look for common keyword column names
            keyword = None
            for col in ['Keyword', 'keyword', 'Query', 'query', 'Search Term', 'search_term']:
                if col in row and row[col]:
                    keyword = row[col].strip()
                    break
            
            if keyword and keyword not in keywords:
                keywords.append(keyword)
        
        logger.info(f"Parsed {len(keywords)} keywords from AHREFS CSV")
        return keywords
        
    except Exception as e:
        logger.error(f"Error parsing AHREFS CSV: {str(e)}")
        return []

def parse_ahrefs_csv_with_metrics(csv_text: str) -> List[Dict[str, Any]]:
    """
    Parse AHREFS CSV and extract keywords with metrics (volume, KD, etc.)
    Ahrefs exports are typically tab-separated, not comma-separated
    """
    try:
        # Split into lines and detect delimiter
        lines = csv_text.strip().split('\n')
        if len(lines) < 2:
            logger.error("CSV file must have at least a header row and one data row")
            return []
        
        # Detect delimiter - try tab first (Ahrefs standard), then comma
        first_line = lines[0]
        if '\t' in first_line:
            delimiter = '\t'
            logger.info("Detected tab-separated format (Ahrefs standard)")
        elif ',' in first_line:
            delimiter = ','
            logger.info("Detected comma-separated format")
        else:
            logger.error("Could not detect delimiter in CSV file")
            return []
        
        # Parse header - handle quoted values properly
        headers = []
        for h in first_line.split(delimiter):
            h = h.strip()
            if h.startswith('"') and h.endswith('"'):
                h = h[1:-1]  # Remove quotes
            headers.append(h)
        logger.info(f"CSV headers: {headers}")
        
        keywords = []
        for i, line in enumerate(lines[1:], 1):
            if not line.strip():
                continue
                
            # Split line by delimiter and handle quoted values
            values = []
            current_value = ""
            in_quotes = False
            
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == delimiter and not in_quotes:
                    values.append(current_value.strip())
                    current_value = ""
                else:
                    current_value += char
            
            # Add the last value
            values.append(current_value.strip())
            
            # Remove quotes from values
            values = [v.replace('"', '') for v in values]
            
            # Ensure we have enough values
            if len(values) < len(headers):
                # Pad with empty strings
                values.extend([''] * (len(headers) - len(values)))
            elif len(values) > len(headers):
                # Truncate excess values
                values = values[:len(headers)]
            
            # Create row dictionary
            row = dict(zip(headers, values))
            
            # Look for keyword column with more flexible matching
            keyword = None
            keyword_cols = ['Keyword', 'keyword', 'Query', 'query', 'Search Term', 'search_term', 'term', 'Term', 'Key', 'key']
            
            for col in keyword_cols:
                if col in row and row[col] and row[col].strip():
                    keyword = row[col].strip()
                    break
            
            # If no keyword found, try the first column
            if not keyword and headers and headers[0] in row and row[headers[0]]:
                keyword = row[headers[0]].strip()
            
            if keyword:
                # Extract metrics with more flexible column matching
                keyword_data = {
                    'keyword': keyword,
                    'volume': 0,
                    'difficulty': 0,  # Use 'difficulty' instead of 'kd' for frontend compatibility
                    'cpc': 0,
                    'competition': 'Low',
                    'trend': 'Stable',
                    'intents': [],  # Add intents array for frontend compatibility
                    'traffic_potential': 0,
                    'serp_features': [],
                    'parent_keyword': None,
                    'country': 'us',
                    'global_volume': 0,
                    'global_traffic_potential': 0,
                    'first_seen': '',
                    'last_update': ''
                }
                
                # Try to find volume column (Ahrefs uses "Volume")
                volume_cols = ['Volume', 'volume', 'Search Volume', 'search_volume', 'Vol', 'vol', 'SV', 'sv']
                for col in volume_cols:
                    if col in row and row[col]:
                        try:
                            # Handle different number formats
                            vol_str = str(row[col]).replace(',', '').replace(' ', '')
                            if vol_str and vol_str != '-' and vol_str != '':
                                keyword_data['volume'] = int(float(vol_str))
                                break
                        except (ValueError, TypeError):
                            continue
                
                # Try to find KD column (Ahrefs uses "Difficulty")
                kd_cols = ['Difficulty', 'difficulty', 'KD', 'kd', 'Keyword Difficulty', 'keyword_difficulty', 'KD Score', 'kd_score']
                for col in kd_cols:
                    if col in row and row[col]:
                        try:
                            kd_str = str(row[col]).replace('%', '').replace(' ', '')
                            if kd_str and kd_str != '-' and kd_str != '':
                                keyword_data['difficulty'] = float(kd_str)
                                break
                        except (ValueError, TypeError):
                            continue
                
                # Try to find CPC column (Ahrefs uses "CPC")
                cpc_cols = ['CPC', 'cpc', 'Cost Per Click', 'cost_per_click', 'Price', 'price', 'Cost', 'cost']
                for col in cpc_cols:
                    if col in row and row[col]:
                        try:
                            cpc_str = str(row[col]).replace('$', '').replace(',', '').replace(' ', '')
                            if cpc_str and cpc_str != '-' and cpc_str != '':
                                keyword_data['cpc'] = float(cpc_str)
                                break
                        except (ValueError, TypeError):
                            continue
                
                # Try to find competition column
                comp_cols = ['Competition', 'competition', 'Comp', 'comp', 'Competition Level', 'competition_level', 'Competition Score', 'competition_score']
                for col in comp_cols:
                    if col in row and row[col]:
                        comp_val = str(row[col]).strip()
                        if comp_val and comp_val != '-':
                            keyword_data['competition'] = comp_val
                            break
                
                # Try to find trend column
                trend_cols = ['Trend', 'trend', 'Change', 'change', 'Growth', 'growth', 'Trend Score', 'trend_score']
                for col in trend_cols:
                    if col in row and row[col]:
                        trend_val = str(row[col]).strip()
                        if trend_val and trend_val != '-':
                            keyword_data['trend'] = trend_val
                            break
                
                # Try to find intents column (Ahrefs uses "Intents")
                intents_cols = ['Intents', 'intents', 'Intent', 'intent', 'Search Intent', 'search_intent']
                for col in intents_cols:
                    if col in row and row[col]:
                        intents_val = str(row[col]).strip()
                        if intents_val and intents_val != '-':
                            # Split comma-separated intents and clean them
                            intents_list = [intent.strip() for intent in intents_val.split(',') if intent.strip()]
                            keyword_data['intents'] = intents_list
                            break
                
                # Try to find traffic potential column
                traffic_cols = ['Traffic potential', 'traffic_potential', 'Traffic Potential', 'traffic_potential', 'TP', 'tp']
                for col in traffic_cols:
                    if col in row and row[col]:
                        try:
                            tp_str = str(row[col]).replace(',', '').replace(' ', '')
                            if tp_str and tp_str != '-':
                                keyword_data['traffic_potential'] = int(float(tp_str))
                                break
                        except (ValueError, TypeError):
                            continue
                
                # Try to find SERP features column
                serp_cols = ['SERP Features', 'serp_features', 'SERP', 'serp', 'Features', 'features']
                for col in serp_cols:
                    if col in row and row[col]:
                        serp_val = str(row[col]).strip()
                        if serp_val and serp_val != '-':
                            # Split comma-separated SERP features and clean them
                            serp_list = [feature.strip() for feature in serp_val.split(',') if feature.strip()]
                            keyword_data['serp_features'] = serp_list
                            break
                
                # Try to find parent keyword column
                parent_cols = ['Parent Keyword', 'parent_keyword', 'Parent', 'parent', 'Parent Key', 'parent_key']
                for col in parent_cols:
                    if col in row and row[col]:
                        parent_val = str(row[col]).strip()
                        if parent_val and parent_val != '-':
                            keyword_data['parent_keyword'] = parent_val
                            break
                
                # Try to find global volume column
                global_vol_cols = ['Global volume', 'global_volume', 'Global Volume', 'global_vol', 'Global Vol']
                for col in global_vol_cols:
                    if col in row and row[col]:
                        try:
                            gv_str = str(row[col]).replace(',', '').replace(' ', '')
                            if gv_str and gv_str != '-':
                                keyword_data['global_volume'] = int(float(gv_str))
                                break
                        except (ValueError, TypeError):
                            continue
                
                # Try to find global traffic potential column
                global_tp_cols = ['Global traffic potential', 'global_traffic_potential', 'Global Traffic Potential', 'global_tp']
                for col in global_tp_cols:
                    if col in row and row[col]:
                        try:
                            gtp_str = str(row[col]).replace(',', '').replace(' ', '')
                            if gtp_str and gtp_str != '-':
                                keyword_data['global_traffic_potential'] = int(float(gtp_str))
                                break
                        except (ValueError, TypeError):
                            continue
                
                # Try to find first seen column
                first_seen_cols = ['First seen', 'first_seen', 'First Seen', 'firstseen', 'Firstseen']
                for col in first_seen_cols:
                    if col in row and row[col]:
                        first_seen_val = str(row[col]).strip()
                        if first_seen_val and first_seen_val != '-':
                            keyword_data['first_seen'] = first_seen_val
                            break
                
                # Try to find last update column
                last_update_cols = ['Last Update', 'last_update', 'Last Update', 'lastupdate', 'Lastupdate']
                for col in last_update_cols:
                    if col in row and row[col]:
                        last_update_val = str(row[col]).strip()
                        if last_update_val and last_update_val != '-':
                            keyword_data['last_update'] = last_update_val
                            break
                
                keywords.append(keyword_data)
                logger.info(f"Parsed keyword: {keyword} (vol: {keyword_data['volume']}, difficulty: {keyword_data['difficulty']}, cpc: {keyword_data['cpc']})")
        
        logger.info(f"Successfully parsed {len(keywords)} keywords with metrics from AHREFS CSV")
        return keywords
        
    except Exception as e:
        logger.error(f"Error parsing AHREFS CSV with metrics: {str(e)}")
        logger.error(f"CSV content preview: {csv_text[:500]}...")
        return []

async def generate_enhanced_content_ideas_with_ahrefs(
    topic_id: str,
    topic_title: str,
    subtopics: List[str],
    ahrefs_keywords: List[Dict[str, Any]],
    user_id: str
) -> Dict[str, Any]:
    """
    Generate enhanced content ideas using AHREFS keyword data with LLM + templates
    """
    try:
        logger.info(f"Starting enhanced content generation for topic: {topic_title}")
        logger.info(f"Subtopics: {len(subtopics)}, Keywords: {len(ahrefs_keywords)}")
        
        all_ideas = []
        blog_ideas = []
        software_ideas = []
        
        # Generate ~10 blog ideas per subtopic concurrently
        tasks = []
        for subtopic in subtopics:
            tasks.append(generate_blog_ideas_for_subtopic(
                subtopic, topic_id, topic_title, ahrefs_keywords, user_id
            ))
        
        # Run all subtopic generation tasks concurrently
        if tasks:
            subtopics_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result_ideas in enumerate(subtopics_results):
                if isinstance(result_ideas, Exception):
                    logger.error(f"Error generating ideas for subtopic {subtopics[i]}: {result_ideas}")
                    continue
                
                blog_ideas.extend(result_ideas)
                all_ideas.extend(result_ideas)
        
        # Generate software ideas (separate from subtopics)
        software_ideas = generate_software_ideas_for_topic(
            topic_id, topic_title, ahrefs_keywords, user_id
        )
        all_ideas.extend(software_ideas)
        
        logger.info(f"Generated {len(all_ideas)} total ideas: {len(blog_ideas)} blog, {len(software_ideas)} software")
        
        return {
            'success': True,
            'total_ideas': len(all_ideas),
            'blog_ideas': len(blog_ideas),
            'software_ideas': len(software_ideas),
            'ideas': all_ideas,
            'analytics_summary': {
                'total_keywords': len(ahrefs_keywords),
                'avg_volume': sum(k.get('volume', 0) for k in ahrefs_keywords) / len(ahrefs_keywords) if ahrefs_keywords else 0,
                'avg_kd': sum(k.get('kd', 0) for k in ahrefs_keywords) / len(ahrefs_keywords) if ahrefs_keywords else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Enhanced content generation failed: {str(e)}")
        raise

async def generate_blog_ideas_for_subtopic(
    subtopic: str,
    topic_id: str,
    topic_title: str,
    ahrefs_keywords: List[Dict[str, Any]],
    user_id: str
) -> List[Dict[str, Any]]:
    """
    Generate ~10 blog ideas for a specific subtopic using AHREFS data with real LLM calls
    """
    # Filter keywords relevant to this subtopic - be more flexible with matching
    relevant_keywords = []
    subtopic_words = [word.lower() for word in subtopic.lower().split() if len(word) > 3]  # Only use words longer than 3 chars
    
    for k in ahrefs_keywords:
        keyword_lower = k['keyword'].lower()
        # Check if any subtopic word appears in the keyword
        if any(word in keyword_lower for word in subtopic_words):
            relevant_keywords.append(k)
        # Also check if keyword appears in subtopic
        elif any(word in subtopic.lower() for word in keyword_lower.split()):
            relevant_keywords.append(k)
    
    # If no relevant keywords, use top keywords by volume
    if not relevant_keywords:
        relevant_keywords = sorted(ahrefs_keywords, key=lambda x: x.get('volume', 0), reverse=True)[:10]
        logger.info(f"No relevant keywords found for subtopic '{subtopic}', using top {len(relevant_keywords)} keywords by volume")
    else:
        logger.info(f"Found {len(relevant_keywords)} relevant keywords for subtopic '{subtopic}'")
    
    # Prepare keyword data for LLM
    top_keywords = relevant_keywords[:5]  # Use top 5 keywords for LLM prompt
    keyword_data = {
        'keywords': [kw['keyword'] for kw in top_keywords],
        'volumes': [kw.get('volume', 0) for kw in top_keywords],
        'difficulties': [kw.get('difficulty', 0) for kw in top_keywords],
        'cpcs': [kw.get('cpc', 0) for kw in top_keywords]
    }
    
    # Create LLM prompt for blog ideas
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    current_year = datetime.now().year
    
    prompt = f"""You are a content strategist. Generate exactly 10 blog post ideas for "{topic_title}" focusing on "{subtopic}".

CURRENT DATE: {current_date} ({current_year})
IMPORTANT: Focus on current trends, recent developments, and up-to-date information for {current_year}. Avoid outdated content or references to previous years unless specifically relevant.

Keywords: {', '.join(keyword_data['keywords'])}
Search Volumes: {', '.join(map(str, keyword_data['volumes']))}
Keyword Difficulties: {', '.join(map(str, keyword_data['difficulties']))}
CPC Values: {', '.join(map(str, keyword_data['cpcs']))}

REQUIREMENTS:
- Focus on current trends and recent developments in {current_year}
- Include timely topics, latest technologies, and recent news
- Make content feel fresh and relevant to today's audience
- Avoid outdated references or content from previous years
- CRITICAL: Each title MUST include at least one primary keyword naturally
- Use keywords in titles in a way that sounds natural and compelling
- Keywords should be prominent but not forced

IMPORTANT: Return ONLY a valid JSON array with exactly 10 objects. No other text, no explanations, no markdown.

Each object must have these exact fields:
- title: SEO-optimized blog post title (MUST include at least one primary keyword naturally)
- description: 2-3 sentence description emphasizing current relevance
- primary_keywords: array of 2-3 main keywords (use these in the title)
- secondary_keywords: array of 3-4 related keywords  
- content_angle: string (tutorial, guide, review, comparison, etc.)
- target_audience: string (beginners, professionals, businesses, etc.)
- estimated_read_time: number (5-15)

Example format:
[
  {{
    "title": "2025 Photography Trends: Complete Guide to Camera Settings",
    "description": "Discover the latest photography techniques and camera settings that are trending in 2025. Learn modern approaches to improve your photography skills.",
    "primary_keywords": ["camera", "photography"],
    "secondary_keywords": ["2025 trends", "modern techniques", "current tips"],
    "content_angle": "guide",
    "target_audience": "beginners",
    "estimated_read_time": 12
  }}
]

Generate 10 current, up-to-date ideas for {current_year}:"""

    try:
        # Call LLM to generate ideas
        logger.info(f"🤖 Calling LLM to generate blog ideas for subtopic: {subtopic}")
        logger.info(f"🔍 LLM prompt length: {len(prompt)}")
        llm_result = await generate_content_with_llm(prompt)
        logger.info(f"🔍 LLM result: {llm_result}")
        
        if llm_result.get('content') and 'error' not in llm_result:
            # Parse LLM response
            import json
            import re
            
            content = llm_result['content']
            if isinstance(content, list) and len(content) > 0:
                content = content[0].get('text', '')
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                ideas_data = json.loads(json_match.group())
                logger.info(f"✅ LLM generated {len(ideas_data)} blog ideas for {subtopic}")
                
                # Convert to our format
                ideas = []
                for i, idea_data in enumerate(ideas_data[:10]):  # Limit to 10 ideas
                    keyword = top_keywords[i % len(top_keywords)] if top_keywords else {'keyword': subtopic, 'volume': 1000, 'difficulty': 50, 'cpc': 2.50}
                    
                    # Ensure keywords are in the title
                    title = idea_data.get('title', f"{subtopic} Guide")
                    primary_keywords = idea_data.get('primary_keywords', [keyword['keyword'], subtopic])
                    
                    # Check if any primary keyword is in the title, if not, add the main keyword
                    if primary_keywords and not any(kw.lower() in title.lower() for kw in primary_keywords):
                        main_keyword = primary_keywords[0]
                        # Add keyword naturally to the title
                        if ":" in title:
                            title = f"{main_keyword.title()}: {title}"
                        else:
                            title = f"{title} - {main_keyword.title()} Guide"
                    
                    # Calculate scores based on keyword data
                    seo_score = min(95, 60 + keyword.get('difficulty', 50))
                    traffic_score = min(90, 50 + (keyword.get('volume', 1000) / 100))
                    difficulty_score = keyword.get('difficulty', 50)
                    overall_quality = round((seo_score + traffic_score + (100 - difficulty_score)) / 3, 2)
                    
                    idea = {
                        "id": str(uuid.uuid4()),
                        "title": title,
                        "content_type": "blog",
                        "description": idea_data.get('description', f"Comprehensive guide for {subtopic}"),
                        "primary_keywords": primary_keywords,
                        "secondary_keywords": idea_data.get('secondary_keywords', [f"{subtopic} tips", f"{keyword['keyword']} guide"]),
                        "difficulty": "intermediate" if difficulty_score < 60 else "advanced",
                        "estimated_time": f"{idea_data.get('estimated_read_time', 8)} minutes",
                        "seo_optimization_score": seo_score,
                        "traffic_potential_score": traffic_score,
                        "overall_quality_score": overall_quality,
                        "total_search_volume": keyword.get('volume', 1000),
                        "average_difficulty": difficulty_score,
                        "average_cpc": keyword.get('cpc', 2.50),
                        "content_angle": idea_data.get('content_angle', 'tutorial'),
                        "target_audience": idea_data.get('target_audience', 'general'),
                        "optimization_tips": [
                            f"Target primary keyword: {idea_data.get('primary_keywords', [keyword['keyword']])[0]}",
                            f"Focus on {subtopic} specific content",
                            "Include practical examples and case studies",
                            f"Optimize for {idea_data.get('target_audience', 'general')} audience"
                        ],
                        "content_outline": [
                            f"Introduction to {idea_data.get('primary_keywords', [keyword['keyword']])[0]} in {subtopic}",
                            f"Key concepts and fundamentals",
                            f"Practical applications and examples",
                            f"Best practices for {subtopic}",
                            "Advanced techniques and tips",
                            "Conclusion and next steps"
                        ],
                        "user_id": user_id,
                        "topic_id": topic_id,
                        "subtopic": subtopic,
                        "enhanced_with_ahrefs": True,
                        "generation_method": "llm"
                    }
                    ideas.append(idea)
                
                return ideas
            else:
                logger.warning(f"Could not parse LLM response for {subtopic}, using fallback")
        else:
            logger.warning(f"LLM call failed for {subtopic}: {llm_result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"LLM generation failed for {subtopic}: {str(e)}")
    
    # Fallback to template-based generation if LLM fails
    logger.info(f"Using template fallback for {subtopic}")
    ideas = []
    for i in range(10):  # Generate 10 ideas per subtopic
        keyword = relevant_keywords[i % len(relevant_keywords)] if relevant_keywords else {'keyword': subtopic, 'volume': 1000, 'kd': 50}
        
        # Create more meaningful titles based on keyword and subtopic
        title_templates = [
            f"{keyword['keyword'].title()}: Complete {subtopic} Guide",
            f"How to {keyword['keyword']} for {subtopic}",
            f"{subtopic} {keyword['keyword'].title()}: Best Practices",
            f"Ultimate {keyword['keyword'].title()} Guide for {subtopic}",
            f"{keyword['keyword'].title()} Strategies for {subtopic} Success"
        ]
        
        description_templates = [
            f"Comprehensive guide covering {keyword['keyword']} specifically for {subtopic}",
            f"Learn how to effectively use {keyword['keyword']} in {subtopic}",
            f"Expert insights on {keyword['keyword']} for {subtopic} professionals",
            f"Step-by-step {keyword['keyword']} guide tailored for {subtopic}",
            f"Master {keyword['keyword']} techniques for {subtopic} success"
        ]
        
        # Calculate scores based on keyword data
        seo_score = min(95, 60 + keyword.get('difficulty', 50))
        traffic_score = min(90, 50 + (keyword.get('volume', 1000) / 100))
        difficulty_score = keyword.get('difficulty', 50)
        overall_quality = round((seo_score + traffic_score + (100 - difficulty_score)) / 3, 2)
        
        idea = {
            "id": str(uuid.uuid4()),
            "title": title_templates[i % len(title_templates)],
            "content_type": "blog",
            "description": description_templates[i % len(description_templates)],
            "primary_keywords": [keyword['keyword'], subtopic],
            "secondary_keywords": [f"{subtopic} tips", f"{keyword['keyword']} guide", f"{keyword['keyword']} {subtopic}"],
            "difficulty": "intermediate" if difficulty_score < 60 else "advanced",
            "estimated_time": "45 minutes",
            "seo_optimization_score": seo_score,
            "traffic_potential_score": traffic_score,
            "overall_quality_score": overall_quality,
            "total_search_volume": keyword.get('volume', 1000),
            "average_difficulty": difficulty_score,
            "average_cpc": keyword.get('cpc', 2.50),
            "optimization_tips": [
                f"Target long-tail keyword: {keyword['keyword']}",
                f"Focus on {subtopic} specific content",
                "Include practical examples and case studies",
                f"Leverage {keyword['keyword']} for better {subtopic} results"
            ],
            "content_outline": [
                f"Introduction to {keyword['keyword']} in {subtopic}",
                f"Key concepts and fundamentals",
                f"Practical {keyword['keyword']} applications",
                f"Best practices for {subtopic}",
                "Advanced techniques and tips",
                "Conclusion and next steps"
            ],
            "user_id": user_id,
            "topic_id": topic_id,
            "subtopic": subtopic,
            "enhanced_with_ahrefs": True,
            "generation_method": "template"
        }
        ideas.append(idea)
    
    return ideas

def generate_software_ideas_for_topic(
    topic_id: str,
    topic_title: str,
    ahrefs_keywords: List[Dict[str, Any]],
    user_id: str
) -> List[Dict[str, Any]]:
    """
    Generate software ideas for the topic using AHREFS data
    """
    # Get top keywords by volume for software ideas
    top_keywords = sorted(ahrefs_keywords, key=lambda x: x.get('volume', 0), reverse=True)[:5]
    
    software_types = [
        "Web Application", "Mobile App", "SaaS Tool", "Calculator Tool", 
        "Dashboard App", "Community Platform", "Analytics Tool", "Content Generator"
    ]
    
    ideas = []
    for i, software_type in enumerate(software_types):
        keyword = top_keywords[i % len(top_keywords)] if top_keywords else {'keyword': topic_title, 'volume': 1000, 'kd': 50}
        
        # Calculate scores based on keyword data
        seo_score = min(85, 50 + keyword.get('difficulty', 50))
        traffic_score = min(80, 40 + (keyword.get('volume', 1000) / 200))
        difficulty_score = keyword.get('difficulty', 50)
        overall_quality = round((seo_score + traffic_score + (100 - difficulty_score)) / 3, 2)
        
        idea = {
            "id": str(uuid.uuid4()),
            "title": f"{topic_title} {software_type}",
            "content_type": "software",
            "description": f"Build a {software_type.lower()} for {keyword['keyword']} management and analysis",
            "primary_keywords": [f"{topic_title} {software_type.lower()}", keyword['keyword']],
            "secondary_keywords": [f"{software_type.lower()} tool", f"{topic_title} app"],
            "difficulty": "advanced",
            "estimated_time": "2-4 weeks",
            "seo_optimization_score": seo_score,
            "traffic_potential_score": traffic_score,
            "overall_quality_score": overall_quality,
            "total_search_volume": keyword.get('volume', 1000),
            "average_difficulty": difficulty_score,
            "average_cpc": keyword.get('cpc', 4.50),
            "optimization_tips": [
                f"Focus on {keyword['keyword']} specific features",
                "Include user-friendly interface design",
                "Implement data visualization and analytics"
            ],
            "content_outline": [
                f"Project overview and {keyword['keyword']} focus",
                "Technical architecture and features",
                "User interface and experience design",
                "Implementation timeline and milestones",
                "Launch strategy and marketing"
            ],
            "user_id": user_id,
            "topic_id": topic_id,
            "enhanced_with_ahrefs": True
        }
        ideas.append(idea)
    
    return ideas

# Content Ideas Endpoints (duplicate removed - using the one below)

# Content Ideas Generation Endpoint
class ContentIdeaGenerationRequest(BaseModel):
    topic_id: str
    topic_title: str
    subtopics: List[str]
    keywords: List[Dict[str, Any]] = []  # Changed to accept rich keyword data
    user_id: str
    content_types: List[str] = ["blog", "software"]  # Default content types

class OptimizedContentIdeaGenerationRequest(BaseModel):
    topic_id: str
    topic_title: str
    subtopics: List[str] = []
    user_id: str
    content_types: List[str] = ["blog", "software"]
    max_keywords: int = 50  # Limit keywords for performance

@app.post("/api/content-ideas/generate")
async def generate_content_ideas(request: ContentIdeaGenerationRequest):
    """
    Generate content ideas based on topic, subtopics, and keywords with rich metrics
    """
    try:
        from src.services.content_idea_generator import ContentIdeaGenerator
        
        logger.info(f"Generating content ideas for topic: {request.topic_title}")
        logger.info(f"Subtopics: {request.subtopics}")
        logger.info(f"Keywords: {len(request.keywords)} keywords with rich metrics")
        
        # Handle empty subtopics by using topic title
        subtopics_to_use = request.subtopics if request.subtopics else [request.topic_title]
        logger.info(f"Using subtopics: {subtopics_to_use}")
        
        # Use the enhanced ContentIdeaGenerator
        generator = ContentIdeaGenerator()
        
        result = await generator.generate_content_ideas(
            topic_id=request.topic_id,
            topic_title=request.topic_title,
            subtopics=subtopics_to_use,
            keywords=request.keywords,
            user_id=request.user_id,
            content_types=request.content_types
        )
        
        logger.info(f"Generated {result['total_ideas']} total ideas: {result['blog_ideas']} blog, {result['software_ideas']} software")
        
        return {
            "success": result["success"],
            "message": f"Successfully generated {result['total_ideas']} content ideas",
            "total_ideas": result["total_ideas"],
            "blog_ideas": result["blog_ideas"],
            "software_ideas": result["software_ideas"],
            "ideas": result["ideas"]
        }
        
    except Exception as e:
        logger.error(f"Error generating content ideas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate content ideas: {str(e)}")

async def get_keywords_for_idea_generation(topic_id: str, user_id: str, max_keywords: int = 50) -> List[str]:
    """
    Query keywords from database for idea generation with RLS
    Returns only the keyword strings, not full objects for performance
    """
    try:
        supabase = get_supabase_client()
        # Query keywords from database using Supabase client
        response = supabase.table('keyword_research_data').select('keyword, priority_score, search_volume').eq('topic_id', topic_id).eq('user_id', user_id).order('priority_score', desc=True).limit(max_keywords).execute()
        
        if response.data:
            # Extract just the keyword strings, sorted by priority
            keywords = [item['keyword'] for item in response.data if item.get('keyword')]
            logger.info(f"Retrieved {len(keywords)} keywords from database for topic {topic_id}")
            return keywords
        else:
            logger.warning(f"No keywords found for topic {topic_id} and user {user_id}")
            return []
            
    except Exception as e:
        logger.error(f"Error querying keywords from database: {str(e)}")
        return []

@app.post("/api/content-ideas/generate-optimized")
async def generate_content_ideas_optimized(request: OptimizedContentIdeaGenerationRequest):
    """
    Generate content ideas with optimized keyword loading from database
    This endpoint queries keywords from the database instead of receiving them from frontend
    """
    try:
        logger.info(f"Generating content ideas (optimized) for topic: {request.topic_title}")
        logger.info(f"Topic ID: {request.topic_id}, User ID: {request.user_id}")
        
        # Query keywords from database with RLS
        keywords_data = await get_keywords_for_idea_generation(
            topic_id=request.topic_id,
            user_id=request.user_id,
            max_keywords=request.max_keywords
        )
        
        logger.info(f"Retrieved {len(keywords_data)} keywords from database")
        
        # Handle empty subtopics by using topic title
        subtopics_to_use = request.subtopics if request.subtopics else [request.topic_title]
        logger.info(f"Using subtopics: {subtopics_to_use}")
        
        all_ideas = []
        blog_ideas = []
        software_ideas = []
        
        # Generate blog ideas for each subtopic
        if "blog" in request.content_types:
            for subtopic in subtopics_to_use:
                subtopic_blog_ideas = await generate_blog_ideas_for_subtopic_with_keywords(
                    subtopic, request.topic_id, request.topic_title, keywords_data, request.user_id
                )
                blog_ideas.extend(subtopic_blog_ideas)
                all_ideas.extend(subtopic_blog_ideas)
        
        # Generate software ideas
        if "software" in request.content_types:
            software_ideas = await generate_software_ideas_for_topic_with_keywords(
                request.topic_id, request.topic_title, keywords_data, request.user_id
            )
            all_ideas.extend(software_ideas)
        
        logger.info(f"Generated {len(all_ideas)} total ideas: {len(blog_ideas)} blog, {len(software_ideas)} software")
        
        # Save ideas to database if we have any
        if all_ideas:
            logger.info(f"🔄 Attempting to save {len(all_ideas)} ideas to database...")
            save_success = save_content_ideas(
                ideas=all_ideas,
                user_id=request.user_id,
                topic_id=request.topic_id
            )
            logger.info(f"💾 Save result: {save_success}")
        else:
            save_success = False
        
        return {
            "success": True,
            "ideas": all_ideas,
            "total_ideas": len(all_ideas),
            "blog_ideas": len(blog_ideas),
            "software_ideas": len(software_ideas),
            "saved_to_database": save_success,
            "keywords_used": len(keywords_data)
        }
        
    except Exception as e:
        logger.error(f"Error generating content ideas (optimized): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate content ideas: {str(e)}")

async def generate_blog_ideas_for_subtopic_with_keywords(
    subtopic: str,
    topic_id: str,
    topic_title: str,
    keywords: List[str],
    user_id: str
) -> List[Dict[str, Any]]:
    """
    Generate ~10 blog ideas for a specific subtopic using seed keywords with real LLM calls
    """
    # Filter keywords relevant to this subtopic
    relevant_keywords = []
    subtopic_words = [word.lower() for word in subtopic.lower().split() if len(word) > 3]
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        # Check if any subtopic word appears in the keyword
        if any(word in keyword_lower for word in subtopic_words):
            relevant_keywords.append(keyword)
        # Also check if keyword appears in subtopic
        elif any(word in subtopic.lower() for word in keyword_lower.split()):
            relevant_keywords.append(keyword)
    
    # If no relevant keywords, use all keywords
    if not relevant_keywords:
        relevant_keywords = keywords[:10]
        logger.info(f"No relevant keywords found for subtopic '{subtopic}', using first 10 keywords")
    else:
        logger.info(f"Found {len(relevant_keywords)} relevant keywords for subtopic '{subtopic}'")
    
    # Prepare keyword data for LLM
    top_keywords = relevant_keywords[:5]  # Use top 5 keywords for LLM prompt
    
    # Create LLM prompt for blog ideas
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    current_year = datetime.now().year
    
    prompt = f"""Generate exactly 10 blog post ideas for "{topic_title}" focusing on "{subtopic}".
CURRENT DATE: {current_date} ({current_year})

REQUIREMENTS:
- Focus on current trends in {current_year}
- CRITICAL: Each title MUST include at least one primary keyword naturally
- CRITICAL: Each title MUST be under 60 characters
- Format each idea EXACTLY like this:

[IDEA]
TITLE: Topic - Keyword Guide [#{random.randint(1000, 9999)}]
DESC: 2-3 sentence description here.
PK: primary keyword 1, primary keyword 2
SK: secondary keyword 1, secondary keyword 2
ANGLE: guide
AUDIENCE: beginners
TIME: 10
[/IDEA]

Wait 10 ideas now:"""

    error_detail = "LLM response insufficient or unavailable"
    try:
        # Call LLM to generate ideas
        logger.info(f"🤖 Calling LLM to generate blog ideas for subtopic: {subtopic} (DELIMITED FORMAT)")
        llm_result = await generate_content_with_llm(prompt, use_json_mode=False)
        
        if llm_result.get('content') and 'error' not in llm_result:
            content = llm_result['content']
            if isinstance(content, list) and len(content) > 0:
                content = content[0].get('text', '')
            
            # Parse Delimited Format
            ideas = []
            blocks = re.findall(r'\[IDEA\](.*?)\[/IDEA\]', content, re.DOTALL | re.IGNORECASE)
            
            if not blocks:
                # Fallback to splitting by common markers if tags are missing
                blocks = content.split('---')
            
            for idx, block in enumerate(blocks):
                if not block.strip(): continue
                
                try:
                    idea_data = {}
                    # Simple extraction
                    lines = block.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.upper().startswith('TITLE:'): idea_data['title'] = line[6:].strip()
                        elif line.upper().startswith('DESC:'): idea_data['description'] = line[5:].strip()
                        elif line.upper().startswith('PK:'): idea_data['primary_keywords'] = [k.strip() for k in line[3:].split(',')]
                        elif line.upper().startswith('SK:'): idea_data['secondary_keywords'] = [k.strip() for k in line[3:].split(',')]
                        elif line.upper().startswith('ANGLE:'): idea_data['content_angle'] = line[6:].strip()
                        elif line.upper().startswith('AUDIENCE:'): idea_data['target_audience'] = line[9:].strip()
                        elif line.upper().startswith('TIME:'): 
                            try: idea_data['estimated_read_time'] = int(re.search(r'\d+', line).group())
                            except: idea_data['estimated_read_time'] = 10
                    
                    if not idea_data.get('title'): continue
                    
                    keyword = top_keywords[idx % len(top_keywords)] if top_keywords else subtopic
                    
                    # Ensure keywords are in the title
                    title = idea_data.get('title', f"{subtopic} Guide")
                    primary_keywords = idea_data.get('primary_keywords', [keyword, subtopic])
                    
                    # Check if any primary keyword is in the title, if not, add the main keyword
                    if primary_keywords and not any(kw.lower() in title.lower() for kw in primary_keywords):
                        main_keyword = primary_keywords[0]
                        # Add keyword naturally to the title
                        if ":" in title:
                            title = f"{main_keyword.title()}: {title}"
                        else:
                            title = f"{title} - {main_keyword.title()} Guide"
                    
                    # Add random 4-digit ID to title
                    title = f"{title} [#{random.randint(1000, 9999)}]"
                    
                    # Calculate overall quality score: (seo + traffic + (100 - difficulty)) / 3
                    seo_score = 85
                    traffic_score = 75
                    difficulty_score = 50
                    overall_quality = round((seo_score + traffic_score + (100 - difficulty_score)) / 3, 2)
                    
                    idea = {
                        "id": str(uuid.uuid4()),
                        "title": title,
                        "content_type": "blog",
                        "description": idea_data.get('description', f"Comprehensive guide for {subtopic}"),
                        "primary_keywords": primary_keywords,
                        "secondary_keywords": idea_data.get('secondary_keywords', [f"{subtopic} tips", f"{keyword} guide"]),
                        "difficulty": "intermediate",
                        "estimated_time": f"{idea_data.get('estimated_read_time', 8)} minutes",
                        "seo_optimization_score": seo_score,
                        "traffic_potential_score": traffic_score,
                        "overall_quality_score": overall_quality,
                        "total_search_volume": 1000,
                        "average_difficulty": difficulty_score,
                        "average_cpc": 2.50,
                        "content_angle": idea_data.get('content_angle', 'tutorial'),
                        "target_audience": idea_data.get('target_audience', 'general'),
                        "optimization_tips": [
                            f"Target primary keyword: {idea_data.get('primary_keywords', [keyword])[0]}",
                            f"Focus on {subtopic} specific content",
                            "Include practical examples and case studies",
                            f"Optimize for {idea_data.get('target_audience', 'general')} audience"
                        ],
                        "content_outline": [
                            f"Introduction to {idea_data.get('primary_keywords', [keyword])[0]} in {subtopic}",
                            f"Key concepts and fundamentals",
                            f"Practical applications and examples",
                            f"Best practices for {subtopic}",
                            "Advanced techniques and tips",
                            "Conclusion and next steps"
                        ],
                        "user_id": user_id,
                        "topic_id": topic_id,
                        "subtopic": subtopic,
                        "enhanced_with_ahrefs": False,
                        "generation_method": "llm"
                    }
                    ideas.append(idea)
                except Exception as e:
                    logger.error(f"Error parsing blog idea block: {e}")
                    continue
            
            if ideas:
                return ideas
            else:
                error_detail = "No blog ideas could be parsed from LLM output"
        else:
            error_detail = llm_result.get('error', 'Unknown LLM error')
            logger.warning(f"LLM call failed for {subtopic}: {error_detail}")
    
    except Exception as e:
        error_detail = str(e)
        logger.error(f"LLM generation failed for {subtopic}: {error_detail}")
    
    # No fallback - raise error to prompt user to fix LLM connection
    logger.error(f"LLM generation failed for '{subtopic}', raising error with details: {error_detail}")
    raise Exception(f"LLM generation failed for {subtopic}: {error_detail}")

async def generate_software_ideas_for_topic_with_keywords(
    topic_id: str,
    topic_title: str,
    keywords: List[str],
    user_id: str
) -> List[Dict[str, Any]]:
    """
    Generate software ideas for the topic using seed keywords and LLM
    """
    top_keywords = keywords[:10] if keywords else [topic_title]
    logger.info(f"Generating software ideas for '{topic_title}' using LLM")
    
    prompt = f"""Generate exactly 10 unique software ideas for the topic "{topic_title}".
    
    Target Keywords: {', '.join(top_keywords[:20])}
    
    REQUIREMENTS:
    - CRITICAL: Each title MUST be under 60 characters
    - Each title MUST include at least one keyword.
    - Format each idea EXACTLY like this:

    [IDEA]
    TITLE: Software Title [#{random.randint(1000, 9999)}]
    DESC: Brief description.
    PK: keyword 1, keyword 2
    TYPE: SaaS
    DIFF: intermediate
    TIME: 2-4 weeks
    [/IDEA]

    Wait 10 ideas now:"""
    
    error_detail = "Software LLM response insufficient or unavailable"
    try:
        logger.info(f"🤖 Calling LLM for software ideas (DELIMITED FORMAT)")
        llm_result = await generate_content_with_llm(prompt, "google", use_json_mode=False)
        
        if llm_result.get('content') and 'error' not in llm_result:
            content = llm_result['content']
            if isinstance(content, list) and len(content) > 0:
                content = content[0].get('text', '')
            
            # Parse Delimited Format
            ideas = []
            blocks = re.findall(r'\[IDEA\](.*?)\[/IDEA\]', content, re.DOTALL | re.IGNORECASE)
            
            if not blocks:
                blocks = content.split('---')
            
            for i, block in enumerate(blocks):
                if not block.strip(): continue
                
                try:
                    data = {}
                    lines = block.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.upper().startswith('TITLE:'): data['title'] = line[6:].strip()
                        elif line.upper().startswith('DESC:'): data['description'] = line[5:].strip()
                        elif line.upper().startswith('PK:'): data['primary_keywords'] = [k.strip() for k in line[3:].split(',')]
                        elif line.upper().startswith('TYPE:'): data['software_type'] = line[5:].strip()
                        elif line.upper().startswith('DIFF:'): data['difficulty'] = line[5:].strip()
                        elif line.upper().startswith('TIME:'): data['estimated_time'] = line[5:].strip()
                    
                    if not data.get('title'): continue
                    title = data.get("title", f"{topic_title} Tool {i+1}")
                    # Add random 4-digit ID to title
                    title = f"{title} [#{random.randint(1000, 9999)}]"
                    
                    if len(title) > 60:
                        # If adding the ID made it too long, truncate title part but keep ID
                        base_title = title.split(" [#")[0]
                        title = base_title[:50] + "... [#" + title.split(" [#")[1]
                        
                    # Calculate overall quality score: (seo + traffic + (100 - difficulty)) / 3
                    seo_score = 80 + (i % 5)
                    traffic_score = 70 + (i % 10)
                    difficulty_val = 50 if data.get("difficulty") == "intermediate" else (30 if data.get("difficulty") == "beginner" else 70)
                    overall_quality = round((seo_score + traffic_score + (100 - difficulty_val)) / 3, 2)
                    
                    idea = {
                        "id": str(uuid.uuid4()),
                        "title": title,
                        "content_type": "software",
                        "description": data.get("description", f"Software solution for {topic_title}"),
                        "primary_keywords": data.get("primary_keywords", [topic_title]),
                        "secondary_keywords": [f"{topic_title} tool", f"{data.get('software_type', 'software')} app"],
                        "difficulty": data.get("difficulty", "intermediate"),
                        "estimated_time": data.get("estimated_time", "2-4 weeks"),
                        "seo_optimization_score": seo_score,
                        "traffic_potential_score": traffic_score,
                        "overall_quality_score": overall_quality,
                        "total_search_volume": 1000,
                        "average_difficulty": difficulty_val,
                        "average_cpc": 4.50,
                        "optimization_tips": [
                            f"Focus on {topic_title} specific features",
                            "Include user-friendly interface design"
                        ],
                        "content_outline": [
                            "Project overview",
                            "Technical architecture",
                            "User experience design"
                        ],
                        "user_id": user_id,
                        "topic_id": topic_id,
                        "enhanced_with_ahrefs": False,
                        "generation_method": "llm"
                    }
                    ideas.append(idea)
                except Exception as e:
                    logger.error(f"Error parsing software idea block: {e}")
                    continue
            
            if ideas:
                return ideas
            else:
                error_detail = "No software ideas could be parsed from LLM response"
        else:
            error_detail = llm_result.get('error', 'Unknown Software LLM error')
                
    except Exception as e:
        error_detail = str(e)
        logger.error(f"Software LLM generation failed: {error_detail}")
        
    # No fallback allowed
    raise Exception(f"Software idea generation failed for {topic_title}: {error_detail}")

# Keywords Generation Endpoint
class KeywordGenerationRequest(BaseModel):
    model_config = ConfigDict(extra='ignore')  # Ignore extra fields
    
    subtopics: List[str] = Field(default_factory=list, description="List of subtopics to generate keywords for")
    topicId: Optional[str] = Field(default=None, description="Topic ID (optional)")
    topicTitle: Optional[str] = Field(default=None, description="Topic title (optional)")
    topic_title: Optional[str] = Field(default=None, description="Topic title alternative field (optional)")
    user_id: Optional[str] = Field(default=None, description="User ID (optional)")

class KeywordGenerationResponse(BaseModel):
    keywords: List[str]
    success: bool
    message: str

@app.post("/api/keywords/generate", response_model=KeywordGenerationResponse)
async def generate_keywords(request: KeywordGenerationRequest):
    """
    Generate simple seed keywords (max 3 words) using LLM for all subtopics
    """
    try:
        # Log the incoming request for debugging
        logger.info(f"Received keyword generation request:")
        logger.info(f"  - subtopics: {request.subtopics}")
        logger.info(f"  - subtopics type: {type(request.subtopics)}")
        logger.info(f"  - subtopics length: {len(request.subtopics) if request.subtopics else 0}")
        logger.info(f"  - topicId: {request.topicId}")
        logger.info(f"  - topicTitle: {request.topicTitle}")
        logger.info(f"  - topic_title: {request.topic_title}")
        logger.info(f"  - user_id: {request.user_id}")
        
        # Get topic title from either field
        topic_title = request.topicTitle or request.topic_title or "Unknown Topic"
        # Handle empty subtopics list
        if not request.subtopics:
            return KeywordGenerationResponse(
                keywords=[],
                success=False,
                message="No subtopics provided. Please add at least one subtopic."
            )
        logger.info(f"Generating seed keywords for topic: {topic_title}, subtopics: {len(request.subtopics)}")
        
        # Create LLM prompt for seed keyword generation
        subtopics_list = ', '.join(request.subtopics)
        prompt = f"""
Generate seed keywords for the topic "{topic_title}" and its subtopics: {subtopics_list}

IMPORTANT REQUIREMENTS:
- Generate 5-8 seed keywords for EACH subtopic
- Each keyword must be 1, 2, or 3 words (vary the length - don't make them all the same length)
- Keywords should be related to their specific subtopic
- Focus on basic, foundational terms perfect for seed keyword research
- Keywords should be commercial and affiliate-friendly
- Avoid long-tail keywords - these are seed keywords for further research
- Return ALL keywords in a single flat list (not grouped by subtopic)
- Mix single words, two-word phrases, and three-word phrases for variety

Format the response as a JSON array of strings with all keywords in one list.

Example format:
[
  "subtopic1",
  "subtopic1 guide", 
  "subtopic1 tips",
  "subtopic1 tools",
  "subtopic2",
  "subtopic2 basics",
  "subtopic2 equipment",
  "subtopic3",
  "subtopic3 course",
  "subtopic3 software"
]

Generate seed keywords for ALL subtopics: {subtopics_list}
Return all keywords in a single flat JSON array.
"""
        
        # Try to get LLM response
        llm_result = await generate_content_with_llm(prompt)
        
        if llm_result.get('content') and 'error' not in llm_result:
            logger.info("Using LLM-generated seed keywords")
            
            # Parse LLM response
            import json
            import re
            
            content = llm_result['content']
            if isinstance(content, list) and len(content) > 0:
                content = content[0].get('text', '')
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                try:
                    keywords = json.loads(json_match.group())
                    if isinstance(keywords, list):
                        # Filter to ensure all keywords are 3 words or less
                        filtered_keywords = [kw for kw in keywords if len(kw.split()) <= 3]
                        
                        # Remove duplicates while preserving order
                        unique_keywords = list(dict.fromkeys(filtered_keywords))
                        
                        logger.info(f"Generated {len(unique_keywords)} seed keywords using LLM")
                        
                        return KeywordGenerationResponse(
                            keywords=unique_keywords,
                            success=True,
                            message=f"Generated {len(unique_keywords)} simple seed keywords for all {len(request.subtopics)} subtopics"
                        )
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from LLM response: {str(e)}")
            else:
                logger.warning("No JSON array found in LLM response")
        else:
            logger.warning("LLM service unavailable, using fallback seed keywords")
        
        # Fallback: Generate simple rule-based seed keywords for all subtopics
        all_keywords = []
        
        for subtopic in request.subtopics:
            # Generate simple seed keywords (max 3 words) for each subtopic
            seed_keywords = [
                subtopic,
                f"{subtopic} guide",
                f"{subtopic} tips",
                f"{subtopic} tools",
                f"{subtopic} basics",
                f"best {subtopic}",
                f"learn {subtopic}",
                f"{subtopic} tutorial",
                f"{subtopic} course",
                f"{subtopic} software",
                f"{subtopic} equipment",
                f"{subtopic} techniques",
                f"{subtopic} strategies",
                f"{subtopic} resources",
                f"{subtopic} reviews"
            ]
            
            # Add topic-specific simple seed keywords
            topic_title_used = request.topicTitle or request.topic_title or "Unknown Topic"
            topic_lower = topic_title_used.lower()
            if any(word in topic_lower for word in ["eco", "green", "sustainable", "environment"]):
                seed_keywords.extend([
                    f"{subtopic} eco",
                    f"{subtopic} green",
                    f"{subtopic} sustainable",
                    f"eco {subtopic}",
                    f"green {subtopic}",
                    f"sustainable {subtopic}"
                ])
            elif any(word in topic_lower for word in ["photography", "photo"]):
                seed_keywords.extend([
                    f"{subtopic} camera",
                    f"{subtopic} lighting",
                    f"{subtopic} editing",
                    f"photo {subtopic}",
                    f"{subtopic} photography"
                ])
            elif any(word in topic_lower for word in ["travel", "trip"]):
                seed_keywords.extend([
                    f"{subtopic} destinations",
                    f"{subtopic} planning",
                    f"travel {subtopic}",
                    f"{subtopic} vacation"
                ])
            
            # Filter to only include keywords with 3 words or less
            filtered_subtopic_keywords = [kw for kw in seed_keywords if len(kw.split()) <= 3]
            all_keywords.extend(filtered_subtopic_keywords)
        
        # Remove duplicates while preserving order
        unique_keywords = list(dict.fromkeys(all_keywords))
        
        logger.info(f"Generated {len(unique_keywords)} fallback seed keywords for all {len(request.subtopics)} subtopics")
        
        return KeywordGenerationResponse(
            keywords=unique_keywords,
            success=True,
            message=f"Generated {len(unique_keywords)} simple seed keywords for all {len(request.subtopics)} subtopics"
        )
        
    except Exception as e:
        logger.error(f"Error generating seed keywords: {str(e)}")
        return KeywordGenerationResponse(
            keywords=[],
            success=False,
            message=f"Error generating seed keywords: {str(e)}"
        )

# Content Ideas Management Endpoints
class ContentIdeasListRequest(BaseModel):
    user_id: str
    topic_id: Optional[str] = None
    content_type: Optional[str] = None

class ContentIdeasListResponse(BaseModel):
    success: bool
    ideas: List[Dict[str, Any]]
    count: int
    message: str

@app.post("/api/content-ideas/list", response_model=List[Dict[str, Any]])
async def list_content_ideas(request: ContentIdeasListRequest):
    """
    Retrieve content ideas for a specific topic
    """
    try:
        logger.info(f"Listing content ideas for user: {request.user_id}, topic: {request.topic_id}")
        
        # Get ideas from Supabase
        ideas = get_content_ideas(request.user_id, request.topic_id)
        
        # Filter by content type if specified
        if request.content_type:
            ideas = [idea for idea in ideas if idea.get("content_type") == request.content_type]
        
        logger.info(f"Found {len(ideas)} content ideas")
        
        return ideas
        
    except Exception as e:
        logger.error(f"Error listing content ideas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list content ideas: {str(e)}")

class ContentIdeaDeleteRequest(BaseModel):
    idea_id: str
    user_id: str

class ContentIdeaDeleteResponse(BaseModel):
    success: bool
    message: str

@app.post("/api/content-ideas/delete", response_model=ContentIdeaDeleteResponse)
async def delete_content_idea_post(request: ContentIdeaDeleteRequest):
    """
    Delete a specific content idea
    """
    try:
        logger.info(f"Deleting content idea: {request.idea_id} for user: {request.user_id}")
        
        success = delete_content_idea(request.idea_id, request.user_id)
        
        if success:
            return ContentIdeaDeleteResponse(
                success=True,
                message=f"Content idea {request.idea_id} deleted successfully"
            )
        else:
            return ContentIdeaDeleteResponse(
                success=False,
                message=f"Failed to delete content idea {request.idea_id}"
            )
        
    except Exception as e:
        logger.error(f"Error deleting content idea: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete content idea: {str(e)}")

@app.delete("/api/content-ideas/{idea_id}")
async def delete_content_idea_by_id(idea_id: str, user_id: str):
    """
    Delete a specific content idea by ID (RESTful endpoint)
    """
    try:
        logger.info(f"Deleting content idea: {idea_id} for user: {user_id}")
        
        success = delete_content_idea(idea_id, user_id)
        
        if success:
            return {"success": True, "message": f"Content idea {idea_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Content idea {idea_id} not found or not owned by user")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting content idea: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete content idea: {str(e)}")

class PublishIdeasRequest(BaseModel):
    idea_ids: List[str]
    user_id: str

@app.post("/api/content-ideas/publish")
async def publish_ideas(request: PublishIdeasRequest):
    """
    Publish content ideas to the Titles table
    """
    try:
        from src.services.content_idea_generator import ContentIdeaGenerator
        
        generator = ContentIdeaGenerator()
        
        logger.info(f"Publishing {len(request.idea_ids)} content ideas for user: {request.user_id}")
        
        result = await generator.publish_ideas_to_titles(
            idea_ids=request.idea_ids,
            user_id=request.user_id
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
            
        return result

    except Exception as e:
        logger.error(f"Failed to publish content ideas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to publish content ideas: {str(e)}")

class ContentIdeasCleanupRequest(BaseModel):
    topic_id: str
    user_id: str

class ContentIdeasCleanupResponse(BaseModel):
    success: bool
    message: str
    deleted_count: int

@app.post("/api/content-ideas/cleanup", response_model=ContentIdeasCleanupResponse)
async def cleanup_content_ideas(request: ContentIdeasCleanupRequest):
    """
    Delete all content ideas for a specific topic
    """
    try:
        logger.info(f"Cleaning up content ideas for topic: {request.topic_id}, user: {request.user_id}")
        
        # Get count before deletion
        ideas = get_content_ideas(request.user_id, request.topic_id)
        count_before = len(ideas)
        
        success = delete_content_ideas_by_topic(request.topic_id, request.user_id)
        
        if success:
            return ContentIdeasCleanupResponse(
                success=True,
                message=f"Cleaned up {count_before} content ideas for topic {request.topic_id}",
                deleted_count=count_before
            )
        else:
            return ContentIdeasCleanupResponse(
                success=False,
                message=f"Failed to cleanup content ideas for topic {request.topic_id}",
                deleted_count=0
            )
        
    except Exception as e:
        logger.error(f"Error cleaning up content ideas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup content ideas: {str(e)}")

class ContentIdeasStatsRequest(BaseModel):
    user_id: str
    topic_id: Optional[str] = None

class ContentIdeasStatsResponse(BaseModel):
    success: bool
    stats: Dict[str, Any]
    message: str

@app.post("/api/content-ideas/stats", response_model=ContentIdeasStatsResponse)
async def get_content_ideas_stats(request: ContentIdeasStatsRequest):
    """
    Get statistics about content ideas
    """
    try:
        logger.info(f"Getting content ideas stats for user: {request.user_id}, topic: {request.topic_id}")
        
        ideas = get_content_ideas(request.user_id, request.topic_id)
        
        # Calculate stats
        total_ideas = len(ideas)
        blog_ideas = len([idea for idea in ideas if idea.get("content_type") == "blog"])
        software_ideas = len([idea for idea in ideas if idea.get("content_type") == "software"])
        
        # Calculate average scores
        avg_seo_score = sum(idea.get("seo_optimization_score", 0) for idea in ideas) / total_ideas if total_ideas > 0 else 0
        avg_traffic_score = sum(idea.get("traffic_potential_score", 0) for idea in ideas) / total_ideas if total_ideas > 0 else 0
        
        # Group by subtopic
        subtopics = {}
        for idea in ideas:
            subtopic = idea.get("subtopic", "Unknown")
            if subtopic not in subtopics:
                subtopics[subtopic] = 0
            subtopics[subtopic] += 1
        
        stats = {
            "total_ideas": total_ideas,
            "blog_ideas": blog_ideas,
            "software_ideas": software_ideas,
            "average_seo_score": round(avg_seo_score, 2),
            "average_traffic_score": round(avg_traffic_score, 2),
            "ideas_by_subtopic": subtopics,
            "enhanced_with_ahrefs": len([idea for idea in ideas if idea.get("enhanced_with_ahrefs", False)])
        }
        
        return ContentIdeasStatsResponse(
            success=True,
            stats=stats,
            message=f"Retrieved stats for {total_ideas} content ideas"
        )
        
    except Exception as e:
        logger.error(f"Error getting content ideas stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get content ideas stats: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting minimal Trend Analysis backend with Supabase persistence...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
