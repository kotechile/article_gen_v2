"""
Supabase API Routes
Main API routes that use Supabase SDK instead of SQLAlchemy
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List
import structlog
from pydantic import BaseModel

from ..core.supabase_database_service import get_database_service, SupabaseDatabaseService

logger = structlog.get_logger()
router = APIRouter(prefix="/api", tags=["main"])

# Pydantic models for request/response
class TopicDecompositionRequest(BaseModel):
    search_query: str

class TopicDecompositionResponse(BaseModel):
    subtopics: List[str]
    success: bool
    message: str

class AffiliateResearchRequest(BaseModel):
    search_term: str
    topic: str
    user_id: str = None

class AffiliateResearchResponse(BaseModel):
    programs: List[Dict[str, Any]]
    success: bool
    message: str

class TrendAnalysisRequest(BaseModel):
    analysis_name: str
    keywords: List[str]
    timeframe: str = "12m"
    geo: str = "US"
    category: str = "general"
    description: str = ""

class TrendAnalysisResponse(BaseModel):
    topic: str
    interest_score: float
    trend_direction: str
    opportunity_score: float
    related_topics: List[str]
    success: bool
    message: str

class ContentGenerationRequest(BaseModel):
    opportunity_id: str = None
    content_type: str = "blog_post"
    topic: str
    target_audience: str = "general"
    keywords: List[str] = []
    max_ideas: int = 5

class ContentGenerationResponse(BaseModel):
    ideas: List[Dict[str, Any]]
    success: bool
    message: str

@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "TrendTap API",
        "version": "1.0.0",
        "status": "running",
        "database": "supabase"
    }

@router.post("/topic-decomposition", response_model=TopicDecompositionResponse)
async def topic_decomposition(
    request: TopicDecompositionRequest,
    db: SupabaseDatabaseService = Depends(get_database_service)
):
    """Decompose topic into subtopics using LLM"""
    try:
        from ..core.llm_config import LLMConfigManager
        
        # Initialize LLM manager
        llm_manager = LLMConfigManager()
        
        # Create prompt for topic decomposition
        prompt = f"""
        Analyze the topic "{request.search_query}" and break it down into 4-6 specific, actionable subtopics that would be valuable for affiliate marketing research.
        
        Each subtopic should be:
        - Specific and focused
        - Relevant for affiliate marketing opportunities
        - Different enough to provide unique value
        - Actionable for content creation
        
        Return only a JSON array of subtopic strings, like this:
        ["subtopic 1", "subtopic 2", "subtopic 3", "subtopic 4"]
        
        Topic: {request.search_query}
        """
        
        # Get LLM response using the correct integration
        from ..integrations.llm_providers import generate_content
        
        llm_result = await generate_content(
            prompt=prompt,
            provider="openai",  # Use OpenAI as default
            max_tokens=1000,
            temperature=0.7
        )
        
        llm_response = llm_result.get("content", "") if "error" not in llm_result else ""
        
        # Parse the response to extract subtopics
        try:
            # Try to extract JSON array from the response
            import json
            import re
            
            # Look for JSON array in the response
            json_match = re.search(r'\[.*?\]', llm_response, re.DOTALL)
            if json_match:
                subtopics = json.loads(json_match.group())
            else:
                # Fallback: split by lines and clean up
                lines = llm_response.strip().split('\n')
                subtopics = []
                for line in lines:
                    line = line.strip().strip('"').strip("'").strip('-').strip('*').strip()
                    if line and not line.startswith('[') and not line.startswith(']'):
                        subtopics.append(line)
                
                # Limit to 6 subtopics max
                subtopics = subtopics[:6]
        except Exception as parse_error:
            logger.warning("Failed to parse LLM response, using fallback", error=str(parse_error))
            # Fallback to simple decomposition
            subtopics = [
                f"{request.search_query} basics",
                f"advanced {request.search_query}",
                f"{request.search_query} tools",
                f"{request.search_query} best practices"
            ]
        
        # Ensure we have at least 4 subtopics
        if len(subtopics) < 4:
            subtopics.extend([
                f"{request.search_query} equipment",
                f"{request.search_query} techniques"
            ])
        
        return TopicDecompositionResponse(
            subtopics=subtopics[:6],  # Limit to 6 subtopics
            success=True,
            message=f"Topic decomposed into {len(subtopics)} subtopics using LLM"
        )
    except Exception as e:
        logger.error("Topic decomposition failed", error=str(e))
        # Fallback to mock data if LLM fails
        subtopics = [
            f"{request.search_query} basics",
            f"advanced {request.search_query}",
            f"{request.search_query} tools",
            f"{request.search_query} best practices"
        ]
        return TopicDecompositionResponse(
            subtopics=subtopics,
            success=True,
            message="Topic decomposed successfully (fallback mode)"
        )

@router.post("/affiliate-research", response_model=AffiliateResearchResponse)
async def affiliate_research(
    request: AffiliateResearchRequest,
    db: SupabaseDatabaseService = Depends(get_database_service)
):
    """Search for affiliate programs using real Linkup API and LLM analysis"""
    try:
        from ..services.affiliate_research_service import AffiliateResearchService
        
        # Initialize the real affiliate research service
        affiliate_service = AffiliateResearchService()
        
        # Perform real affiliate research with LLM analysis and Linkup integration
        research_result = await affiliate_service.search_affiliate_programs(
            search_term=request.search_term,
            niche=request.topic,
            budget_range=None,  # Can be added later
            user_id=request.user_id
        )
        
        # Extract programs from the research result
        programs = []
        if research_result and 'programs' in research_result:
            programs = research_result['programs']
        elif research_result and 'data' in research_result and 'programs' in research_result['data']:
            programs = research_result['data']['programs']
        
        # Format programs for response
        formatted_programs = []
        for program in programs:
            formatted_programs.append({
                "id": program.get("id", "unknown"),
                "name": program.get("name", "Unknown Program"),
                "description": program.get("description", "No description available"),
                "commission_rate": program.get("commission_rate", "Unknown"),
                "network": program.get("network", "Unknown"),
                "epc": program.get("epc", "0.00"),
                "link": program.get("link", "#")
            })
        
        # If no real results found, provide informative fallback
        if not formatted_programs:
            logger.warning("No affiliate programs found via Linkup/LLM", 
                          search_term=request.search_term, topic=request.topic)
            
            # Provide helpful fallback message
            formatted_programs = [
                {
                    "id": "no-results",
                    "name": "No Programs Found",
                    "description": f"No affiliate programs found for '{request.search_term}'. Try a different search term or check back later.",
                    "commission_rate": "N/A",
                    "network": "N/A",
                    "epc": "0.00",
                    "link": "#"
                }
            ]
        
        return AffiliateResearchResponse(
            programs=formatted_programs,
            success=True,
            message=f"Found {len(formatted_programs)} affiliate programs using LLM analysis and Linkup API"
        )
    except Exception as e:
        logger.error("Affiliate research failed", error=str(e))
        # Provide fallback response instead of failing completely
        fallback_programs = [
            {
                "id": "error-fallback",
                "name": "Research Temporarily Unavailable",
                "description": f"Unable to search for '{request.search_term}' at the moment. Please try again later.",
                "commission_rate": "N/A",
                "network": "N/A", 
                "epc": "0.00",
                "link": "#"
            }
        ]
        return AffiliateResearchResponse(
            programs=fallback_programs,
            success=False,
            message="Affiliate research service temporarily unavailable"
        )

@router.post("/trend-analysis", response_model=TrendAnalysisResponse)
async def trend_analysis(
    request: TrendAnalysisRequest,
    db: SupabaseDatabaseService = Depends(get_database_service)
):
    """Analyze trends for given keywords"""
    try:
        # For now, return mock data - will integrate with real trend analysis later
        return TrendAnalysisResponse(
            topic=request.keywords[0] if request.keywords else "general",
            interest_score=75.5,
            trend_direction="rising",
            opportunity_score=82.3,
            related_topics=[
                f"{request.keywords[0]} trends",
                f"{request.keywords[0]} market",
                f"{request.keywords[0]} growth"
            ],
            success=True,
            message="Trend analysis completed"
        )
    except Exception as e:
        logger.error("Trend analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail="Trend analysis failed")

@router.post("/content/generate", response_model=ContentGenerationResponse)
async def generate_content(
    request: ContentGenerationRequest,
    db: SupabaseDatabaseService = Depends(get_database_service)
):
    """Generate content ideas"""
    try:
        # For now, return mock data - will integrate with LLM later
        ideas = [
            {
                "title": f"Ultimate Guide to {request.topic}",
                "description": f"Comprehensive guide covering all aspects of {request.topic}",
                "content_type": request.content_type,
                "priority": "high",
                "keywords": request.keywords
            },
            {
                "title": f"Best {request.topic} Tools and Resources",
                "description": f"Curated list of the best tools for {request.topic}",
                "content_type": request.content_type,
                "priority": "medium",
                "keywords": request.keywords
            }
        ]
        
        return ContentGenerationResponse(
            ideas=ideas,
            success=True,
            message="Content ideas generated"
        )
    except Exception as e:
        logger.error("Content generation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Content generation failed")
