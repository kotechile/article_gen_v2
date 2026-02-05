"""
DataForSEO API Router

This router handles all DataForSEO-related API endpoints for trend analysis and keyword research.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import logging
import asyncio

# Import our DataForSEO models and services
from ..models.trend_data import TrendData
from ..models.keyword_data import KeywordData
from ..dataforseo.database import dataforseo_repository
from ..dataforseo.api_integration import DataForSEOAPIClient
from ..services.dataforseo_service import dataforseo_service
from ..core.supabase_auth import get_current_user

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["DataForSEO"])

# Request models
class TrendAnalysisRequest(BaseModel):
    subtopics: List[str]
    location: str = "United States"
    time_range: str = "12m"

class KeywordIdeasRequest(BaseModel):
    seed_keywords: List[str]
    location_code: int = 2840
    language_code: str = "en"
    limit: int = 100

class RelatedKeywordsRequest(BaseModel):
    keywords: List[str]
    location_code: int = 2840
    language_code: str = "en"
    depth: int = 2  # Changed default to 2 for better results
    limit: int = 10

@router.post("/trend-analysis/dataforseo")
async def get_trend_analysis(request: TrendAnalysisRequest):
    """
    Get trend analysis data for specified subtopics using DataForSEO API.
    
    Args:
        request: TrendAnalysisRequest containing subtopics, location, and time_range
    
    Returns:
        List of trend data for the specified subtopics
    """
    try:
        logger.info(f"🎯 Received trend analysis request")
        logger.info(f"📊 Request subtopics: {request.subtopics}")
        logger.info(f"📊 Request subtopics count: {len(request.subtopics)}")
        logger.info(f"📍 Request location: {request.location}")
        logger.info(f"⏰ Request time range: {request.time_range}")
        
        # Ensure we don't exceed the 5-keyword limit per API call
        if len(request.subtopics) > 5:
            logger.warning(f"⚠️ Received {len(request.subtopics)} subtopics, but DataForSEO API limit is 5. Processing first 5 only.")
            request.subtopics = request.subtopics[:5]
            logger.info(f"📊 Limited subtopics: {request.subtopics}")
        
        # Get trend data from DataForSEO API
        logger.info(f"🌐 Initializing DataForSEO API client...")
        api_client = DataForSEOAPIClient()
        await api_client.initialize()
        logger.info(f"✅ DataForSEO API client initialized")
        
        logger.info(f"🔄 Calling get_trend_data...")
        trend_data = await api_client.get_trend_data(request.subtopics, request.location, request.time_range)
        logger.info(f"✅ get_trend_data completed")
        
        # Log trend data details
        logger.info(f"📊 Trend data received: {len(trend_data)} items")
        logger.info(f"📊 Trend data type: {type(trend_data)}")
        
        if trend_data:
            logger.info(f"📊 First trend data item type: {type(trend_data[0])}")
            logger.info(f"📊 First trend data item keyword: {trend_data[0].keyword if hasattr(trend_data[0], 'keyword') else 'N/A'}")
            
            # Log details of each trend data item
            for i, data in enumerate(trend_data):
                logger.info(f"📊 Trend data {i}: keyword='{data.keyword}', time_series_length={len(data.time_series) if data.time_series else 0}, avg_interest={data.average_interest}, peak_interest={data.peak_interest}")
        else:
            logger.warning(f"⚠️ No trend data received from API client")
        
        # Save to database
        logger.info(f"💾 Saving trend data to database...")
        for i, data in enumerate(trend_data):
            logger.info(f"💾 Saving trend data {i}: {data.keyword}")
            await dataforseo_repository.save_trend_data(data)
        logger.info(f"✅ All trend data saved to database")
        
        logger.info(f"✅ Successfully retrieved {len(trend_data)} trend data points")
        
        # Log final response structure
        logger.info(f"📤 Preparing response...")
        logger.info(f"📊 Response type: {type(trend_data)}")
        logger.info(f"📊 Response length: {len(trend_data)}")
        
        # Verify serialization
        try:
            # Test serialization by converting to dict
            serialized_data = [item.dict() for item in trend_data]
            logger.info(f"✅ Trend data serialization successful")
            logger.info(f"📊 Serialized data length: {len(serialized_data)}")
            
            # Log sample of serialized data
            if serialized_data:
                sample = serialized_data[0]
                logger.info(f"📊 Sample serialized item keys: {list(sample.keys())}")
                logger.info(f"📊 Sample serialized item keyword: {sample.get('keyword', 'N/A')}")
                logger.info(f"📊 Sample serialized item time_series length: {len(sample.get('time_series', []))}")
                logger.info(f"📊 Sample serialized item average_interest: {sample.get('average_interest', 'N/A')}")
                logger.info(f"📊 Sample serialized item peak_interest: {sample.get('peak_interest', 'N/A')}")
        except Exception as serialization_error:
            logger.error(f"❌ Trend data serialization failed: {serialization_error}")
        
        logger.info(f"📤 Returning trend data to frontend...")
        return trend_data
        
    except Exception as e:
        logger.error(f"❌ Error getting trend analysis: {e}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=f"Failed to get trend analysis: {str(e)}")

@router.post("/trend-analysis/dataforseo/compare")
async def compare_subtopics(
    subtopics: List[str],
    location: str = "United States",
    time_range: str = "12m"
):
    """
    Compare multiple subtopics' trend data.
    
    Args:
        subtopics: List of subtopics to compare
        location: Geographic location for analysis
        time_range: Time range for analysis
    
    Returns:
        Comparison data for the specified subtopics
    """
    try:
        logger.info(f"Comparing subtopics: {subtopics}")
        
        # Get trend data for all subtopics
        api_client = DataForSEOAPIClient()
        await api_client.initialize()
        trend_data = await api_client.get_trend_data(subtopics, location, time_range)
        
        # Save to database
        for data in trend_data:
            await dataforseo_repository.save_trend_data(data)
        
        # Create comparison data
        comparison_data = {
            "subtopics": subtopics,
            "location": location,
            "time_range": time_range,
            "data": trend_data,
            "comparison_metrics": {
                "highest_interest": max(trend_data, key=lambda x: x.average_interest) if trend_data else None,
                "lowest_interest": min(trend_data, key=lambda x: x.average_interest) if trend_data else None,
                "total_subtopics": len(trend_data)
            }
        }
        
        logger.info(f"Successfully compared {len(trend_data)} subtopics")
        return comparison_data
        
    except Exception as e:
        logger.error(f"Error comparing subtopics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compare subtopics: {str(e)}")

@router.post("/trend-analysis/dataforseo/suggestions")
async def get_trending_suggestions(
    topic: str,
    location: str = "United States",
    time_range: str = "12m",
    max_suggestions: int = 10
):
    """
    Get trending subtopic suggestions for a given topic.
    
    Args:
        topic: Base topic to find suggestions for
        location: Geographic location for analysis
        time_range: Time range for analysis
        max_suggestions: Maximum number of suggestions to return
    
    Returns:
        List of trending subtopic suggestions
    """
    try:
        logger.info(f"Getting trending suggestions for topic: {topic}")
        
        # Get related topics and queries
        api_client = DataForSEOAPIClient()
        await api_client.initialize()
        suggestions = await api_client.get_suggestions([topic], location, max_suggestions)
        
        # Save suggestions to database
        for suggestion in suggestions:
            await dataforseo_repository.save_subtopic_suggestion(suggestion)
        
        logger.info(f"Successfully retrieved {len(suggestions)} trending suggestions")
        return suggestions
        
    except Exception as e:
        logger.error(f"Error getting trending suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get trending suggestions: {str(e)}")

@router.post("/keyword-research/dataforseo")
async def research_keywords(
    seed_keywords: List[str],
    max_difficulty: int = 50,
    max_keywords: int = 20,
    location: str = "United States"
):
    """
    Research keywords using DataForSEO Labs API with 2-call approach.
    
    Args:
        seed_keywords: List of seed keywords to research
        max_difficulty: Maximum keyword difficulty (0-100)
        max_keywords: Maximum number of keywords to return
        location: Geographic location for analysis
    
    Returns:
        List of researched keywords with metrics
    """
    try:
        logger.info(f"Researching keywords for seeds: {seed_keywords}")
        logger.info(f"Max difficulty: {max_difficulty}, Max keywords: {max_keywords}, Location: {location}")
        
        # Initialize the new DataForSEO service
        if not dataforseo_service.client:
            await dataforseo_service.initialize()
        
        # Get keyword ideas using the new 2-call service
        keyword_ideas = await dataforseo_service.get_keyword_ideas(
            seed_keywords=seed_keywords,
            location_code=2840,  # United States
            language_code="en",
            limit=max_keywords
        )
        
        # Get related keywords for additional context
        related_keywords = await dataforseo_service.get_related_keywords(
            keywords=seed_keywords,
            location_code=2840,  # United States
            language_code="en",
            depth=1,
            limit=5
        )
        
        # Combine and process the data
        all_keywords = []
        
        # Process keyword ideas
        for item in keyword_ideas:
            all_keywords.append({
                "keyword": item.keyword,
                "search_volume": item.search_volume,
                "keyword_difficulty": item.keyword_difficulty,
                "cpc": item.cpc,
                "competition_value": int(float(item.competition) * 100) if item.competition is not None else 0,
                "trend_percentage": 0,
                "intent_type": item.intent_type.value if item.intent_type else "COMMERCIAL",
                "priority_score": 100,  # Default priority
                "related_keywords": item.related_keywords or [],
                "search_volume_trend": {
                    "monthly": 0,
                    "quarterly": 0,
                    "yearly": 0
                },
                "created_at": item.created_at.isoformat() if item.created_at else None
            })
        
        # Process related keywords
        for item in related_keywords:
            # Check if this keyword already exists
            existing_keyword = next((k for k in all_keywords if k["keyword"] == item.keyword), None)
            if existing_keyword:
                # Add to related keywords
                if not existing_keyword["related_keywords"]:
                    existing_keyword["related_keywords"] = []
                existing_keyword["related_keywords"].extend(item.related_keywords or [])
            else:
                # Create new keyword entry
                all_keywords.append({
                    "keyword": item.keyword,
                    "search_volume": item.search_volume,
                    "keyword_difficulty": item.keyword_difficulty,
                    "cpc": item.cpc,
                    "competition_value": int(float(item.competition) * 100) if item.competition is not None else 0,
                    "trend_percentage": 0,
                    "intent_type": item.intent_type.value if item.intent_type else "INFORMATIONAL",
                    "priority_score": 50,  # Lower priority for related keywords
                    "related_keywords": item.related_keywords or [],
                    "search_volume_trend": {
                        "monthly": 0,
                        "quarterly": 0,
                        "yearly": 0
                    },
                    "created_at": item.created_at.isoformat() if item.created_at else None
                })
        
        # Filter by max_difficulty if specified
        if max_difficulty < 100:
            all_keywords = [k for k in all_keywords if (k.get("keyword_difficulty", 0) or 0) <= max_difficulty]
        
        # Limit results
        all_keywords = all_keywords[:max_keywords]
        
        # Save to database
        for keyword_data in all_keywords:
            # Convert to KeywordData object for database storage
            from ..models.keyword_data import KeywordData, IntentType, CompetitionLevel
            from datetime import datetime
            
            keyword_obj = KeywordData(
                keyword=keyword_data["keyword"],
                search_volume=keyword_data["search_volume"],
                keyword_difficulty=keyword_data["keyword_difficulty"],
                cpc=keyword_data["cpc"],
                competition_value=keyword_data["competition_value"],
                trend_percentage=keyword_data["trend_percentage"],
                intent_type=IntentType(keyword_data["intent_type"]) if keyword_data["intent_type"] in ["COMMERCIAL", "INFORMATIONAL", "TRANSACTIONAL"] else IntentType.COMMERCIAL,
                priority_score=keyword_data["priority_score"],
                related_keywords=keyword_data["related_keywords"],
                search_volume_trend=keyword_data["search_volume_trend"],
                created_at=datetime.fromisoformat(keyword_data["created_at"]) if keyword_data["created_at"] else datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            await dataforseo_repository.save_keyword_data(keyword_obj)
        
        logger.info(f"Successfully researched {len(all_keywords)} keywords")
        return all_keywords
        
    except Exception as e:
        logger.error(f"Error researching keywords: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to research keywords: {str(e)}")

@router.post("/keyword-research/dataforseo/prioritize")
async def prioritize_keywords(
    keywords: List[str],
    priority_factors: Optional[dict] = None
):
    """
    Prioritize keywords based on various factors.
    
    Args:
        keywords: List of keywords to prioritize
        priority_factors: Factors to consider for prioritization
    
    Returns:
        Prioritized list of keywords
    """
    try:
        logger.info(f"Prioritizing {len(keywords)} keywords")
        
        # Get keyword data from database
        keyword_data = await dataforseo_repository.get_keywords_by_list(keywords)
        
        # Apply prioritization algorithm
        api_client = DataForSEOAPIClient()
        await api_client.initialize()
        prioritized = await api_client.prioritize_keywords(keyword_data, priority_factors)
        
        logger.info(f"Successfully prioritized {len(prioritized)} keywords")
        return prioritized
        
    except Exception as e:
        logger.error(f"Error prioritizing keywords: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prioritize keywords: {str(e)}")

@router.get("/dataforseo/health")
async def dataforseo_health():
    """
    Check DataForSEO API health and connectivity.
    
    Returns:
        Health status of DataForSEO integration
    """
    try:
        # Check API credentials
        credentials = await dataforseo_repository.get_api_credentials("dataforseo")
        
        if not credentials:
            return {
                "status": "unhealthy",
                "message": "No DataForSEO API credentials found",
                "api_connected": False
            }
        
        # Test API connectivity
        try:
            api_client = DataForSEOAPIClient()
            await api_client.initialize()
            return {
                "status": "healthy",
                "message": "DataForSEO API is connected and ready",
                "api_connected": True,
                "provider": credentials.provider,
                "base_url": credentials.base_url
            }
        except Exception as api_error:
            return {
                "status": "unhealthy",
                "message": f"DataForSEO API connection failed: {str(api_error)}",
                "api_connected": False
            }
            
    except Exception as e:
        logger.error(f"Error checking DataForSEO health: {e}")
        return {
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}",
            "api_connected": False
        }

@router.post("/keyword-research/related-keywords")
async def get_related_keywords_endpoint(request: RelatedKeywordsRequest):
    """
    Get related keywords using DataForSEO Labs API with single-step live endpoint.
    
    Args:
        keywords: List of seed keywords
        location_code: Location code (2840 = United States)
        language_code: Language code (en = English)
        depth: Search depth (1 = direct related, 2 = related to related)
        limit: Maximum keywords per seed
    
    Returns:
        List of related keyword data with comprehensive fields
    """
    try:
        logger.info(f"Getting related keywords for: {request.keywords}")
        
        # Initialize service if needed
        if not dataforseo_service.client:
            await dataforseo_service.initialize()
        
        # Get related keywords
        related_keywords = await dataforseo_service.get_related_keywords(
            keywords=request.keywords,
            location_code=request.location_code,
            language_code=request.language_code,
            depth=request.depth,
            limit=request.limit
        )
        
        
        logger.info(f"Successfully retrieved {len(related_keywords)} related keywords")
        
        # Note: Data storage is handled by the frontend through the store endpoint
        # to ensure proper user_id and topic_id association
        
        return related_keywords
        
    except Exception as e:
        logger.error(f"Error getting related keywords: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get related keywords: {str(e)}")

@router.post("/keyword-research/keyword-ideas")
async def get_keyword_ideas_endpoint(request: KeywordIdeasRequest):
    """
    Get keyword ideas using DataForSEO Labs API with 2-call approach.
    
    Args:
        request: KeywordIdeasRequest with seed_keywords, location_code, language_code, and limit
    
    Returns:
        List of keyword idea data
    """
    try:
        logger.info(f"Getting keyword ideas for: {request.seed_keywords}")
        
        # Initialize service if needed
        if not dataforseo_service.client:
            await dataforseo_service.initialize()
        
        # Get keyword ideas
        keyword_ideas = await dataforseo_service.get_keyword_ideas(
            seed_keywords=request.seed_keywords,
            location_code=request.location_code,
            language_code=request.language_code,
            limit=request.limit
        )
        
        
        logger.info(f"Successfully retrieved {len(keyword_ideas)} keyword ideas")
        return keyword_ideas
        
    except Exception as e:
        logger.error(f"Error getting keyword ideas: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get keyword ideas: {str(e)}")

@router.post("/keyword-research/store")
async def store_keyword_data(
    keywords: List[Dict[str, Any]]
):
    """
    Store keyword data in Supabase database.
    
    Args:
        keywords: List of keyword data objects to store (must include user_id and topic_id)
    
    Returns:
        Success status and count of stored keywords
    """
    try:
        logger.info(f"Storing {len(keywords)} keywords in Supabase")
        
        # Validate input
        if not keywords or len(keywords) == 0:
            logger.error("No keywords provided for storage")
            raise HTTPException(status_code=400, detail="No keywords provided for storage")
        
        # Extract topic and user info for logging and validation
        topic_id = keywords[0].get("topic_id")
        user_id = keywords[0].get("user_id")
        logger.info(f"Processing keywords for topic_id: {topic_id}, user_id: {user_id}")
        
        # Validate required fields
        if not topic_id:
            logger.error("Missing topic_id in keyword data")
            raise HTTPException(status_code=400, detail="Missing topic_id in keyword data")
        
        if not user_id:
            logger.error("Missing user_id in keyword data")
            raise HTTPException(status_code=400, detail="Missing user_id in keyword data")
        
        # Log sample keyword data for debugging
        if len(keywords) > 0:
            sample_keyword = keywords[0]
            logger.info(f"Sample keyword data: keyword='{sample_keyword.get('keyword')}', source='{sample_keyword.get('source')}', difficulty={sample_keyword.get('difficulty')}")
        
        # Store keywords using the repository (includes cleanup of existing records)
        logger.info(f"Calling save_keyword_data_batch with {len(keywords)} keywords")
        success = await dataforseo_repository.save_keyword_data_batch(keywords)
        
        if success:
            logger.info(f"Successfully stored {len(keywords)} keywords in Supabase")
            # Fetch the updated full list of keywords for this topic
            all_keywords = await dataforseo_repository.get_keywords_by_topic_and_user(topic_id, user_id)
            logger.info(f"Retrieved {len(all_keywords)} total keywords for topic {topic_id} after merge")
            
            return {
                "success": True,
                "message": f"Successfully stored {len(keywords)} keywords and retrieved {len(all_keywords)} total",
                "count": len(keywords),
                "total_count": len(all_keywords),
                "keywords": all_keywords,
                "topic_id": topic_id,
                "user_id": user_id
            }
        else:
            logger.error("Failed to store keywords in Supabase - repository returned False")
            raise HTTPException(status_code=500, detail="Failed to store keywords in database - repository operation failed")
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error storing keyword data: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to store keyword data: {str(e)}")

@router.get("/keyword-research/test")
async def test_endpoint():
    """Test endpoint to verify router is working"""
    return {"message": "Test endpoint working"}

@router.get("/keyword-research/debug")
async def debug_keywords():
    """Debug endpoint to check database structure and data"""
    try:
        # Ensure database is initialized
        if not dataforseo_repository.db_manager.client:
            await dataforseo_repository.db_manager.initialize()
        
        client = dataforseo_repository.db_manager.get_client()
        
        # Try to get table info
        result = client.table("keyword_research_data")\
            .select("*")\
            .limit(5)\
            .execute()
        
        return {
            "success": True,
            "total_keywords": len(result.data),
            "sample_keywords": result.data,
            "columns": list(result.data[0].keys()) if result.data else []
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@router.get("/keyword-research/by-topic/{topic_id}")
async def get_keywords_by_topic(topic_id: str, user_id: str = Query(..., description="User ID")):
    """
    Get all keywords for a specific topic and user.
    
    Args:
        topic_id: The topic ID to get keywords for
        user_id: The user ID to filter keywords by
    
    Returns:
        List of keyword data for the specified topic and user
    """
    try:
        logger.info(f"Getting keywords for topic_id: {topic_id}, user_id: {user_id}")
        
        # Validate input
        if not topic_id:
            raise HTTPException(status_code=400, detail="topic_id is required")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # Get keywords from repository
        # Note: This endpoint requires the database migration to add user_id and topic_id columns
        # If the migration hasn't been applied, this will return an empty list
        keywords = await dataforseo_repository.get_keywords_by_topic_and_user(topic_id, user_id)
        
        logger.info(f"Retrieved {len(keywords)} keywords for topic {topic_id}")
        
        return {
            "success": True,
            "keywords": keywords,
            "count": len(keywords),
            "topic_id": topic_id,
            "user_id": user_id
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error getting keywords by topic: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get keywords: {str(e)}")
