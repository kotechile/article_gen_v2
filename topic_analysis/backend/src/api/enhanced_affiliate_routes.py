"""
Enhanced Affiliate Research API Routes
Provides intelligent, research-driven affiliate offer discovery and management
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import structlog

from ..core.supabase_database import get_supabase_db, SupabaseDatabaseService
from ..services.enhanced_affiliate_research_service import EnhancedAffiliateResearchService
from ..core.auth import get_current_user
from src.core.supabase_database_service import SupabaseDatabaseService

logger = structlog.get_logger()
router = APIRouter(prefix="/api/enhanced-affiliate", tags=["Enhanced Affiliate Research"])

# Request/Response Models
class IntelligentDiscoveryRequest(BaseModel):
    search_terms: List[str] = Field(..., description="List of search terms for offer discovery")
    research_scope: str = Field("comprehensive", description="Research scope: quick, comprehensive, deep")
    max_offers: int = Field(20, ge=1, le=100, description="Maximum number of offers to return")
    user_id: str = Field(..., description="User ID for personalization")

class OfferRecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User ID for personalization")
    search_terms: List[str] = Field(..., description="Search terms for recommendations")
    max_offers: int = Field(10, ge=1, le=50, description="Maximum number of recommendations")
    include_analytics: bool = Field(False, description="Include analytics data")

class OfferRefreshRequest(BaseModel):
    offer_ids: List[str] = Field(..., description="List of offer IDs to refresh")
    user_id: str = Field(..., description="User ID")

class UserPreferencesRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    preferred_networks: Optional[List[str]] = Field(None, description="Preferred affiliate networks")
    preferred_commission_ranges: Optional[List[str]] = Field(None, description="Preferred commission ranges")
    preferred_categories: Optional[List[str]] = Field(None, description="Preferred categories")
    preferred_difficulty_levels: Optional[List[str]] = Field(None, description="Preferred difficulty levels")
    learning_enabled: Optional[bool] = Field(None, description="Enable learning from user behavior")

class OfferAnalyticsRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    days: int = Field(30, ge=1, le=365, description="Number of days to analyze")
    include_top_performers: bool = Field(True, description="Include top performing offers")

# Response Models
class OfferRecommendation(BaseModel):
    id: str
    program_name: str
    company_name: str
    description: str
    commission_rate: float
    network_name: str
    website_url: str
    program_url: str
    target_audience: List[str]
    content_opportunities: List[str]
    difficulty: str
    research_score: float
    quality_score: float
    relevance_score: float
    personalization_score: float
    overall_score: float
    personalization_reasons: List[str]
    verification_status: str
    last_verified: Optional[str]

class IntelligentDiscoveryResponse(BaseModel):
    success: bool
    session_id: str
    discovered_programs: int
    recommended_offers: List[OfferRecommendation]
    research_quality_score: float
    personalization_score: float
    timestamp: str
    message: str

class OfferAnalyticsResponse(BaseModel):
    success: bool
    total_offers_viewed: int
    total_clicks: int
    total_conversions: int
    total_revenue: float
    total_commission: float
    avg_time_spent: float
    top_performing_offers: List[Dict[str, Any]]
    timestamp: str

class UserPreferencesResponse(BaseModel):
    success: bool
    preferences: Dict[str, Any]
    message: str

# API Endpoints
@router.post("/discover", response_model=IntelligentDiscoveryResponse)
async def intelligent_offer_discovery(
    request: IntelligentDiscoveryRequest,
    background_tasks: BackgroundTasks,
    db: SupabaseDatabaseService = Depends(get_supabase_db)
):
    """
    Intelligently discover affiliate offers using multiple research methods
    """
    try:
        service = EnhancedAffiliateResearchService()
        
        # Perform intelligent discovery
        result = await service.intelligent_offer_discovery(
            search_terms=request.search_terms,
            user_id=request.user_id,
            research_scope=request.research_scope,
            max_offers=request.max_offers
        )
        
        # Convert offers to response format
        recommended_offers = []
        for offer in result.get('recommended_offers', []):
            recommended_offers.append(OfferRecommendation(
                id=offer.get('id', ''),
                program_name=offer.get('program_name', ''),
                company_name=offer.get('company_name', ''),
                description=offer.get('description', ''),
                commission_rate=offer.get('commission_rate', 0.0),
                network_name=offer.get('network_name', ''),
                website_url=offer.get('website_url', ''),
                program_url=offer.get('program_url', ''),
                target_audience=offer.get('target_audience', []),
                content_opportunities=offer.get('content_opportunities', []),
                difficulty=offer.get('difficulty', 'Medium'),
                research_score=offer.get('research_score', 0.0),
                quality_score=offer.get('quality_score', 0.0),
                relevance_score=offer.get('relevance_score', 0.0),
                personalization_score=offer.get('personalization_score', 0.0),
                overall_score=offer.get('overall_score', 0.0),
                personalization_reasons=offer.get('personalization_reasons', []),
                verification_status=offer.get('verification_status', 'unverified'),
                last_verified=offer.get('last_verified')
            ))
        
        return IntelligentDiscoveryResponse(
            success=True,
            session_id=result.get('session_id', ''),
            discovered_programs=result.get('discovered_programs', 0),
            recommended_offers=recommended_offers,
            research_quality_score=result.get('research_quality_score', 0.0),
            personalization_score=result.get('personalization_score', 0.0),
            timestamp=result.get('timestamp', datetime.utcnow().isoformat()),
            message="Intelligent offer discovery completed successfully"
        )
        
    except Exception as e:
        logger.error("Intelligent offer discovery failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")

@router.post("/recommendations", response_model=IntelligentDiscoveryResponse)
async def get_personalized_recommendations(
    request: OfferRecommendationRequest,
    db: SupabaseDatabaseService = Depends(get_supabase_db)
):
    """
    Get personalized affiliate offer recommendations
    """
    try:
        service = EnhancedAffiliateResearchService()
        
        # Get user preferences
        user_preferences = await service._get_user_preferences(request.user_id)
        
        # Search existing programs
        existing_programs = await service._search_existing_programs(request.search_terms)
        
        # Analyze and score offers
        analyzed_offers = await service._analyze_and_score_offers(
            existing_programs, request.search_terms, user_preferences
        )
        
        # Generate personalized recommendations
        recommendations = await service._generate_personalized_recommendations(
            analyzed_offers, user_preferences, request.max_offers
        )
        
        # Convert to response format
        recommended_offers = []
        for offer in recommendations:
            recommended_offers.append(OfferRecommendation(
                id=offer.get('id', ''),
                program_name=offer.get('program_name', ''),
                company_name=offer.get('company_name', ''),
                description=offer.get('description', ''),
                commission_rate=offer.get('commission_rate', 0.0),
                network_name=offer.get('network_name', ''),
                website_url=offer.get('website_url', ''),
                program_url=offer.get('program_url', ''),
                target_audience=offer.get('target_audience', []),
                content_opportunities=offer.get('content_opportunities', []),
                difficulty=offer.get('difficulty', 'Medium'),
                research_score=offer.get('research_score', 0.0),
                quality_score=offer.get('quality_score', 0.0),
                relevance_score=offer.get('relevance_score', 0.0),
                personalization_score=offer.get('personalization_score', 0.0),
                overall_score=offer.get('overall_score', 0.0),
                personalization_reasons=offer.get('personalization_reasons', []),
                verification_status=offer.get('verification_status', 'unverified'),
                last_verified=offer.get('last_verified')
            ))
        
        return IntelligentDiscoveryResponse(
            success=True,
            session_id="",
            discovered_programs=len(existing_programs),
            recommended_offers=recommended_offers,
            research_quality_score=service._calculate_research_quality_score(analyzed_offers),
            personalization_score=service._calculate_personalization_score(user_preferences, recommendations),
            timestamp=datetime.utcnow().isoformat(),
            message="Personalized recommendations generated successfully"
        )
        
    except Exception as e:
        logger.error("Personalized recommendations failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")

@router.get("/analytics/{user_id}", response_model=OfferAnalyticsResponse)
async def get_offer_analytics(
    user_id: str,
    days: int = Query(30, ge=1, le=365),
    db: SupabaseDatabaseService = Depends(get_supabase_db)
):
    """
    Get offer analytics for a user
    """
    try:
        service = EnhancedAffiliateResearchService()
        analytics = await service.get_offer_analytics(user_id, days)
        
        return OfferAnalyticsResponse(
            success=True,
            total_offers_viewed=analytics.get('total_offers_viewed', 0),
            total_clicks=analytics.get('total_clicks', 0),
            total_conversions=analytics.get('total_conversions', 0),
            total_revenue=analytics.get('total_revenue', 0.0),
            total_commission=analytics.get('total_commission', 0.0),
            avg_time_spent=analytics.get('avg_time_spent', 0.0),
            top_performing_offers=analytics.get('top_performing_offers', []),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error("Offer analytics failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

@router.post("/refresh", response_model=Dict[str, Any])
async def refresh_offer_data(
    request: OfferRefreshRequest,
    background_tasks: BackgroundTasks,
    db: SupabaseDatabaseService = Depends(get_supabase_db)
):
    """
    Refresh offer data from external sources
    """
    try:
        service = EnhancedAffiliateResearchService()
        results = []
        
        for offer_id in request.offer_ids:
            result = await service.refresh_offer_data(offer_id)
            results.append({
                "offer_id": offer_id,
                "result": result
            })
        
        return {
            "success": True,
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Offer refresh failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Refresh failed: {str(e)}")

@router.get("/preferences/{user_id}", response_model=UserPreferencesResponse)
async def get_user_preferences(
    user_id: str,
    db: SupabaseDatabaseService = Depends(get_supabase_db)
):
    """
    Get user preferences for personalization
    """
    try:
        service = EnhancedAffiliateResearchService()
        preferences = await service._get_user_preferences(user_id)
        
        return UserPreferencesResponse(
            success=True,
            preferences=preferences,
            message="User preferences retrieved successfully"
        )
        
    except Exception as e:
        logger.error("Get user preferences failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Preferences retrieval failed: {str(e)}")

@router.post("/preferences", response_model=UserPreferencesResponse)
async def update_user_preferences(
    request: UserPreferencesRequest,
    db: SupabaseDatabaseService = Depends(get_supabase_db)
):
    """
    Update user preferences for personalization
    """
    try:
        # Prepare update data
        update_data = {}
        if request.preferred_networks is not None:
            update_data['preferred_networks'] = request.preferred_networks
        if request.preferred_commission_ranges is not None:
            update_data['preferred_commission_ranges'] = request.preferred_commission_ranges
        if request.preferred_categories is not None:
            update_data['preferred_categories'] = request.preferred_categories
        if request.preferred_difficulty_levels is not None:
            update_data['preferred_difficulty_levels'] = request.preferred_difficulty_levels
        if request.learning_enabled is not None:
            update_data['learning_enabled'] = request.learning_enabled
        
        update_data['last_updated'] = datetime.utcnow().isoformat()
        
        # Update preferences
        result = db.client.table("user_offer_preferences").update(update_data).eq(
            "user_id", request.user_id
        ).execute()
        
        if result.data:
            return UserPreferencesResponse(
                success=True,
                preferences=result.data[0],
                message="User preferences updated successfully"
            )
        else:
            # Create new preferences if they don't exist
            service = EnhancedAffiliateResearchService()
            preferences = await service._create_default_preferences(request.user_id)
            preferences.update(update_data)
            
            create_result = db.client.table("user_offer_preferences").insert(preferences).execute()
            
            return UserPreferencesResponse(
                success=True,
                preferences=create_result.data[0] if create_result.data else preferences,
                message="User preferences created successfully"
            )
        
    except Exception as e:
        logger.error("Update user preferences failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Preferences update failed: {str(e)}")

@router.get("/programs/search")
async def search_programs(
    query: str = Query(..., description="Search query"),
    category: Optional[str] = Query(None, description="Category filter"),
    network: Optional[str] = Query(None, description="Network filter"),
    min_commission: Optional[float] = Query(None, description="Minimum commission rate"),
    max_commission: Optional[float] = Query(None, description="Maximum commission rate"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results"),
    db: SupabaseDatabaseService = Depends(get_supabase_db)
):
    """
    Search affiliate programs with filters
    """
    try:
        # Build query
        search_query = db.client.table("affiliate_programs").select("*").eq("status", "active")
        
        if query:
            search_query = search_query.or_(
                f"program_name.ilike.%{query}%,"
                f"company_name.ilike.%{query}%,"
                f"description.ilike.%{query}%"
            )
        
        if category:
            search_query = search_query.eq("category", category)
        
        if network:
            search_query = search_query.eq("network_name", network)
        
        if min_commission is not None:
            search_query = search_query.gte("commission_rate", min_commission)
        
        if max_commission is not None:
            search_query = search_query.lte("commission_rate", max_commission)
        
        # Execute query
        result = search_query.order("research_score", desc=True).limit(limit).execute()
        
        return {
            "success": True,
            "programs": result.data or [],
            "total": len(result.data or []),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Program search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/programs/{program_id}")
async def get_program_details(
    program_id: str,
    db: SupabaseDatabaseService = Depends(get_supabase_db)
):
    """
    Get detailed information about a specific program
    """
    try:
        result = db.client.table("affiliate_programs").select("*").eq("id", program_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Program not found")
        
        return {
            "success": True,
            "program": result.data[0],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get program details failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Program details retrieval failed: {str(e)}")

@router.get("/research-sessions/{user_id}")
async def get_research_sessions(
    user_id: str,
    limit: int = Query(10, ge=1, le=50),
    db: SupabaseDatabaseService = Depends(get_supabase_db)
):
    """
    Get user's research sessions
    """
    try:
        result = db.client.table("offer_research_sessions").select("*").eq(
            "user_id", user_id
        ).order("created_at", desc=True).limit(limit).execute()
        
        return {
            "success": True,
            "sessions": result.data or [],
            "total": len(result.data or []),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Get research sessions failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Research sessions retrieval failed: {str(e)}")
