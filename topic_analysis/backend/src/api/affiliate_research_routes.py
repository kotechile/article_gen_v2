"""
Affiliate Research API Routes
Handles affiliate program discovery and content generation
"""

from fastapi import APIRouter, HTTPException, status
from typing import List, Optional
import structlog

from ..services.enhanced_affiliate_research_service import EnhancedAffiliateResearchService
from ..schemas.affiliate_schemas import (
    AffiliateSearchRequest,
    AffiliateSearchResponse,
    ContentIdeasRequest,
    ContentIdeasResponse,
    ResearchHistoryResponse
)

logger = structlog.get_logger()
router = APIRouter(prefix="/api/affiliate-research", tags=["affiliate-research"])

@router.post("/search", response_model=AffiliateSearchResponse)
async def search_affiliate_programs(
    request: AffiliateSearchRequest
):
    """
    Search for affiliate programs using Enhanced Intelligent Discovery
    """
    try:
        logger.info("Enhanced Affiliate search request", 
                   search_term=request.search_term, 
                   niche=request.niche,
                   budget_range=request.budget_range)
        
        service = EnhancedAffiliateResearchService()
        
        # Use intelligent discovery
        # Map request parameters to expected args
        search_terms = [request.search_term]
        if request.niche:
            search_terms.append(request.niche)
            
        result = await service.intelligent_offer_discovery(
            search_terms=search_terms,
            user_id=request.user_id,
            research_scope="comprehensive", # Default to comprehensive
            ignore_cache=True # Always fresh for manual search
        )
        
        # Transform result to match AffiliateSearchResponse structure if needed
        # The service returns {session_id, discovered_programs, recommended_offers...}
        # The frontend expects a specific structure. 
        # Checking AffiliateSearchResponse model... assumed generic data field or similar.
        # Ideally we map 'recommended_offers' to the list the UI expects.
        
        return AffiliateSearchResponse(
            success=True,
            message="Affiliate programs found successfully",
            data={
                "programs": result["recommended_offers"],
                "count": len(result["recommended_offers"]),
                "session_id": result.get("session_id")
            }
        )
        
    except Exception as e:
        logger.error("Affiliate search failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )
        
    except Exception as e:
        logger.error("Affiliate search failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@router.post("/content-ideas", response_model=ContentIdeasResponse)
async def generate_content_ideas(
    request: ContentIdeasRequest
):
    """
    Generate content ideas based on selected affiliate programs
    """
    try:
        logger.info("Content ideas request", 
                   selected_programs=request.selected_programs,
                   user_id=request.user_id)
        
        service = AffiliateResearchService()
        result = await service.generate_content_ideas(
            selected_programs=request.selected_programs,
            user_id=request.user_id
        )
        
        return ContentIdeasResponse(
            success=True,
            message="Content ideas generated successfully",
            data=result
        )
        
    except Exception as e:
        logger.error("Content ideas generation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Content generation failed: {str(e)}"
        )

@router.get("/history/{user_id}", response_model=ResearchHistoryResponse)
async def get_research_history(
    user_id: str,
    limit: int = 10
):
    """
    Get user's affiliate research history
    """
    try:
        logger.info("Research history request", user_id=user_id, limit=limit)
        
        service = AffiliateResearchService()
        history = service.get_research_history(user_id=user_id, limit=limit)
        
        return ResearchHistoryResponse(
            success=True,
            message="Research history retrieved successfully",
            data={
                "history": history,
                "total_records": len(history)
            }
        )
        
    except Exception as e:
        logger.error("Research history retrieval failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"History retrieval failed: {str(e)}"
        )

@router.get("/programs/{program_id}")
async def get_program_details(
    program_id: str
):
    """
    Get detailed information about a specific affiliate program
    """
    try:
        logger.info("Program details request", program_id=program_id)
        
        service = AffiliateResearchService()
        # In real implementation, this would fetch from database
        # For now, return mock data
        mock_program = {
            "id": program_id,
            "name": "Sample Affiliate Program",
            "description": "Detailed program information",
            "commission": "5-10%",
            "cookie_duration": "30 days",
            "payment_terms": "Net 30",
            "min_payout": "$50",
            "category": "General",
            "rating": 4.5,
            "estimated_earnings": "$200-500/month",
            "difficulty": "Medium",
            "affiliate_network": "ShareASale",
            "tracking_method": "Cookie-based",
            "payment_methods": ["PayPal", "Bank Transfer"],
            "support_level": "High",
            "promotional_materials": ["Banners", "Text links"],
            "restrictions": "Must disclose affiliate relationship"
        }
        
        return {
            "success": True,
            "message": "Program details retrieved successfully",
            "data": mock_program
        }
        
    except Exception as e:
        logger.error("Program details retrieval failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Program details retrieval failed: {str(e)}"
        )

@router.get("/categories")
async def get_affiliate_categories():
    """
    Get list of available affiliate program categories
    """
    try:
        categories = [
            "Home & Garden",
            "Technology",
            "Health & Fitness",
            "Fashion & Beauty",
            "Food & Beverage",
            "Travel & Tourism",
            "Finance & Insurance",
            "Education & Training",
            "Automotive",
            "Sports & Recreation",
            "Pets & Animals",
            "Business & Professional Services"
        ]
        
        return {
            "success": True,
            "message": "Categories retrieved successfully",
            "data": {
                "categories": categories,
                "total_categories": len(categories)
            }
        }
        
    except Exception as e:
        logger.error("Categories retrieval failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Categories retrieval failed: {str(e)}"
        )

@router.get("/networks")
async def get_affiliate_networks():
    """
    Get list of available affiliate networks
    """
    try:
        networks = [
            {
                "id": "shareasale",
                "name": "ShareASale",
                "description": "Leading affiliate marketing network",
                "commission_range": "3-15%",
                "payment_terms": "Net 30",
                "min_payout": "$50"
            },
            {
                "id": "cj_affiliate",
                "name": "CJ Affiliate",
                "description": "Performance marketing platform",
                "commission_range": "2-12%",
                "payment_terms": "Net 30",
                "min_payout": "$25"
            },
            {
                "id": "impact_radius",
                "name": "Impact Radius",
                "description": "Partnership automation platform",
                "commission_range": "5-20%",
                "payment_terms": "Net 30",
                "min_payout": "$100"
            },
            {
                "id": "awin",
                "name": "Awin",
                "description": "Global affiliate network",
                "commission_range": "2-10%",
                "payment_terms": "Net 30",
                "min_payout": "$50"
            }
        ]
        
        return {
            "success": True,
            "message": "Networks retrieved successfully",
            "data": {
                "networks": networks,
                "total_networks": len(networks)
            }
        }
        
    except Exception as e:
        logger.error("Networks retrieval failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Networks retrieval failed: {str(e)}"
        )

