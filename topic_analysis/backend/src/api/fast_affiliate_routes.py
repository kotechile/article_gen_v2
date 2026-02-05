"""
Fast Affiliate Research API Routes
Provides fast affiliate research without external API calls
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import structlog
from ..services.fast_affiliate_service import FastAffiliateService

logger = structlog.get_logger()
router = APIRouter()

class FastAffiliateSearchRequest(BaseModel):
    search_term: str
    niche: Optional[str] = None
    budget_range: Optional[str] = None
    user_id: Optional[str] = None

class FastAffiliateSearchResponse(BaseModel):
    search_term: str
    niche: Optional[str]
    budget_range: Optional[str]
    programs: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    research_id: str
    total_programs: int
    timestamp: str
    success: bool = True
    message: str = "Fast affiliate search completed successfully"

@router.post("/fast-search", response_model=FastAffiliateSearchResponse)
async def fast_affiliate_search(request: FastAffiliateSearchRequest):
    """
    Fast affiliate program search with mock data - no external API calls
    Returns results in under 1 second
    """
    try:
        logger.info("Fast affiliate search request", 
                   search_term=request.search_term, 
                   niche=request.niche)
        
        # Initialize fast service
        fast_service = FastAffiliateService()
        
        # Perform fast search
        result = await fast_service.search_affiliate_programs(
            search_term=request.search_term,
            niche=request.niche,
            budget_range=request.budget_range,
            user_id=request.user_id
        )
        
        # Format response
        response = FastAffiliateSearchResponse(
            search_term=result["search_term"],
            niche=result["niche"],
            budget_range=result["budget_range"],
            programs=result["programs"],
            analysis=result["analysis"],
            research_id=result["research_id"],
            total_programs=result["total_programs"],
            timestamp=result["timestamp"],
            success=True,
            message=f"Found {result['total_programs']} affiliate programs in under 1 second"
        )
        
        logger.info("Fast affiliate search completed", 
                   search_term=request.search_term, 
                   programs_found=result["total_programs"])
        
        return response
        
    except Exception as e:
        logger.error("Fast affiliate search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Fast search failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "fast_affiliate_research"}
