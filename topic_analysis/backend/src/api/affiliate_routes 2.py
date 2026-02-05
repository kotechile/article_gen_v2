"""
Affiliate Research API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog

from ..services.affiliate_research_service import AffiliateResearchService
from ..core.supabase_database_service import get_supabase_db, SupabaseDatabaseService
from ..schemas.affiliate_schemas import (
    AffiliateResearchRequest,
    AffiliateResearchResponse,
    AffiliateResearchListResponse,
    AffiliateProgramResponse,
    AffiliateResearchUpdate,
    ResearchStatus
)
from pydantic import BaseModel
from src.core.supabase_database_service import SupabaseDatabaseService

logger = structlog.get_logger()
router = APIRouter(prefix="/api/affiliate", tags=["affiliate-research"])

def get_affiliate_service() -> AffiliateResearchService:
    """Get affiliate service dependency"""
    return AffiliateResearchService()

from ..core.supabase_auth import get_current_user

@router.post("/research", response_model=AffiliateResearchResponse)
async def start_affiliate_research(
    request: AffiliateResearchRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    affiliate_service: AffiliateResearchService = Depends(get_affiliate_service)
):
    """Start affiliate research for a niche"""
    try:
        user_id = current_user.get("id") or current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in token")
        
        logger.info("Starting affiliate research", user_id=user_id, niche=request.niche)
        
        # Check user limits
        if not await affiliate_service.check_user_limits(user_id, "affiliate_researches"):
            raise HTTPException(
                status_code=403,
                detail="Affiliate research limit reached for your subscription tier"
            )
        
        # Start research
        research = await affiliate_service.start_affiliate_research(
            user_id=user_id,
            niche=request.niche,
            target_audience=request.target_audience,
            budget_range=request.budget_range,
            preferred_networks=request.preferred_networks
        )
        
        logger.info("Affiliate research started", research_id=research.id, user_id=user_id)
        
        return AffiliateResearchResponse(
            id=research.id,
            niche=research.niche,
            status=research.status.value,
            programs_found=len(research.programs_data.get("programs", [])),
            created_at=research.created_at.isoformat(),
            estimated_completion_time=research.estimated_completion_time
        )
        
    except ValueError as e:
        logger.error("Invalid request for affiliate research", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to start affiliate research", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/search", response_model=Dict[str, Any])
async def search_affiliate_programs(
    request: AffiliateResearchRequest,
    affiliate_service: AffiliateResearchService = Depends(get_affiliate_service)
):
    """Search for affiliate programs and return immediate results"""
    try:
        logger.info("Searching affiliate programs", query=request.query)
        
        # Search for affiliate programs
        result = await affiliate_service.search_affiliate_programs(
            search_term=request.query,
            niche=request.category,
            budget_range=None,
            user_id="demo-user"
        )
        
        # Extract programs from the result
        programs = result.get('programs', [])
        
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
        
        return {
            "programs": formatted_programs,
            "success": True,
            "message": f"Found {len(formatted_programs)} affiliate programs"
        }
        
    except Exception as e:
        logger.error("Affiliate search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/save-research", response_model=Dict[str, Any])
async def save_affiliate_research(
    request: Dict[str, Any],
    db: SupabaseDatabaseService = Depends(get_supabase_db)
):
    """Save affiliate research data to database"""
    try:
        logger.info("Saving affiliate research data", research_data=request)
        
        # Prepare research data for database
        research_data = {
            "user_id": request.get("user_id", "demo-user"),
            "search_term": request.get("main_topic", ""),
            "niche": request.get("category"),
            "budget_range": None,
            "results": {
                "subtopics": request.get("subtopics", []),
                "programs": request.get("programs", []),
                "affiliate_offers": request.get("programs", []),  # Also save as affiliate_offers
                "created_at": request.get("created_at")
            },
            "status": "completed"
        }
        
        # Save to database
        result = db.create_affiliate_research(research_data)
        
        if result:
            research_id = result.get("id")
            logger.info("Research data saved successfully", research_id=research_id)
            
            # Also save individual affiliate offers (if table exists)
            programs = request.get("programs", [])
            saved_offers = []
            if programs:
                try:
                    for program in programs:
                        try:
                            offer_data = {
                                "user_id": request.get("user_id", "demo-user"),
                                "workflow_session_id": research_id,  # Use research_id as session_id
                                "offer_name": program.get("name", "Unknown Program"),
                                "offer_description": program.get("description", ""),
                                "commission_rate": program.get("commission_rate", "Unknown"),
                                "access_instructions": program.get("link", ""),
                                "linkup_data": program,
                                "status": "active"
                            }
                            offer_result = db.create_affiliate_offer(offer_data)
                            if offer_result:
                                saved_offers.append(offer_result)
                        except Exception as e:
                            logger.warning("Failed to save individual offer", offer=program.get("name"), error=str(e))
                    
                    logger.info("Affiliate offers saved", count=len(saved_offers))
                except Exception as e:
                    logger.warning("Affiliate offers table may not exist, skipping individual offer saves", error=str(e))
                    # Don't fail the entire request if affiliate_offers table doesn't exist
            
            return {
                "success": True,
                "message": "Research data and affiliate offers saved successfully",
                "research_id": research_id,
                "data": result,
                "saved_offers_count": len(saved_offers)
            }
        else:
            raise Exception("Failed to save research data")
            
    except Exception as e:
        logger.error("Failed to save research data", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to save research: {str(e)}")

@router.get("/research/{research_id}", response_model=AffiliateResearchResponse)
async def get_affiliate_research(
    research_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    affiliate_service: AffiliateResearchService = Depends(get_affiliate_service)
):
    """Get affiliate research by ID"""
    try:
        user_id = current_user.get("id") or current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in token")
        
        research = await affiliate_service.get_affiliate_research(research_id)
        
        if not research:
            raise HTTPException(status_code=404, detail="Research not found")
        
        # Check if user owns this research
        if research.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return AffiliateResearchResponse(
            id=research.id,
            niche=research.niche,
            status=research.status.value,
            programs_found=len(research.programs_data.get("programs", [])),
            created_at=research.created_at.isoformat(),
            estimated_completion_time=research.estimated_completion_time,
            programs_data=research.programs_data,
            selected_programs=research.selected_programs
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get affiliate research", research_id=research_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/research", response_model=AffiliateResearchListResponse)
async def list_affiliate_researches(
    skip: int = 0,
    limit: int = 20,
    status: Optional[ResearchStatus] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    affiliate_service: AffiliateResearchService = Depends(get_affiliate_service)
):
    """List user's affiliate researches"""
    try:
        user_id = current_user.get("id") or current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in token")
        
        researches = await affiliate_service.list_user_researches(
            user_id=user_id,
            skip=skip,
            limit=limit,
            status=status
        )
        
        return AffiliateResearchListResponse(
            researches=[
                AffiliateResearchResponse(
                    id=research.id,
                    niche=research.niche,
                    status=research.status.value,
                    programs_found=len(research.programs_data.get("programs", [])),
                    created_at=research.created_at.isoformat(),
                    estimated_completion_time=research.estimated_completion_time
                )
                for research in researches
            ],
            total=len(researches),
            page=skip // limit + 1,
            per_page=limit,
            total_pages=(len(researches) + limit - 1) // limit
        )
        
    except Exception as e:
        logger.error("Failed to list affiliate researches", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/research-by-user", response_model=Dict[str, Any])
async def get_researches_by_user_id(
    user_id: str,
    db: SupabaseDatabaseService = Depends(get_supabase_db)
):
    """Get researches by user_id (for frontend without auth)"""
    try:
        logger.info("Getting researches for user", user_id=user_id)
        
        # Query the database directly for researches by user_id
        result = db.client.table("affiliate_research").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        
        if result.data:
            # Format the data for frontend
            researches = []
            for research in result.data:
                research_data = {
                    "id": research.get("id"),
                    "main_topic": research.get("search_term", ""),
                    "subtopics": research.get("results", {}).get("subtopics", []),
                    "programs": research.get("results", {}).get("programs", []),
                    "created_at": research.get("created_at"),
                    "status": research.get("status", "completed")
                }
                researches.append(research_data)
            
            logger.info("Found researches", count=len(researches), user_id=user_id)
            return {
                "success": True,
                "researches": researches,
                "total": len(researches)
            }
        else:
            logger.info("No researches found", user_id=user_id)
            return {
                "success": True,
                "researches": [],
                "total": 0
            }
            
    except Exception as e:
        logger.error("Failed to get researches by user_id", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get researches: {str(e)}")

@router.put("/research/{research_id}", response_model=AffiliateResearchResponse)
async def update_affiliate_research(
    research_id: str,
    request: AffiliateResearchUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    affiliate_service: AffiliateResearchService = Depends(get_affiliate_service)
):
    """Update affiliate research"""
    try:
        user_id = current_user.get("id") or current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in token")
        
        research = await affiliate_service.get_affiliate_research(research_id)
        
        if not research:
            raise HTTPException(status_code=404, detail="Research not found")
        
        # Check if user owns this research
        if research.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update research
        updated_research = await affiliate_service.update_affiliate_research(
            research_id=research_id,
            selected_programs=request.selected_programs,
            notes=request.notes
        )
        
        return AffiliateResearchResponse(
            id=updated_research.id,
            niche=updated_research.niche,
            status=updated_research.status.value,
            programs_found=len(updated_research.programs_data.get("programs", [])),
            created_at=updated_research.created_at.isoformat(),
            estimated_completion_time=updated_research.estimated_completion_time,
            programs_data=updated_research.programs_data,
            selected_programs=updated_research.selected_programs
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update affiliate research", research_id=research_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/research/{research_id}")
async def delete_affiliate_research(
    research_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    affiliate_service: AffiliateResearchService = Depends(get_affiliate_service)
):
    """Delete affiliate research"""
    try:
        user_id = current_user.get("id") or current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in token")
        
        research = await affiliate_service.get_affiliate_research(research_id)
        
        if not research:
            raise HTTPException(status_code=404, detail="Research not found")
        
        # Check if user owns this research
        if research.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        success = await affiliate_service.delete_affiliate_research(research_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete research")
        
        return {"message": "Research deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete affiliate research", research_id=research_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/research/{research_id}/programs", response_model=List[AffiliateProgramResponse])
async def get_affiliate_programs(
    research_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    affiliate_service: AffiliateResearchService = Depends(get_affiliate_service)
):
    """Get affiliate programs from research"""
    try:
        user_id = current_user.get("id") or current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in token")
        
        research = await affiliate_service.get_affiliate_research(research_id)
        
        if not research:
            raise HTTPException(status_code=404, detail="Research not found")
        
        # Check if user owns this research
        if research.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        programs = research.programs_data.get("programs", [])
        
        return [
            AffiliateProgramResponse(
                id=program.get("id", ""),
                name=program.get("name", ""),
                network=program.get("network", ""),
                epc=program.get("epc", 0.0),
                reversal_rate=program.get("reversal_rate", 0.0),
                cookie_length=program.get("cookie_length", 0),
                commission=program.get("commission", 0.0),
                description=program.get("description", ""),
                url=program.get("url", ""),
                category=program.get("category", ""),
                is_selected=program.get("id", "") in research.selected_programs
            )
            for program in programs
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get affiliate programs", research_id=research_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/research/{research_id}/select-programs")
async def select_affiliate_programs(
    research_id: str,
    program_ids: List[str],
    current_user: Dict[str, Any] = Depends(get_current_user),
    affiliate_service: AffiliateResearchService = Depends(get_affiliate_service)
):
    """Select affiliate programs from research"""
    try:
        user_id = current_user.get("id") or current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in token")
        
        research = await affiliate_service.get_affiliate_research(research_id)
        
        if not research:
            raise HTTPException(status_code=404, detail="Research not found")
        
        # Check if user owns this research
        if research.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        success = await affiliate_service.select_affiliate_programs(
            research_id=research_id,
            program_ids=program_ids
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to select programs")
        
        return {"message": "Programs selected successfully", "selected_count": len(program_ids)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to select affiliate programs", research_id=research_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/networks", response_model=List[Dict[str, Any]])
async def get_affiliate_networks():
    """Get available affiliate networks"""
    try:
        networks = [
            {
                "id": "shareasale",
                "name": "ShareASale",
                "description": "Leading affiliate marketing network",
                "categories": ["fashion", "home", "health", "beauty"],
                "min_commission": 0.01,
                "max_commission": 0.50
            },
            {
                "id": "impact",
                "name": "Impact",
                "description": "Global partnership automation platform",
                "categories": ["technology", "finance", "travel", "retail"],
                "min_commission": 0.05,
                "max_commission": 0.30
            },
            {
                "id": "amazon",
                "name": "Amazon Associates",
                "description": "Amazon's affiliate program",
                "categories": ["all"],
                "min_commission": 0.01,
                "max_commission": 0.10
            },
            {
                "id": "cj",
                "name": "CJ Affiliate",
                "description": "Commission Junction affiliate network",
                "categories": ["retail", "travel", "finance", "health"],
                "min_commission": 0.01,
                "max_commission": 0.40
            },
            {
                "id": "partnerize",
                "name": "Partnerize",
                "description": "Partnership automation platform",
                "categories": ["retail", "fashion", "beauty", "home"],
                "min_commission": 0.02,
                "max_commission": 0.25
            }
        ]
        
        return networks
        
    except Exception as e:
        logger.error("Failed to get affiliate networks", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/research/{research_id}/analytics", response_model=Dict[str, Any])
async def get_research_analytics(
    research_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    affiliate_service: AffiliateResearchService = Depends(get_affiliate_service)
):
    """Get research analytics"""
    try:
        user_id = current_user.get("id") or current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in token")
        
        research = await affiliate_service.get_affiliate_research(research_id)
        
        if not research:
            raise HTTPException(status_code=404, detail="Research not found")
        
        # Check if user owns this research
        if research.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        analytics = await affiliate_service.get_research_analytics(research_id)
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get research analytics", research_id=research_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# Cache Management Endpoints

class CacheClearRequest(BaseModel):
    search_term: str
    niche: Optional[str] = None
    budget_range: Optional[str] = None

@router.get("/cache/stats")
async def get_cache_stats(affiliate_service: AffiliateResearchService = Depends(get_affiliate_service)):
    """Get cache statistics"""
    try:
        stats = affiliate_service.get_cache_stats()
        return {"success": True, "data": stats}
    except Exception as e:
        logger.error("Failed to get cache stats", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/cache/clear")
async def clear_search_cache(
    request: CacheClearRequest,
    affiliate_service: AffiliateResearchService = Depends(get_affiliate_service)
):
    """Clear cache for specific search parameters"""
    try:
        success = affiliate_service.clear_search_cache(
            request.search_term,
            request.niche,
            request.budget_range
        )
        return {
            "success": success,
            "message": "Cache cleared successfully" if success else "Failed to clear cache"
        }
    except Exception as e:
        logger.error("Failed to clear search cache", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/cache/clear-all")
async def clear_all_cache(affiliate_service: AffiliateResearchService = Depends(get_affiliate_service)):
    """Clear all affiliate search cache"""
    try:
        success = affiliate_service.clear_all_search_cache()
        return {
            "success": success,
            "message": "All cache cleared successfully" if success else "Failed to clear all cache"
        }
    except Exception as e:
        logger.error("Failed to clear all cache", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
