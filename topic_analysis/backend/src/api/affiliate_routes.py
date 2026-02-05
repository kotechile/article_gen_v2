"""
Affiliate Research API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog

from ..core.database import get_db
from ..services.affiliate_service import AffiliateService
from ..models.user import User
from ..models.affiliate_research import AffiliateResearch, ResearchStatus
from src.core.supabase_database_service import SupabaseDatabaseService
from ..schemas.affiliate_schemas import (
    AffiliateResearchRequest,
    AffiliateResearchResponse,
    AffiliateResearchListResponse,
    AffiliateProgramResponse,
    AffiliateResearchUpdate
)

logger = structlog.get_logger()
router = APIRouter(prefix="/api/affiliate", tags=["affiliate-research"])

def get_affiliate_service(db: SupabaseDatabaseService = Depends(get_db)) -> AffiliateService:
    """Get affiliate service dependency"""
    return AffiliateService(db)

def get_current_user(db: SupabaseDatabaseService = Depends(get_db)) -> User:
    """Get current authenticated user (placeholder - implement auth middleware)"""
    # This is a placeholder - in real implementation, this would extract user from JWT token
    user = db.query(User).first()
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user

@router.post("/research", response_model=AffiliateResearchResponse)
async def start_affiliate_research(
    request: AffiliateResearchRequest,
    current_user: User = Depends(get_current_user),
    affiliate_service: AffiliateService = Depends(get_affiliate_service)
):
    """Start affiliate research for a niche"""
    try:
        logger.info("Starting affiliate research", user_id=current_user.id, niche=request.niche)
        
        # Check user limits
        if not await affiliate_service.check_user_limits(current_user.id, "affiliate_researches"):
            raise HTTPException(
                status_code=403,
                detail="Affiliate research limit reached for your subscription tier"
            )
        
        # Start research
        research = await affiliate_service.start_affiliate_research(
            user_id=current_user.id,
            niche=request.niche,
            target_audience=request.target_audience,
            budget_range=request.budget_range,
            preferred_networks=request.preferred_networks
        )
        
        logger.info("Affiliate research started", research_id=research.id, user_id=current_user.id)
        
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

@router.get("/research/{research_id}", response_model=AffiliateResearchResponse)
async def get_affiliate_research(
    research_id: str,
    current_user: User = Depends(get_current_user),
    affiliate_service: AffiliateService = Depends(get_affiliate_service)
):
    """Get affiliate research by ID"""
    try:
        research = await affiliate_service.get_affiliate_research(research_id)
        
        if not research:
            raise HTTPException(status_code=404, detail="Research not found")
        
        # Check if user owns this research
        if research.user_id != current_user.id:
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
    current_user: User = Depends(get_current_user),
    affiliate_service: AffiliateService = Depends(get_affiliate_service)
):
    """List user's affiliate researches"""
    try:
        researches = await affiliate_service.list_user_researches(
            user_id=current_user.id,
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
            total=len(researches)
        )
        
    except Exception as e:
        logger.error("Failed to list affiliate researches", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/research/{research_id}", response_model=AffiliateResearchResponse)
async def update_affiliate_research(
    research_id: str,
    request: AffiliateResearchUpdate,
    current_user: User = Depends(get_current_user),
    affiliate_service: AffiliateService = Depends(get_affiliate_service)
):
    """Update affiliate research"""
    try:
        research = await affiliate_service.get_affiliate_research(research_id)
        
        if not research:
            raise HTTPException(status_code=404, detail="Research not found")
        
        # Check if user owns this research
        if research.user_id != current_user.id:
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
    current_user: User = Depends(get_current_user),
    affiliate_service: AffiliateService = Depends(get_affiliate_service)
):
    """Delete affiliate research"""
    try:
        research = await affiliate_service.get_affiliate_research(research_id)
        
        if not research:
            raise HTTPException(status_code=404, detail="Research not found")
        
        # Check if user owns this research
        if research.user_id != current_user.id:
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
    current_user: User = Depends(get_current_user),
    affiliate_service: AffiliateService = Depends(get_affiliate_service)
):
    """Get affiliate programs from research"""
    try:
        research = await affiliate_service.get_affiliate_research(research_id)
        
        if not research:
            raise HTTPException(status_code=404, detail="Research not found")
        
        # Check if user owns this research
        if research.user_id != current_user.id:
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
    current_user: User = Depends(get_current_user),
    affiliate_service: AffiliateService = Depends(get_affiliate_service)
):
    """Select affiliate programs from research"""
    try:
        research = await affiliate_service.get_affiliate_research(research_id)
        
        if not research:
            raise HTTPException(status_code=404, detail="Research not found")
        
        # Check if user owns this research
        if research.user_id != current_user.id:
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
    current_user: User = Depends(get_current_user),
    affiliate_service: AffiliateService = Depends(get_affiliate_service)
):
    """Get research analytics"""
    try:
        research = await affiliate_service.get_affiliate_research(research_id)
        
        if not research:
            raise HTTPException(status_code=404, detail="Research not found")
        
        # Check if user owns this research
        if research.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        analytics = await affiliate_service.get_research_analytics(research_id)
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get research analytics", research_id=research_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
