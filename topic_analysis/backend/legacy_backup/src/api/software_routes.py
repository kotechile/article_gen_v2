"""
Software Generation API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog

from ..core.database import get_db
from ..services.software_service import SoftwareService
from ..models.user import User
from ..models.software_solutions import SoftwareSolutions, SoftwareType, DevelopmentStatus
from ..schemas.software_schemas import (
    SoftwareGenerationRequest,
    SoftwareGenerationResponse,
    SoftwareSolutionsResponse,
    SoftwareSolutionsListResponse,
    SoftwareSolutionResponse,
    SoftwareUpdateRequest
)

logger = structlog.get_logger()
router = APIRouter(prefix="/api/software", tags=["software-generation"])


def get_software_service(db: Session = Depends(get_db)) -> SoftwareService:
    """Get software service dependency"""
    return SoftwareService(db)


def get_current_user(db: Session = Depends(get_db)) -> User:
    """Get current authenticated user (placeholder - implement auth middleware)"""
    # This is a placeholder - in real implementation, this would extract user from JWT token
    user = db.query(User).first()
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


@router.post("/generate", response_model=SoftwareGenerationResponse)
async def generate_software_solutions(
    request: SoftwareGenerationRequest,
    current_user: User = Depends(get_current_user),
    software_service: SoftwareService = Depends(get_software_service)
):
    """Generate software solution ideas"""
    try:
        logger.info("Generating software solutions", user_id=current_user.id, opportunity_id=request.opportunity_id)
        
        # Check user limits
        if not await software_service.check_user_limits(current_user.id, "software_solutions"):
            raise HTTPException(
                status_code=403,
                detail="Software solutions limit reached for your subscription tier"
            )
        
        # Generate software solutions
        result = await software_service.generate_solutions(
            user_id=current_user.id,
            trend_analysis_id=request.trend_analysis_id,
            keyword_data_id=request.keyword_data_id,
            software_types=request.software_types,
            max_solutions=request.max_solutions
        )
        
        logger.info("Software solutions generated successfully", user_id=current_user.id, software_solutions_id=result["id"])
        
        return SoftwareGenerationResponse(
            success=True,
            software_solutions_id=result["id"],
            solutions_generated=len(result.get("software_solutions", [])),
            estimated_completion_time=result.get("estimated_completion_time"),
            generated_at=result.get("generated_at")
        )
        
    except ValueError as e:
        logger.error("Invalid request for software generation", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to generate software solutions", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/solutions/{software_solutions_id}", response_model=SoftwareSolutionsResponse)
async def get_software_solutions(
    software_solutions_id: str,
    current_user: User = Depends(get_current_user),
    software_service: SoftwareService = Depends(get_software_service)
):
    """Get software solutions by ID"""
    try:
        software_solutions = await software_service.get_software_solution(software_solutions_id)
        
        if not software_solutions:
            raise HTTPException(status_code=404, detail="Software solutions not found")
        
        # Check if user owns this data
        if software_solutions["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return SoftwareSolutionsResponse(
            id=software_solutions["id"],
            software_solutions=software_solutions.get("software_solutions", []),
            created_at=software_solutions["created_at"],
            updated_at=software_solutions.get("updated_at")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get software solutions", software_solutions_id=software_solutions_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/solutions", response_model=SoftwareSolutionsListResponse)
async def list_software_solutions(
    skip: int = 0,
    limit: int = 20,
    software_type: Optional[SoftwareType] = None,
    current_user: User = Depends(get_current_user),
    software_service: SoftwareService = Depends(get_software_service)
):
    """List user's software solutions"""
    try:
        software_solutions_list = await software_service.get_user_software_solutions(
            user_id=current_user.id,
            limit=limit,
            offset=skip
        )
        
        return SoftwareSolutionsListResponse(
            software_solutions=[
                SoftwareSolutionsResponse(
                    id=ss["id"],
                    software_solutions=ss.get("software_solutions", []),
                    created_at=ss["created_at"],
                    updated_at=ss.get("updated_at")
                )
                for ss in software_solutions_list
            ],
            total=len(software_solutions_list)
        )
        
    except Exception as e:
        logger.error("Failed to list software solutions", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/solution/{solution_id}", response_model=SoftwareSolutionResponse)
async def get_software_solution(
    solution_id: str,
    current_user: User = Depends(get_current_user),
    software_service: SoftwareService = Depends(get_software_service)
):
    """Get individual software solution by ID"""
    try:
        solution = await software_service.get_software_solution(solution_id)
        
        if not solution:
            raise HTTPException(status_code=404, detail="Software solution not found")
        
        # Check if user owns this solution
        if solution["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return SoftwareSolutionResponse(
            id=solution["id"],
            name=solution["name"],
            description=solution["description"],
            software_type=solution["software_type"],
            complexity_score=solution["complexity_score"],
            priority_score=solution["priority_score"],
            target_keywords=solution["target_keywords"],
            technical_requirements=solution["technical_requirements"],
            estimated_development_time=solution["estimated_development_time"],
            development_phases=solution["development_phases"],
            monetization_strategy=solution["monetization_strategy"],
            seo_optimization=solution["seo_optimization"],
            status=solution["status"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get software solution", solution_id=solution_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/solution/{solution_id}", response_model=SoftwareSolutionResponse)
async def update_software_solution(
    solution_id: str,
    request: SoftwareUpdateRequest,
    current_user: User = Depends(get_current_user),
    software_service: SoftwareService = Depends(get_software_service)
):
    """Update software solution"""
    try:
        solution = await software_service.get_software_solution(solution_id)
        
        if not solution:
            raise HTTPException(status_code=404, detail="Software solution not found")
        
        # Check if user owns this solution
        if solution["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update solution
        success = await software_service.update_software_status(
            software_solution_id=solution_id,
            status=request.status,
            planned_start_date=request.planned_start_date,
            notes=request.notes
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update software solution")
        
        # Get updated solution
        updated_solution = await software_service.get_software_solution(solution_id)
        
        return SoftwareSolutionResponse(
            id=updated_solution["id"],
            name=updated_solution["name"],
            description=updated_solution["description"],
            software_type=updated_solution["software_type"],
            complexity_score=updated_solution["complexity_score"],
            priority_score=updated_solution["priority_score"],
            target_keywords=updated_solution["target_keywords"],
            technical_requirements=updated_solution["technical_requirements"],
            estimated_development_time=updated_solution["estimated_development_time"],
            development_phases=updated_solution["development_phases"],
            monetization_strategy=updated_solution["monetization_strategy"],
            seo_optimization=updated_solution["seo_optimization"],
            status=updated_solution["status"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update software solution", solution_id=solution_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/solution/{solution_id}")
async def delete_software_solution(
    solution_id: str,
    current_user: User = Depends(get_current_user),
    software_service: SoftwareService = Depends(get_software_service)
):
    """Delete software solution"""
    try:
        solution = await software_service.get_software_solution(solution_id)
        
        if not solution:
            raise HTTPException(status_code=404, detail="Software solution not found")
        
        # Check if user owns this solution
        if solution["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        success = await software_service.delete_software_solution(solution_id, current_user.id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete software solution")
        
        return {"message": "Software solution deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete software solution", solution_id=solution_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/solution/{solution_id}/development-plan", response_model=Dict[str, Any])
async def get_development_plan(
    solution_id: str,
    current_user: User = Depends(get_current_user),
    software_service: SoftwareService = Depends(get_software_service)
):
    """Get detailed development plan for software solution"""
    try:
        solution = await software_service.get_software_solution(solution_id)
        
        if not solution:
            raise HTTPException(status_code=404, detail="Software solution not found")
        
        # Check if user owns this solution
        if solution["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        development_plan = {
            "solution_id": solution_id,
            "name": solution["name"],
            "complexity_score": solution["complexity_score"],
            "estimated_development_time": solution["estimated_development_time"],
            "development_phases": solution["development_phases"],
            "technical_requirements": solution["technical_requirements"],
            "monetization_strategy": solution["monetization_strategy"],
            "seo_optimization": solution["seo_optimization"]
        }
        
        return development_plan
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get development plan", solution_id=solution_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/solution/{solution_id}/monetization", response_model=Dict[str, Any])
async def get_monetization_strategy(
    solution_id: str,
    current_user: User = Depends(get_current_user),
    software_service: SoftwareService = Depends(get_software_service)
):
    """Get monetization strategy for software solution"""
    try:
        solution = await software_service.get_software_solution(solution_id)
        
        if not solution:
            raise HTTPException(status_code=404, detail="Software solution not found")
        
        # Check if user owns this solution
        if solution["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return solution["monetization_strategy"]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get monetization strategy", solution_id=solution_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/solution/{solution_id}/seo-optimization", response_model=Dict[str, Any])
async def get_seo_optimization(
    solution_id: str,
    current_user: User = Depends(get_current_user),
    software_service: SoftwareService = Depends(get_software_service)
):
    """Get SEO optimization data for software solution"""
    try:
        solution = await software_service.get_software_solution(solution_id)
        
        if not solution:
            raise HTTPException(status_code=404, detail="Software solution not found")
        
        # Check if user owns this solution
        if solution["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return solution["seo_optimization"]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get SEO optimization", solution_id=solution_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/types", response_model=List[Dict[str, Any]])
async def get_software_types():
    """Get available software types"""
    try:
        software_types = [
            {
                "id": "calculator",
                "name": "Calculator",
                "description": "Mathematical calculation tools",
                "examples": ["Mortgage Calculator", "BMI Calculator", "Tip Calculator"],
                "complexity_range": "1-5"
            },
            {
                "id": "analyzer",
                "name": "Analyzer",
                "description": "Data analysis and visualization tools",
                "examples": ["SEO Analyzer", "Website Speed Analyzer", "Text Analyzer"],
                "complexity_range": "3-8"
            },
            {
                "id": "generator",
                "name": "Generator",
                "description": "Content and data generation tools",
                "examples": ["Password Generator", "QR Code Generator", "Color Palette Generator"],
                "complexity_range": "2-6"
            },
            {
                "id": "converter",
                "name": "Converter",
                "description": "Format and unit conversion tools",
                "examples": ["Currency Converter", "Image Converter", "Unit Converter"],
                "complexity_range": "2-5"
            },
            {
                "id": "estimator",
                "name": "Estimator",
                "description": "Estimation and prediction tools",
                "examples": ["Tax Estimator", "ROI Calculator", "Time Estimator"],
                "complexity_range": "3-7"
            }
        ]
        
        return software_types
        
    except Exception as e:
        logger.error("Failed to get software types", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/solution/{solution_id}/analytics", response_model=Dict[str, Any])
async def get_software_analytics(
    solution_id: str,
    current_user: User = Depends(get_current_user),
    software_service: SoftwareService = Depends(get_software_service)
):
    """Get software solution analytics"""
    try:
        solution = await software_service.get_software_solution(solution_id)
        
        if not solution:
            raise HTTPException(status_code=404, detail="Software solution not found")
        
        # Check if user owns this solution
        if solution["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        analytics = {
            "solution_id": solution_id,
            "complexity_score": solution["complexity_score"],
            "priority_score": solution["priority_score"],
            "target_keywords_count": len(solution["target_keywords"]),
            "estimated_development_time": solution["estimated_development_time"],
            "development_phases_count": len(solution["development_phases"]),
            "monetization_potential": solution["monetization_strategy"].get("potential", "medium"),
            "seo_opportunity": solution["seo_optimization"].get("opportunity", "medium")
        }
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get software analytics", solution_id=solution_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
