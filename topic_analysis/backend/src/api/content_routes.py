"""
Content Generation API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog

from ..core.database import get_db
from ..services.content_service import ContentService
from ..models.user import User
from ..models.content_ideas import ContentIdeas, ContentStatus
from src.core.supabase_database_service import SupabaseDatabaseService
from ..schemas.content_schemas import (
    ContentGenerationRequest,
    ContentGenerationResponse,
    ContentIdeasResponse,
    ContentIdeasListResponse,
    ContentOutlineResponse,
    ContentUpdateRequest
)

logger = structlog.get_logger()
router = APIRouter(prefix="/api/content", tags=["content-generation"])

def get_content_service(db: SupabaseDatabaseService = Depends(get_db)) -> ContentService:
    """Get content service dependency"""
    return ContentService(db)

def get_current_user(db: SupabaseDatabaseService = Depends(get_db)) -> User:
    """Get current authenticated user (placeholder - implement auth middleware)"""
    # This is a placeholder - in real implementation, this would extract user from JWT token
    user = db.query(User).first()
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user

@router.post("/generate", response_model=ContentGenerationResponse)
async def generate_content(
    request: ContentGenerationRequest,
    current_user: User = Depends(get_current_user),
    content_service: ContentService = Depends(get_content_service)
):
    """Generate content ideas and outlines"""
    try:
        logger.info("Generating content", user_id=current_user.id, opportunity_id=request.opportunity_id)
        
        # Check user limits
        if not await content_service.check_user_limits(current_user.id, "content_ideas"):
            raise HTTPException(
                status_code=403,
                detail="Content ideas limit reached for your subscription tier"
            )
        
        # Generate content
        result = await content_service.generate_content(
            user_id=current_user.id,
            opportunity_id=request.opportunity_id,
            content_types=request.content_types,
            target_audience=request.target_audience,
            tone=request.tone,
            length=request.length
        )
        
        logger.info("Content generated successfully", user_id=current_user.id, content_id=result["content_id"])
        
        return ContentGenerationResponse(
            success=True,
            content_id=result["content_id"],
            article_angles=result["article_angles"],
            content_outlines=result["content_outlines"],
            seo_recommendations=result["seo_recommendations"],
            target_audience=result["target_audience"],
            opportunity_score=result["opportunity_score"],
            generated_at=result["generated_at"]
        )
        
    except ValueError as e:
        logger.error("Invalid request for content generation", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to generate content", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/ideas/{content_id}", response_model=ContentIdeasResponse)
async def get_content_ideas(
    content_id: str,
    current_user: User = Depends(get_current_user),
    content_service: ContentService = Depends(get_content_service)
):
    """Get content ideas by ID"""
    try:
        content_ideas = await content_service.get_content_ideas(content_id)
        
        if not content_ideas:
            raise HTTPException(status_code=404, detail="Content ideas not found")
        
        # Check if user owns this content
        if content_ideas.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return ContentIdeasResponse(
            id=content_ideas.id,
            title=content_ideas.title,
            opportunity_score=content_ideas.opportunity_score,
            article_angles=content_ideas.article_angles,
            content_outlines=content_ideas.content_outlines,
            seo_recommendations=content_ideas.seo_recommendations,
            status=content_ideas.status.value,
            created_at=content_ideas.created_at.isoformat(),
            updated_at=content_ideas.updated_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get content ideas", content_id=content_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/ideas", response_model=ContentIdeasListResponse)
async def list_content_ideas(
    skip: int = 0,
    limit: int = 20,
    status: Optional[ContentStatus] = None,
    current_user: User = Depends(get_current_user),
    content_service: ContentService = Depends(get_content_service)
):
    """List user's content ideas"""
    try:
        content_ideas_list = await content_service.list_user_content_ideas(
            user_id=current_user.id,
            skip=skip,
            limit=limit,
            status=status
        )
        
        return ContentIdeasListResponse(
            content_ideas=[
                ContentIdeasResponse(
                    id=ci.id,
                    title=ci.title,
                    opportunity_score=ci.opportunity_score,
                    article_angles=ci.article_angles,
                    content_outlines=ci.content_outlines,
                    seo_recommendations=ci.seo_recommendations,
                    status=ci.status.value,
                    created_at=ci.created_at.isoformat(),
                    updated_at=ci.updated_at.isoformat()
                )
                for ci in content_ideas_list
            ],
            total=len(content_ideas_list)
        )
        
    except Exception as e:
        logger.error("Failed to list content ideas", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/ideas/{content_id}", response_model=ContentIdeasResponse)
async def update_content_ideas(
    content_id: str,
    request: ContentUpdateRequest,
    current_user: User = Depends(get_current_user),
    content_service: ContentService = Depends(get_content_service)
):
    """Update content ideas"""
    try:
        content_ideas = await content_service.get_content_ideas(content_id)
        
        if not content_ideas:
            raise HTTPException(status_code=404, detail="Content ideas not found")
        
        # Check if user owns this content
        if content_ideas.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update content
        updated_content = await content_service.update_content_ideas(
            content_id=content_id,
            title=request.title,
            article_angles=request.article_angles,
            content_outlines=request.content_outlines,
            seo_recommendations=request.seo_recommendations,
            status=request.status
        )
        
        return ContentIdeasResponse(
            id=updated_content.id,
            title=updated_content.title,
            opportunity_score=updated_content.opportunity_score,
            article_angles=updated_content.article_angles,
            content_outlines=updated_content.content_outlines,
            seo_recommendations=updated_content.seo_recommendations,
            status=updated_content.status.value,
            created_at=updated_content.created_at.isoformat(),
            updated_at=updated_content.updated_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update content ideas", content_id=content_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/ideas/{content_id}")
async def delete_content_ideas(
    content_id: str,
    current_user: User = Depends(get_current_user),
    content_service: ContentService = Depends(get_content_service)
):
    """Delete content ideas"""
    try:
        content_ideas = await content_service.get_content_ideas(content_id)
        
        if not content_ideas:
            raise HTTPException(status_code=404, detail="Content ideas not found")
        
        # Check if user owns this content
        if content_ideas.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        success = await content_service.delete_content_ideas(content_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete content ideas")
        
        return {"message": "Content ideas deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete content ideas", content_id=content_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/ideas/{content_id}/outline/{angle_id}", response_model=ContentOutlineResponse)
async def get_content_outline(
    content_id: str,
    angle_id: str,
    current_user: User = Depends(get_current_user),
    content_service: ContentService = Depends(get_content_service)
):
    """Get detailed content outline for specific angle"""
    try:
        content_ideas = await content_service.get_content_ideas(content_id)
        
        if not content_ideas:
            raise HTTPException(status_code=404, detail="Content ideas not found")
        
        # Check if user owns this content
        if content_ideas.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        outline = await content_service.get_content_outline(content_id, angle_id)
        
        return ContentOutlineResponse(
            content_id=content_id,
            angle_id=angle_id,
            title=outline["title"],
            sections=outline["sections"],
            word_count=outline["word_count"],
            seo_score=outline["seo_score"],
            readability_score=outline["readability_score"],
            target_keywords=outline["target_keywords"],
            meta_description=outline["meta_description"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get content outline", content_id=content_id, angle_id=angle_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/ideas/{content_id}/regenerate")
async def regenerate_content(
    content_id: str,
    angle_ids: List[str],
    current_user: User = Depends(get_current_user),
    content_service: ContentService = Depends(get_content_service)
):
    """Regenerate content for specific angles"""
    try:
        content_ideas = await content_service.get_content_ideas(content_id)
        
        if not content_ideas:
            raise HTTPException(status_code=404, detail="Content ideas not found")
        
        # Check if user owns this content
        if content_ideas.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        success = await content_service.regenerate_content(content_id, angle_ids)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to regenerate content")
        
        return {"message": "Content regeneration initiated", "content_id": content_id, "angle_ids": angle_ids}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to regenerate content", content_id=content_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/ideas/{content_id}/seo-analysis", response_model=Dict[str, Any])
async def get_seo_analysis(
    content_id: str,
    current_user: User = Depends(get_current_user),
    content_service: ContentService = Depends(get_content_service)
):
    """Get SEO analysis for content"""
    try:
        content_ideas = await content_service.get_content_ideas(content_id)
        
        if not content_ideas:
            raise HTTPException(status_code=404, detail="Content ideas not found")
        
        # Check if user owns this content
        if content_ideas.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        seo_analysis = await content_service.get_seo_analysis(content_id)
        
        return seo_analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get SEO analysis", content_id=content_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/ideas/{content_id}/competitor-analysis", response_model=Dict[str, Any])
async def get_competitor_analysis(
    content_id: str,
    current_user: User = Depends(get_current_user),
    content_service: ContentService = Depends(get_content_service)
):
    """Get competitor analysis for content"""
    try:
        content_ideas = await content_service.get_content_ideas(content_id)
        
        if not content_ideas:
            raise HTTPException(status_code=404, detail="Content ideas not found")
        
        # Check if user owns this content
        if content_ideas.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        competitor_analysis = await content_service.get_competitor_analysis(content_id)
        
        return competitor_analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get competitor analysis", content_id=content_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/ideas/{content_id}/headline-suggestions", response_model=List[Dict[str, Any]])
async def get_headline_suggestions(
    content_id: str,
    angle_id: str,
    current_user: User = Depends(get_current_user),
    content_service: ContentService = Depends(get_content_service)
):
    """Get headline suggestions for content angle"""
    try:
        content_ideas = await content_service.get_content_ideas(content_id)
        
        if not content_ideas:
            raise HTTPException(status_code=404, detail="Content ideas not found")
        
        # Check if user owns this content
        if content_ideas.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        headlines = await content_service.get_headline_suggestions(content_id, angle_id)
        
        return headlines
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get headline suggestions", content_id=content_id, angle_id=angle_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/ideas/{content_id}/analytics", response_model=Dict[str, Any])
async def get_content_analytics(
    content_id: str,
    current_user: User = Depends(get_current_user),
    content_service: ContentService = Depends(get_content_service)
):
    """Get content analytics"""
    try:
        content_ideas = await content_service.get_content_ideas(content_id)
        
        if not content_ideas:
            raise HTTPException(status_code=404, detail="Content ideas not found")
        
        # Check if user owns this content
        if content_ideas.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        analytics = await content_service.get_content_analytics(content_id)
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get content analytics", content_id=content_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/templates", response_model=List[Dict[str, Any]])
async def get_content_templates():
    """Get available content templates"""
    try:
        templates = [
            {
                "id": "how_to",
                "name": "How-To Guide",
                "description": "Step-by-step instructional content",
                "sections": ["Introduction", "Prerequisites", "Steps", "Conclusion", "FAQ"],
                "estimated_word_count": 2000
            },
            {
                "id": "vs_comparison",
                "name": "VS Comparison",
                "description": "Compare two or more products/services",
                "sections": ["Introduction", "Product A", "Product B", "Comparison", "Verdict"],
                "estimated_word_count": 1500
            },
            {
                "id": "listicle",
                "name": "Listicle",
                "description": "List-based content format",
                "sections": ["Introduction", "Item 1", "Item 2", "Item 3", "Conclusion"],
                "estimated_word_count": 1200
            },
            {
                "id": "pain_point",
                "name": "Pain Point Solution",
                "description": "Address specific problems and solutions",
                "sections": ["Problem Identification", "Root Causes", "Solutions", "Implementation", "Results"],
                "estimated_word_count": 1800
            },
            {
                "id": "story",
                "name": "Story-Based Content",
                "description": "Narrative-driven content",
                "sections": ["Setup", "Conflict", "Resolution", "Lessons Learned", "Call to Action"],
                "estimated_word_count": 1600
            }
        ]
        
        return templates
        
    except Exception as e:
        logger.error("Failed to get content templates", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
