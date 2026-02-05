"""
Export Integration API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog

from ..core.database import get_db
from ..services.export_service import ExportService
from ..models.user import User
from ..schemas.export_schemas import (
    ExportRequest,
    ExportResponse,
    ExportTemplateResponse,
    ExportTemplateListResponse,
    ExportStatusResponse
)

logger = structlog.get_logger()
router = APIRouter(prefix="/api/export", tags=["export-integration"])


def get_export_service(db: Session = Depends(get_db)) -> ExportService:
    """Get export service dependency"""
    return ExportService(db)


def get_current_user(db: Session = Depends(get_db)) -> User:
    """Get current authenticated user (placeholder - implement auth middleware)"""
    # This is a placeholder - in real implementation, this would extract user from JWT token
    user = db.query(User).first()
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


@router.post("/google-docs", response_model=ExportResponse)
async def export_to_google_docs(
    request: ExportRequest,
    current_user: User = Depends(get_current_user),
    export_service: ExportService = Depends(get_export_service)
):
    """Export content to Google Docs"""
    try:
        logger.info("Exporting to Google Docs", user_id=current_user.id, content_id=request.content_id)
        
        result = await export_service.export_to_google_docs(
            content_id=request.content_id,
            template_id=request.template_id,
            custom_fields=request.custom_fields
        )
        
        logger.info("Content exported to Google Docs", user_id=current_user.id, document_url=result["document_url"])
        
        return ExportResponse(
            success=result["success"],
            platform=result["platform"],
            export_url=result["document_url"],
            exported_at=result["exported_at"]
        )
        
    except ValueError as e:
        logger.error("Invalid request for Google Docs export", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to export to Google Docs", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/notion", response_model=ExportResponse)
async def export_to_notion(
    request: ExportRequest,
    current_user: User = Depends(get_current_user),
    export_service: ExportService = Depends(get_export_service)
):
    """Export content to Notion"""
    try:
        logger.info("Exporting to Notion", user_id=current_user.id, content_id=request.content_id)
        
        result = await export_service.export_to_notion(
            content_id=request.content_id,
            template_id=request.template_id,
            custom_fields=request.custom_fields
        )
        
        logger.info("Content exported to Notion", user_id=current_user.id, page_url=result["page_url"])
        
        return ExportResponse(
            success=result["success"],
            platform=result["platform"],
            export_url=result["page_url"],
            exported_at=result["exported_at"]
        )
        
    except ValueError as e:
        logger.error("Invalid request for Notion export", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to export to Notion", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/wordpress", response_model=ExportResponse)
async def export_to_wordpress(
    request: ExportRequest,
    current_user: User = Depends(get_current_user),
    export_service: ExportService = Depends(get_export_service)
):
    """Export content to WordPress"""
    try:
        logger.info("Exporting to WordPress", user_id=current_user.id, content_id=request.content_id)
        
        result = await export_service.export_to_wordpress(
            content_id=request.content_id,
            template_id=request.template_id,
            custom_fields=request.custom_fields
        )
        
        logger.info("Content exported to WordPress", user_id=current_user.id, post_url=result["post_url"])
        
        return ExportResponse(
            success=result["success"],
            platform=result["platform"],
            export_url=result["post_url"],
            exported_at=result["exported_at"]
        )
        
    except ValueError as e:
        logger.error("Invalid request for WordPress export", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to export to WordPress", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/software/{software_solution_id}")
async def export_software_solution(
    software_solution_id: str,
    platform: str,
    template_id: int,
    custom_fields: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_user),
    export_service: ExportService = Depends(get_export_service)
):
    """Export software solution to platform"""
    try:
        logger.info("Exporting software solution", user_id=current_user.id, software_solution_id=software_solution_id, platform=platform)
        
        result = await export_service.export_software_solution(
            software_solution_id=software_solution_id,
            platform=platform,
            template_id=template_id,
            custom_fields=custom_fields
        )
        
        logger.info("Software solution exported", user_id=current_user.id, platform=platform, export_url=result["export_url"])
        
        return ExportResponse(
            success=result["success"],
            platform=result["platform"],
            export_url=result["export_url"],
            exported_at=result["exported_at"]
        )
        
    except ValueError as e:
        logger.error("Invalid request for software export", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to export software solution", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/calendar")
async def export_calendar_entries(
    start_date: datetime,
    end_date: datetime,
    platform: str,
    template_id: int,
    current_user: User = Depends(get_current_user),
    export_service: ExportService = Depends(get_export_service)
):
    """Export calendar entries to platform"""
    try:
        logger.info("Exporting calendar entries", user_id=current_user.id, platform=platform, start_date=start_date, end_date=end_date)
        
        result = await export_service.export_calendar_entries(
            user_id=current_user.id,
            start_date=start_date,
            end_date=end_date,
            platform=platform,
            template_id=template_id
        )
        
        logger.info("Calendar entries exported", user_id=current_user.id, platform=platform, entries_count=result["entries_count"])
        
        return ExportResponse(
            success=result["success"],
            platform=result["platform"],
            export_url=result["export_url"],
            exported_at=result["exported_at"],
            additional_data={"entries_count": result["entries_count"]}
        )
        
    except ValueError as e:
        logger.error("Invalid request for calendar export", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to export calendar entries", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/templates", response_model=ExportTemplateListResponse)
async def get_export_templates(
    platform: Optional[str] = None,
    content_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    export_service: ExportService = Depends(get_export_service)
):
    """Get available export templates"""
    try:
        templates = await export_service.get_export_templates(
            platform=platform,
            content_type=content_type
        )
        
        return ExportTemplateListResponse(
            templates=[
                ExportTemplateResponse(
                    id=template["id"],
                    name=template["name"],
                    platform=template["platform"],
                    content_type=template["content_type"],
                    description=template["description"],
                    fields=template["fields"],
                    created_at=template["created_at"]
                )
                for template in templates
            ],
            total=len(templates)
        )
        
    except Exception as e:
        logger.error("Failed to get export templates", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/templates/{template_id}", response_model=ExportTemplateResponse)
async def get_export_template(
    template_id: int,
    current_user: User = Depends(get_current_user),
    export_service: ExportService = Depends(get_export_service)
):
    """Get export template by ID"""
    try:
        template = await export_service.get_export_template(template_id)
        
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return ExportTemplateResponse(
            id=template["id"],
            name=template["name"],
            platform=template["platform"],
            content_type=template["content_type"],
            description=template["description"],
            fields=template["fields"],
            created_at=template["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get export template", template_id=template_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/templates", response_model=ExportTemplateResponse)
async def create_export_template(
    name: str,
    platform: str,
    content_type: str,
    description: str,
    fields: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    export_service: ExportService = Depends(get_export_service)
):
    """Create new export template"""
    try:
        logger.info("Creating export template", user_id=current_user.id, name=name, platform=platform)
        
        template = await export_service.create_export_template(
            user_id=current_user.id,
            name=name,
            platform=platform,
            content_type=content_type,
            description=description,
            fields=fields
        )
        
        logger.info("Export template created", user_id=current_user.id, template_id=template["id"])
        
        return ExportTemplateResponse(
            id=template["id"],
            name=template["name"],
            platform=template["platform"],
            content_type=template["content_type"],
            description=template["description"],
            fields=template["fields"],
            created_at=template["created_at"]
        )
        
    except ValueError as e:
        logger.error("Invalid request for template creation", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to create export template", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/templates/{template_id}", response_model=ExportTemplateResponse)
async def update_export_template(
    template_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
    fields: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_user),
    export_service: ExportService = Depends(get_export_service)
):
    """Update export template"""
    try:
        template = await export_service.get_export_template(template_id)
        
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        # Check if user owns this template
        if template["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        updated_template = await export_service.update_export_template(
            template_id=template_id,
            name=name,
            description=description,
            fields=fields
        )
        
        return ExportTemplateResponse(
            id=updated_template["id"],
            name=updated_template["name"],
            platform=updated_template["platform"],
            content_type=updated_template["content_type"],
            description=updated_template["description"],
            fields=updated_template["fields"],
            created_at=updated_template["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update export template", template_id=template_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/templates/{template_id}")
async def delete_export_template(
    template_id: int,
    current_user: User = Depends(get_current_user),
    export_service: ExportService = Depends(get_export_service)
):
    """Delete export template"""
    try:
        template = await export_service.get_export_template(template_id)
        
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        # Check if user owns this template
        if template["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        success = await export_service.delete_export_template(template_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete template")
        
        return {"message": "Template deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete export template", template_id=template_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status/{export_id}", response_model=ExportStatusResponse)
async def get_export_status(
    export_id: str,
    current_user: User = Depends(get_current_user),
    export_service: ExportService = Depends(get_export_service)
):
    """Get export status"""
    try:
        status = await export_service.get_export_status(export_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Export not found")
        
        return ExportStatusResponse(
            export_id=export_id,
            status=status["status"],
            progress=status["progress"],
            platform=status["platform"],
            export_url=status.get("export_url"),
            error_message=status.get("error_message"),
            created_at=status["created_at"],
            completed_at=status.get("completed_at")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get export status", export_id=export_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/platforms", response_model=List[Dict[str, Any]])
async def get_export_platforms():
    """Get available export platforms"""
    try:
        platforms = [
            {
                "id": "google_docs",
                "name": "Google Docs",
                "description": "Export to Google Docs documents",
                "supported_formats": ["docx", "pdf"],
                "features": ["Collaborative editing", "Version history", "Comments"]
            },
            {
                "id": "notion",
                "name": "Notion",
                "description": "Export to Notion pages",
                "supported_formats": ["notion_page"],
                "features": ["Database integration", "Templates", "Sharing"]
            },
            {
                "id": "wordpress",
                "name": "WordPress",
                "description": "Export to WordPress posts",
                "supported_formats": ["post", "page"],
                "features": ["SEO optimization", "Categories", "Tags"]
            }
        ]
        
        return platforms
        
    except Exception as e:
        logger.error("Failed to get export platforms", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/history", response_model=List[Dict[str, Any]])
async def get_export_history(
    skip: int = 0,
    limit: int = 20,
    platform: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    export_service: ExportService = Depends(get_export_service)
):
    """Get user's export history"""
    try:
        history = await export_service.get_export_history(
            user_id=current_user.id,
            skip=skip,
            limit=limit,
            platform=platform
        )
        
        return history
        
    except Exception as e:
        logger.error("Failed to get export history", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
