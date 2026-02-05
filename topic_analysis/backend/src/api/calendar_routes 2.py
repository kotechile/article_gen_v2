"""
Calendar API Routes
Provides endpoints for content calendar management
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog
from ..services.calendar_service import CalendarService
from ..core.supabase_database import get_supabase_db
from ..schemas.calendar_schemas import (
    CalendarCreateRequest,
    CalendarResponse,
    CalendarEntryUpdateRequest,
    CalendarAnalyticsResponse
)

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/calendar", tags=["calendar"])

@router.post("/create", response_model=CalendarResponse)
async def create_content_calendar(
    request: CalendarCreateRequest,
    db: Any = Depends(get_supabase_db)
):
    """
    Create a content calendar with scheduled content and software projects
    """
    try:
        logger.info("Creating content calendar", 
                   user_id=request.user_id,
                   content_count=len(request.content_ideas),
                   software_count=len(request.software_solutions))
        
        calendar_service = CalendarService()
        
        result = await calendar_service.create_content_calendar(
            user_id=request.user_id,
            content_ideas=request.content_ideas,
            software_solutions=request.software_solutions,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Failed to create calendar")
            )
        
        logger.info("Content calendar created successfully", 
                   calendar_id=result["calendar_id"])
        
        return CalendarResponse(
            success=True,
            calendar_id=result["calendar_id"],
            entries_count=result["entries_count"],
            content_entries=result["content_entries"],
            software_entries=result["software_entries"],
            start_date=result["start_date"],
            end_date=result["end_date"],
            calendar_entries=result["calendar_entries"]
        )
        
    except Exception as e:
        logger.error("Failed to create content calendar", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get("/entries", response_model=Dict[str, Any])
async def get_calendar_entries(
    user_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    entry_type: Optional[str] = None,
    db: Any = Depends(get_supabase_db)
):
    """
    Get calendar entries for a user with optional filtering
    """
    try:
        logger.info("Getting calendar entries", 
                   user_id=user_id,
                   start_date=start_date.isoformat() if start_date else None,
                   end_date=end_date.isoformat() if end_date else None,
                   entry_type=entry_type)
        
        calendar_service = CalendarService()
        
        result = await calendar_service.get_calendar_entries(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            entry_type=entry_type
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to get calendar entries"
            )
        
        logger.info("Calendar entries retrieved successfully", 
                   total_entries=result["total_entries"])
        
        return {
            "success": True,
            "total_entries": result["total_entries"],
            "date_groups": result["date_groups"],
            "entries": result["entries"],
            "grouped_entries": result["grouped_entries"]
        }
        
    except Exception as e:
        logger.error("Failed to get calendar entries", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.put("/entries/{entry_id}", response_model=Dict[str, Any])
async def update_calendar_entry(
    entry_id: str,
    request: CalendarEntryUpdateRequest,
    user_id: str,
    db: Any = Depends(get_supabase_db)
):
    """
    Update a calendar entry
    """
    try:
        logger.info("Updating calendar entry", 
                   entry_id=entry_id,
                   user_id=user_id,
                   updates=request.dict())
        
        calendar_service = CalendarService()
        
        result = await calendar_service.update_calendar_entry(
            user_id=user_id,
            entry_id=entry_id,
            updates=request.dict()
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update calendar entry"
            )
        
        logger.info("Calendar entry updated successfully", 
                   entry_id=entry_id)
        
        return {
            "success": True,
            "entry_id": result["entry_id"],
            "updates_applied": result["updates_applied"],
            "updated_entry": result["updated_entry"]
        }
        
    except Exception as e:
        logger.error("Failed to update calendar entry", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.delete("/entries/{entry_id}", response_model=Dict[str, Any])
async def delete_calendar_entry(
    entry_id: str,
    user_id: str,
    db: Any = Depends(get_supabase_db)
):
    """
    Delete a calendar entry
    """
    try:
        logger.info("Deleting calendar entry", 
                   entry_id=entry_id,
                   user_id=user_id)
        
        calendar_service = CalendarService()
        
        result = await calendar_service.delete_calendar_entry(
            user_id=user_id,
            entry_id=entry_id
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to delete calendar entry"
            )
        
        logger.info("Calendar entry deleted successfully", 
                   entry_id=entry_id)
        
        return {
            "success": True,
            "entry_id": result["entry_id"],
            "deleted_at": result["deleted_at"]
        }
        
    except Exception as e:
        logger.error("Failed to delete calendar entry", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get("/analytics", response_model=CalendarAnalyticsResponse)
async def get_calendar_analytics(
    user_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Any = Depends(get_supabase_db)
):
    """
    Get calendar analytics and insights
    """
    try:
        logger.info("Getting calendar analytics", 
                   user_id=user_id,
                   start_date=start_date.isoformat() if start_date else None,
                   end_date=end_date.isoformat() if end_date else None)
        
        calendar_service = CalendarService()
        
        result = await calendar_service.get_calendar_analytics(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to get calendar analytics"
            )
        
        logger.info("Calendar analytics retrieved successfully", 
                   user_id=user_id)
        
        return CalendarAnalyticsResponse(
            success=True,
            analytics=result["analytics"],
            period=result["period"]
        )
        
    except Exception as e:
        logger.error("Failed to get calendar analytics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

