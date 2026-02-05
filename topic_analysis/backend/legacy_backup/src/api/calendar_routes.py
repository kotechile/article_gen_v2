"""
Calendar Management API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import structlog

from ..core.database import get_db
from ..services.calendar_service import CalendarService
from ..models.user import User
from ..models.content_calendar import ContentCalendar, CalendarItemType, CalendarStatus
from ..schemas.calendar_schemas import (
    CalendarScheduleRequest,
    CalendarScheduleResponse,
    CalendarEntryResponse,
    CalendarEntryListResponse,
    CalendarUpdateRequest,
    CalendarAnalyticsResponse
)

logger = structlog.get_logger()
router = APIRouter(prefix="/api/calendar", tags=["calendar-management"])


def get_calendar_service(db: Session = Depends(get_db)) -> CalendarService:
    """Get calendar service dependency"""
    return CalendarService(db)


def get_current_user(db: Session = Depends(get_db)) -> User:
    """Get current authenticated user (placeholder - implement auth middleware)"""
    # This is a placeholder - in real implementation, this would extract user from JWT token
    user = db.query(User).first()
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


@router.post("/schedule", response_model=CalendarScheduleResponse)
async def schedule_content(
    request: CalendarScheduleRequest,
    current_user: User = Depends(get_current_user),
    calendar_service: CalendarService = Depends(get_calendar_service)
):
    """Schedule content for publication"""
    try:
        logger.info("Scheduling content", user_id=current_user.id, content_id=request.content_id, scheduled_date=request.scheduled_date)
        
        result = await calendar_service.schedule_content(
            user_id=current_user.id,
            content_id=request.content_id,
            scheduled_date=request.scheduled_date,
            content_type=request.content_type,
            notes=request.notes
        )
        
        logger.info("Content scheduled successfully", user_id=current_user.id, calendar_entry_id=result["calendar_entry_id"])
        
        return CalendarScheduleResponse(
            success=result["success"],
            calendar_entry_id=result["calendar_entry_id"],
            scheduled_date=result["scheduled_date"],
            status=result["status"]
        )
        
    except ValueError as e:
        logger.error("Invalid request for content scheduling", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to schedule content", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/schedule-software")
async def schedule_software_development(
    software_solution_id: str,
    planned_start_date: datetime,
    estimated_completion_date: datetime,
    notes: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    calendar_service: CalendarService = Depends(get_calendar_service)
):
    """Schedule software development"""
    try:
        logger.info("Scheduling software development", user_id=current_user.id, software_solution_id=software_solution_id)
        
        result = await calendar_service.schedule_software_development(
            user_id=current_user.id,
            software_solution_id=software_solution_id,
            planned_start_date=planned_start_date,
            estimated_completion_date=estimated_completion_date,
            notes=notes
        )
        
        logger.info("Software development scheduled", user_id=current_user.id, calendar_entry_id=result["calendar_entry_id"])
        
        return CalendarScheduleResponse(
            success=result["success"],
            calendar_entry_id=result["calendar_entry_id"],
            scheduled_date=result["planned_start_date"],
            status=result["status"]
        )
        
    except ValueError as e:
        logger.error("Invalid request for software scheduling", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to schedule software development", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/entries", response_model=CalendarEntryListResponse)
async def get_calendar_entries(
    start_date: datetime,
    end_date: datetime,
    content_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    calendar_service: CalendarService = Depends(get_calendar_service)
):
    """Get calendar entries for date range"""
    try:
        entries = await calendar_service.get_calendar_entries(
            user_id=current_user.id,
            start_date=start_date,
            end_date=end_date,
            content_type=content_type
        )
        
        return CalendarEntryListResponse(
            entries=[
                CalendarEntryResponse(
                    id=entry["id"],
                    item_type=entry["item_type"],
                    scheduled_date=entry["scheduled_date"],
                    status=entry["status"],
                    notes=entry["notes"],
                    created_at=entry["created_at"],
                    content=entry.get("content"),
                    software=entry.get("software")
                )
                for entry in entries
            ],
            total=len(entries)
        )
        
    except Exception as e:
        logger.error("Failed to get calendar entries", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/entries/{entry_id}", response_model=CalendarEntryResponse)
async def get_calendar_entry(
    entry_id: str,
    current_user: User = Depends(get_current_user),
    calendar_service: CalendarService = Depends(get_calendar_service)
):
    """Get calendar entry by ID"""
    try:
        entry = await calendar_service.get_calendar_entry(entry_id)
        
        if not entry:
            raise HTTPException(status_code=404, detail="Calendar entry not found")
        
        # Check if user owns this entry
        if entry["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return CalendarEntryResponse(
            id=entry["id"],
            item_type=entry["item_type"],
            scheduled_date=entry["scheduled_date"],
            status=entry["status"],
            notes=entry["notes"],
            created_at=entry["created_at"],
            content=entry.get("content"),
            software=entry.get("software")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get calendar entry", entry_id=entry_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/entries/{entry_id}", response_model=CalendarEntryResponse)
async def update_calendar_entry(
    entry_id: str,
    request: CalendarUpdateRequest,
    current_user: User = Depends(get_current_user),
    calendar_service: CalendarService = Depends(get_calendar_service)
):
    """Update calendar entry"""
    try:
        entry = await calendar_service.get_calendar_entry(entry_id)
        
        if not entry:
            raise HTTPException(status_code=404, detail="Calendar entry not found")
        
        # Check if user owns this entry
        if entry["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        success = await calendar_service.update_calendar_entry(
            calendar_entry_id=entry_id,
            user_id=current_user.id,
            scheduled_date=request.scheduled_date,
            status=request.status,
            notes=request.notes
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update calendar entry")
        
        # Get updated entry
        updated_entry = await calendar_service.get_calendar_entry(entry_id)
        
        return CalendarEntryResponse(
            id=updated_entry["id"],
            item_type=updated_entry["item_type"],
            scheduled_date=updated_entry["scheduled_date"],
            status=updated_entry["status"],
            notes=updated_entry["notes"],
            created_at=updated_entry["created_at"],
            content=updated_entry.get("content"),
            software=updated_entry.get("software")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update calendar entry", entry_id=entry_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/entries/{entry_id}")
async def delete_calendar_entry(
    entry_id: str,
    current_user: User = Depends(get_current_user),
    calendar_service: CalendarService = Depends(get_calendar_service)
):
    """Delete calendar entry"""
    try:
        entry = await calendar_service.get_calendar_entry(entry_id)
        
        if not entry:
            raise HTTPException(status_code=404, detail="Calendar entry not found")
        
        # Check if user owns this entry
        if entry["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        success = await calendar_service.delete_calendar_entry(entry_id, current_user.id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete calendar entry")
        
        return {"message": "Calendar entry deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete calendar entry", entry_id=entry_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/reminders", response_model=List[Dict[str, Any]])
async def get_upcoming_reminders(
    hours_ahead: int = 24,
    current_user: User = Depends(get_current_user),
    calendar_service: CalendarService = Depends(get_calendar_service)
):
    """Get upcoming reminders"""
    try:
        reminders = await calendar_service.get_upcoming_reminders(
            user_id=current_user.id,
            hours_ahead=hours_ahead
        )
        
        return reminders
        
    except Exception as e:
        logger.error("Failed to get upcoming reminders", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/auto-schedule")
async def auto_schedule_content(
    content_ideas: List[Dict[str, Any]],
    start_date: datetime,
    frequency_days: int = 7,
    current_user: User = Depends(get_current_user),
    calendar_service: CalendarService = Depends(get_calendar_service)
):
    """Auto-schedule content based on priority and frequency"""
    try:
        logger.info("Auto-scheduling content", user_id=current_user.id, content_count=len(content_ideas))
        
        scheduled_entries = await calendar_service.auto_schedule_content(
            user_id=current_user.id,
            content_ideas=content_ideas,
            start_date=start_date,
            frequency_days=frequency_days
        )
        
        logger.info("Content auto-scheduled", user_id=current_user.id, scheduled_count=len(scheduled_entries))
        
        return {
            "success": True,
            "scheduled_entries": scheduled_entries,
            "scheduled_count": len(scheduled_entries)
        }
        
    except Exception as e:
        logger.error("Failed to auto-schedule content", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics", response_model=CalendarAnalyticsResponse)
async def get_calendar_analytics(
    start_date: datetime,
    end_date: datetime,
    current_user: User = Depends(get_current_user),
    calendar_service: CalendarService = Depends(get_calendar_service)
):
    """Get calendar analytics"""
    try:
        analytics = await calendar_service.get_calendar_analytics(
            user_id=current_user.id,
            start_date=start_date,
            end_date=end_date
        )
        
        return CalendarAnalyticsResponse(
            total_entries=analytics["total_entries"],
            content_entries=analytics["content_entries"],
            software_entries=analytics["software_entries"],
            status_breakdown=analytics["status_breakdown"],
            monthly_breakdown=analytics["monthly_breakdown"],
            completion_rate=analytics["completion_rate"],
            date_range=analytics["date_range"]
        )
        
    except Exception as e:
        logger.error("Failed to get calendar analytics", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sync-status", response_model=Dict[str, Any])
async def get_sync_status(
    current_user: User = Depends(get_current_user),
    calendar_service: CalendarService = Depends(get_calendar_service)
):
    """Get external calendar sync status"""
    try:
        sync_status = await calendar_service.get_sync_status(current_user.id)
        
        return sync_status
        
    except Exception as e:
        logger.error("Failed to get sync status", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/sync")
async def sync_with_external_calendars(
    current_user: User = Depends(get_current_user),
    calendar_service: CalendarService = Depends(get_calendar_service)
):
    """Sync with external calendars"""
    try:
        logger.info("Syncing with external calendars", user_id=current_user.id)
        
        result = await calendar_service.sync_with_external_calendars(current_user.id)
        
        logger.info("External calendar sync completed", user_id=current_user.id, synced_count=result.get("synced_count", 0))
        
        return {
            "success": True,
            "synced_count": result.get("synced_count", 0),
            "sync_errors": result.get("sync_errors", [])
        }
        
    except Exception as e:
        logger.error("Failed to sync with external calendars", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/templates", response_model=List[Dict[str, Any]])
async def get_calendar_templates():
    """Get available calendar templates"""
    try:
        templates = [
            {
                "id": "content_calendar",
                "name": "Content Calendar",
                "description": "Standard content publishing schedule",
                "frequency": "weekly",
                "default_days": ["Monday", "Wednesday", "Friday"]
            },
            {
                "id": "software_development",
                "name": "Software Development",
                "description": "Software development project timeline",
                "frequency": "daily",
                "default_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            },
            {
                "id": "mixed_content",
                "name": "Mixed Content",
                "description": "Mix of content and software development",
                "frequency": "flexible",
                "default_days": ["Monday", "Wednesday", "Friday"]
            }
        ]
        
        return templates
        
    except Exception as e:
        logger.error("Failed to get calendar templates", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/export")
async def export_calendar(
    start_date: datetime,
    end_date: datetime,
    format: str = "csv",
    current_user: User = Depends(get_current_user),
    calendar_service: CalendarService = Depends(get_calendar_service)
):
    """Export calendar to file"""
    try:
        if format not in ["csv", "json", "ics"]:
            raise HTTPException(status_code=400, detail="Unsupported export format")
        
        export_data = await calendar_service.export_calendar(
            user_id=current_user.id,
            start_date=start_date,
            end_date=end_date,
            format=format
        )
        
        return {
            "success": True,
            "format": format,
            "download_url": export_data["download_url"],
            "expires_at": export_data["expires_at"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to export calendar", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
