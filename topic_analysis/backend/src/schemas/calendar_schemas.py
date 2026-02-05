"""
Calendar Schemas
Pydantic models for calendar-related requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..models.content_calendar import EntryType, CalendarStatus

class CalendarCreateRequest(BaseModel):
    """Request model for creating a content calendar"""
    user_id: str = Field(..., description="User ID")
    content_ideas: List[Dict[str, Any]] = Field(..., description="List of content ideas to schedule")
    software_solutions: List[Dict[str, Any]] = Field(..., description="List of software solutions to schedule")
    start_date: datetime = Field(..., description="Calendar start date")
    end_date: datetime = Field(..., description="Calendar end date")

class CalendarEntryUpdateRequest(BaseModel):
    """Request model for updating a calendar entry"""
    title: Optional[str] = Field(None, description="Entry title")
    description: Optional[str] = Field(None, description="Entry description")
    scheduled_date: Optional[datetime] = Field(None, description="Scheduled date")
    status: Optional[str] = Field(None, description="Entry status")
    priority: Optional[float] = Field(None, description="Entry priority")
    estimated_time: Optional[str] = Field(None, description="Estimated time")
    platform: Optional[str] = Field(None, description="Platform")
    notes: Optional[str] = Field(None, description="Additional notes")

class CalendarResponse(BaseModel):
    """Response model for calendar operations"""
    success: bool = Field(..., description="Operation success status")
    calendar_id: Optional[str] = Field(None, description="Calendar ID")
    entries_count: Optional[int] = Field(None, description="Total number of entries")
    content_entries: Optional[int] = Field(None, description="Number of content entries")
    software_entries: Optional[int] = Field(None, description="Number of software entries")
    start_date: Optional[str] = Field(None, description="Start date")
    end_date: Optional[str] = Field(None, description="End date")
    calendar_entries: Optional[List[Dict[str, Any]]] = Field(None, description="Calendar entries")

class CalendarAnalyticsResponse(BaseModel):
    """Response model for calendar analytics"""
    success: bool = Field(..., description="Operation success status")
    analytics: Dict[str, Any] = Field(..., description="Analytics data")
    period: Dict[str, Any] = Field(..., description="Analytics period")

class CalendarEntry(BaseModel):
    """Model for a calendar entry"""
    id: str = Field(..., description="Entry ID")
    type: str = Field(..., description="Entry type (content/software)")
    title: str = Field(..., description="Entry title")
    description: str = Field(..., description="Entry description")
    scheduled_date: str = Field(..., description="Scheduled date")
    status: str = Field(..., description="Entry status")
    priority: float = Field(..., description="Entry priority")
    estimated_time: str = Field(..., description="Estimated time")
    platform: str = Field(..., description="Platform")
    keywords: Optional[List[str]] = Field(None, description="Keywords")
    seo_recommendations: Optional[List[str]] = Field(None, description="SEO recommendations")
    complexity: Optional[int] = Field(None, description="Complexity score")
    technical_requirements: Optional[List[str]] = Field(None, description="Technical requirements")

class CalendarAnalytics(BaseModel):
    """Model for calendar analytics"""
    total_entries: int = Field(..., description="Total number of entries")
    content_entries: int = Field(..., description="Number of content entries")
    software_entries: int = Field(..., description="Number of software entries")
    completed_entries: int = Field(..., description="Number of completed entries")
    scheduled_entries: int = Field(..., description="Number of scheduled entries")
    in_progress_entries: int = Field(..., description="Number of in-progress entries")
    overdue_entries: int = Field(..., description="Number of overdue entries")
    completion_rate: float = Field(..., description="Completion rate")
    average_priority: float = Field(..., description="Average priority")
    time_distribution: Dict[str, int] = Field(..., description="Time distribution")
    platform_distribution: Dict[str, int] = Field(..., description="Platform distribution")

class CalendarScheduleRequest(BaseModel):
    """Request model for scheduling calendar entries"""
    content_idea_id: Optional[int] = Field(None, description="Content idea ID")
    software_solution_id: Optional[int] = Field(None, description="Software solution ID")
    title: str = Field(..., description="Entry title")
    description: Optional[str] = Field(None, description="Entry description")
    entry_type: EntryType = Field(..., description="Entry type")
    scheduled_date: datetime = Field(..., description="Scheduled date")
    status: CalendarStatus = Field(default=CalendarStatus.SCHEDULED, description="Entry status")
    priority: int = Field(default=3, description="Priority (1-5)")
    platform: Optional[str] = Field(None, description="Platform")
    notes: Optional[str] = Field(None, description="Additional notes")

class CalendarScheduleResponse(BaseModel):
    """Response model for calendar scheduling"""
    success: bool = Field(..., description="Operation success status")
    calendar_id: Optional[int] = Field(None, description="Calendar entry ID")
    message: Optional[str] = Field(None, description="Response message")

class CalendarEntryResponse(BaseModel):
    """Response model for calendar entry"""
    id: int = Field(..., description="Entry ID")
    user_id: int = Field(..., description="User ID")
    content_idea_id: Optional[int] = Field(None, description="Content idea ID")
    software_solution_id: Optional[int] = Field(None, description="Software solution ID")
    title: str = Field(..., description="Entry title")
    description: Optional[str] = Field(None, description="Entry description")
    entry_type: str = Field(..., description="Entry type")
    platform: Optional[str] = Field(None, description="Platform")
    scheduled_date: str = Field(..., description="Scheduled date")
    status: str = Field(..., description="Entry status")
    priority: int = Field(..., description="Priority")
    progress_percentage: int = Field(..., description="Progress percentage")
    current_phase: Optional[str] = Field(None, description="Current phase")
    notes: Optional[str] = Field(None, description="Notes")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")

class CalendarEntryListResponse(BaseModel):
    """Response model for calendar entry list"""
    success: bool = Field(..., description="Operation success status")
    entries: List[CalendarEntryResponse] = Field(..., description="Calendar entries")
    total_count: int = Field(..., description="Total number of entries")
    page: int = Field(..., description="Current page")
    per_page: int = Field(..., description="Entries per page")

class CalendarUpdateRequest(BaseModel):
    """Request model for updating calendar entries"""
    title: Optional[str] = Field(None, description="Entry title")
    description: Optional[str] = Field(None, description="Entry description")
    scheduled_date: Optional[datetime] = Field(None, description="Scheduled date")
    status: Optional[CalendarStatus] = Field(None, description="Entry status")
    priority: Optional[int] = Field(None, description="Priority (1-5)")
    platform: Optional[str] = Field(None, description="Platform")
    progress_percentage: Optional[int] = Field(None, description="Progress percentage (0-100)")
    current_phase: Optional[str] = Field(None, description="Current phase")
    notes: Optional[str] = Field(None, description="Additional notes")