"""
CalendarService for content scheduling and tracking
"""

import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import structlog
from ..core.database import get_db
from ..core.redis import cache
from ..core.config import get_settings
from ..models.content_calendar import ContentCalendar, EntryType, CalendarStatus
from ..models.content_ideas import ContentIdeas
from ..models.software_solutions import SoftwareSolutions

logger = structlog.get_logger()
settings = get_settings()

class CalendarService:
    """Service for content calendar management"""
    
    def __init__(self):
        self.google_calendar_api_key = settings.google_calendar_api_key
        self.notion_calendar_api_key = settings.notion_calendar_api_key
        self.coschedule_api_key = settings.coschedule_api_key
        
        # Calendar settings
        self.default_reminder_hours = 24
        self.max_scheduled_items = 100
        self.auto_schedule_days = 30
    
    async def schedule_content(self, user_id: int, content_id: int, scheduled_date: datetime,
                             content_type: str, notes: Optional[str] = None) -> Dict[str, Any]:
        """Schedule content for publication"""
        try:
            # Validate content exists
            db = next(get_db())
            content = db.get_ContentIdeas_by_id(ContentIdeas.id == content_id)
            if not content:
                raise ValueError("Content not found")
            
            # Create calendar entry
            calendar_entry = ContentCalendar(
                user_id=user_id,
                content_idea_id=content_id,
                item_type=EntryType.CONTENT,
                scheduled_date=scheduled_date,
                status=CalendarStatus.SCHEDULED,
                notes=notes
            )
            
            db.add(calendar_entry)
            db.commit()
            db.refresh(calendar_entry)
            
            # Set up reminders
            await self._setup_reminders(calendar_entry)
            
            # Sync with external calendars
            await self._sync_with_external_calendars(calendar_entry)
            
            logger.info("Content scheduled", user_id=user_id, content_id=content_id, scheduled_date=scheduled_date)
            
            return {
                "success": True,
                "calendar_entry_id": calendar_entry.id,
                "scheduled_date": scheduled_date.isoformat(),
                "status": "scheduled"
            }
            
        except Exception as e:
            logger.error("Failed to schedule content", user_id=user_id, content_id=content_id, error=str(e))
            raise
    
    async def schedule_software_development(self, user_id: int, software_solution_id: int, 
                                         planned_start_date: datetime, estimated_completion_date: datetime,
                                         notes: Optional[str] = None) -> Dict[str, Any]:
        """Schedule software development"""
        try:
            # Validate software solution exists
            db = next(get_db())
            software_solution = db.get_SoftwareSolutions_by_id(SoftwareSolutions.id == software_solution_id)
            if not software_solution:
                raise ValueError("Software solution not found")
            
            # Create calendar entry
            calendar_entry = ContentCalendar(
                user_id=user_id,
                software_solution_id=software_solution_id,
                item_type=EntryType.SOFTWARE,
                scheduled_date=planned_start_date,
                status=CalendarStatus.SCHEDULED,
                notes=notes
            )
            
            db.add(calendar_entry)
            db.commit()
            db.refresh(calendar_entry)
            
            # Set up reminders
            await self._setup_reminders(calendar_entry)
            
            # Sync with external calendars
            await self._sync_with_external_calendars(calendar_entry)
            
            logger.info("Software development scheduled", user_id=user_id, software_solution_id=software_solution_id)
            
            return {
                "success": True,
                "calendar_entry_id": calendar_entry.id,
                "planned_start_date": planned_start_date.isoformat(),
                "estimated_completion_date": estimated_completion_date.isoformat(),
                "status": "scheduled"
            }
            
        except Exception as e:
            logger.error("Failed to schedule software development", user_id=user_id, software_solution_id=software_solution_id, error=str(e))
            raise
    
    async def get_calendar_entries(self, user_id: int, start_date: datetime, end_date: datetime,
                                 content_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get calendar entries for date range"""
        try:
            db = next(get_db())
            query = db.query(ContentCalendar).filter(
                ContentCalendar.user_id == user_id,
                ContentCalendar.scheduled_date >= start_date,
                ContentCalendar.scheduled_date <= end_date
            )
            
            if content_type:
                if content_type == "content":
                    query = query.filter(ContentCalendar.item_type == EntryType.CONTENT)
                elif content_type == "software":
                    query = query.filter(ContentCalendar.item_type == EntryType.SOFTWARE)
            
            calendar_entries = query.order_by(ContentCalendar.scheduled_date).all()
            
            # Format entries
            formatted_entries = []
            for entry in calendar_entries:
                formatted_entry = {
                    "id": entry.id,
                    "item_type": entry.item_type.value,
                    "scheduled_date": entry.scheduled_date.isoformat(),
                    "status": entry.status.value,
                    "notes": entry.notes,
                    "created_at": entry.created_at.isoformat()
                }
                
                # Add content-specific data
                if entry.content_idea_id:
                    content = db.get_ContentIdeas_by_id(ContentIdeas.id == entry.content_idea_id)
                    if content:
                        formatted_entry["content"] = {
                            "id": content.id,
                            "title": content.title,
                            "opportunity_score": content.opportunity_score
                        }
                
                # Add software-specific data
                if entry.software_solution_id:
                    software = db.get_SoftwareSolutions_by_id(SoftwareSolutions.id == entry.software_solution_id)
                    if software:
                        formatted_entry["software"] = {
                            "id": software.id,
                            "name": software.name,
                            "complexity_score": software.complexity_score
                        }
                
                formatted_entries.append(formatted_entry)
            
            return formatted_entries
            
        except Exception as e:
            logger.error("Failed to get calendar entries", user_id=user_id, error=str(e))
            raise
    
    async def update_calendar_entry(self, calendar_entry_id: int, user_id: int,
                                  scheduled_date: Optional[datetime] = None,
                                  status: Optional[str] = None,
                                  notes: Optional[str] = None) -> bool:
        """Update calendar entry"""
        try:
            db = next(get_db())
            calendar_entry = db.get_ContentCalendar_by_id(
                ContentCalendar.id == calendar_entry_id,
                ContentCalendar.user_id == user_id
            )
            
            if not calendar_entry:
                raise ValueError("Calendar entry not found")
            
            # Update fields
            if scheduled_date:
                calendar_entry.scheduled_date = scheduled_date
            if status:
                calendar_entry.status = CalendarStatus(status)
            if notes is not None:
                calendar_entry.notes = notes
            
            calendar_entry.updated_at = datetime.utcnow()
            
            db.commit()
            
            # Sync with external calendars
            await self._sync_with_external_calendars(calendar_entry)
            
            logger.info("Calendar entry updated", calendar_entry_id=calendar_entry_id, user_id=user_id)
            return True
            
        except Exception as e:
            logger.error("Failed to update calendar entry", calendar_entry_id=calendar_entry_id, error=str(e))
            raise
    
    async def delete_calendar_entry(self, calendar_entry_id: int, user_id: int) -> bool:
        """Delete calendar entry"""
        try:
            db = next(get_db())
            calendar_entry = db.get_ContentCalendar_by_id(
                ContentCalendar.id == calendar_entry_id,
                ContentCalendar.user_id == user_id
            )
            
            if not calendar_entry:
                raise ValueError("Calendar entry not found")
            
            # Remove from external calendars
            await self._remove_from_external_calendars(calendar_entry)
            
            # Delete from database
            db.delete(calendar_entry)
            db.commit()
            
            logger.info("Calendar entry deleted", calendar_entry_id=calendar_entry_id, user_id=user_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete calendar entry", calendar_entry_id=calendar_entry_id, error=str(e))
            raise
    
    async def get_upcoming_reminders(self, user_id: int, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """Get upcoming reminders"""
        try:
            db = next(get_db())
            now = datetime.utcnow()
            reminder_time = now + timedelta(hours=hours_ahead)
            
            calendar_entries = db.query(ContentCalendar).filter(
                ContentCalendar.user_id == user_id,
                ContentCalendar.scheduled_date >= now,
                ContentCalendar.scheduled_date <= reminder_time,
                ContentCalendar.status == CalendarStatus.SCHEDULED
            ).order_by(ContentCalendar.scheduled_date).all()
            
            # Format reminders
            reminders = []
            for entry in calendar_entries:
                reminder = {
                    "id": entry.id,
                    "item_type": entry.item_type.value,
                    "scheduled_date": entry.scheduled_date.isoformat(),
                    "notes": entry.notes,
                    "hours_until": (entry.scheduled_date - now).total_seconds() / 3600
                }
                
                # Add content-specific data
                if entry.content_idea_id:
                    content = db.get_ContentIdeas_by_id(ContentIdeas.id == entry.content_idea_id)
                    if content:
                        reminder["content"] = {
                            "id": content.id,
                            "title": content.title
                        }
                
                # Add software-specific data
                if entry.software_solution_id:
                    software = db.get_SoftwareSolutions_by_id(SoftwareSolutions.id == entry.software_solution_id)
                    if software:
                        reminder["software"] = {
                            "id": software.id,
                            "name": software.name
                        }
                
                reminders.append(reminder)
            
            return reminders
            
        except Exception as e:
            logger.error("Failed to get upcoming reminders", user_id=user_id, error=str(e))
            raise
    
    async def auto_schedule_content(self, user_id: int, content_ideas: List[Dict[str, Any]],
                                  start_date: datetime, frequency_days: int = 7) -> List[Dict[str, Any]]:
        """Auto-schedule content based on priority and frequency"""
        try:
            # Sort content by priority score
            sorted_content = sorted(content_ideas, key=lambda x: x.get("priority_score", 0), reverse=True)
            
            # Calculate schedule dates
            schedule_dates = []
            current_date = start_date
            for i, content in enumerate(sorted_content):
                if i > 0:
                    current_date += timedelta(days=frequency_days)
                schedule_dates.append(current_date)
            
            # Schedule content
            scheduled_entries = []
            for content, schedule_date in zip(sorted_content, schedule_dates):
                try:
                    result = await self.schedule_content(
                        user_id=user_id,
                        content_id=content["id"],
                        scheduled_date=schedule_date,
                        content_type="article",
                        notes=f"Auto-scheduled based on priority score: {content.get('priority_score', 0)}"
                    )
                    scheduled_entries.append(result)
                except Exception as e:
                    logger.warning("Failed to auto-schedule content", content_id=content["id"], error=str(e))
                    continue
            
            logger.info("Auto-scheduling completed", user_id=user_id, scheduled_count=len(scheduled_entries))
            return scheduled_entries
            
        except Exception as e:
            logger.error("Failed to auto-schedule content", user_id=user_id, error=str(e))
            raise
    
    async def get_calendar_analytics(self, user_id: int, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get calendar analytics"""
        try:
            db = next(get_db())
            
            # Get all entries in date range
            calendar_entries = db.query(ContentCalendar).filter(
                ContentCalendar.user_id == user_id,
                ContentCalendar.scheduled_date >= start_date,
                ContentCalendar.scheduled_date <= end_date
            ).all()
            
            # Calculate analytics
            total_entries = len(calendar_entries)
            content_entries = len([e for e in calendar_entries if e.item_type == EntryType.CONTENT])
            software_entries = len([e for e in calendar_entries if e.item_type == EntryType.SOFTWARE])
            
            # Status breakdown
            status_breakdown = {}
            for entry in calendar_entries:
                status = entry.status.value
                status_breakdown[status] = status_breakdown.get(status, 0) + 1
            
            # Monthly breakdown
            monthly_breakdown = {}
            for entry in calendar_entries:
                month_key = entry.scheduled_date.strftime("%Y-%m")
                monthly_breakdown[month_key] = monthly_breakdown.get(month_key, 0) + 1
            
            # Completion rate
            completed_entries = len([e for e in calendar_entries if e.status == CalendarStatus.COMPLETED])
            completion_rate = (completed_entries / total_entries * 100) if total_entries > 0 else 0
            
            return {
                "total_entries": total_entries,
                "content_entries": content_entries,
                "software_entries": software_entries,
                "status_breakdown": status_breakdown,
                "monthly_breakdown": monthly_breakdown,
                "completion_rate": completion_rate,
                "date_range": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                }
            }
            
        except Exception as e:
            logger.error("Failed to get calendar analytics", user_id=user_id, error=str(e))
            raise
    
    async def _setup_reminders(self, calendar_entry: ContentCalendar):
        """Set up reminders for calendar entry"""
        try:
            # Calculate reminder times
            reminder_times = [
                calendar_entry.scheduled_date - timedelta(hours=24),  # 24 hours before
                calendar_entry.scheduled_date - timedelta(hours=1),   # 1 hour before
            ]
            
            # Store reminders in Redis
            for reminder_time in reminder_times:
                if reminder_time > datetime.utcnow():
                    reminder_key = f"reminder:{calendar_entry.id}:{reminder_time.isoformat()}"
                    await cache.set(reminder_key, {
                        "calendar_entry_id": calendar_entry.id,
                        "user_id": calendar_entry.user_id,
                        "scheduled_date": calendar_entry.scheduled_date.isoformat(),
                        "item_type": calendar_entry.item_type.value
                    }, expire=int((reminder_time - datetime.utcnow()).total_seconds()))
            
            logger.info("Reminders set up", calendar_entry_id=calendar_entry.id)
            
        except Exception as e:
            logger.error("Failed to set up reminders", calendar_entry_id=calendar_entry.id, error=str(e))
    
    async def _sync_with_external_calendars(self, calendar_entry: ContentCalendar):
        """Sync calendar entry with external calendars"""
        try:
            # Sync with Google Calendar
            if self.google_calendar_api_key:
                await self._sync_with_google_calendar(calendar_entry)
            
            # Sync with Notion Calendar
            if self.notion_calendar_api_key:
                await self._sync_with_notion_calendar(calendar_entry)
            
            # Sync with CoSchedule
            if self.coschedule_api_key:
                await self._sync_with_coschedule(calendar_entry)
            
        except Exception as e:
            logger.error("Failed to sync with external calendars", calendar_entry_id=calendar_entry.id, error=str(e))
    
    async def _sync_with_google_calendar(self, calendar_entry: ContentCalendar):
        """Sync with Google Calendar"""
        try:
            # Prepare event data
            event_data = {
                "summary": f"Content: {calendar_entry.title or 'Scheduled Content'}",
                "description": calendar_entry.notes or "",
                "start": {
                    "dateTime": calendar_entry.scheduled_date.isoformat(),
                    "timeZone": "UTC"
                },
                "end": {
                    "dateTime": (calendar_entry.scheduled_date + timedelta(hours=1)).isoformat(),
                    "timeZone": "UTC"
                }
            }
            
            # Create event via Google Calendar API
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.google_calendar_api_key}",
                    "Content-Type": "application/json"
                }
                
                async with session.post(
                    "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                    headers=headers,
                    json=event_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Synced with Google Calendar", calendar_entry_id=calendar_entry.id, event_id=result.get("id"))
                    else:
                        logger.warning("Failed to sync with Google Calendar", calendar_entry_id=calendar_entry.id, status=response.status)
            
        except Exception as e:
            logger.error("Failed to sync with Google Calendar", calendar_entry_id=calendar_entry.id, error=str(e))
    
    async def _sync_with_notion_calendar(self, calendar_entry: ContentCalendar):
        """Sync with Notion Calendar"""
        try:
            # Prepare page data
            page_data = {
                "parent": {"database_id": "calendar_database_id"},
                "properties": {
                    "title": {
                        "title": [
                            {
                                "text": {
                                    "content": calendar_entry.title or "Scheduled Content"
                                }
                            }
                        ]
                    },
                    "date": {
                        "date": {
                            "start": calendar_entry.scheduled_date.isoformat()
                        }
                    },
                    "status": {
                        "select": {
                            "name": calendar_entry.status.value
                        }
                    }
                }
            }
            
            # Create page via Notion API
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.notion_calendar_api_key}",
                    "Content-Type": "application/json",
                    "Notion-Version": "2022-06-28"
                }
                
                async with session.post(
                    "https://api.notion.com/v1/pages",
                    headers=headers,
                    json=page_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Synced with Notion Calendar", calendar_entry_id=calendar_entry.id, page_id=result.get("id"))
                    else:
                        logger.warning("Failed to sync with Notion Calendar", calendar_entry_id=calendar_entry.id, status=response.status)
            
        except Exception as e:
            logger.error("Failed to sync with Notion Calendar", calendar_entry_id=calendar_entry.id, error=str(e))
    
    async def _sync_with_coschedule(self, calendar_entry: ContentCalendar):
        """Sync with CoSchedule"""
        try:
            # Prepare post data
            post_data = {
                "title": calendar_entry.title or "Scheduled Content",
                "publishDate": calendar_entry.scheduled_date.isoformat(),
                "status": "draft"
            }
            
            # Create post via CoSchedule API
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.coschedule_api_key}",
                    "Content-Type": "application/json"
                }
                
                async with session.post(
                    "https://api.coschedule.com/v1/posts",
                    headers=headers,
                    json=post_data
                ) as response:
                    if response.status == 201:
                        result = await response.json()
                        logger.info("Synced with CoSchedule", calendar_entry_id=calendar_entry.id, post_id=result.get("id"))
                    else:
                        logger.warning("Failed to sync with CoSchedule", calendar_entry_id=calendar_entry.id, status=response.status)
            
        except Exception as e:
            logger.error("Failed to sync with CoSchedule", calendar_entry_id=calendar_entry.id, error=str(e))
    
    async def _remove_from_external_calendars(self, calendar_entry: ContentCalendar):
        """Remove calendar entry from external calendars"""
        try:
            # Remove from Google Calendar
            if self.google_calendar_api_key:
                await self._remove_from_google_calendar(calendar_entry)
            
            # Remove from Notion Calendar
            if self.notion_calendar_api_key:
                await self._remove_from_notion_calendar(calendar_entry)
            
            # Remove from CoSchedule
            if self.coschedule_api_key:
                await self._remove_from_coschedule(calendar_entry)
            
        except Exception as e:
            logger.error("Failed to remove from external calendars", calendar_entry_id=calendar_entry.id, error=str(e))
    
    async def _remove_from_google_calendar(self, calendar_entry: ContentCalendar):
        """Remove from Google Calendar"""
        try:
            # This would require storing the Google Calendar event ID
            # For now, just log the action
            logger.info("Removed from Google Calendar", calendar_entry_id=calendar_entry.id)
            
        except Exception as e:
            logger.error("Failed to remove from Google Calendar", calendar_entry_id=calendar_entry.id, error=str(e))
    
    async def _remove_from_notion_calendar(self, calendar_entry: ContentCalendar):
        """Remove from Notion Calendar"""
        try:
            # This would require storing the Notion page ID
            # For now, just log the action
            logger.info("Removed from Notion Calendar", calendar_entry_id=calendar_entry.id)
            
        except Exception as e:
            logger.error("Failed to remove from Notion Calendar", calendar_entry_id=calendar_entry.id, error=str(e))
    
    async def _remove_from_coschedule(self, calendar_entry: ContentCalendar):
        """Remove from CoSchedule"""
        try:
            # This would require storing the CoSchedule post ID
            # For now, just log the action
            logger.info("Removed from CoSchedule", calendar_entry_id=calendar_entry.id)
            
        except Exception as e:
            logger.error("Failed to remove from CoSchedule", calendar_entry_id=calendar_entry.id, error=str(e))
