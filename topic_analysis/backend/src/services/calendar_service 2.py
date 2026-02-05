"""
Content Calendar Service
Manages content scheduling, calendar events, and publication tracking
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import structlog
from ..core.supabase_database import get_supabase_db
from ..core.redis import cache

logger = structlog.get_logger()

class CalendarService:
    """Service for content calendar management and scheduling"""
    
    def __init__(self):
        self.db = get_supabase_db()
        self.cache_ttl = 3600  # 1 hour cache TTL
    
    async def create_content_calendar(
        self,
        user_id: str,
        content_ideas: List[Dict[str, Any]],
        software_solutions: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Create a content calendar with scheduled content and software projects
        
        Args:
            user_id: User ID
            content_ideas: List of content ideas to schedule
            software_solutions: List of software solutions to schedule
            start_date: Calendar start date
            end_date: Calendar end date
            
        Returns:
            Dict containing calendar creation results
        """
        try:
            logger.info("Creating content calendar", 
                       user_id=user_id, 
                       content_count=len(content_ideas),
                       software_count=len(software_solutions))
            
            # Generate calendar entries
            calendar_entries = []
            
            # Schedule content ideas
            for i, content in enumerate(content_ideas):
                entry = {
                    "id": f"content_{i}_{datetime.utcnow().timestamp()}",
                    "type": "content",
                    "title": content.get("title", "Untitled Content"),
                    "description": content.get("description", ""),
                    "content_id": content.get("id"),
                    "scheduled_date": self._calculate_schedule_date(start_date, i),
                    "status": "scheduled",
                    "priority": content.get("priority_score", 0.5),
                    "estimated_time": content.get("estimated_time", "2 hours"),
                    "platform": content.get("platform", "blog"),
                    "keywords": content.get("keywords", []),
                    "seo_recommendations": content.get("seo_recommendations", [])
                }
                calendar_entries.append(entry)
            
            # Schedule software solutions
            for i, software in enumerate(software_solutions):
                entry = {
                    "id": f"software_{i}_{datetime.utcnow().timestamp()}",
                    "type": "software",
                    "title": software.get("title", "Untitled Software"),
                    "description": software.get("description", ""),
                    "software_id": software.get("id"),
                    "scheduled_date": self._calculate_schedule_date(start_date, len(content_ideas) + i),
                    "status": "scheduled",
                    "priority": software.get("priority_score", 0.5),
                    "estimated_time": self._calculate_development_time(software.get("complexity_score", 5)),
                    "platform": "web",
                    "complexity": software.get("complexity_score", 5),
                    "technical_requirements": software.get("technical_requirements", [])
                }
                calendar_entries.append(entry)
            
            # Save calendar to database
            calendar_data = {
                "user_id": user_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "entries": calendar_entries,
                "created_at": datetime.utcnow().isoformat(),
                "status": "active"
            }
            
            result = await self.db.create_calendar(calendar_data)
            
            logger.info("Content calendar created successfully", 
                       calendar_id=result.get("id"),
                       entries_count=len(calendar_entries))
            
            return {
                "success": True,
                "calendar_id": result.get("id"),
                "entries_count": len(calendar_entries),
                "content_entries": len([e for e in calendar_entries if e["type"] == "content"]),
                "software_entries": len([e for e in calendar_entries if e["type"] == "software"]),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "calendar_entries": calendar_entries
            }
            
        except Exception as e:
            logger.error("Failed to create content calendar", error=str(e))
            raise
    
    async def get_calendar_entries(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        entry_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get calendar entries for a user
        
        Args:
            user_id: User ID
            start_date: Optional start date filter
            end_date: Optional end date filter
            entry_type: Optional entry type filter (content/software)
            
        Returns:
            Dict containing calendar entries
        """
        try:
            logger.info("Getting calendar entries", 
                       user_id=user_id,
                       start_date=start_date.isoformat() if start_date else None,
                       end_date=end_date.isoformat() if end_date else None,
                       entry_type=entry_type)
            
            # Build query filters
            filters = {"user_id": user_id}
            if start_date:
                filters["start_date"] = start_date.isoformat()
            if end_date:
                filters["end_date"] = end_date.isoformat()
            if entry_type:
                filters["entry_type"] = entry_type
            
            # Get calendar entries from database
            entries = await self.db.get_calendar_entries(filters)
            
            # Group entries by date
            grouped_entries = self._group_entries_by_date(entries)
            
            logger.info("Retrieved calendar entries", 
                       total_entries=len(entries),
                       date_groups=len(grouped_entries))
            
            return {
                "success": True,
                "total_entries": len(entries),
                "date_groups": len(grouped_entries),
                "entries": entries,
                "grouped_entries": grouped_entries
            }
            
        except Exception as e:
            logger.error("Failed to get calendar entries", error=str(e))
            raise
    
    async def update_calendar_entry(
        self,
        user_id: str,
        entry_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update a calendar entry
        
        Args:
            user_id: User ID
            entry_id: Entry ID to update
            updates: Dictionary of updates to apply
            
        Returns:
            Dict containing update results
        """
        try:
            logger.info("Updating calendar entry", 
                       user_id=user_id,
                       entry_id=entry_id,
                       updates=updates)
            
            # Validate updates
            allowed_updates = [
                "title", "description", "scheduled_date", "status", 
                "priority", "estimated_time", "platform", "notes"
            ]
            
            filtered_updates = {
                k: v for k, v in updates.items() 
                if k in allowed_updates
            }
            
            if not filtered_updates:
                raise ValueError("No valid updates provided")
            
            # Add update timestamp
            filtered_updates["updated_at"] = datetime.utcnow().isoformat()
            
            # Update entry in database
            result = await self.db.update_calendar_entry(
                user_id=user_id,
                entry_id=entry_id,
                updates=filtered_updates
            )
            
            logger.info("Calendar entry updated successfully", 
                       entry_id=entry_id,
                       updates_applied=len(filtered_updates))
            
            return {
                "success": True,
                "entry_id": entry_id,
                "updates_applied": filtered_updates,
                "updated_entry": result
            }
            
        except Exception as e:
            logger.error("Failed to update calendar entry", error=str(e))
            raise
    
    async def delete_calendar_entry(
        self,
        user_id: str,
        entry_id: str
    ) -> Dict[str, Any]:
        """
        Delete a calendar entry
        
        Args:
            user_id: User ID
            entry_id: Entry ID to delete
            
        Returns:
            Dict containing deletion results
        """
        try:
            logger.info("Deleting calendar entry", 
                       user_id=user_id,
                       entry_id=entry_id)
            
            # Delete entry from database
            result = await self.db.delete_calendar_entry(
                user_id=user_id,
                entry_id=entry_id
            )
            
            logger.info("Calendar entry deleted successfully", 
                       entry_id=entry_id)
            
            return {
                "success": True,
                "entry_id": entry_id,
                "deleted_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to delete calendar entry", error=str(e))
            raise
    
    async def get_calendar_analytics(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get calendar analytics and insights
        
        Args:
            user_id: User ID
            start_date: Optional start date for analytics
            end_date: Optional end date for analytics
            
        Returns:
            Dict containing calendar analytics
        """
        try:
            logger.info("Getting calendar analytics", 
                       user_id=user_id,
                       start_date=start_date.isoformat() if start_date else None,
                       end_date=end_date.isoformat() if end_date else None)
            
            # Get calendar entries
            entries_result = await self.get_calendar_entries(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date
            )
            
            entries = entries_result.get("entries", [])
            
            # Calculate analytics
            analytics = {
                "total_entries": len(entries),
                "content_entries": len([e for e in entries if e.get("type") == "content"]),
                "software_entries": len([e for e in entries if e.get("type") == "software"]),
                "completed_entries": len([e for e in entries if e.get("status") == "completed"]),
                "scheduled_entries": len([e for e in entries if e.get("status") == "scheduled"]),
                "in_progress_entries": len([e for e in entries if e.get("status") == "in_progress"]),
                "overdue_entries": self._count_overdue_entries(entries),
                "completion_rate": self._calculate_completion_rate(entries),
                "average_priority": self._calculate_average_priority(entries),
                "time_distribution": self._analyze_time_distribution(entries),
                "platform_distribution": self._analyze_platform_distribution(entries)
            }
            
            logger.info("Calendar analytics calculated", 
                       total_entries=analytics["total_entries"],
                       completion_rate=analytics["completion_rate"])
            
            return {
                "success": True,
                "analytics": analytics,
                "period": {
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None
                }
            }
            
        except Exception as e:
            logger.error("Failed to get calendar analytics", error=str(e))
            raise
    
    def _calculate_schedule_date(self, start_date: datetime, offset_days: int) -> str:
        """Calculate schedule date with offset"""
        schedule_date = start_date + timedelta(days=offset_days)
        return schedule_date.isoformat()
    
    def _calculate_development_time(self, complexity_score: int) -> str:
        """Calculate estimated development time based on complexity"""
        if complexity_score <= 3:
            return "1-2 weeks"
        elif complexity_score <= 6:
            return "2-4 weeks"
        elif complexity_score <= 8:
            return "1-2 months"
        else:
            return "2-3 months"
    
    def _group_entries_by_date(self, entries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group entries by date"""
        grouped = {}
        for entry in entries:
            date = entry.get("scheduled_date", "").split("T")[0]
            if date not in grouped:
                grouped[date] = []
            grouped[date].append(entry)
        return grouped
    
    def _count_overdue_entries(self, entries: List[Dict[str, Any]]) -> int:
        """Count overdue entries"""
        now = datetime.utcnow()
        overdue = 0
        for entry in entries:
            if entry.get("status") in ["scheduled", "in_progress"]:
                scheduled_date = datetime.fromisoformat(entry.get("scheduled_date", "").replace("Z", ""))
                if scheduled_date < now:
                    overdue += 1
        return overdue
    
    def _calculate_completion_rate(self, entries: List[Dict[str, Any]]) -> float:
        """Calculate completion rate"""
        if not entries:
            return 0.0
        completed = len([e for e in entries if e.get("status") == "completed"])
        return completed / len(entries)
    
    def _calculate_average_priority(self, entries: List[Dict[str, Any]]) -> float:
        """Calculate average priority score"""
        if not entries:
            return 0.0
        priorities = [e.get("priority", 0.5) for e in entries if e.get("priority")]
        return sum(priorities) / len(priorities) if priorities else 0.0
    
    def _analyze_time_distribution(self, entries: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze time distribution of entries"""
        distribution = {"morning": 0, "afternoon": 0, "evening": 0}
        for entry in entries:
            scheduled_date = entry.get("scheduled_date", "")
            if scheduled_date:
                try:
                    dt = datetime.fromisoformat(scheduled_date.replace("Z", ""))
                    hour = dt.hour
                    if 6 <= hour < 12:
                        distribution["morning"] += 1
                    elif 12 <= hour < 18:
                        distribution["afternoon"] += 1
                    else:
                        distribution["evening"] += 1
                except:
                    pass
        return distribution
    
    def _analyze_platform_distribution(self, entries: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze platform distribution of entries"""
        distribution = {}
        for entry in entries:
            platform = entry.get("platform", "unknown")
            distribution[platform] = distribution.get(platform, 0) + 1
        return distribution