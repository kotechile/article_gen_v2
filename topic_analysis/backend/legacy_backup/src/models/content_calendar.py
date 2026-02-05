"""
ContentCalendar model for TrendTap
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Enum, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from ..core.database import Base


class EntryType(PyEnum):
    """Entry type enumeration"""
    CONTENT = "content"
    SOFTWARE_PROJECT = "software_project"


class CalendarItemType(PyEnum):
    """Calendar item type enumeration"""
    CONTENT = "content"
    SOFTWARE = "software"


class CalendarStatus(PyEnum):
    """Calendar status enumeration"""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"


class ContentCalendar(Base):
    """Content calendar model"""
    __tablename__ = "content_calendar"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    content_idea_id = Column(Integer, ForeignKey("content_ideas.id"), nullable=True, index=True)
    software_solution_id = Column(Integer, ForeignKey("software_solutions.id"), nullable=True, index=True)
    
    # Entry details
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    entry_type = Column(Enum(EntryType), nullable=False)
    platform = Column(String(100), nullable=True)  # wordpress, web, etc.
    
    # Scheduling
    scheduled_date = Column(DateTime(timezone=True), nullable=False)
    status = Column(Enum(CalendarStatus), default=CalendarStatus.SCHEDULED, nullable=False)
    priority = Column(Integer, default=3, nullable=False)  # 1-5 scale
    
    # Progress tracking
    progress_percentage = Column(Integer, default=0, nullable=False)  # 0-100
    current_phase = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)
    
    # Completion tracking
    completed_date = Column(DateTime(timezone=True), nullable=True)
    actual_duration = Column(Integer, nullable=True)  # Duration in minutes
    estimated_duration = Column(Integer, nullable=True)  # Estimated duration in minutes
    
    # Dependencies
    dependencies = Column(JSON, nullable=True)  # Array of dependent entry IDs
    blocking_entries = Column(JSON, nullable=True)  # Array of entries this blocks
    
    # Metadata
    tags = Column(JSON, nullable=True)  # Array of tags
    category = Column(String(100), nullable=True)
    assignee = Column(String(100), nullable=True)  # For team collaboration
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="content_calendar")
    content_idea = relationship("ContentIdeas", back_populates="content_calendar")
    software_solution = relationship("SoftwareSolutions", back_populates="content_calendar")
    
    def __repr__(self):
        return f"<ContentCalendar(id={self.id}, title='{self.title}', type='{self.entry_type.value}')>"
    
    def to_dict(self):
        """Convert calendar entry to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content_idea_id": self.content_idea_id,
            "software_solution_id": self.software_solution_id,
            "title": self.title,
            "description": self.description,
            "entry_type": self.entry_type.value,
            "platform": self.platform,
            "scheduled_date": self.scheduled_date.isoformat(),
            "status": self.status.value,
            "priority": self.priority,
            "progress_percentage": self.progress_percentage,
            "current_phase": self.current_phase,
            "notes": self.notes,
            "completed_date": self.completed_date.isoformat() if self.completed_date else None,
            "actual_duration": self.actual_duration,
            "estimated_duration": self.estimated_duration,
            "dependencies": self.dependencies,
            "blocking_entries": self.blocking_entries,
            "tags": self.tags,
            "category": self.category,
            "assignee": self.assignee,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def is_completed(self) -> bool:
        """Check if entry is completed"""
        return self.status == CalendarStatus.COMPLETED
    
    def is_in_progress(self) -> bool:
        """Check if entry is in progress"""
        return self.status == CalendarStatus.IN_PROGRESS
    
    def is_scheduled(self) -> bool:
        """Check if entry is scheduled"""
        return self.status == CalendarStatus.SCHEDULED
    
    def is_cancelled(self) -> bool:
        """Check if entry is cancelled"""
        return self.status == CalendarStatus.CANCELLED
    
    def is_on_hold(self) -> bool:
        """Check if entry is on hold"""
        return self.status == CalendarStatus.ON_HOLD
    
    def get_priority_category(self) -> str:
        """Get priority category"""
        if self.priority <= 2:
            return "high"
        elif self.priority <= 3:
            return "medium"
        else:
            return "low"
    
    def get_progress_category(self) -> str:
        """Get progress category"""
        if self.progress_percentage >= 90:
            return "almost_done"
        elif self.progress_percentage >= 70:
            return "good_progress"
        elif self.progress_percentage >= 40:
            return "in_progress"
        elif self.progress_percentage >= 10:
            return "started"
        else:
            return "not_started"
    
    def get_duration_status(self) -> str:
        """Get duration status"""
        if not self.estimated_duration or not self.actual_duration:
            return "unknown"
        
        if self.actual_duration <= self.estimated_duration:
            return "on_time"
        elif self.actual_duration <= self.estimated_duration * 1.2:
            return "slightly_over"
        else:
            return "overdue"
    
    def get_dependencies_list(self) -> list:
        """Get dependencies as list"""
        if not self.dependencies:
            return []
        
        return self.dependencies
    
    def get_blocking_entries_list(self) -> list:
        """Get blocking entries as list"""
        if not self.blocking_entries:
            return []
        
        return self.blocking_entries
    
    def get_tags_list(self) -> list:
        """Get tags as list"""
        if not self.tags:
            return []
        
        return self.tags
    
    def add_dependency(self, entry_id: int):
        """Add dependency"""
        if not self.dependencies:
            self.dependencies = []
        
        if entry_id not in self.dependencies:
            self.dependencies.append(entry_id)
    
    def remove_dependency(self, entry_id: int):
        """Remove dependency"""
        if not self.dependencies:
            return
        
        if entry_id in self.dependencies:
            self.dependencies.remove(entry_id)
    
    def add_blocking_entry(self, entry_id: int):
        """Add blocking entry"""
        if not self.blocking_entries:
            self.blocking_entries = []
        
        if entry_id not in self.blocking_entries:
            self.blocking_entries.append(entry_id)
    
    def remove_blocking_entry(self, entry_id: int):
        """Remove blocking entry"""
        if not self.blocking_entries:
            return
        
        if entry_id in self.blocking_entries:
            self.blocking_entries.remove(entry_id)
    
    def add_tag(self, tag: str):
        """Add tag"""
        if not self.tags:
            self.tags = []
        
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str):
        """Remove tag"""
        if not self.tags:
            return
        
        if tag in self.tags:
            self.tags.remove(tag)
    
    def update_status(self, status: CalendarStatus, notes: str = None):
        """Update entry status"""
        self.status = status
        
        if status == CalendarStatus.COMPLETED:
            self.completed_date = func.now()
            self.progress_percentage = 100
        elif status == CalendarStatus.IN_PROGRESS and self.progress_percentage == 0:
            self.progress_percentage = 10  # Mark as started
        
        if notes:
            self.notes = notes
    
    def update_progress(self, percentage: int, current_phase: str = None, notes: str = None):
        """Update progress"""
        self.progress_percentage = max(0, min(percentage, 100))
        
        if current_phase:
            self.current_phase = current_phase
        
        if notes:
            self.notes = notes
        
        # Auto-update status based on progress
        if self.progress_percentage == 100 and self.status != CalendarStatus.COMPLETED:
            self.update_status(CalendarStatus.COMPLETED)
        elif self.progress_percentage > 0 and self.status == CalendarStatus.SCHEDULED:
            self.update_status(CalendarStatus.IN_PROGRESS)
    
    def add_note(self, note: str):
        """Add note"""
        if not self.notes:
            self.notes = note
        else:
            self.notes += f"\n\n{note}"
    
    def calculate_actual_duration(self):
        """Calculate actual duration if not set"""
        if self.actual_duration or not self.completed_date:
            return
        
        # Calculate duration from scheduled_date to completed_date
        if self.completed_date and self.scheduled_date:
            duration = self.completed_date - self.scheduled_date
            self.actual_duration = int(duration.total_seconds() / 60)  # Convert to minutes
    
    def is_overdue(self) -> bool:
        """Check if entry is overdue"""
        if self.status in [CalendarStatus.COMPLETED, CalendarStatus.CANCELLED]:
            return False
        
        from datetime import datetime
        now = datetime.now(self.scheduled_date.tzinfo)
        return now > self.scheduled_date
    
    def get_days_until_due(self) -> int:
        """Get days until due date"""
        if self.status in [CalendarStatus.COMPLETED, CalendarStatus.CANCELLED]:
            return 0
        
        from datetime import datetime
        now = datetime.now(self.scheduled_date.tzinfo)
        delta = self.scheduled_date - now
        return max(0, delta.days)
    
    def get_calendar_summary(self) -> dict:
        """Get calendar summary"""
        return {
            "id": self.id,
            "title": self.title,
            "entry_type": self.entry_type.value,
            "platform": self.platform,
            "scheduled_date": self.scheduled_date.isoformat(),
            "status": self.status.value,
            "priority": self.priority,
            "priority_category": self.get_priority_category(),
            "progress_percentage": self.progress_percentage,
            "progress_category": self.get_progress_category(),
            "current_phase": self.current_phase,
            "is_overdue": self.is_overdue(),
            "days_until_due": self.get_days_until_due(),
            "tags": self.get_tags_list(),
            "category": self.category,
            "assignee": self.assignee
        }
