"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
ContentCalendar model for TrendTap
"""

from enum import Enum as PyEnum
from ..core.database import Base

class EntryType(PyEnum):
    """Entry type enumeration"""
    CONTENT = "content"
    SOFTWARE_PROJECT = "software_project"

class CalendarStatus(PyEnum):
    """Calendar status enumeration"""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"

class ContentCalendar:
    """Simple data class for ContentCalendar - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
