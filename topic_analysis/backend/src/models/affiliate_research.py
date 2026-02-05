"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
AffiliateResearch model for TrendTap
"""

from enum import Enum as PyEnum
from ..core.database import Base

class ResearchStatus(PyEnum):
    """Research status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AffiliateResearch:
    """Simple data class for AffiliateResearch - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
