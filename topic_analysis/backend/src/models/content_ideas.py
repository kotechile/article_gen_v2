"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
ContentIdeas model for TrendTap
"""

from enum import Enum as PyEnum
from ..core.database import Base

class ContentType(PyEnum):
    """Content type enumeration"""
    ARTICLE = "article"
    GUIDE = "guide"
    REVIEW = "review"
    TUTORIAL = "tutorial"
    LISTICLE = "listicle"

class ContentAngle(PyEnum):
    """Content angle enumeration"""
    HOW_TO = "how-to"
    VS = "vs"
    LISTICLE = "listicle"
    PAIN_POINT = "pain-point"
    STORY = "story"

class ContentStatus(PyEnum):
    """Content status enumeration"""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class ContentIdeas:
    """Simple data class for ContentIdeas - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
