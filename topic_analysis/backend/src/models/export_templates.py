"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
Export Templates model for TrendTap
"""

from enum import Enum as PyEnum
from ..core.database import Base

class ExportFormat(PyEnum):
    """Export format enumeration"""
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"

class ExportStatus(PyEnum):
    """Export status enumeration"""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"

class ExportTemplate:
    """Simple data class for ExportTemplate - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
