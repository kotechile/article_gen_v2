"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
SoftwareSolutions model for TrendTap
"""

from enum import Enum as PyEnum
from ..core.database import Base

class SoftwareType(PyEnum):
    """Software type enumeration"""
    CALCULATOR = "calculator"
    ANALYZER = "analyzer"
    GENERATOR = "generator"
    CONVERTER = "converter"
    ESTIMATOR = "estimator"

class DevelopmentStatus(PyEnum):
    """Development status enumeration"""
    IDEA = "idea"
    PLANNED = "planned"
    IN_DEVELOPMENT = "in_development"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class SoftwareSolutions:
    """Simple data class for SoftwareSolutions - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
