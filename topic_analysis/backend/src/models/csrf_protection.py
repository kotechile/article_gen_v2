"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
CSRF protection models for token management
"""
from datetime import datetime, timedelta
from typing import Optional, List
import enum

# Base = declarative_base()  # Disabled for Supabase-only

class CSRFTokenStatus(enum.Enum):
    """CSRF token status enumeration"""
    ACTIVE = "active"
    USED = "used"
    EXPIRED = "expired"
    REVOKED = "revoked"

class CSRFProtection:
    """Simple data class for CSRFProtection - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class CSRFWhitelist:
    """Simple data class for CSRFWhitelist - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class CSRFViolation:
    """Simple data class for CSRFViolation - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class CSRFConfiguration:
    """Simple data class for CSRFConfiguration - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
