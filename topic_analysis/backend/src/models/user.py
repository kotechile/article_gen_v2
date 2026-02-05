"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
User model for TrendTap
"""

from enum import Enum as PyEnum
from ..core.database import Base

class UserRole(PyEnum):
    """User role enumeration"""
    USER = "user"
    PREMIUM_USER = "premium_user"
    ADMIN = "admin"
    MODERATOR = "moderator"

class SubscriptionTier(PyEnum):
    """Subscription tier enumeration"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

class User:
    """Simple data class for User - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
