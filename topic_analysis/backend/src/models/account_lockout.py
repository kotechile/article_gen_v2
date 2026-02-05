"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
Account lockout models for tracking failed login attempts
"""
from datetime import datetime, timedelta
from typing import Optional, List
import enum

# Base = declarative_base()  # Disabled for Supabase-only

class LockoutReason(enum.Enum):
    """Reasons for account lockout"""
    FAILED_LOGIN = "failed_login"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ADMIN_LOCK = "admin_lock"
    PASSWORD_BREACH = "password_breach"
    RATE_LIMIT = "rate_limit"

class AccountLockout:
    """Simple data class for AccountLockout - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class FailedLoginAttempt:
    """Simple data class for FailedLoginAttempt - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class SecurityEvent:
    """Simple data class for SecurityEvent - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class PasswordBreach:
    """Simple data class for PasswordBreach - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class SuspiciousActivity:
    """Simple data class for SuspiciousActivity - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
