"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
JWT token blacklist models for token revocation
"""
from datetime import datetime, timedelta
from typing import Optional, List
import enum

# Base = declarative_base()  # Disabled for Supabase-only

class TokenStatus(enum.Enum):
    """Token status enumeration"""
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"
    BLACKLISTED = "blacklisted"

class RevocationReason(enum.Enum):
    """Token revocation reasons"""
    USER_LOGOUT = "user_logout"
    ADMIN_REVOKE = "admin_revoke"
    SECURITY_BREACH = "security_breach"
    PASSWORD_CHANGE = "password_change"
    ACCOUNT_LOCKED = "account_locked"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    TOKEN_EXPIRED = "token_expired"
    MANUAL_REVOKE = "manual_revoke"

class JWTBlacklist:
    """Simple data class for JWTBlacklist - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class TokenWhitelist:
    """Simple data class for TokenWhitelist - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class TokenSession:
    """Simple data class for TokenSession - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class TokenAuditLog:
    """Simple data class for TokenAuditLog - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
