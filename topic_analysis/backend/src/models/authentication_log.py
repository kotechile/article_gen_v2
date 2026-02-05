"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
AuthenticationLog model for audit logging and security monitoring.
"""
import uuid
import enum
from datetime import datetime

class AuthenticationEventType(str, enum.Enum):
    """Authentication event types for logging."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    PASSWORD_RESET_REQUESTED = "password_reset_requested"
    PASSWORD_RESET_USED = "password_reset_used"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    EMAIL_VERIFIED = "email_verified"
    ROLE_CHANGED = "role_changed"
    ACCOUNT_DEACTIVATED = "account_deactivated"
    TOKEN_BLACKLISTED = "token_blacklisted"

class AuthenticationLog:
    """Simple data class for AuthenticationLog - use Supabase for database operations"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def log_event(cls, event_type, success=True, user_id=None, email=None, 
                  ip_address=None, user_agent=None, device_info=None, 
                  error_message=None, details=None, event_metadata=None):
        """Log an authentication event"""
        return cls(
            event_type=event_type,
            success=success,
            user_id=user_id,
            email=email,
            ip_address=ip_address,
            user_agent=user_agent,
            device_info=device_info,
            error_message=error_message,
            details=details,
            event_metadata=event_metadata or {},
            created_at=datetime.utcnow()
        )
    
    @classmethod
    def log_login_success(cls, user_id, ip_address=None, user_agent=None, device_info=None):
        return cls.log_event(
            event_type=AuthenticationEventType.LOGIN_SUCCESS,
            success=True,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            device_info=device_info
        )
    
    @classmethod
    def log_login_failed(cls, email, ip_address=None, user_agent=None, error_message=None):
        return cls.log_event(
            event_type=AuthenticationEventType.LOGIN_FAILED,
            success=False,
            email=email,
            ip_address=ip_address,
            user_agent=user_agent,
            error_message=error_message
        )
    
    @classmethod
    def log_logout(cls, user_id, ip_address=None, user_agent=None):
        return cls.log_event(
            event_type=AuthenticationEventType.LOGOUT,
            success=True,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    @classmethod
    def log_password_reset_requested(cls, user_id, ip_address=None, user_agent=None):
        return cls.log_event(
            event_type=AuthenticationEventType.PASSWORD_RESET_REQUESTED,
            success=True,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    @classmethod
    def log_password_reset_used(cls, user_id, ip_address=None, user_agent=None):
        return cls.log_event(
            event_type=AuthenticationEventType.PASSWORD_RESET_USED,
            success=True,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    @classmethod
    def log_email_verified(cls, user_id, ip_address=None, user_agent=None):
        return cls.log_event(
            event_type=AuthenticationEventType.EMAIL_VERIFIED,
            success=True,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent
        )

    @classmethod
    def log_security_event(cls, user_id, event_type, ip_address=None, user_agent=None, details=None):
        return cls.log_event(
            event_type=event_type,
            success=True,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details
        )
