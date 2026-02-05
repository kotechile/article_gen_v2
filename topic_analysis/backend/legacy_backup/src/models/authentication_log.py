"""
AuthenticationLog model for audit logging and security monitoring.
"""
import uuid
from sqlalchemy import (
    Column,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    Enum as SqlEnum,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from src.core.database import Base
import enum

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

class AuthenticationLog(Base):
    """AuthenticationLog model representing a log entry for authentication events and security monitoring."""
    
    __tablename__ = "authentication_logs"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to User (nullable for failed attempts on non-existent users)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    
    # Event information
    event_type = Column(SqlEnum(AuthenticationEventType), nullable=False, index=True)  # Type of authentication event
    success = Column(Boolean, nullable=False)  # Whether the event was successful
    error_message = Column(String(500), nullable=True)  # Error message if unsuccessful
    
    # Request metadata
    ip_address = Column(String(45), nullable=True)  # IP address of the request (supports IPv6)
    user_agent = Column(String(500), nullable=True)  # User agent string
    device_info = Column(JSON, nullable=True)  # Device and browser information
    
    # Additional event data
    event_metadata = Column(JSON, nullable=True)  # Additional event-specific data
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)  # Event timestamp
    
    # Relationship to User
    user = relationship("User", back_populates="authentication_logs")
    
    def __repr__(self):
        return f"<AuthenticationLog(id={self.id}, event_type={self.event_type}, success={self.success})>"
    
    @classmethod
    def log_event(cls, event_type, success, user_id=None, ip_address=None, user_agent=None, 
                  device_info=None, error_message=None, event_metadata=None):
        """Create a new authentication log entry."""
        return cls(
            user_id=user_id,
            event_type=event_type,
            success=success,
            error_message=error_message,
            ip_address=ip_address,
            user_agent=user_agent,
            device_info=device_info,
            event_metadata=event_metadata
        )
    
    @classmethod
    def log_login_success(cls, user_id, ip_address=None, user_agent=None, device_info=None):
        """Log a successful login."""
        return cls.log_event(
            event_type=AuthenticationEventType.LOGIN_SUCCESS,
            success=True,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            device_info=device_info
        )
    
    @classmethod
    def log_login_failed(cls, email, ip_address=None, user_agent=None, error_message="Invalid credentials"):
        """Log a failed login attempt."""
        return cls.log_event(
            event_type=AuthenticationEventType.LOGIN_FAILED,
            success=False,
            user_id=None,  # No user for failed attempts
            ip_address=ip_address,
            user_agent=user_agent,
            error_message=error_message,
            event_metadata={"email": email}
        )
    
    @classmethod
    def log_logout(cls, user_id, ip_address=None, user_agent=None):
        """Log a logout event."""
        return cls.log_event(
            event_type=AuthenticationEventType.LOGOUT,
            success=True,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    @classmethod
    def log_password_reset_requested(cls, user_id, ip_address=None, user_agent=None):
        """Log a password reset request."""
        return cls.log_event(
            event_type=AuthenticationEventType.PASSWORD_RESET_REQUESTED,
            success=True,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    @classmethod
    def log_password_reset_used(cls, user_id, ip_address=None, user_agent=None):
        """Log a password reset completion."""
        return cls.log_event(
            event_type=AuthenticationEventType.PASSWORD_RESET_USED,
            success=True,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    @classmethod
    def log_account_locked(cls, user_id, ip_address=None, user_agent=None, reason="Too many failed attempts"):
        """Log an account lockout."""
        return cls.log_event(
            event_type=AuthenticationEventType.ACCOUNT_LOCKED,
            success=False,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            error_message=reason
        )
    
    @classmethod
    def log_email_verified(cls, user_id, ip_address=None, user_agent=None):
        """Log an email verification."""
        return cls.log_event(
            event_type=AuthenticationEventType.EMAIL_VERIFIED,
            success=True,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    @classmethod
    def log_role_changed(cls, user_id, old_role, new_role, changed_by_user_id, ip_address=None, user_agent=None):
        """Log a role change."""
        return cls.log_event(
            event_type=AuthenticationEventType.ROLE_CHANGED,
            success=True,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            event_metadata={
                "old_role": old_role,
                "new_role": new_role,
                "changed_by": str(changed_by_user_id)
            }
        )
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "event_type": self.event_type.value,
            "success": self.success,
            "error_message": self.error_message,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "device_info": self.device_info,
            "event_metadata": self.event_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
