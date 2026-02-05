"""
Account lockout models for tracking failed login attempts
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
from typing import Optional, List
import enum

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class LockoutReason(enum.Enum):
    """Reasons for account lockout"""
    FAILED_LOGIN = "failed_login"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ADMIN_LOCK = "admin_lock"
    PASSWORD_BREACH = "password_breach"
    RATE_LIMIT = "rate_limit"

class AccountLockout(Base):
    """Account lockout tracking"""
    __tablename__ = "account_lockouts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    reason = Column(String(50), nullable=False, default=LockoutReason.FAILED_LOGIN.value)
    locked_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    is_permanent = Column(Boolean, default=False, nullable=False)
    unlock_token = Column(String(255), nullable=True, unique=True, index=True)
    locked_by_admin = Column(Integer, ForeignKey("users.id"), nullable=True)
    reason_description = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    user = relationship("User", foreign_keys=[user_id], back_populates="lockouts")
    admin = relationship("User", foreign_keys=[locked_by_admin])

    def is_locked(self) -> bool:
        """Check if account is currently locked"""
        if not self.is_active:
            return False
        
        if self.is_permanent:
            return True
        
        if self.locked_until and self.locked_until > datetime.utcnow():
            return True
        
        return False

    def get_remaining_lockout_time(self) -> Optional[timedelta]:
        """Get remaining lockout time"""
        if not self.is_locked():
            return None
        
        if self.is_permanent:
            return None
        
        if self.locked_until:
            remaining = self.locked_until - datetime.utcnow()
            return remaining if remaining.total_seconds() > 0 else None
        
        return None

    def unlock(self) -> None:
        """Unlock the account"""
        self.is_active = False
        self.updated_at = datetime.utcnow()

class FailedLoginAttempt(Base):
    """Failed login attempt tracking"""
    __tablename__ = "failed_login_attempts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)  # Nullable for unknown users
    email = Column(String(255), nullable=False, index=True)
    ip_address = Column(String(45), nullable=False, index=True)
    user_agent = Column(Text, nullable=True)
    attempt_time = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    failure_reason = Column(String(100), nullable=True)  # wrong_password, user_not_found, etc.
    is_suspicious = Column(Boolean, default=False, nullable=False)
    geolocation = Column(String(100), nullable=True)  # Country, city
    device_fingerprint = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", foreign_keys=[user_id])

    def is_recent(self, minutes: int = 15) -> bool:
        """Check if attempt is within recent time window"""
        return self.attempt_time > datetime.utcnow() - timedelta(minutes=minutes)

class SecurityEvent(Base):
    """Security events tracking"""
    __tablename__ = "security_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    event_description = Column(Text, nullable=False)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    severity = Column(String(20), nullable=False, default="medium")  # low, medium, high, critical
    event_metadata = Column(Text, nullable=True)  # JSON string for additional data
    is_resolved = Column(Boolean, default=False, nullable=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolved_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    resolver = relationship("User", foreign_keys=[resolved_by])

class PasswordBreach(Base):
    """Password breach tracking"""
    __tablename__ = "password_breaches"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    breach_source = Column(String(100), nullable=False)  # haveibeenpwned, internal, etc.
    breach_date = Column(DateTime(timezone=True), nullable=True)
    breach_description = Column(Text, nullable=True)
    password_hash = Column(String(255), nullable=True)  # Hash of breached password
    is_resolved = Column(Boolean, default=False, nullable=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", foreign_keys=[user_id])

class SuspiciousActivity(Base):
    """Suspicious activity tracking"""
    __tablename__ = "suspicious_activities"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    activity_type = Column(String(50), nullable=False, index=True)
    description = Column(Text, nullable=False)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    geolocation = Column(String(100), nullable=True)
    risk_score = Column(Integer, default=0, nullable=False)  # 0-100
    is_investigated = Column(Boolean, default=False, nullable=False)
    investigation_notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", foreign_keys=[user_id])

# Update User model to include relationships
# This would be added to the existing User model
"""
# Add these relationships to the User model:
lockouts = relationship("AccountLockout", foreign_keys="AccountLockout.user_id", back_populates="user")
failed_login_attempts = relationship("FailedLoginAttempt", foreign_keys="FailedLoginAttempt.user_id")
security_events = relationship("SecurityEvent", foreign_keys="SecurityEvent.user_id")
password_breaches = relationship("PasswordBreach", foreign_keys="PasswordBreach.user_id")
suspicious_activities = relationship("SuspiciousActivity", foreign_keys="SuspiciousActivity.user_id")
"""
