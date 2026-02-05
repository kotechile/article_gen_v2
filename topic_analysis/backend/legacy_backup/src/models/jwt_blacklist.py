"""
JWT token blacklist models for token revocation
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, Index
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
from typing import Optional, List
import enum

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

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

class JWTBlacklist(Base):
    """JWT token blacklist for revoked tokens"""
    __tablename__ = "jwt_blacklist"

    id = Column(Integer, primary_key=True, index=True)
    jti = Column(String(255), nullable=False, unique=True, index=True)  # JWT ID
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    token_type = Column(String(20), nullable=False, default="access")  # access, refresh
    token_hash = Column(String(255), nullable=False, index=True)  # Hash of the token
    revoked_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    reason = Column(String(50), nullable=False, default=RevocationReason.USER_LOGOUT.value)
    revoked_by = Column(Integer, ForeignKey("users.id"), nullable=True)  # Admin who revoked
    ip_address = Column(String(45), nullable=True)  # IP where token was revoked
    user_agent = Column(Text, nullable=True)  # User agent when revoked
    is_permanent = Column(Boolean, default=False, nullable=False)  # Permanent blacklist
    token_metadata = Column(Text, nullable=True)  # Additional metadata as JSON
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", foreign_keys=[user_id], back_populates="blacklisted_tokens")
    revoker = relationship("User", foreign_keys=[revoked_by])

    # Indexes for performance
    __table_args__ = (
        Index('idx_jwt_blacklist_user_expires', 'user_id', 'expires_at'),
        Index('idx_jwt_blacklist_jti_expires', 'jti', 'expires_at'),
        Index('idx_jwt_blacklist_token_hash', 'token_hash'),
    )

    def is_expired(self) -> bool:
        """Check if blacklisted token has expired"""
        return datetime.utcnow() > self.expires_at

    def is_active_blacklist(self) -> bool:
        """Check if blacklist entry is still active (not expired)"""
        return not self.is_expired()

class TokenWhitelist(Base):
    """Token whitelist for active tokens (optional optimization)"""
    __tablename__ = "token_whitelist"

    id = Column(Integer, primary_key=True, index=True)
    jti = Column(String(255), nullable=False, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    token_type = Column(String(20), nullable=False, default="access")
    token_hash = Column(String(255), nullable=False, index=True)
    issued_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    last_used = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", foreign_keys=[user_id])

    # Indexes for performance
    __table_args__ = (
        Index('idx_token_whitelist_user_expires', 'user_id', 'expires_at'),
        Index('idx_token_whitelist_jti_expires', 'jti', 'expires_at'),
        Index('idx_token_whitelist_active', 'is_active', 'expires_at'),
    )

    def is_expired(self) -> bool:
        """Check if token has expired"""
        return datetime.utcnow() > self.expires_at

    def is_valid(self) -> bool:
        """Check if token is valid (active and not expired)"""
        return self.is_active and not self.is_expired()

class TokenSession(Base):
    """Token session tracking for advanced security"""
    __tablename__ = "token_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), nullable=False, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    device_fingerprint = Column(String(255), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True, index=True)
    user_agent = Column(Text, nullable=True)
    geolocation = Column(String(100), nullable=True)  # Country, city
    is_active = Column(Boolean, default=True, nullable=False)
    last_activity = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)

    # Relationships
    user = relationship("User", foreign_keys=[user_id])

    def is_expired(self) -> bool:
        """Check if session has expired"""
        return datetime.utcnow() > self.expires_at

    def is_valid(self) -> bool:
        """Check if session is valid (active and not expired)"""
        return self.is_active and not self.is_expired()

class TokenAuditLog(Base):
    """Token audit log for security monitoring"""
    __tablename__ = "token_audit_log"

    id = Column(Integer, primary_key=True, index=True)
    jti = Column(String(255), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    action = Column(String(50), nullable=False, index=True)  # issued, used, revoked, expired
    token_type = Column(String(20), nullable=False)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    geolocation = Column(String(100), nullable=True)
    log_metadata = Column(Text, nullable=True)  # Additional context
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", foreign_keys=[user_id])

    # Indexes for performance
    __table_args__ = (
        Index('idx_token_audit_user_action', 'user_id', 'action'),
        Index('idx_token_audit_jti_action', 'jti', 'action'),
        Index('idx_token_audit_created', 'created_at'),
    )

# Update User model to include relationships
# This would be added to the existing User model:
"""
# Add these relationships to the User model:
blacklisted_tokens = relationship("JWTBlacklist", foreign_keys="JWTBlacklist.user_id", back_populates="user")
whitelisted_tokens = relationship("TokenWhitelist", foreign_keys="TokenWhitelist.user_id")
token_sessions = relationship("TokenSession", foreign_keys="TokenSession.user_id")
token_audit_logs = relationship("TokenAuditLog", foreign_keys="TokenAuditLog.user_id")
"""
