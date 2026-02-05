"""
UserSession model for JWT session management.
"""
import uuid
from sqlalchemy import (
    Column,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from src.core.database import Base

class UserSession(Base):
    """UserSession model representing an active user session with JWT token information."""
    
    __tablename__ = "user_sessions"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to User
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    
    # JWT token information
    token_jti = Column(String(255), unique=True, nullable=False, index=True)  # JWT ID for token identification
    refresh_token = Column(String(255), unique=True, nullable=False, index=True)  # Refresh token for session renewal
    
    # Session metadata
    device_info = Column(JSON, nullable=True)  # Device and browser information
    ip_address = Column(String(45), nullable=True)  # IP address of the session (supports IPv6)
    user_agent = Column(String(500), nullable=True)  # User agent string
    
    # Session status
    is_active = Column(Boolean, default=True, nullable=False)  # Whether the session is active
    
    # Timestamps
    expires_at = Column(DateTime(timezone=True), nullable=False)  # Session expiration time
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)  # Session creation timestamp
    last_accessed = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)  # Last activity timestamp
    
    # Relationship to User
    user = relationship("User", back_populates="sessions")
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, user_id={self.user_id}, is_active={self.is_active})>"
    
    def is_expired(self):
        """Check if the session is expired."""
        from datetime import datetime
        return self.expires_at < datetime.utcnow()
    
    def is_valid(self):
        """Check if the session is valid (active and not expired)."""
        return self.is_active and not self.is_expired()
    
    def deactivate(self):
        """Deactivate the session."""
        self.is_active = False
    
    def update_last_accessed(self):
        """Update the last accessed timestamp."""
        from datetime import datetime
        self.last_accessed = datetime.utcnow()

