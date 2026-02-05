"""
AuthenticationContext model for managing Supabase sessions
"""
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class AuthenticationContext(Base):
    """AuthenticationContext model for managing Supabase sessions"""
    
    __tablename__ = "authentication_contexts"
    
    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    access_token = Column(String(1000), nullable=True)
    refresh_token = Column(String(1000), nullable=True)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_used = Column(DateTime, nullable=True)
    
    # Foreign keys
    supabase_client_id = Column(UUID(as_uuid=True), ForeignKey("supabase_clients.id"), nullable=False)
    
    # Relationships
    supabase_client = relationship("SupabaseClient", back_populates="authentication_contexts")
    
    def __repr__(self):
        return f"<AuthenticationContext(id={self.session_id}, user_id={self.user_id}, active={self.is_active})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "session_id": str(self.session_id),
            "user_id": str(self.user_id) if self.user_id else None,
            "is_active": self.is_active,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None
        }
    
    def is_expired(self) -> bool:
        """Check if session is expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if session is valid (active and not expired)"""
        return self.is_active and not self.is_expired()
    
    def refresh_session(self, access_token: str, refresh_token: str, expires_at: datetime):
        """Refresh session with new tokens"""
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires_at = expires_at
        self.last_used = datetime.utcnow()
    
    def revoke_session(self):
        """Revoke the session"""
        self.is_active = False
        self.access_token = None
        self.refresh_token = None
        self.expires_at = None
    
    def update_activity(self):
        """Update last used timestamp"""
        self.last_used = datetime.utcnow()
    
    def get_remaining_time(self) -> int:
        """Get remaining time in seconds until expiration"""
        if self.expires_at is None:
            return -1  # No expiration
        remaining = (self.expires_at - datetime.utcnow()).total_seconds()
        return max(0, int(remaining))
    
    def needs_refresh(self, buffer_seconds: int = 300) -> bool:
        """Check if session needs refresh (within buffer time)"""
        if self.expires_at is None:
            return False
        remaining = self.get_remaining_time()
        return remaining > 0 and remaining <= buffer_seconds

