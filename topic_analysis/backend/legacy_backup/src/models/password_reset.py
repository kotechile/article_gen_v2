"""
PasswordReset model for password reset functionality.
"""
import uuid
from sqlalchemy import (
    Column,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from src.core.database import Base

class PasswordReset(Base):
    """PasswordReset model representing a password reset request with secure token."""
    
    __tablename__ = "password_resets"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to User
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    
    # Reset token information
    token = Column(String(255), unique=True, nullable=False, index=True)  # Secure reset token
    
    # Token status
    is_used = Column(Boolean, default=False, nullable=False)  # Whether the token has been used
    
    # Timestamps
    expires_at = Column(DateTime(timezone=True), nullable=False)  # Token expiration time (typically 1 hour)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)  # Request creation timestamp
    used_at = Column(DateTime(timezone=True), nullable=True)  # When the token was used
    
    # Relationship to User
    user = relationship("User", back_populates="password_resets")
    
    def __repr__(self):
        return f"<PasswordReset(id={self.id}, user_id={self.user_id}, is_used={self.is_used})>"
    
    def is_expired(self):
        """Check if the reset token is expired."""
        from datetime import datetime
        return self.expires_at < datetime.utcnow()
    
    def is_valid(self):
        """Check if the reset token is valid (not used and not expired)."""
        return not self.is_used and not self.is_expired()
    
    def mark_as_used(self):
        """Mark the token as used."""
        from datetime import datetime
        self.is_used = True
        self.used_at = datetime.utcnow()
    
    @classmethod
    def create_reset_token(cls, user_id, token, expires_in_hours=1):
        """Create a new password reset token."""
        from datetime import datetime, timedelta
        
        expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
        
        return cls(
            user_id=user_id,
            token=token,
            expires_at=expires_at
        )
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "is_used": self.is_used,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "used_at": self.used_at.isoformat() if self.used_at else None,
            "is_valid": self.is_valid()
        }

