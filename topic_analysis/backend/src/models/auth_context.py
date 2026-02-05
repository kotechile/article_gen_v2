"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
AuthenticationContext Model

This module defines the AuthenticationContext data model for managing user
authentication sessions and tokens with Supabase.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import uuid

class AuthenticationContext(BaseModel):
    """
    AuthenticationContext model representing a user authentication session.
    
    This model tracks user sessions, JWT tokens, and authentication state.
    """
    
    # Primary identifier
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique session identifier")
    
    # User information
    user_id: str = Field(..., description="Supabase user ID")
    
    # Token information
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_at: datetime = Field(..., description="Token expiration timestamp")
    
    # Session state
    is_active: bool = Field(default=True, description="Whether session is active")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation timestamp")
    last_activity: datetime = Field(default_factory=datetime.utcnow, description="Last activity timestamp")
    
    # Client information
    ip_address: Optional[str] = Field(default=None, description="Client IP address")
    user_agent: Optional[str] = Field(default=None, description="Client user agent")
    
    # Permissions and roles
    permissions: Optional[Dict[str, Any]] = Field(default=None, description="User permissions and roles")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('access_token')
    def validate_access_token(cls, v):
        """Validate access token format."""
        if not v or len(v) < 10:
            raise ValueError('Access token must be non-empty and substantial')
        return v
    
    @validator('refresh_token')
    def validate_refresh_token(cls, v):
        """Validate refresh token format."""
        if not v or len(v) < 10:
            raise ValueError('Refresh token must be non-empty and substantial')
        return v
    
    @validator('expires_at')
    def validate_expires_at(cls, v):
        """Validate expiration timestamp."""
        if v <= datetime.utcnow():
            raise ValueError('expires_at must be in the future')
        return v
    
    @validator('last_activity')
    def validate_last_activity(cls, v, values):
        """Validate last_activity timestamp."""
        if 'created_at' in values and v < values['created_at']:
            raise ValueError('last_activity must be after created_at')
        return v
    
    @validator('ip_address')
    def validate_ip_address(cls, v):
        """Validate IP address format."""
        if v is not None:
            # Basic IP address validation
            parts = v.split('.')
            if len(parts) != 4:
                raise ValueError('IP address must be in IPv4 format')
            for part in parts:
                if not part.isdigit() or not 0 <= int(part) <= 255:
                    raise ValueError('IP address must be in IPv4 format')
        return v
    
    def is_expired(self) -> bool:
        """Check if the session is expired."""
        return datetime.utcnow() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if the session is valid (active and not expired)."""
        return self.is_active and not self.is_expired()
    
    def update_activity(self, ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
        if ip_address:
            self.ip_address = ip_address
        if user_agent:
            self.user_agent = user_agent
    
    def refresh_tokens(self, new_access_token: str, new_refresh_token: str, expires_at: datetime) -> None:
        """Refresh authentication tokens."""
        self.access_token = new_access_token
        self.refresh_token = new_refresh_token
        self.expires_at = expires_at
        self.update_activity()
    
    def logout(self) -> None:
        """Logout user (deactivate session)."""
        self.is_active = False
    
    def get_remaining_time(self) -> Optional[timedelta]:
        """Get remaining time until expiration."""
        if self.is_expired():
            return None
        return self.expires_at - datetime.utcnow()
    
    def get_session_duration(self) -> timedelta:
        """Get total session duration."""
        return datetime.utcnow() - self.created_at
    
    def get_idle_time(self) -> timedelta:
        """Get idle time since last activity."""
        return datetime.utcnow() - self.last_activity
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        if not self.permissions:
            return False
        return self.permissions.get(permission, False)
    
    def get_permissions(self) -> Dict[str, Any]:
        """Get user permissions."""
        return self.permissions or {}
    
    def set_permissions(self, permissions: Dict[str, Any]) -> None:
        """Set user permissions."""
        self.permissions = permissions
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information for monitoring."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "is_active": self.is_active,
            "is_valid": self.is_valid(),
            "is_expired": self.is_expired(),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "remaining_time_seconds": self.get_remaining_time().total_seconds() if self.get_remaining_time() else 0,
            "session_duration_seconds": self.get_session_duration().total_seconds(),
            "idle_time_seconds": self.get_idle_time().total_seconds(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "access_token": self.access_token[:10] + "...",  # Mask token for security
            "refresh_token": self.refresh_token[:10] + "...",  # Mask token for security
            "token_type": self.token_type,
            "expires_at": self.expires_at.isoformat(),
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "permissions": self.permissions
        }
    
    @classmethod
    def create_session(cls, user_id: str, access_token: str, refresh_token: str, expires_at: datetime, 
                      ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                      permissions: Optional[Dict[str, Any]] = None) -> 'AuthenticationContext':
        """Create a new authentication session."""
        return cls(
            user_id=user_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
            permissions=permissions
        )
    
    @classmethod
    def create_from_supabase_auth(cls, auth_response: Dict[str, Any], 
                                 ip_address: Optional[str] = None, 
                                 user_agent: Optional[str] = None) -> 'AuthenticationContext':
        """Create session from Supabase auth response."""
        user = auth_response.get('user', {})
        session = auth_response.get('session', {})
        
        return cls(
            user_id=user.get('id', ''),
            access_token=session.get('access_token', ''),
            refresh_token=session.get('refresh_token', ''),
            expires_at=datetime.fromisoformat(session.get('expires_at', '').replace('Z', '+00:00')),
            ip_address=ip_address,
            user_agent=user_agent,
            permissions=user.get('user_metadata', {}).get('permissions')
        )
    
    def __str__(self) -> str:
        """String representation."""
        return f"AuthenticationContext(session_id={self.session_id}, user_id={self.user_id}, active={self.is_active})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"AuthenticationContext("
            f"session_id='{self.session_id}', "
            f"user_id='{self.user_id}', "
            f"is_active={self.is_active}, "
            f"is_valid={self.is_valid()}, "
            f"expires_at='{self.expires_at.isoformat()}'"
            f")"
        )
