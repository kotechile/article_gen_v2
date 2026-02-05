"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
SupabaseClient Model

This module defines the SupabaseClient data model for tracking client instances,
connection status, and performance metrics.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid

class ClientType(str, Enum):
    """Supabase client types."""
    SERVICE_ROLE = "service_role"
    ANON = "anon"

class ConnectionStatus(str, Enum):
    """Connection status values."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"

class SupabaseClient(BaseModel):
    """
    SupabaseClient model representing a database connection instance.
    
    This model tracks client configuration, connection status, and performance metrics.
    """
    
    # Primary identifier
    client_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique client identifier")
    
    # Connection configuration
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_key: str = Field(..., description="Service role or anon key")
    client_type: ClientType = Field(..., description="Type of client (service_role or anon)")
    
    # Status tracking
    is_active: bool = Field(default=True, description="Whether client is currently active")
    connection_status: ConnectionStatus = Field(default=ConnectionStatus.DISCONNECTED, description="Current connection status")
    
    # Performance metrics
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Client creation timestamp")
    last_used: Optional[datetime] = Field(default=None, description="Last successful operation timestamp")
    error_count: int = Field(default=0, ge=0, le=10, description="Number of consecutive errors")
    retry_count: int = Field(default=0, ge=0, le=5, description="Number of retry attempts")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('supabase_url')
    def validate_supabase_url(cls, v):
        """Validate Supabase URL format."""
        if not v.startswith('https://'):
            raise ValueError('Supabase URL must be a valid HTTPS URL')
        if not v.endswith('.supabase.co'):
            raise ValueError('Supabase URL must end with .supabase.co')
        return v
    
    @validator('supabase_key')
    def validate_supabase_key(cls, v):
        """Validate Supabase key format."""
        if not v or len(v) < 10:
            raise ValueError('Supabase key must be non-empty and substantial')
        return v
    
    @validator('last_used')
    def validate_last_used(cls, v, values):
        """Validate last_used timestamp."""
        if v and 'created_at' in values and v < values['created_at']:
            raise ValueError('last_used must be after created_at')
        return v
    
    def mark_connection_success(self) -> None:
        """Mark a successful connection."""
        self.connection_status = ConnectionStatus.CONNECTED
        self.last_used = datetime.utcnow()
        self.error_count = 0
        self.retry_count = 0
    
    def mark_connection_error(self) -> None:
        """Mark a connection error."""
        self.connection_status = ConnectionStatus.ERROR
        self.error_count += 1
    
    def mark_retry_attempt(self) -> None:
        """Mark a retry attempt."""
        self.retry_count += 1
    
    def should_retry(self, max_retries: int = 5) -> bool:
        """Check if client should retry connection."""
        return self.retry_count < max_retries and self.error_count < 10
    
    def is_healthy(self) -> bool:
        """Check if client is healthy."""
        return (
            self.is_active and 
            self.connection_status == ConnectionStatus.CONNECTED and
            self.error_count < 5
        )
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics for monitoring."""
        return {
            "client_id": self.client_id,
            "client_type": self.client_type,
            "is_active": self.is_active,
            "connection_status": self.connection_status,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "is_healthy": self.is_healthy()
        }
    
    def reset_connection(self) -> None:
        """Reset connection state."""
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.error_count = 0
        self.retry_count = 0
        self.last_used = None
    
    def deactivate(self) -> None:
        """Deactivate client."""
        self.is_active = False
        self.connection_status = ConnectionStatus.DISCONNECTED
    
    def activate(self) -> None:
        """Activate client."""
        self.is_active = True
        self.connection_status = ConnectionStatus.DISCONNECTED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "client_id": self.client_id,
            "supabase_url": self.supabase_url,
            "supabase_key": self.supabase_key[:10] + "...",  # Mask key for security
            "client_type": self.client_type,
            "is_active": self.is_active,
            "connection_status": self.connection_status,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "error_count": self.error_count,
            "retry_count": self.retry_count
        }
    
    @classmethod
    def create_service_role_client(cls, url: str, key: str) -> 'SupabaseClient':
        """Create a service role client."""
        return cls(
            supabase_url=url,
            supabase_key=key,
            client_type=ClientType.SERVICE_ROLE
        )
    
    @classmethod
    def create_anon_client(cls, url: str, key: str) -> 'SupabaseClient':
        """Create an anonymous client."""
        return cls(
            supabase_url=url,
            supabase_key=key,
            client_type=ClientType.ANON
        )
    
    def __str__(self) -> str:
        """String representation."""
        return f"SupabaseClient(id={self.client_id}, type={self.client_type}, status={self.connection_status})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"SupabaseClient("
            f"client_id='{self.client_id}', "
            f"client_type='{self.client_type}', "
            f"connection_status='{self.connection_status}', "
            f"is_active={self.is_active}, "
            f"error_count={self.error_count}"
            f")"
        )