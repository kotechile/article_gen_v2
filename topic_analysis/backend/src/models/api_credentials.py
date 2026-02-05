"""
APICredentials model for DataForSEO API integration.

Represents DataForSEO API configuration stored in Supabase.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class APICredentialStatus(str, Enum):
    """Enum for API credential status."""
    ACTIVE = "active"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    INACTIVE = "inactive"


class APICredentials(BaseModel):
    """API credentials model for DataForSEO configuration."""
    
    id: str = Field(..., min_length=1, description="Unique identifier")
    provider: str = Field(..., description="API provider")
    base_url: str = Field(..., description="API base URL")
    key_value: str = Field(..., min_length=1, description="API key (encrypted)")
    is_active: bool = Field(..., description="Whether the credentials are active")
    rate_limit: Optional[int] = Field(None, gt=0, description="Rate limit per minute")
    quota_used: Optional[int] = Field(None, ge=0, description="Quota used this month")
    quota_limit: Optional[int] = Field(None, gt=0, description="Monthly quota limit")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the credentials were created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When the credentials were last updated")
    
    @validator('id')
    def validate_id(cls, v):
        """Validate ID is not empty and trimmed."""
        if not v or not v.strip():
            raise ValueError('ID must not be empty')
        return v.strip()
    
    @validator('provider')
    def validate_provider(cls, v):
        """Validate provider is dataforseo."""
        if v != "dataforseo":
            raise ValueError('Provider must be "dataforseo"')
        return v
    
    @validator('base_url')
    def validate_base_url(cls, v):
        """Validate base URL is a valid URL."""
        if not v or not v.strip():
            raise ValueError('Base URL must not be empty')
        v = v.strip()
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Base URL must start with http:// or https://')
        return v
    
    @validator('key_value')
    def validate_key_value(cls, v):
        """Validate key value is not empty."""
        if not v or not v.strip():
            raise ValueError('Key value must not be empty')
        return v.strip()
    
    @validator('rate_limit')
    def validate_rate_limit(cls, v):
        """Validate rate limit is positive."""
        if v is not None and v <= 0:
            raise ValueError('Rate limit must be positive')
        return v
    
    @validator('quota_used')
    def validate_quota_used(cls, v):
        """Validate quota used is non-negative."""
        if v is not None and v < 0:
            raise ValueError('Quota used must be non-negative')
        return v
    
    @validator('quota_limit')
    def validate_quota_limit(cls, v):
        """Validate quota limit is positive."""
        if v is not None and v <= 0:
            raise ValueError('Quota limit must be positive')
        return v
    
    @validator('quota_used', 'quota_limit')
    def validate_quota_consistency(cls, v, values):
        """Validate quota used doesn't exceed quota limit."""
        if 'quota_used' in values and 'quota_limit' in values:
            quota_used = values.get('quota_used')
            quota_limit = values.get('quota_limit')
            if quota_used is not None and quota_limit is not None and quota_used > quota_limit:
                raise ValueError('Quota used cannot exceed quota limit')
        return v
    
    def get_status(self) -> APICredentialStatus:
        """Get current status of the API credentials."""
        if not self.is_active:
            return APICredentialStatus.INACTIVE
        
        if self.quota_used is not None and self.quota_limit is not None:
            if self.quota_used >= self.quota_limit:
                return APICredentialStatus.QUOTA_EXCEEDED
        
        # Note: Rate limited status would be set by external monitoring
        return APICredentialStatus.ACTIVE
    
    def is_quota_available(self) -> bool:
        """Check if quota is available."""
        if self.quota_used is None or self.quota_limit is None:
            return True
        return self.quota_used < self.quota_limit
    
    def get_quota_remaining(self) -> Optional[int]:
        """Get remaining quota."""
        if self.quota_used is None or self.quota_limit is None:
            return None
        return max(0, self.quota_limit - self.quota_used)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "id": "dataforseo_001",
                "provider": "dataforseo",
                "base_url": "https://api.dataforseo.com",
                "key_value": "encrypted_api_key_here",
                "is_active": True,
                "rate_limit": 1000,
                "quota_used": 500,
                "quota_limit": 10000,
                "created_at": "2025-01-14T10:00:00Z",
                "updated_at": "2025-01-14T10:00:00Z"
            }
        }


class APICredentialsResponse(BaseModel):
    """Response model for API credentials API."""
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[APICredentials] = Field(None, description="API credentials data")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {
                    "id": "dataforseo_001",
                    "provider": "dataforseo",
                    "base_url": "https://api.dataforseo.com",
                    "key_value": "encrypted_api_key_here",
                    "is_active": True,
                    "rate_limit": 1000,
                    "quota_used": 500,
                    "quota_limit": 10000,
                    "created_at": "2025-01-14T10:00:00Z",
                    "updated_at": "2025-01-14T10:00:00Z"
                },
                "metadata": {
                    "status": "active",
                    "quota_remaining": 9500
                }
            }
        }
