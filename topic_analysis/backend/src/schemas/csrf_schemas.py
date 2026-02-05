"""
Pydantic schemas for CSRF protection
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

class CSRFTokenResponse(BaseModel):
    """Response schema for CSRF token generation"""
    token: str = Field(..., description="Generated CSRF token")
    is_single_use: bool = Field(..., description="Whether token is single-use")
    is_permanent: bool = Field(..., description="Whether token is permanent")
    message: str = Field(..., description="Response message")

class CSRFValidationRequest(BaseModel):
    """Request schema for CSRF token validation"""
    token: str = Field(..., description="CSRF token to validate", min_length=1)
    session_id: Optional[str] = Field(None, description="Session ID")
    
    @validator('token')
    def validate_token(cls, v):
        if not v or not v.strip():
            raise ValueError('CSRF token cannot be empty')
        return v.strip()

class CSRFValidationResponse(BaseModel):
    """Response schema for CSRF token validation"""
    is_valid: bool = Field(..., description="Whether token is valid")
    message: str = Field(..., description="Validation message")

class CSRFWhitelistRequest(BaseModel):
    """Request schema for adding origin to whitelist"""
    origin: str = Field(..., description="Origin to whitelist", min_length=1)
    description: Optional[str] = Field(None, description="Description of the origin")
    expires_at: Optional[datetime] = Field(None, description="Expiration date for whitelist entry")
    
    @validator('origin')
    def validate_origin(cls, v):
        if not v or not v.strip():
            raise ValueError('Origin cannot be empty')
        return v.strip()

class CSRFWhitelistResponse(BaseModel):
    """Response schema for whitelist operations"""
    id: int = Field(..., description="Whitelist entry ID")
    origin: str = Field(..., description="Whitelisted origin")
    description: Optional[str] = Field(None, description="Description of the origin")
    is_active: bool = Field(..., description="Whether entry is active")
    created_at: str = Field(..., description="Creation timestamp")
    message: str = Field(..., description="Response message")

class CSRFViolationStats(BaseModel):
    """Response schema for violation statistics"""
    total_violations: int = Field(..., description="Total number of violations")
    violation_types: Dict[str, int] = Field(..., description="Violations by type")
    severity_counts: Dict[str, int] = Field(..., description="Violations by severity")
    time_period_hours: int = Field(..., description="Time period for statistics")

class CSRFConfigRequest(BaseModel):
    """Request schema for CSRF configuration updates"""
    setting_name: str = Field(..., description="Configuration setting name")
    setting_value: str = Field(..., description="Configuration setting value")
    setting_type: str = Field(default="string", description="Configuration setting type")
    description: Optional[str] = Field(None, description="Configuration description")
    
    @validator('setting_type')
    def validate_setting_type(cls, v):
        allowed_types = ['string', 'int', 'bool', 'json']
        if v not in allowed_types:
            raise ValueError(f'Setting type must be one of: {allowed_types}')
        return v

class CSRFConfigResponse(BaseModel):
    """Response schema for CSRF configuration"""
    id: int = Field(..., description="Configuration ID")
    setting_name: str = Field(..., description="Configuration setting name")
    setting_value: str = Field(..., description="Configuration setting value")
    setting_type: str = Field(..., description="Configuration setting type")
    description: Optional[str] = Field(None, description="Configuration description")
    is_active: bool = Field(..., description="Whether configuration is active")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    message: str = Field(..., description="Response message")

class CSRFTokenInfo(BaseModel):
    """Schema for CSRF token information"""
    id: int = Field(..., description="Token ID")
    token: str = Field(..., description="Masked token")
    session_id: Optional[str] = Field(None, description="Session ID")
    ip_address: Optional[str] = Field(None, description="IP address")
    status: str = Field(..., description="Token status")
    created_at: str = Field(..., description="Creation timestamp")
    expires_at: str = Field(..., description="Expiration timestamp")
    is_single_use: bool = Field(..., description="Whether token is single-use")
    is_permanent: bool = Field(..., description="Whether token is permanent")

class CSRFViolationInfo(BaseModel):
    """Schema for CSRF violation information"""
    id: int = Field(..., description="Violation ID")
    user_id: Optional[int] = Field(None, description="User ID")
    ip_address: str = Field(..., description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent")
    origin: Optional[str] = Field(None, description="Origin")
    referer: Optional[str] = Field(None, description="Referer")
    violation_type: str = Field(..., description="Type of violation")
    token_provided: Optional[str] = Field(None, description="Token that was provided")
    expected_token: Optional[str] = Field(None, description="Expected token")
    request_path: Optional[str] = Field(None, description="Request path")
    request_method: Optional[str] = Field(None, description="Request method")
    severity: str = Field(..., description="Violation severity")
    is_blocked: bool = Field(..., description="Whether IP is blocked")
    blocked_until: Optional[str] = Field(None, description="Block expiration")
    created_at: str = Field(..., description="Creation timestamp")

class CSRFWhitelistInfo(BaseModel):
    """Schema for whitelist entry information"""
    id: int = Field(..., description="Whitelist entry ID")
    origin: str = Field(..., description="Whitelisted origin")
    description: Optional[str] = Field(None, description="Description")
    is_active: bool = Field(..., description="Whether entry is active")
    created_at: str = Field(..., description="Creation timestamp")
    created_by: Optional[int] = Field(None, description="Creator user ID")
    expires_at: Optional[str] = Field(None, description="Expiration timestamp")

class CSRFConfigInfo(BaseModel):
    """Schema for configuration information"""
    id: int = Field(..., description="Configuration ID")
    setting_name: str = Field(..., description="Setting name")
    setting_value: str = Field(..., description="Setting value")
    setting_type: str = Field(..., description="Setting type")
    description: Optional[str] = Field(None, description="Description")
    is_active: bool = Field(..., description="Whether setting is active")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    updated_by: Optional[int] = Field(None, description="Last updater user ID")

class CSRFStatsResponse(BaseModel):
    """Response schema for CSRF statistics"""
    total_tokens: int = Field(..., description="Total number of tokens")
    active_tokens: int = Field(..., description="Number of active tokens")
    expired_tokens: int = Field(..., description="Number of expired tokens")
    revoked_tokens: int = Field(..., description="Number of revoked tokens")
    total_violations: int = Field(..., description="Total number of violations")
    blocked_ips: int = Field(..., description="Number of blocked IPs")
    whitelist_entries: int = Field(..., description="Number of whitelist entries")

class CSRFHealthResponse(BaseModel):
    """Response schema for CSRF health check"""
    status: str = Field(..., description="Health status")
    message: str = Field(..., description="Health message")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="CSRF protection version")
    features: List[str] = Field(..., description="Enabled features")
