"""
Security-related Pydantic schemas
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class PasswordStrength(str, Enum):
    """Password strength levels"""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    FAIR = "fair"
    GOOD = "good"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class PasswordValidationRequest(BaseModel):
    """Request schema for password validation"""
    password: str = Field(..., min_length=1, max_length=128, description="Password to validate")
    user_id: Optional[int] = Field(None, description="User ID for context checking")

class PasswordValidationResponse(BaseModel):
    """Response schema for password validation"""
    is_valid: bool = Field(..., description="Whether password meets all requirements")
    strength: PasswordStrength = Field(..., description="Password strength level")
    score: int = Field(..., ge=0, le=100, description="Password strength score (0-100)")
    feedback: List[str] = Field(default_factory=list, description="Validation feedback messages")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    requirements_met: Dict[str, bool] = Field(..., description="Individual requirement status")
    entropy: float = Field(..., ge=0, description="Password entropy in bits")
    crack_time: str = Field(..., description="Estimated time to crack password")
    crack_time_seconds: float = Field(..., ge=0, description="Crack time in seconds")

class PasswordGenerationRequest(BaseModel):
    """Request schema for password generation"""
    length: int = Field(16, ge=8, le=128, description="Password length")
    include_special: bool = Field(True, description="Include special characters")

class PasswordGenerationResponse(BaseModel):
    """Response schema for password generation"""
    password: str = Field(..., description="Generated password")
    strength: PasswordStrength = Field(..., description="Password strength level")
    score: int = Field(..., ge=0, le=100, description="Password strength score")
    entropy: float = Field(..., ge=0, description="Password entropy in bits")
    crack_time: str = Field(..., description="Estimated time to crack password")

class AccountLockoutInfo(BaseModel):
    """Account lockout information"""
    is_locked: bool = Field(..., description="Whether account is currently locked")
    reason: Optional[str] = Field(None, description="Lockout reason")
    locked_at: Optional[datetime] = Field(None, description="When account was locked")
    locked_until: Optional[datetime] = Field(None, description="When account will be unlocked")
    is_permanent: bool = Field(False, description="Whether lockout is permanent")
    remaining_time_seconds: Optional[float] = Field(None, description="Remaining lockout time in seconds")
    reason_description: Optional[str] = Field(None, description="Detailed lockout description")

class SecurityEventResponse(BaseModel):
    """Security event response"""
    id: int = Field(..., description="Event ID")
    event_type: str = Field(..., description="Type of security event")
    event_description: str = Field(..., description="Event description")
    ip_address: Optional[str] = Field(None, description="IP address associated with event")
    severity: str = Field(..., description="Event severity level")
    created_at: datetime = Field(..., description="When event occurred")
    is_resolved: bool = Field(False, description="Whether event is resolved")

class SecuritySummaryResponse(BaseModel):
    """Comprehensive security summary"""
    user_id: int = Field(..., description="User ID")
    is_locked: bool = Field(..., description="Whether account is locked")
    lockout_info: Optional[AccountLockoutInfo] = Field(None, description="Lockout details")
    recent_failed_attempts: int = Field(..., description="Failed attempts in last 24 hours")
    recent_security_events: int = Field(..., description="Security events in last 24 hours")
    suspicious_activities_30d: int = Field(..., description="Suspicious activities in last 30 days")
    unresolved_password_breaches: int = Field(..., description="Unresolved password breaches")
    security_score: int = Field(..., ge=0, le=100, description="Overall security score")

class UnlockAccountRequest(BaseModel):
    """Request to unlock an account"""
    user_id: int = Field(..., description="User ID to unlock")
    reason: str = Field(..., min_length=1, max_length=500, description="Reason for unlocking")

class UnlockAccountResponse(BaseModel):
    """Response for account unlock"""
    success: bool = Field(..., description="Whether unlock was successful")
    message: str = Field(..., description="Unlock result message")

class PasswordRequirementsRequest(BaseModel):
    """Request to get password requirements"""
    user_id: Optional[int] = Field(None, description="User ID for context")

class PasswordRequirementsResponse(BaseModel):
    """Password requirements configuration"""
    min_length: int = Field(..., description="Minimum password length")
    max_length: int = Field(..., description="Maximum password length")
    require_uppercase: bool = Field(..., description="Require uppercase letters")
    require_lowercase: bool = Field(..., description="Require lowercase letters")
    require_digits: bool = Field(..., description="Require digits")
    require_special_chars: bool = Field(..., description="Require special characters")
    min_special_chars: int = Field(..., description="Minimum special character count")
    max_consecutive_chars: int = Field(..., description="Maximum consecutive characters")
    max_repeated_chars: int = Field(..., description="Maximum repeated characters")
    min_entropy: float = Field(..., description="Minimum entropy requirement")
    min_zxcvbn_score: int = Field(..., description="Minimum zxcvbn score")

class FailedLoginAttemptResponse(BaseModel):
    """Failed login attempt response"""
    success: bool = Field(..., description="Whether recording was successful")
    attempt_id: int = Field(..., description="Attempt ID")
    is_suspicious: bool = Field(..., description="Whether attempt is suspicious")

class AccountStatusResponse(BaseModel):
    """Account status response"""
    is_locked: bool = Field(..., description="Whether account is locked")
    lockout_info: Optional[AccountLockoutInfo] = Field(None, description="Lockout details")

class ClearFailedAttemptsResponse(BaseModel):
    """Clear failed attempts response"""
    success: bool = Field(..., description="Whether clearing was successful")
    message: str = Field(..., description="Result message")

class SecurityMetricsResponse(BaseModel):
    """Security metrics response"""
    total_failed_attempts: int = Field(..., description="Total failed attempts")
    total_lockouts: int = Field(..., description="Total account lockouts")
    active_lockouts: int = Field(..., description="Currently active lockouts")
    suspicious_activities: int = Field(..., description="Suspicious activities")
    password_breaches: int = Field(..., description="Password breaches")
    security_events_24h: int = Field(..., description="Security events in last 24 hours")
    average_security_score: float = Field(..., description="Average security score")

class PasswordBreachCheckRequest(BaseModel):
    """Request to check password breach"""
    password_hash: str = Field(..., description="Password hash to check")

class PasswordBreachCheckResponse(BaseModel):
    """Password breach check response"""
    is_breached: bool = Field(..., description="Whether password is breached")
    breach_sources: List[str] = Field(default_factory=list, description="Sources where password was found")
    breach_date: Optional[datetime] = Field(None, description="Date of breach")

class SuspiciousActivityRequest(BaseModel):
    """Request to record suspicious activity"""
    user_id: Optional[int] = Field(None, description="User ID")
    activity_type: str = Field(..., description="Type of suspicious activity")
    description: str = Field(..., description="Activity description")
    ip_address: Optional[str] = Field(None, description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent")
    risk_score: int = Field(50, ge=0, le=100, description="Risk score")

class SuspiciousActivityResponse(BaseModel):
    """Suspicious activity response"""
    success: bool = Field(..., description="Whether recording was successful")
    activity_id: int = Field(..., description="Activity ID")
    risk_level: str = Field(..., description="Risk level")

class SecurityConfigResponse(BaseModel):
    """Security configuration response"""
    max_failed_attempts: int = Field(..., description="Maximum failed login attempts")
    lockout_duration: int = Field(..., description="Lockout duration in minutes")
    failed_attempt_window: int = Field(..., description="Failed attempt window in minutes")
    suspicious_activity_threshold: int = Field(..., description="Suspicious activity threshold")
    password_requirements: PasswordRequirementsResponse = Field(..., description="Password requirements")

class SecurityHealthCheckResponse(BaseModel):
    """Security health check response"""
    status: str = Field(..., description="Overall security status")
    services: Dict[str, str] = Field(..., description="Individual service status")
    last_check: datetime = Field(..., description="Last health check time")
    recommendations: List[str] = Field(default_factory=list, description="Security recommendations")
