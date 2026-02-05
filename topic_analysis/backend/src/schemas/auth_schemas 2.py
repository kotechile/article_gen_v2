"""
Authentication schemas for API request/response models.
"""
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional
from datetime import datetime
from uuid import UUID
from .user_schemas import UserResponse

class LoginRequest(BaseModel):
    """Schema for user login request."""
    email: EmailStr
    password: str = Field(..., min_length=1)

class LoginResponse(BaseModel):
    """Schema for login response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

class LogoutResponse(BaseModel):
    """Schema for logout response."""
    message: str

class RefreshTokenRequest(BaseModel):
    """Schema for refresh token request."""
    refresh_token: str = Field(..., min_length=1)

class RefreshTokenResponse(LoginResponse):
    """Schema for refresh token response (same as login)."""
    pass

class UserRegistrationRequest(BaseModel):
    """Schema for user registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v):
            raise ValueError('Password must contain at least one special character')
        return v

class UserRegistrationResponse(BaseModel):
    """Schema for user registration response."""
    message: str
    user: UserResponse

class EmailVerificationRequest(BaseModel):
    """Schema for email verification request."""
    token: str = Field(..., min_length=1)

class EmailVerificationResponse(BaseModel):
    """Schema for email verification response."""
    message: str

class PasswordResetRequest(BaseModel):
    """Schema for password reset request."""
    email: EmailStr

class PasswordResetResponse(BaseModel):
    """Schema for password reset response."""
    message: str

class PasswordResetConfirmRequest(BaseModel):
    """Schema for password reset confirmation request."""
    token: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)
    
    @validator('new_password')
    def validate_new_password(cls, v):
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v):
            raise ValueError('Password must contain at least one special character')
        return v

class PasswordResetConfirmResponse(BaseModel):
    """Schema for password reset confirmation response."""
    message: str

class TokenData(BaseModel):
    """Schema for JWT token data."""
    user_id: Optional[UUID] = None
    email: Optional[str] = None
    role: Optional[str] = None
    jti: Optional[str] = None  # JWT ID for token identification

class AuthErrorResponse(BaseModel):
    """Schema for authentication error response."""
    error: str
    message: str
    details: Optional[dict] = None

class ValidationErrorDetail(BaseModel):
    """Schema for validation error detail."""
    field: str
    reason: str

class ValidationErrorResponse(BaseModel):
    """Schema for validation error response."""
    error: str = "VALIDATION_ERROR"
    message: str
    details: list[ValidationErrorDetail]

