"""
User schemas for API request/response models.
"""
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List
from datetime import datetime
from uuid import UUID
from enum import Enum

class UserRole(str, Enum):
    """User role enumeration."""
    USER = "user"
    ADMIN = "admin"

class UserBase(BaseModel):
    """Base user schema with common fields."""
    email: EmailStr
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    role: UserRole = UserRole.USER

class UserCreate(UserBase):
    """Schema for creating a new user."""
    password: str = Field(..., min_length=8, max_length=128)
    
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

class UserUpdate(BaseModel):
    """Schema for updating user profile."""
    first_name: Optional[str] = Field(None, min_length=1, max_length=50)
    last_name: Optional[str] = Field(None, min_length=1, max_length=50)

class AdminUserUpdate(BaseModel):
    """Schema for admin updating user information."""
    first_name: Optional[str] = Field(None, min_length=1, max_length=50)
    last_name: Optional[str] = Field(None, min_length=1, max_length=50)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None

class UserResponse(UserBase):
    """Schema for user response."""
    id: UUID
    is_active: bool
    is_verified: bool
    last_login: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class UserProfile(UserResponse):
    """Schema for user profile response."""
    pass

class UserListResponse(BaseModel):
    """Schema for user list response with pagination."""
    users: List[UserResponse]
    pagination: dict = Field(..., description="Pagination information")

class ChangePasswordRequest(BaseModel):
    """Schema for changing password."""
    current_password: str = Field(..., min_length=1)
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

class ChangePasswordResponse(BaseModel):
    """Schema for change password response."""
    message: str

class UserDeactivationResponse(BaseModel):
    """Schema for user deactivation response."""
    message: str

class UserRegistrationRequest(BaseModel):
    """Request schema for user registration"""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., min_length=8, max_length=128, description="User password")
    
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
    """Response schema for user registration"""
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="User email")
    message: str = Field(..., description="Registration message")

class UserLoginRequest(BaseModel):
    """Request schema for user login"""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="User password")

class UserLoginResponse(BaseModel):
    """Response schema for user login"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(..., description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: UserResponse = Field(..., description="User information")

class UserProfileResponse(BaseModel):
    """Response schema for user profile"""
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="User email")
    first_name: str = Field(..., description="First name")
    last_name: str = Field(..., description="Last name")
    role: str = Field(..., description="User role")
    subscription_tier: str = Field(..., description="Subscription tier")
    is_active: bool = Field(..., description="Is user active")
    created_at: datetime = Field(..., description="Account creation date")
    last_login: Optional[datetime] = Field(None, description="Last login date")

class UserProfileUpdateRequest(BaseModel):
    """Request schema for updating user profile"""
    first_name: Optional[str] = Field(None, min_length=1, max_length=50, description="First name")
    last_name: Optional[str] = Field(None, min_length=1, max_length=50, description="Last name")
    email: Optional[EmailStr] = Field(None, description="User email")

class UserDashboardResponse(BaseModel):
    """Response schema for user dashboard data"""
    user: UserProfileResponse = Field(..., description="User profile")
    stats: dict = Field(..., description="User statistics")
    recent_activity: List[dict] = Field(..., description="Recent activity")

class UserAnalyticsResponse(BaseModel):
    """Response schema for user analytics"""
    total_activities: int = Field(..., description="Total number of activities")
    activity_breakdown: dict = Field(..., description="Activity breakdown by type")
    monthly_activity: List[dict] = Field(..., description="Monthly activity data")
    top_keywords: List[dict] = Field(..., description="Top keywords used")
    content_created: int = Field(..., description="Number of content pieces created")
    software_generated: int = Field(..., description="Number of software solutions generated")
    export_count: int = Field(..., description="Number of exports performed")
    last_activity: Optional[datetime] = Field(None, description="Last activity date")

class MessageResponse(BaseModel):
    """Generic message response."""
    message: str

class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    message: str
    details: Optional[dict] = None
