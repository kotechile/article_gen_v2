"""
Authentication API routes.
"""
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Dict, Any

from src.core.database import get_db
from src.core.config import settings
from src.services.auth_service import auth_service
from src.schemas.auth_schemas import (
    UserRegistrationRequest, UserRegistrationResponse,
    LoginRequest, LoginResponse,
    LogoutResponse, RefreshTokenRequest, RefreshTokenResponse,
    EmailVerificationRequest, EmailVerificationResponse,
    PasswordResetRequest, PasswordResetResponse,
    PasswordResetConfirmRequest, PasswordResetConfirmResponse,
    AuthErrorResponse
)
from src.schemas.user_schemas import UserResponse

router = APIRouter()
security = HTTPBearer()

def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Get current user ID from JWT token."""
    token = credentials.credentials
    token_data = auth_service.verify_access_token(token)
    
    if not token_data or not token_data.user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return str(token_data.user_id)

@router.post("/register", response_model=UserRegistrationResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserRegistrationRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """Register a new user."""
    if not settings.enable_user_registration:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User registration is currently disabled"
        )
    
    # Get client information
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent")
    
    success, message, user_response = auth_service.register_user(
        db=db,
        user_data=user_data,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return UserRegistrationResponse(
        message=message,
        user=user_response
    )

@router.post("/verify-email", response_model=EmailVerificationResponse)
async def verify_email(
    verification_data: EmailVerificationRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """Verify user email with token."""
    if not settings.enable_email_verification:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification is currently disabled"
        )
    
    # Get client information
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent")
    
    success, message = auth_service.verify_email(
        db=db,
        token=verification_data.token,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return EmailVerificationResponse(message=message)

@router.post("/login", response_model=LoginResponse)
async def login_user(
    login_data: LoginRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """Authenticate user and return tokens."""
    # Get client information
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent")
    device_info = {
        "user_agent": user_agent,
        "ip_address": ip_address
    }
    
    success, message, response_data = auth_service.login_user(
        db=db,
        login_data=login_data,
        ip_address=ip_address,
        user_agent=user_agent,
        device_info=device_info
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=message
        )
    
    return LoginResponse(**response_data)

@router.post("/refresh", response_model=RefreshTokenResponse)
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """Refresh access token using refresh token."""
    # Get client information
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent")
    
    success, message, response_data = auth_service.refresh_token(
        db=db,
        refresh_token=refresh_data.refresh_token,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=message
        )
    
    return RefreshTokenResponse(**response_data)

@router.post("/logout", response_model=LogoutResponse)
async def logout_user(
    request: Request,
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """Logout user and invalidate session."""
    # Get access token from Authorization header
    authorization = request.headers.get("Authorization")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header"
        )
    
    access_token = authorization.split(" ")[1]
    
    # Get client information
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent")
    
    success, message = auth_service.logout_user(
        db=db,
        access_token=access_token,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return LogoutResponse(message=message)

@router.post("/request-password-reset", response_model=PasswordResetResponse)
async def request_password_reset(
    reset_data: PasswordResetRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """Request password reset email."""
    if not settings.enable_password_reset:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Password reset is currently disabled"
        )
    
    # Get client information
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent")
    
    success, message = auth_service.request_password_reset(
        db=db,
        reset_data=reset_data,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return PasswordResetResponse(message=message)

@router.post("/confirm-password-reset", response_model=PasswordResetConfirmResponse)
async def confirm_password_reset(
    reset_data: PasswordResetConfirmRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """Confirm password reset with new password."""
    if not settings.enable_password_reset:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Password reset is currently disabled"
        )
    
    # Get client information
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent")
    
    success, message = auth_service.confirm_password_reset(
        db=db,
        reset_data=reset_data,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return PasswordResetConfirmResponse(message=message)

@router.get("/me", response_model=UserResponse)
async def get_current_user(
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """Get current user information."""
    user = auth_service.get_user_by_id(db, current_user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse.from_orm(user)

