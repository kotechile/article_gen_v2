"""
Google OAuth API Routes

This module provides API endpoints for Google OAuth authentication integration.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
import structlog
from ..services.google_auth_service import get_google_auth_service, GoogleAuthService

logger = structlog.get_logger()
router = APIRouter(prefix="/api/auth/google", tags=["Google OAuth"])

class GoogleUserData(BaseModel):
    """Google user data from OAuth"""
    id: str
    email: str
    name: str
    picture: Optional[str] = None
    verified_email: bool = True

class GoogleAuthResponse(BaseModel):
    """Response for Google authentication"""
    success: bool
    user: Optional[Dict[str, Any]] = None
    message: str
    token: Optional[str] = None

@router.post("/callback", response_model=GoogleAuthResponse)
async def google_auth_callback(
    user_data: GoogleUserData,
    google_auth_service: GoogleAuthService = Depends(get_google_auth_service)
):
    """
    Handle Google OAuth callback
    
    This endpoint processes the Google OAuth callback and creates/updates the user.
    """
    try:
        logger.info("Processing Google OAuth callback", user_id=user_data.id, email=user_data.email)
        
        # Create or update user from Google data
        user = await google_auth_service.create_or_update_user_from_google({
            'id': user_data.id,
            'email': user_data.email,
            'name': user_data.name,
            'picture': user_data.picture,
            'verified_email': user_data.verified_email
        })
        
        return GoogleAuthResponse(
            success=True,
            user=user,
            message="Google authentication successful",
            token="mock_jwt_token"  # In real implementation, this would be a JWT token
        )
        
    except Exception as e:
        logger.error("Google OAuth callback error", error=str(e), user_data=user_data.dict())
        raise HTTPException(
            status_code=400,
            detail=f"Google authentication failed: {str(e)}"
        )

@router.get("/profile/{user_id}")
async def get_google_user_profile(
    user_id: str,
    google_auth_service: GoogleAuthService = Depends(get_google_auth_service)
):
    """
    Get Google user profile
    
    Args:
        user_id: User ID
        
    Returns:
        User profile data
    """
    try:
        profile = await google_auth_service.get_user_profile(user_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "success": True,
            "user": profile
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting Google user profile", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get user profile: {str(e)}"
        )

@router.post("/validate-token")
async def validate_google_token(
    token: str,
    google_auth_service: GoogleAuthService = Depends(get_google_auth_service)
):
    """
    Validate Google OAuth token
    
    Args:
        token: Google OAuth token
        
    Returns:
        Validation result
    """
    try:
        user_data = await google_auth_service.validate_google_token(token)
        
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid Google token")
        
        return {
            "success": True,
            "user": user_data,
            "message": "Token is valid"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error validating Google token", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Token validation failed: {str(e)}"
        )

@router.get("/health")
async def google_auth_health():
    """Health check for Google OAuth service"""
    return {
        "status": "healthy",
        "service": "Google OAuth",
        "message": "Google OAuth service is running"
    }

