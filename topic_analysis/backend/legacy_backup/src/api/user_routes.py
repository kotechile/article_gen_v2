"""
User Management API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog

from ..core.database import get_db
from ..services.user_service import UserService
from ..models.user import User, UserRole, SubscriptionTier
from ..schemas.user_schemas import (
    UserRegistrationRequest,
    UserRegistrationResponse,
    UserLoginRequest,
    UserLoginResponse,
    UserProfileResponse,
    UserProfileUpdateRequest,
    UserDashboardResponse,
    UserAnalyticsResponse
)

logger = structlog.get_logger()
router = APIRouter(prefix="/api/users", tags=["user-management"])


def get_user_service(db: Session = Depends(get_db)) -> UserService:
    """Get user service dependency"""
    return UserService(db)


def get_current_user(db: Session = Depends(get_db)) -> User:
    """Get current authenticated user (placeholder - implement auth middleware)"""
    # This is a placeholder - in real implementation, this would extract user from JWT token
    user = db.query(User).first()
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


@router.post("/register", response_model=UserRegistrationResponse)
async def register_user(
    request: UserRegistrationRequest,
    user_service: UserService = Depends(get_user_service)
):
    """Register new user"""
    try:
        logger.info("Registering new user", username=request.username, email=request.email)
        
        result = await user_service.create_user(
            username=request.username,
            email=request.email,
            password=request.password,
            subscription_tier=SubscriptionTier.FREE
        )
        
        logger.info("User registered successfully", user_id=result["user_id"], username=request.username)
        
        return UserRegistrationResponse(
            success=True,
            user_id=result["user_id"],
            username=result["username"],
            email=result["email"],
            subscription_tier=result["subscription_tier"],
            is_active=result["is_active"],
            created_at=result["created_at"],
            token=result["token"]
        )
        
    except ValueError as e:
        logger.error("Invalid registration request", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to register user", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/login", response_model=UserLoginResponse)
async def login_user(
    request: UserLoginRequest,
    user_service: UserService = Depends(get_user_service)
):
    """Authenticate user"""
    try:
        logger.info("User login attempt", username=request.username)
        
        result = await user_service.authenticate_user(
            username=request.username,
            password=request.password
        )
        
        logger.info("User authenticated successfully", user_id=result["user_id"], username=request.username)
        
        return UserLoginResponse(
            success=True,
            user_id=result["user_id"],
            username=result["username"],
            email=result["email"],
            subscription_tier=result["subscription_tier"],
            is_active=result["is_active"],
            last_login=result["last_login"],
            token=result["token"]
        )
        
    except ValueError as e:
        logger.error("Invalid login credentials", error=str(e))
        raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception as e:
        logger.error("Failed to authenticate user", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/profile", response_model=UserProfileResponse)
async def get_user_profile(
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    """Get user profile"""
    try:
        profile = await user_service.get_user_profile(current_user.id)
        
        return UserProfileResponse(
            user_id=profile["user_id"],
            username=profile["username"],
            email=profile["email"],
            subscription_tier=profile["subscription_tier"],
            is_active=profile["is_active"],
            created_at=profile["created_at"],
            last_login=profile["last_login"],
            statistics=profile["statistics"],
            limits=profile["limits"]
        )
        
    except Exception as e:
        logger.error("Failed to get user profile", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/profile", response_model=UserProfileResponse)
async def update_user_profile(
    request: UserProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    """Update user profile"""
    try:
        logger.info("Updating user profile", user_id=current_user.id)
        
        success = await user_service.update_user_profile(
            user_id=current_user.id,
            username=request.username,
            email=request.email,
            password=request.password
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update profile")
        
        # Get updated profile
        profile = await user_service.get_user_profile(current_user.id)
        
        logger.info("User profile updated successfully", user_id=current_user.id)
        
        return UserProfileResponse(
            user_id=profile["user_id"],
            username=profile["username"],
            email=profile["email"],
            subscription_tier=profile["subscription_tier"],
            is_active=profile["is_active"],
            created_at=profile["created_at"],
            last_login=profile["last_login"],
            statistics=profile["statistics"],
            limits=profile["limits"]
        )
        
    except ValueError as e:
        logger.error("Invalid profile update request", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to update user profile", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/upgrade-subscription")
async def upgrade_subscription(
    new_tier: SubscriptionTier,
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    """Upgrade user subscription"""
    try:
        logger.info("Upgrading subscription", user_id=current_user.id, new_tier=new_tier.value)
        
        success = await user_service.upgrade_subscription(
            user_id=current_user.id,
            new_tier=new_tier
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to upgrade subscription")
        
        logger.info("Subscription upgraded successfully", user_id=current_user.id, new_tier=new_tier.value)
        
        return {
            "success": True,
            "message": f"Subscription upgraded to {new_tier.value}",
            "new_tier": new_tier.value
        }
        
    except ValueError as e:
        logger.error("Invalid subscription upgrade request", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to upgrade subscription", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/deactivate")
async def deactivate_user(
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    """Deactivate user account"""
    try:
        logger.info("Deactivating user account", user_id=current_user.id)
        
        success = await user_service.deactivate_user(current_user.id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to deactivate account")
        
        logger.info("User account deactivated", user_id=current_user.id)
        
        return {
            "success": True,
            "message": "Account deactivated successfully"
        }
        
    except Exception as e:
        logger.error("Failed to deactivate user account", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/dashboard", response_model=UserDashboardResponse)
async def get_user_dashboard(
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    """Get user dashboard data"""
    try:
        dashboard = await user_service.get_user_dashboard(current_user.id)
        
        return UserDashboardResponse(
            user=dashboard["user"],
            recent_activity=dashboard["recent_activity"],
            statistics=dashboard["statistics"],
            upcoming_reminders=dashboard["upcoming_reminders"],
            subscription=dashboard["subscription"]
        )
        
    except Exception as e:
        logger.error("Failed to get user dashboard", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics", response_model=UserAnalyticsResponse)
async def get_user_analytics(
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    """Get user analytics"""
    try:
        analytics = await user_service.get_user_analytics(current_user.id)
        
        return UserAnalyticsResponse(
            total_activities=analytics["total_activities"],
            activity_breakdown=analytics["activity_breakdown"],
            monthly_activity=analytics["monthly_activity"],
            subscription_usage=analytics["subscription_usage"],
            productivity_metrics=analytics["productivity_metrics"]
        )
        
    except Exception as e:
        logger.error("Failed to get user analytics", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/limits", response_model=Dict[str, Any])
async def get_user_limits(
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    """Get user limits and usage"""
    try:
        limits = await user_service.get_user_limits(current_user.id)
        
        return limits
        
    except Exception as e:
        logger.error("Failed to get user limits", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/subscription-tiers", response_model=List[Dict[str, Any]])
async def get_subscription_tiers():
    """Get available subscription tiers"""
    try:
        tiers = [
            {
                "id": "free",
                "name": "Free",
                "price": 0,
                "currency": "USD",
                "features": [
                    "5 affiliate researches",
                    "10 trend analyses",
                    "20 content ideas",
                    "10 software solutions",
                    "50 calendar entries"
                ],
                "limits": {
                    "affiliate_researches": 5,
                    "trend_analyses": 10,
                    "content_ideas": 20,
                    "software_solutions": 10,
                    "calendar_entries": 50
                }
            },
            {
                "id": "basic",
                "name": "Basic",
                "price": 29,
                "currency": "USD",
                "features": [
                    "25 affiliate researches",
                    "50 trend analyses",
                    "100 content ideas",
                    "50 software solutions",
                    "250 calendar entries",
                    "Priority support"
                ],
                "limits": {
                    "affiliate_researches": 25,
                    "trend_analyses": 50,
                    "content_ideas": 100,
                    "software_solutions": 50,
                    "calendar_entries": 250
                }
            },
            {
                "id": "pro",
                "name": "Pro",
                "price": 79,
                "currency": "USD",
                "features": [
                    "100 affiliate researches",
                    "200 trend analyses",
                    "500 content ideas",
                    "200 software solutions",
                    "1000 calendar entries",
                    "Advanced analytics",
                    "API access",
                    "Priority support"
                ],
                "limits": {
                    "affiliate_researches": 100,
                    "trend_analyses": 200,
                    "content_ideas": 500,
                    "software_solutions": 200,
                    "calendar_entries": 1000
                }
            },
            {
                "id": "enterprise",
                "name": "Enterprise",
                "price": 199,
                "currency": "USD",
                "features": [
                    "Unlimited everything",
                    "Custom integrations",
                    "Dedicated support",
                    "White-label options",
                    "Custom training"
                ],
                "limits": {
                    "affiliate_researches": -1,
                    "trend_analyses": -1,
                    "content_ideas": -1,
                    "software_solutions": -1,
                    "calendar_entries": -1
                }
            }
        ]
        
        return tiers
        
    except Exception as e:
        logger.error("Failed to get subscription tiers", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/activity", response_model=List[Dict[str, Any]])
async def get_user_activity(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    """Get user activity history"""
    try:
        activity = await user_service.get_user_activity(
            user_id=current_user.id,
            skip=skip,
            limit=limit
        )
        
        return activity
        
    except Exception as e:
        logger.error("Failed to get user activity", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/export-data")
async def export_user_data(
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    """Export user data"""
    try:
        logger.info("Exporting user data", user_id=current_user.id)
        
        export_data = await user_service.export_user_data(current_user.id)
        
        logger.info("User data exported", user_id=current_user.id)
        
        return {
            "success": True,
            "download_url": export_data["download_url"],
            "expires_at": export_data["expires_at"]
        }
        
    except Exception as e:
        logger.error("Failed to export user data", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/account")
async def delete_user_account(
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    """Delete user account and all data"""
    try:
        logger.info("Deleting user account", user_id=current_user.id)
        
        success = await user_service.delete_user_account(current_user.id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete account")
        
        logger.info("User account deleted", user_id=current_user.id)
        
        return {
            "success": True,
            "message": "Account and all data deleted successfully"
        }
        
    except Exception as e:
        logger.error("Failed to delete user account", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")