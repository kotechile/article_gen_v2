"""
Security-related API routes for password validation and account lockout
"""
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

from ..core.database import get_db
from ..core.supabase_auth import get_current_user
from src.core.supabase_database_service import SupabaseDatabaseService
from ..models.user import User
from ..services.password_validation import PasswordValidator, PasswordRequirements, validate_password_strength
from ..services.account_lockout import AccountLockoutService
from ..schemas.security_schemas import (
    PasswordValidationRequest,
    PasswordValidationResponse,
    PasswordGenerationRequest,
    PasswordGenerationResponse,
    AccountLockoutInfo,
    SecurityEventResponse,
    SecuritySummaryResponse,
    UnlockAccountRequest,
    UnlockAccountResponse
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/security", tags=["security"])
security = HTTPBearer()

@router.post("/validate-password", response_model=PasswordValidationResponse)
async def validate_password(
    request: PasswordValidationRequest,
    db: SupabaseDatabaseService = Depends(get_db)
):
    """Validate password strength"""
    try:
        # Get user info for context if user is authenticated
        user_info = None
        if request.user_id:
            user = db.get_User_by_id(User.id == request.user_id)
            if user:
                user_info = {
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "username": user.email.split("@")[0]
                }
        
        # Validate password
        result = validate_password_strength(request.password, user_info)
        
        return PasswordValidationResponse(
            is_valid=result.is_valid,
            strength=result.strength.value,
            score=result.score,
            feedback=result.feedback,
            suggestions=result.suggestions,
            requirements_met=result.requirements_met,
            entropy=result.entropy,
            crack_time=result.crack_time,
            crack_time_seconds=result.crack_time_seconds
        )
    
    except Exception as e:
        logger.error(f"Password validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password validation failed"
        )

@router.post("/generate-password", response_model=PasswordGenerationResponse)
async def generate_password(
    request: PasswordGenerationRequest,
    db: SupabaseDatabaseService = Depends(get_db)
):
    """Generate a strong password"""
    try:
        from ..services.password_validation import generate_strong_password
        
        password = generate_strong_password(
            length=request.length,
            include_special=request.include_special
        )
        
        # Validate the generated password
        result = validate_password_strength(password)
        
        return PasswordGenerationResponse(
            password=password,
            strength=result.strength.value,
            score=result.score,
            entropy=result.entropy,
            crack_time=result.crack_time
        )
    
    except Exception as e:
        logger.error(f"Password generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password generation failed"
        )

@router.get("/lockout-info/{user_id}", response_model=AccountLockoutInfo)
async def get_lockout_info(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: SupabaseDatabaseService = Depends(get_db)
):
    """Get account lockout information"""
    try:
        # Check if user can access this information
        if current_user.id != user_id and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this information"
            )
        
        lockout_service = AccountLockoutService(db)
        lockout_info = lockout_service.get_lockout_info(user_id)
        
        if not lockout_info:
            return AccountLockoutInfo(
                is_locked=False,
                reason=None,
                locked_at=None,
                locked_until=None,
                is_permanent=False,
                remaining_time_seconds=None,
                reason_description=None
            )
        
        return AccountLockoutInfo(**lockout_info)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get lockout info error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get lockout information"
        )

@router.post("/unlock-account", response_model=UnlockAccountResponse)
async def unlock_account(
    request: UnlockAccountRequest,
    current_user: User = Depends(get_current_user),
    db: SupabaseDatabaseService = Depends(get_db)
):
    """Unlock an account"""
    try:
        # Only admins can unlock accounts
        if current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can unlock accounts"
            )
        
        lockout_service = AccountLockoutService(db)
        success = lockout_service.unlock_account(
            user_id=request.user_id,
            reason=f"Unlocked by admin {current_user.email}: {request.reason}"
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active lockout found for this user"
            )
        
        return UnlockAccountResponse(
            success=True,
            message=f"Account {request.user_id} unlocked successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unlock account error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to unlock account"
        )

@router.get("/security-events", response_model=List[SecurityEventResponse])
async def get_security_events(
    user_id: Optional[int] = None,
    event_type: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: SupabaseDatabaseService = Depends(get_db)
):
    """Get security events"""
    try:
        # Check permissions
        if user_id and current_user.id != user_id and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this information"
            )
        
        lockout_service = AccountLockoutService(db)
        events = lockout_service.get_security_events(
            user_id=user_id,
            event_type=event_type,
            severity=severity,
            limit=limit
        )
        
        return [
            SecurityEventResponse(
                id=event.id,
                event_type=event.event_type,
                event_description=event.event_description,
                ip_address=event.ip_address,
                severity=event.severity,
                created_at=event.created_at,
                is_resolved=event.is_resolved
            )
            for event in events
        ]
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get security events error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get security events"
        )

@router.get("/security-summary/{user_id}", response_model=SecuritySummaryResponse)
async def get_security_summary(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: SupabaseDatabaseService = Depends(get_db)
):
    """Get comprehensive security summary for account"""
    try:
        # Check permissions
        if current_user.id != user_id and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this information"
            )
        
        lockout_service = AccountLockoutService(db)
        summary = lockout_service.get_account_security_summary(user_id)
        
        return SecuritySummaryResponse(**summary)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get security summary error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get security summary"
        )

@router.post("/record-failed-login")
async def record_failed_login(
    email: str,
    ip_address: str,
    user_agent: Optional[str] = None,
    failure_reason: Optional[str] = None,
    db: SupabaseDatabaseService = Depends(get_db)
):
    """Record a failed login attempt (public endpoint)"""
    try:
        # Get user by email
        user = db.get_User_by_id(User.email == email)
        user_id = user.id if user else None
        
        lockout_service = AccountLockoutService(db)
        attempt = lockout_service.record_failed_login(
            email=email,
            ip_address=ip_address,
            user_agent=user_agent,
            user_id=user_id,
            failure_reason=failure_reason
        )
        
        return {
            "success": True,
            "attempt_id": attempt.id,
            "is_suspicious": attempt.is_suspicious
        }
    
    except Exception as e:
        logger.error(f"Record failed login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record failed login attempt"
        )

@router.get("/check-account-status/{user_id}")
async def check_account_status(
    user_id: int,
    db: SupabaseDatabaseService = Depends(get_db)
):
    """Check if account is locked (public endpoint for login)"""
    try:
        lockout_service = AccountLockoutService(db)
        is_locked = lockout_service.is_account_locked(user_id)
        lockout_info = lockout_service.get_lockout_info(user_id)
        
        return {
            "is_locked": is_locked,
            "lockout_info": lockout_info
        }
    
    except Exception as e:
        logger.error(f"Check account status error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check account status"
        )

@router.post("/clear-failed-attempts/{user_id}")
async def clear_failed_attempts(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: SupabaseDatabaseService = Depends(get_db)
):
    """Clear failed login attempts (after successful login)"""
    try:
        # Check permissions
        if current_user.id != user_id and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to perform this action"
            )
        
        lockout_service = AccountLockoutService(db)
        lockout_service.clear_failed_attempts(user_id)
        
        return {"success": True, "message": "Failed attempts cleared"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clear failed attempts error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear failed attempts"
        )
