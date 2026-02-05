"""
CSRF protection API routes
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import logging

from ..core.database import get_db
from ..core.security import get_current_user
from ..services.csrf_protection import CSRFProtectionService
from ..models.user import User
from ..schemas.csrf_schemas import (
    CSRFTokenResponse,
    CSRFValidationRequest,
    CSRFValidationResponse,
    CSRFWhitelistRequest,
    CSRFWhitelistResponse,
    CSRFViolationStats,
    CSRFConfigRequest,
    CSRFConfigResponse
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/csrf", tags=["CSRF Protection"])

@router.post("/token", response_model=CSRFTokenResponse)
async def generate_csrf_token(
    request: Request,
    response: Response,
    session_id: Optional[str] = None,
    is_single_use: bool = True,
    is_permanent: bool = False,
    custom_ttl: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate a new CSRF protection token"""
    try:
        csrf_service = CSRFProtectionService(db)
        
        # Extract client information
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent")
        
        # Generate CSRF token
        token = csrf_service.generate_csrf_token(
            user_id=current_user.id,
            session_id=session_id,
            ip_address=client_ip,
            user_agent=user_agent,
            is_single_use=is_single_use,
            is_permanent=is_permanent,
            custom_ttl=custom_ttl
        )
        
        # Set token in cookie
        response.set_cookie(
            key="csrf_token",
            value=token,
            httponly=False,  # Allow JavaScript access
            secure=True,
            samesite="strict"
        )
        
        logger.info(f"CSRF token generated for user {current_user.id}")
        
        return CSRFTokenResponse(
            token=token,
            is_single_use=is_single_use,
            is_permanent=is_permanent,
            message="CSRF token generated successfully"
        )
        
    except Exception as e:
        logger.error(f"Error generating CSRF token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate CSRF token"
        )

@router.post("/validate", response_model=CSRFValidationResponse)
async def validate_csrf_token(
    request: CSRFValidationRequest,
    http_request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Validate a CSRF token"""
    try:
        csrf_service = CSRFProtectionService(db)
        
        # Extract client information
        client_ip = http_request.client.host if http_request.client else "unknown"
        user_agent = http_request.headers.get("user-agent")
        origin = http_request.headers.get("origin")
        referer = http_request.headers.get("referer")
        
        # Validate CSRF token
        is_valid, error_message = csrf_service.validate_csrf_token(
            token=request.token,
            user_id=current_user.id,
            session_id=request.session_id,
            ip_address=client_ip,
            user_agent=user_agent,
            origin=origin,
            referer=referer
        )
        
        if is_valid:
            logger.info(f"CSRF token validated successfully for user {current_user.id}")
            return CSRFValidationResponse(
                is_valid=True,
                message="CSRF token is valid"
            )
        else:
            logger.warning(f"CSRF token validation failed for user {current_user.id}: {error_message}")
            return CSRFValidationResponse(
                is_valid=False,
                message=error_message
            )
        
    except Exception as e:
        logger.error(f"Error validating CSRF token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate CSRF token"
        )

@router.get("/tokens", response_model=List[Dict[str, Any]])
async def get_user_tokens(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all active CSRF tokens for the current user"""
    try:
        csrf_service = CSRFProtectionService(db)
        tokens = csrf_service.get_user_active_tokens(current_user.id)
        
        token_list = []
        for token in tokens:
            token_list.append({
                "id": token.id,
                "token": token.token[:8] + "...",  # Masked token
                "session_id": token.session_id,
                "ip_address": token.ip_address,
                "status": token.status,
                "created_at": token.created_at.isoformat(),
                "expires_at": token.expires_at.isoformat(),
                "is_single_use": token.is_single_use,
                "is_permanent": token.is_permanent
            })
        
        return token_list
        
    except Exception as e:
        logger.error(f"Error getting user tokens: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user tokens"
        )

@router.delete("/tokens", response_model=Dict[str, str])
async def revoke_user_tokens(
    reason: str = "manual_revoke",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Revoke all active CSRF tokens for the current user"""
    try:
        csrf_service = CSRFProtectionService(db)
        revoked_count = csrf_service.revoke_user_tokens(
            user_id=current_user.id,
            reason=reason
        )
        
        logger.info(f"Revoked {revoked_count} CSRF tokens for user {current_user.id}")
        
        return {
            "message": f"Successfully revoked {revoked_count} CSRF tokens",
            "revoked_count": str(revoked_count)
        }
        
    except Exception as e:
        logger.error(f"Error revoking user tokens: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke user tokens"
        )

@router.delete("/tokens/{token_id}", response_model=Dict[str, str])
async def revoke_token(
    token_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Revoke a specific CSRF token"""
    try:
        csrf_service = CSRFProtectionService(db)
        
        # Find the token
        token = db.query(csrf_service.db.query(CSRFProtection).filter(
            CSRFProtection.id == token_id,
            CSRFProtection.user_id == current_user.id
        ).first())
        
        if not token:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="CSRF token not found"
            )
        
        success = csrf_service.revoke_token(token.token, current_user.id)
        
        if success:
            logger.info(f"CSRF token {token_id} revoked for user {current_user.id}")
            return {"message": "CSRF token revoked successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to revoke CSRF token"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking CSRF token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke CSRF token"
        )

@router.post("/whitelist", response_model=CSRFWhitelistResponse)
async def add_whitelist_origin(
    request: CSRFWhitelistRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Add an origin to the CSRF whitelist (admin only)"""
    try:
        # Check if user is admin
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        csrf_service = CSRFProtectionService(db)
        whitelist_entry = csrf_service.add_whitelist_origin(
            origin=request.origin,
            description=request.description,
            created_by=current_user.id,
            expires_at=request.expires_at
        )
        
        logger.info(f"Added origin {request.origin} to CSRF whitelist by user {current_user.id}")
        
        return CSRFWhitelistResponse(
            id=whitelist_entry.id,
            origin=whitelist_entry.origin,
            description=whitelist_entry.description,
            is_active=whitelist_entry.is_active,
            created_at=whitelist_entry.created_at.isoformat(),
            message="Origin added to whitelist successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding whitelist origin: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add whitelist origin"
        )

@router.get("/violations/stats", response_model=CSRFViolationStats)
async def get_violation_stats(
    user_id: Optional[int] = None,
    ip_address: Optional[str] = None,
    hours: int = 24,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get CSRF violation statistics (admin only)"""
    try:
        # Check if user is admin
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        csrf_service = CSRFProtectionService(db)
        stats = csrf_service.get_violation_stats(
            user_id=user_id,
            ip_address=ip_address,
            hours=hours
        )
        
        return CSRFViolationStats(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting violation stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get violation statistics"
        )

@router.post("/cleanup", response_model=Dict[str, str])
async def cleanup_expired_tokens(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Clean up expired CSRF tokens (admin only)"""
    try:
        # Check if user is admin
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        csrf_service = CSRFProtectionService(db)
        cleaned_count = csrf_service.cleanup_expired_tokens()
        
        logger.info(f"Cleaned up {cleaned_count} expired CSRF tokens")
        
        return {
            "message": f"Successfully cleaned up {cleaned_count} expired tokens",
            "cleaned_count": str(cleaned_count)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up expired tokens: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clean up expired tokens"
        )
