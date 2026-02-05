"""
Admin API routes for user management.
"""
from fastapi import APIRouter, Depends, HTTPException, Request, status, Query
from sqlalchemy.orm import Session
from uuid import UUID
from typing import Optional

from src.core.database import get_db
from src.core.config import settings
from src.services.user_service import user_service
from src.models.user import UserRole
from src.schemas.user_schemas import (
    UserResponse, UserListResponse, AdminUserUpdate, MessageResponse
)
from src.schemas.auth_schemas import TokenData
# from src.middleware.auth import get_current_user, require_admin
from src.middleware.authorization import (
    Permission, require_permission, require_permissions, require_admin_permission
)

# Placeholder functions for missing auth dependencies
def get_current_user():
    """Placeholder for get_current_user"""
    return None

def require_admin():
    """Placeholder for require_admin"""
    return None

router = APIRouter()

@router.get("/users", response_model=UserListResponse)
async def get_users_list(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search term"),
    role: Optional[UserRole] = Query(None, description="Filter by role"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    current_user: TokenData = Depends(require_admin_permission(Permission.READ_ALL_USERS)),
    db: Session = Depends(get_db)
):
    """Get paginated list of users (admin only)."""
    user_list = user_service.get_users_list(
        db=db,
        page=page,
        per_page=per_page,
        search=search,
        role=role,
        is_active=is_active,
        sort_by=sort_by,
        sort_order=sort_order
    )
    
    return user_list

@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: str,
    current_user: TokenData = Depends(require_admin_permission(Permission.READ_ALL_USERS)),
    db: Session = Depends(get_db)
):
    """Get user by ID (admin only)."""
    try:
        user_uuid = UUID(user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    
    user_response = user_service.get_user_profile(db, user_uuid)
    if not user_response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user_response

@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user_admin(
    user_id: str,
    update_data: AdminUserUpdate,
    request: Request,
    current_user: TokenData = Depends(require_admin_permission(Permission.READ_ALL_USERS)),
    db: Session = Depends(get_db)
):
    """Update user information (admin only)."""
    try:
        user_uuid = UUID(user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    
    # Get client information
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent")
    
    success, message, user_response = user_service.update_user_admin(
        db=db,
        user_id=user_uuid,
        update_data=update_data,
        admin_user_id=UUID(current_user.user_id),
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return user_response

@router.post("/users/{user_id}/deactivate", response_model=MessageResponse)
async def deactivate_user_admin(
    user_id: str,
    request: Request,
    current_user: TokenData = Depends(require_admin_permission(Permission.READ_ALL_USERS)),
    db: Session = Depends(get_db)
):
    """Deactivate user account (admin only)."""
    try:
        user_uuid = UUID(user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    
    # Get client information
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent")
    
    success, message = user_service.deactivate_user(
        db=db,
        user_id=user_uuid,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return MessageResponse(message=message)

@router.post("/users/{user_id}/reactivate", response_model=MessageResponse)
async def reactivate_user_admin(
    user_id: str,
    request: Request,
    current_user: TokenData = Depends(require_admin_permission(Permission.READ_ALL_USERS)),
    db: Session = Depends(get_db)
):
    """Reactivate user account (admin only)."""
    try:
        user_uuid = UUID(user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    
    # Get client information
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent")
    
    success, message = user_service.reactivate_user(
        db=db,
        user_id=user_uuid,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return MessageResponse(message=message)

@router.get("/users/{user_id}/sessions")
async def get_user_sessions_admin(
    user_id: str,
    current_user: TokenData = Depends(require_admin_permission(Permission.READ_ALL_USERS)),
    db: Session = Depends(get_db)
):
    """Get user's active sessions (admin only)."""
    try:
        user_uuid = UUID(user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    
    sessions = user_service.get_user_sessions(db, user_uuid)
    return {"sessions": sessions}

@router.delete("/users/{user_id}/sessions/{session_id}")
async def revoke_user_session_admin(
    user_id: str,
    session_id: str,
    request: Request,
    current_user: TokenData = Depends(require_admin_permission(Permission.READ_ALL_USERS)),
    db: Session = Depends(get_db)
):
    """Revoke a specific user session (admin only)."""
    try:
        user_uuid = UUID(user_id)
        session_uuid = UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID or session ID format"
        )
    
    # Get client information
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent")
    
    success, message = user_service.revoke_user_session(
        db=db,
        user_id=user_uuid,
        session_id=session_uuid,
        admin_user_id=UUID(current_user.user_id),
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return MessageResponse(message=message)

@router.delete("/users/{user_id}/sessions")
async def revoke_all_user_sessions_admin(
    user_id: str,
    request: Request,
    current_user: TokenData = Depends(require_admin_permission(Permission.READ_ALL_USERS)),
    db: Session = Depends(get_db)
):
    """Revoke all user sessions (admin only)."""
    try:
        user_uuid = UUID(user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    
    # Get client information
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent")
    
    success, message = user_service.revoke_all_user_sessions(
        db=db,
        user_id=user_uuid,
        admin_user_id=UUID(current_user.user_id),
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return MessageResponse(message=message)

@router.get("/users/{user_id}/stats")
async def get_user_stats_admin(
    user_id: str,
    current_user: TokenData = Depends(require_admin_permission(Permission.READ_ALL_USERS)),
    db: Session = Depends(get_db)
):
    """Get user statistics (admin only)."""
    try:
        user_uuid = UUID(user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    
    stats = user_service.get_user_stats(db, user_uuid)
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return stats

@router.post("/cleanup/sessions")
async def cleanup_inactive_sessions(
    days_inactive: int = Query(30, ge=1, le=365, description="Days of inactivity"),
    current_user: TokenData = Depends(require_admin_permission(Permission.READ_ALL_USERS)),
    db: Session = Depends(get_db)
):
    """Clean up inactive sessions (admin only)."""
    count = user_service.cleanup_inactive_sessions(db, days_inactive)
    return MessageResponse(message=f"Cleaned up {count} inactive sessions")
