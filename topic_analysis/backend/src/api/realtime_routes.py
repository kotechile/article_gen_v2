"""
Real-time Routes

This module provides API endpoints for real-time subscriptions,
including subscription management and event handling.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from typing import Dict, Any, Optional
from datetime import datetime
from uuid import uuid4

from ..services.realtime_service import RealTimeService
from ..core.logging import db_operation_logger
from ..core.error_handler import DatabaseTimeoutError, DatabaseConnectionError, DatabaseAuthenticationError

router = APIRouter(prefix="/database/real-time", tags=["Real-time"])

def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    """
    Extract user ID from authorization header.
    
    Args:
        authorization: Authorization header value
        
    Returns:
        User ID
        
    Raises:
        HTTPException: If authorization is invalid
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": "Unauthorized", "message": "Missing or invalid authorization header"}
        )
    
    # Validate the JWT token with Supabase
    token = authorization.replace("Bearer ", "")
    try:
        from ..core.supabase_auth import supabase_auth_service
        user = supabase_auth_service.get_user_by_token(token)
        if user and user.get("id"):
            return user["id"]
        else:
            raise HTTPException(
                status_code=401,
                detail={"error": "Unauthorized", "message": "Invalid token"}
            )
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail={"error": "Unauthorized", "message": "Token validation failed"}
        )

@router.post("/subscribe")
async def subscribe_to_realtime(
    table_name: str,
    event_type: str,
    filters: Optional[Dict[str, Any]] = None,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Subscribe to real-time updates for a table.
    
    Args:
        table_name: Table to subscribe to
        event_type: Type of events to listen for (INSERT, UPDATE, DELETE, ALL)
        filters: Optional filters to apply
        user_id: User creating the subscription
        
    Returns:
        Dict containing subscription information
    """
    try:
        # Initialize service
        realtime_service = RealTimeService()
        
        # Validate event type
        valid_event_types = ["INSERT", "UPDATE", "DELETE", "ALL"]
        if event_type not in valid_event_types:
            raise HTTPException(
                status_code=400,
                detail={"error": "Bad Request", "message": f"Invalid event type. Must be one of: {valid_event_types}"}
            )
        
        # Create subscription
        subscription_id = realtime_service.create_subscription(
            table_name=table_name,
            event_type=event_type,
            user_id=user_id,
            filters=filters
        )
        
        return {
            "subscription_id": subscription_id,
            "status": "active"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except DatabaseTimeoutError as e:
        raise HTTPException(
            status_code=408,
            detail={"error": "Request Timeout", "message": str(e)}
        )
    except DatabaseConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail={"error": "Service Unavailable", "message": str(e)}
        )
    except DatabaseAuthenticationError as e:
        raise HTTPException(
            status_code=401,
            detail={"error": "Unauthorized", "message": str(e)}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal Server Error", "message": str(e)}
        )

@router.post("/unsubscribe")
async def unsubscribe_from_realtime(
    subscription_id: str,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Unsubscribe from real-time updates.
    
    Args:
        subscription_id: Subscription identifier
        user_id: User cancelling the subscription
        
    Returns:
        Dict containing cancellation result
    """
    try:
        # Initialize service
        realtime_service = RealTimeService()
        
        # Get subscription to verify ownership
        subscription = realtime_service.get_subscription(subscription_id)
        
        if not subscription:
            raise HTTPException(
                status_code=404,
                detail={"error": "Not Found", "message": "Subscription not found"}
            )
        
        # Check if user owns this subscription
        if subscription.get("user_id") != user_id:
            raise HTTPException(
                status_code=403,
                detail={"error": "Forbidden", "message": "Access denied to this subscription"}
            )
        
        # Cancel subscription
        success = realtime_service.cancel_subscription(subscription_id)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail={"error": "Internal Server Error", "message": "Failed to cancel subscription"}
            )
        
        return {
            "success": True
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except DatabaseTimeoutError as e:
        raise HTTPException(
            status_code=408,
            detail={"error": "Request Timeout", "message": str(e)}
        )
    except DatabaseConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail={"error": "Service Unavailable", "message": str(e)}
        )
    except DatabaseAuthenticationError as e:
        raise HTTPException(
            status_code=401,
            detail={"error": "Unauthorized", "message": str(e)}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal Server Error", "message": str(e)}
        )

@router.get("/subscriptions")
async def get_user_subscriptions(
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get all subscriptions for the current user.
    
    Args:
        user_id: User requesting subscriptions
        
    Returns:
        Dict containing user subscriptions
    """
    try:
        # Initialize service
        realtime_service = RealTimeService()
        
        # Get user subscriptions
        subscriptions = realtime_service.get_user_subscriptions(user_id, active_only=True)
        
        return {
            "subscriptions": subscriptions,
            "count": len(subscriptions)
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal Server Error", "message": str(e)}
        )

@router.post("/subscriptions/{subscription_id}/pause")
async def pause_subscription(
    subscription_id: str,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Pause a subscription.
    
    Args:
        subscription_id: Subscription identifier
        user_id: User pausing the subscription
        
    Returns:
        Dict containing pause result
    """
    try:
        # Initialize service
        realtime_service = RealTimeService()
        
        # Get subscription to verify ownership
        subscription = realtime_service.get_subscription(subscription_id)
        
        if not subscription:
            raise HTTPException(
                status_code=404,
                detail={"error": "Not Found", "message": "Subscription not found"}
            )
        
        # Check if user owns this subscription
        if subscription.get("user_id") != user_id:
            raise HTTPException(
                status_code=403,
                detail={"error": "Forbidden", "message": "Access denied to this subscription"}
            )
        
        # Pause subscription
        success = realtime_service.pause_subscription(subscription_id)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail={"error": "Internal Server Error", "message": "Failed to pause subscription"}
            )
        
        return {
            "success": True,
            "status": "paused"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal Server Error", "message": str(e)}
        )

@router.post("/subscriptions/{subscription_id}/resume")
async def resume_subscription(
    subscription_id: str,
    user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Resume a paused subscription.
    
    Args:
        subscription_id: Subscription identifier
        user_id: User resuming the subscription
        
    Returns:
        Dict containing resume result
    """
    try:
        # Initialize service
        realtime_service = RealTimeService()
        
        # Get subscription to verify ownership
        subscription = realtime_service.get_subscription(subscription_id)
        
        if not subscription:
            raise HTTPException(
                status_code=404,
                detail={"error": "Not Found", "message": "Subscription not found"}
            )
        
        # Check if user owns this subscription
        if subscription.get("user_id") != user_id:
            raise HTTPException(
                status_code=403,
                detail={"error": "Forbidden", "message": "Access denied to this subscription"}
            )
        
        # Resume subscription
        success = realtime_service.resume_subscription(subscription_id)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail={"error": "Internal Server Error", "message": "Failed to resume subscription"}
            )
        
        return {
            "success": True,
            "status": "active"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal Server Error", "message": str(e)}
        )
