"""
Real-time API endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import structlog

from ..core.database import get_db
from ..services.realtime_service import RealTimeService

logger = structlog.get_logger()
router = APIRouter()

# Global real-time service instance
realtime_service = RealTimeService()

@router.post("/database/realtime/subscribe")
async def create_realtime_subscription(
    table_name: str,
    event_types: List[str],
    filter: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Create a real-time subscription
    
    Args:
        table_name: Name of the table to subscribe to
        event_types: List of event types (INSERT, UPDATE, DELETE)
        filter: Optional filter conditions
        db: Database session
        
    Returns:
        Dict containing subscription details
    """
    try:
        logger.info("Creating real-time subscription", 
                   table=table_name, 
                   events=event_types,
                   filter=filter)
        
        # Validate event types
        valid_events = ["INSERT", "UPDATE", "DELETE"]
        for event_type in event_types:
            if event_type not in valid_events:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid subscription request",
                        "message": f"Event type '{event_type}' is not valid. Must be one of: {valid_events}",
                        "details": {"invalid_event_type": event_type},
                        "timestamp": "2024-12-19T10:30:00Z"
                    }
                )
        
        # Create subscription
        subscription_info = realtime_service.create_subscription(
            table_name=table_name,
            event_types=event_types,
            filter_conditions=filter
        )
        
        logger.info("Real-time subscription created successfully", 
                   subscription_id=subscription_info["subscription_id"],
                   table=table_name)
        
        return subscription_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create real-time subscription", 
                    table=table_name,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to create real-time subscription",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        )

@router.delete("/database/realtime/subscribe/{subscription_id}")
async def unsubscribe_realtime(
    subscription_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Unsubscribe from real-time updates
    
    Args:
        subscription_id: Subscription ID to unsubscribe
        db: Database session
        
    Returns:
        Dict containing unsubscription result
    """
    try:
        logger.info("Unsubscribing from real-time updates", subscription_id=subscription_id)
        
        # Unsubscribe
        success = realtime_service.unsubscribe(subscription_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Subscription not found",
                    "message": f"Subscription with ID '{subscription_id}' does not exist",
                    "details": {"subscription_id": subscription_id},
                    "timestamp": "2024-12-19T10:30:00Z"
                }
            )
        
        logger.info("Real-time subscription unsubscribed successfully", 
                   subscription_id=subscription_id)
        
        return {
            "message": "Successfully unsubscribed from real-time updates",
            "subscription_id": subscription_id,
            "timestamp": "2024-12-19T10:30:00Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to unsubscribe from real-time updates", 
                    subscription_id=subscription_id,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to unsubscribe from real-time updates",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        )

@router.get("/database/realtime/subscribe/{subscription_id}")
async def get_realtime_subscription(
    subscription_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get real-time subscription details
    
    Args:
        subscription_id: Subscription ID
        db: Database session
        
    Returns:
        Dict containing subscription details
    """
    try:
        logger.info("Getting real-time subscription", subscription_id=subscription_id)
        
        # Get subscription
        subscription = realtime_service.get_subscription(subscription_id)
        
        if not subscription:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Subscription not found",
                    "message": f"Subscription with ID '{subscription_id}' does not exist",
                    "details": {"subscription_id": subscription_id},
                    "timestamp": "2024-12-19T10:30:00Z"
                }
            )
        
        # Add subscription ID to response
        subscription["subscription_id"] = subscription_id
        
        logger.info("Real-time subscription retrieved successfully", 
                   subscription_id=subscription_id,
                   status=subscription.get("status"))
        
        return subscription
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get real-time subscription", 
                    subscription_id=subscription_id,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to retrieve real-time subscription",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        )

@router.get("/database/realtime/subscribe")
async def get_all_realtime_subscriptions(
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get all active real-time subscriptions
    
    Args:
        db: Database session
        
    Returns:
        List of subscription details
    """
    try:
        logger.info("Getting all real-time subscriptions")
        
        # Get all subscriptions
        subscriptions = realtime_service.get_all_subscriptions()
        
        logger.info("All real-time subscriptions retrieved successfully", 
                   count=len(subscriptions))
        
        return subscriptions
        
    except Exception as e:
        logger.error("Failed to get all real-time subscriptions", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to retrieve real-time subscriptions",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        )

@router.get("/database/realtime/statistics")
async def get_realtime_statistics(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get real-time subscription statistics
    
    Args:
        db: Database session
        
    Returns:
        Dict containing statistics
    """
    try:
        logger.info("Getting real-time statistics")
        
        # Get statistics
        stats = realtime_service.get_subscription_statistics()
        
        logger.info("Real-time statistics retrieved successfully", 
                   total_subscriptions=stats["total_subscriptions"],
                   active_subscriptions=stats["active_subscriptions"])
        
        return stats
        
    except Exception as e:
        logger.error("Failed to get real-time statistics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to retrieve real-time statistics",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        )

@router.post("/database/realtime/test")
async def test_realtime_connection(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Test real-time connection
    
    Args:
        db: Database session
        
    Returns:
        Dict containing test results
    """
    try:
        logger.info("Testing real-time connection")
        
        # Test connection
        test_result = realtime_service.test_realtime_connection()
        
        logger.info("Real-time connection test completed", 
                   status=test_result["status"])
        
        return test_result
        
    except Exception as e:
        logger.error("Failed to test real-time connection", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to test real-time connection",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        )

@router.post("/database/realtime/cleanup")
async def cleanup_realtime_subscriptions(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Clean up inactive real-time subscriptions
    
    Args:
        db: Database session
        
    Returns:
        Dict containing cleanup results
    """
    try:
        logger.info("Cleaning up real-time subscriptions")
        
        # Cleanup inactive subscriptions
        cleaned_count = realtime_service.cleanup_inactive_subscriptions()
        
        logger.info("Real-time subscriptions cleanup completed", 
                   cleaned_count=cleaned_count)
        
        return {
            "message": "Real-time subscriptions cleanup completed",
            "cleaned_count": cleaned_count,
            "timestamp": "2024-12-19T10:30:00Z"
        }
        
    except Exception as e:
        logger.error("Failed to cleanup real-time subscriptions", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to cleanup real-time subscriptions",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        )

