"""
WebSocket API endpoints for real-time features
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, Any
import structlog
import uuid

from ..services.websocket_handler import websocket_manager
from ..services.realtime_notification_service import notification_service

logger = structlog.get_logger()
router = APIRouter()

@router.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time features
    
    Message format:
    {
        "type": "subscribe|unsubscribe|ping|get_status",
        "table_name": "table_name",  # for subscribe
        "event_types": ["INSERT", "UPDATE", "DELETE"],  # for subscribe
        "filter": {...},  # for subscribe
        "subscription_id": "sub_123"  # for unsubscribe
    }
    
    Response format:
    {
        "type": "connection_established|subscription_created|unsubscribed|realtime_event|notification|error|pong|status",
        "data": {...},
        "timestamp": "2024-12-19T10:30:00Z"
    }
    """
    connection_id = str(uuid.uuid4())
    
    try:
        # Accept connection
        await websocket_manager.connect(websocket, connection_id)
        
        logger.info("WebSocket connection established", connection_id=connection_id)
        
        # Handle messages
        while True:
            try:
                # Receive message
                message = await websocket.receive_text()
                
                # Handle message
                await websocket_manager.handle_message(connection_id, message)
                
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected", connection_id=connection_id)
                break
            except Exception as e:
                logger.error("WebSocket message handling error", 
                            connection_id=connection_id,
                            error=str(e))
                
                # Send error message
                await websocket_manager.send_message(connection_id, {
                    "type": "error",
                    "message": "Message processing failed",
                    "timestamp": "2024-12-19T10:30:00Z"
                })
    
    except Exception as e:
        logger.error("WebSocket connection error", 
                    connection_id=connection_id,
                    error=str(e))
    finally:
        # Clean up connection
        await websocket_manager.disconnect(connection_id)

@router.get("/ws/status")
async def websocket_status() -> Dict[str, Any]:
    """
    Get WebSocket connection status
    
    Returns:
        Dict containing WebSocket status information
    """
    try:
        stats = websocket_manager.get_statistics()
        
        return {
            "status": "active",
            "connections": stats["active_connections"],
            "subscriptions": stats["active_subscriptions"],
            "total_subscriptions": stats["total_subscriptions"],
            "timestamp": stats["timestamp"]
        }
        
    except Exception as e:
        logger.error("Failed to get WebSocket status", error=str(e))
        return {
            "status": "error",
            "error": str(e),
            "timestamp": "2024-12-19T10:30:00Z"
        }

@router.get("/ws/notifications/history")
async def get_notification_history(limit: int = 100) -> Dict[str, Any]:
    """
    Get notification history
    
    Args:
        limit: Maximum number of notifications to return
        
    Returns:
        Dict containing notification history
    """
    try:
        history = notification_service.get_notification_history(limit=limit)
        
        return {
            "notifications": history,
            "count": len(history),
            "timestamp": "2024-12-19T10:30:00Z"
        }
        
    except Exception as e:
        logger.error("Failed to get notification history", error=str(e))
        return {
            "notifications": [],
            "count": 0,
            "error": str(e),
            "timestamp": "2024-12-19T10:30:00Z"
        }

@router.get("/ws/notifications/statistics")
async def get_notification_statistics() -> Dict[str, Any]:
    """
    Get notification statistics
    
    Returns:
        Dict containing notification statistics
    """
    try:
        stats = notification_service.get_notification_statistics()
        
        return stats
        
    except Exception as e:
        logger.error("Failed to get notification statistics", error=str(e))
        return {
            "total_notifications": 0,
            "type_counts": {},
            "priority_counts": {},
            "error": str(e),
            "timestamp": "2024-12-19T10:30:00Z"
        }

@router.post("/ws/notifications/send")
async def send_test_notification(
    message: str,
    notification_type: str = "data_change",
    priority: str = "normal"
) -> Dict[str, Any]:
    """
    Send a test notification
    
    Args:
        message: Notification message
        notification_type: Type of notification
        priority: Notification priority
        
    Returns:
        Dict containing notification result
    """
    try:
        from ..services.realtime_notification_service import NotificationType, NotificationPriority
        
        # Parse notification type
        try:
            notif_type = NotificationType(notification_type)
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid notification type: {notification_type}",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        
        # Parse priority
        try:
            notif_priority = NotificationPriority(priority)
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid priority: {priority}",
                "timestamp": "2024-12-19T10:30:00Z"
            }
        
        # Send notification
        notification_id = await notification_service.send_notification(
            notification_type=notif_type,
            message=message,
            priority=notif_priority
        )
        
        return {
            "success": True,
            "notification_id": notification_id,
            "message": message,
            "type": notification_type,
            "priority": priority,
            "timestamp": "2024-12-19T10:30:00Z"
        }
        
    except Exception as e:
        logger.error("Failed to send test notification", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "timestamp": "2024-12-19T10:30:00Z"
        }

