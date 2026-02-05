"""
Real-time notification service for data change notifications
"""
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import structlog
import asyncio
from enum import Enum

from ..services.realtime_service import RealTimeService
from ..services.websocket_handler import websocket_manager

logger = structlog.get_logger()

class NotificationType(Enum):
    """Notification type enumeration"""
    DATA_CHANGE = "data_change"
    SYSTEM_ALERT = "system_alert"
    USER_ACTION = "user_action"
    MIGRATION_STATUS = "migration_status"
    ERROR = "error"

class NotificationPriority(Enum):
    """Notification priority enumeration"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class RealTimeNotificationService:
    """Service for managing real-time notifications"""
    
    def __init__(self):
        self.realtime_service = RealTimeService()
        self.notification_handlers: Dict[str, List[Callable]] = {}
        self.notification_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default notification handlers"""
        self.register_handler(NotificationType.DATA_CHANGE, self._handle_data_change_notification)
        self.register_handler(NotificationType.SYSTEM_ALERT, self._handle_system_alert_notification)
        self.register_handler(NotificationType.USER_ACTION, self._handle_user_action_notification)
        self.register_handler(NotificationType.MIGRATION_STATUS, self._handle_migration_status_notification)
        self.register_handler(NotificationType.ERROR, self._handle_error_notification)
    
    def register_handler(self, notification_type: NotificationType, handler: Callable):
        """Register a notification handler"""
        if notification_type.value not in self.notification_handlers:
            self.notification_handlers[notification_type.value] = []
        self.notification_handlers[notification_type.value].append(handler)
        logger.info("Notification handler registered", type=notification_type.value)
    
    async def send_notification(self, notification_type: NotificationType, 
                              message: str, data: Optional[Dict[str, Any]] = None,
                              priority: NotificationPriority = NotificationPriority.NORMAL,
                              target_connections: Optional[List[str]] = None) -> str:
        """
        Send a real-time notification
        
        Args:
            notification_type: Type of notification
            message: Notification message
            data: Optional additional data
            priority: Notification priority
            target_connections: Specific connections to target (None for all)
            
        Returns:
            Notification ID
        """
        try:
            notification_id = f"notif_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
            
            notification = {
                "id": notification_id,
                "type": notification_type.value,
                "message": message,
                "data": data or {},
                "priority": priority.value,
                "timestamp": datetime.utcnow().isoformat(),
                "target_connections": target_connections
            }
            
            # Add to history
            self._add_to_history(notification)
            
            # Send via WebSocket
            await self._send_websocket_notification(notification)
            
            # Call registered handlers
            await self._call_handlers(notification_type, notification)
            
            logger.info("Notification sent", 
                       id=notification_id,
                       type=notification_type.value,
                       priority=priority.value)
            
            return notification_id
            
        except Exception as e:
            logger.error("Failed to send notification", 
                        type=notification_type.value,
                        error=str(e))
            raise
    
    async def send_data_change_notification(self, table_name: str, event_type: str,
                                          record_id: str, changes: Optional[Dict[str, Any]] = None,
                                          user_id: Optional[str] = None) -> str:
        """Send a data change notification"""
        try:
            message = f"Data changed in {table_name}: {event_type}"
            data = {
                "table_name": table_name,
                "event_type": event_type,
                "record_id": record_id,
                "changes": changes,
                "user_id": user_id
            }
            
            return await self.send_notification(
                notification_type=NotificationType.DATA_CHANGE,
                message=message,
                data=data,
                priority=NotificationPriority.NORMAL
            )
            
        except Exception as e:
            logger.error("Failed to send data change notification", 
                        table_name=table_name,
                        event_type=event_type,
                        error=str(e))
            raise
    
    async def send_system_alert(self, alert_type: str, message: str, 
                               severity: str = "warning",
                               data: Optional[Dict[str, Any]] = None) -> str:
        """Send a system alert notification"""
        try:
            priority = NotificationPriority.HIGH if severity == "critical" else NotificationPriority.NORMAL
            
            notification_data = {
                "alert_type": alert_type,
                "severity": severity,
                **(data or {})
            }
            
            return await self.send_notification(
                notification_type=NotificationType.SYSTEM_ALERT,
                message=message,
                data=notification_data,
                priority=priority
            )
            
        except Exception as e:
            logger.error("Failed to send system alert", 
                        alert_type=alert_type,
                        error=str(e))
            raise
    
    async def send_migration_status(self, migration_id: str, status: str, 
                                  progress: Optional[int] = None,
                                  message: Optional[str] = None) -> str:
        """Send a migration status notification"""
        try:
            status_message = message or f"Migration {migration_id}: {status}"
            if progress is not None:
                status_message += f" ({progress}%)"
            
            data = {
                "migration_id": migration_id,
                "status": status,
                "progress": progress
            }
            
            priority = NotificationPriority.HIGH if status in ["failed", "error"] else NotificationPriority.NORMAL
            
            return await self.send_notification(
                notification_type=NotificationType.MIGRATION_STATUS,
                message=status_message,
                data=data,
                priority=priority
            )
            
        except Exception as e:
            logger.error("Failed to send migration status", 
                        migration_id=migration_id,
                        error=str(e))
            raise
    
    async def send_error_notification(self, error_type: str, error_message: str,
                                    context: Optional[Dict[str, Any]] = None) -> str:
        """Send an error notification"""
        try:
            data = {
                "error_type": error_type,
                "context": context or {}
            }
            
            return await self.send_notification(
                notification_type=NotificationType.ERROR,
                message=error_message,
                data=data,
                priority=NotificationPriority.HIGH
            )
            
        except Exception as e:
            logger.error("Failed to send error notification", 
                        error_type=error_type,
                        error=str(e))
            raise
    
    async def _send_websocket_notification(self, notification: Dict[str, Any]):
        """Send notification via WebSocket"""
        try:
            websocket_message = {
                "type": "notification",
                "notification": notification
            }
            
            if notification.get("target_connections"):
                # Send to specific connections
                for connection_id in notification["target_connections"]:
                    await websocket_manager.send_message(connection_id, websocket_message)
            else:
                # Broadcast to all connections
                await websocket_manager.broadcast_message(websocket_message)
            
        except Exception as e:
            logger.error("Failed to send WebSocket notification", 
                        notification_id=notification["id"],
                        error=str(e))
    
    async def _call_handlers(self, notification_type: NotificationType, notification: Dict[str, Any]):
        """Call registered handlers for notification type"""
        try:
            handlers = self.notification_handlers.get(notification_type.value, [])
            
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(notification)
                    else:
                        handler(notification)
                except Exception as e:
                    logger.error("Notification handler failed", 
                                handler=handler.__name__,
                                notification_id=notification["id"],
                                error=str(e))
                    
        except Exception as e:
            logger.error("Failed to call notification handlers", 
                        type=notification_type.value,
                        error=str(e))
    
    def _add_to_history(self, notification: Dict[str, Any]):
        """Add notification to history"""
        try:
            self.notification_history.append(notification)
            
            # Trim history if too large
            if len(self.notification_history) > self.max_history_size:
                self.notification_history = self.notification_history[-self.max_history_size:]
                
        except Exception as e:
            logger.error("Failed to add notification to history", error=str(e))
    
    async def _handle_data_change_notification(self, notification: Dict[str, Any]):
        """Handle data change notification"""
        try:
            logger.info("Data change notification processed", 
                       table=notification["data"].get("table_name"),
                       event=notification["data"].get("event_type"))
        except Exception as e:
            logger.error("Failed to handle data change notification", error=str(e))
    
    async def _handle_system_alert_notification(self, notification: Dict[str, Any]):
        """Handle system alert notification"""
        try:
            logger.warning("System alert notification processed", 
                          alert_type=notification["data"].get("alert_type"),
                          severity=notification["data"].get("severity"))
        except Exception as e:
            logger.error("Failed to handle system alert notification", error=str(e))
    
    async def _handle_user_action_notification(self, notification: Dict[str, Any]):
        """Handle user action notification"""
        try:
            logger.info("User action notification processed", 
                       user_id=notification["data"].get("user_id"),
                       action=notification["data"].get("action"))
        except Exception as e:
            logger.error("Failed to handle user action notification", error=str(e))
    
    async def _handle_migration_status_notification(self, notification: Dict[str, Any]):
        """Handle migration status notification"""
        try:
            logger.info("Migration status notification processed", 
                       migration_id=notification["data"].get("migration_id"),
                       status=notification["data"].get("status"))
        except Exception as e:
            logger.error("Failed to handle migration status notification", error=str(e))
    
    async def _handle_error_notification(self, notification: Dict[str, Any]):
        """Handle error notification"""
        try:
            logger.error("Error notification processed", 
                        error_type=notification["data"].get("error_type"),
                        message=notification["message"])
        except Exception as e:
            logger.error("Failed to handle error notification", error=str(e))
    
    def get_notification_history(self, limit: int = 100, 
                               notification_type: Optional[NotificationType] = None) -> List[Dict[str, Any]]:
        """Get notification history"""
        try:
            history = self.notification_history.copy()
            
            # Filter by type if specified
            if notification_type:
                history = [n for n in history if n["type"] == notification_type.value]
            
            # Limit results
            return history[-limit:] if limit > 0 else history
            
        except Exception as e:
            logger.error("Failed to get notification history", error=str(e))
            return []
    
    def get_notification_statistics(self) -> Dict[str, Any]:
        """Get notification statistics"""
        try:
            total_notifications = len(self.notification_history)
            
            # Count by type
            type_counts = {}
            for notification in self.notification_history:
                notification_type = notification["type"]
                type_counts[notification_type] = type_counts.get(notification_type, 0) + 1
            
            # Count by priority
            priority_counts = {}
            for notification in self.notification_history:
                priority = notification["priority"]
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            return {
                "total_notifications": total_notifications,
                "type_counts": type_counts,
                "priority_counts": priority_counts,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get notification statistics", error=str(e))
            return {
                "total_notifications": 0,
                "type_counts": {},
                "priority_counts": {},
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def clear_notification_history(self):
        """Clear notification history"""
        try:
            self.notification_history.clear()
            logger.info("Notification history cleared")
        except Exception as e:
            logger.error("Failed to clear notification history", error=str(e))

# Global notification service instance
notification_service = RealTimeNotificationService()

