"""
WebSocket handler for real-time features
"""
import asyncio
import json
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime
import structlog
from fastapi import WebSocket, WebSocketDisconnect
from supabase import Client

from ..database.supabase_client import get_supabase_client
from ..services.realtime_service import RealTimeService

logger = structlog.get_logger()

class WebSocketManager:
    """Manager for WebSocket connections and real-time subscriptions"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_subscriptions: Dict[str, Set[str]] = {}  # connection_id -> set of subscription_ids
        self.subscription_connections: Dict[str, Set[str]] = {}  # subscription_id -> set of connection_ids
        self.realtime_service = RealTimeService()
        self.message_handlers: Dict[str, Callable] = {}
        
        # Register default message handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.message_handlers["subscribe"] = self._handle_subscribe
        self.message_handlers["unsubscribe"] = self._handle_unsubscribe
        self.message_handlers["ping"] = self._handle_ping
        self.message_handlers["get_status"] = self._handle_get_status
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept a WebSocket connection"""
        try:
            await websocket.accept()
            self.active_connections[connection_id] = websocket
            self.connection_subscriptions[connection_id] = set()
            
            logger.info("WebSocket connection established", connection_id=connection_id)
            
            # Send welcome message
            await self.send_message(connection_id, {
                "type": "connection_established",
                "connection_id": connection_id,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error("Failed to establish WebSocket connection", 
                        connection_id=connection_id,
                        error=str(e))
            raise
    
    async def disconnect(self, connection_id: str):
        """Disconnect a WebSocket connection"""
        try:
            if connection_id in self.active_connections:
                # Unsubscribe from all subscriptions
                if connection_id in self.connection_subscriptions:
                    for subscription_id in self.connection_subscriptions[connection_id].copy():
                        await self._unsubscribe_connection(connection_id, subscription_id)
                
                # Remove connection
                del self.active_connections[connection_id]
                if connection_id in self.connection_subscriptions:
                    del self.connection_subscriptions[connection_id]
                
                logger.info("WebSocket connection closed", connection_id=connection_id)
                
        except Exception as e:
            logger.error("Failed to close WebSocket connection", 
                        connection_id=connection_id,
                        error=str(e))
    
    async def send_message(self, connection_id: str, message: Dict[str, Any]):
        """Send a message to a specific connection"""
        try:
            if connection_id in self.active_connections:
                websocket = self.active_connections[connection_id]
                await websocket.send_text(json.dumps(message))
                logger.debug("Message sent", connection_id=connection_id, message_type=message.get("type"))
            else:
                logger.warning("Connection not found", connection_id=connection_id)
                
        except Exception as e:
            logger.error("Failed to send message", 
                        connection_id=connection_id,
                        error=str(e))
    
    async def broadcast_message(self, message: Dict[str, Any], exclude_connections: Optional[Set[str]] = None):
        """Broadcast a message to all active connections"""
        try:
            exclude_connections = exclude_connections or set()
            
            for connection_id, websocket in self.active_connections.items():
                if connection_id not in exclude_connections:
                    try:
                        await websocket.send_text(json.dumps(message))
                    except Exception as e:
                        logger.error("Failed to broadcast to connection", 
                                    connection_id=connection_id,
                                    error=str(e))
            
            logger.debug("Message broadcasted", 
                        connections=len(self.active_connections),
                        excluded=len(exclude_connections))
            
        except Exception as e:
            logger.error("Failed to broadcast message", error=str(e))
    
    async def handle_message(self, connection_id: str, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type in self.message_handlers:
                await self.message_handlers[message_type](connection_id, data)
            else:
                logger.warning("Unknown message type", 
                              connection_id=connection_id,
                              message_type=message_type)
                await self.send_message(connection_id, {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON message", 
                        connection_id=connection_id,
                        error=str(e))
            await self.send_message(connection_id, {
                "type": "error",
                "message": "Invalid JSON format",
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error("Failed to handle message", 
                        connection_id=connection_id,
                        error=str(e))
            await self.send_message(connection_id, {
                "type": "error",
                "message": "Failed to process message",
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def _handle_subscribe(self, connection_id: str, data: Dict[str, Any]):
        """Handle subscription request"""
        try:
            table_name = data.get("table_name")
            event_types = data.get("event_types", ["INSERT", "UPDATE", "DELETE"])
            filter_conditions = data.get("filter")
            
            if not table_name:
                await self.send_message(connection_id, {
                    "type": "error",
                    "message": "table_name is required for subscription",
                    "timestamp": datetime.utcnow().isoformat()
                })
                return
            
            # Create subscription
            subscription_info = self.realtime_service.create_subscription(
                table_name=table_name,
                event_types=event_types,
                filter_conditions=filter_conditions
            )
            
            subscription_id = subscription_info["subscription_id"]
            
            # Track connection subscription
            self.connection_subscriptions[connection_id].add(subscription_id)
            if subscription_id not in self.subscription_connections:
                self.subscription_connections[subscription_id] = set()
            self.subscription_connections[subscription_id].add(connection_id)
            
            # Set up callback for this subscription
            self.realtime_service.set_subscription_callback(
                subscription_id, 
                lambda payload: asyncio.create_task(
                    self._handle_realtime_event(subscription_id, payload)
                )
            )
            
            await self.send_message(connection_id, {
                "type": "subscription_created",
                "subscription_id": subscription_id,
                "table_name": table_name,
                "event_types": event_types,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info("Subscription created", 
                       connection_id=connection_id,
                       subscription_id=subscription_id,
                       table_name=table_name)
            
        except Exception as e:
            logger.error("Failed to handle subscription", 
                        connection_id=connection_id,
                        error=str(e))
            await self.send_message(connection_id, {
                "type": "error",
                "message": f"Failed to create subscription: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def _handle_unsubscribe(self, connection_id: str, data: Dict[str, Any]):
        """Handle unsubscription request"""
        try:
            subscription_id = data.get("subscription_id")
            
            if not subscription_id:
                await self.send_message(connection_id, {
                    "type": "error",
                    "message": "subscription_id is required for unsubscription",
                    "timestamp": datetime.utcnow().isoformat()
                })
                return
            
            await self._unsubscribe_connection(connection_id, subscription_id)
            
            await self.send_message(connection_id, {
                "type": "unsubscribed",
                "subscription_id": subscription_id,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error("Failed to handle unsubscription", 
                        connection_id=connection_id,
                        error=str(e))
            await self.send_message(connection_id, {
                "type": "error",
                "message": f"Failed to unsubscribe: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def _unsubscribe_connection(self, connection_id: str, subscription_id: str):
        """Unsubscribe a connection from a subscription"""
        try:
            # Remove from tracking
            if connection_id in self.connection_subscriptions:
                self.connection_subscriptions[connection_id].discard(subscription_id)
            
            if subscription_id in self.subscription_connections:
                self.subscription_connections[subscription_id].discard(connection_id)
                
                # If no more connections for this subscription, unsubscribe
                if not self.subscription_connections[subscription_id]:
                    self.realtime_service.unsubscribe(subscription_id)
                    del self.subscription_connections[subscription_id]
            
            logger.info("Connection unsubscribed", 
                       connection_id=connection_id,
                       subscription_id=subscription_id)
            
        except Exception as e:
            logger.error("Failed to unsubscribe connection", 
                        connection_id=connection_id,
                        subscription_id=subscription_id,
                        error=str(e))
    
    async def _handle_ping(self, connection_id: str, data: Dict[str, Any]):
        """Handle ping message"""
        await self.send_message(connection_id, {
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def _handle_get_status(self, connection_id: str, data: Dict[str, Any]):
        """Handle status request"""
        try:
            # Get connection status
            subscriptions = list(self.connection_subscriptions.get(connection_id, []))
            
            # Get real-time service statistics
            stats = self.realtime_service.get_subscription_statistics()
            
            await self.send_message(connection_id, {
                "type": "status",
                "connection_id": connection_id,
                "subscriptions": subscriptions,
                "statistics": stats,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error("Failed to handle status request", 
                        connection_id=connection_id,
                        error=str(e))
            await self.send_message(connection_id, {
                "type": "error",
                "message": f"Failed to get status: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def _handle_realtime_event(self, subscription_id: str, payload: Dict[str, Any]):
        """Handle real-time event from Supabase"""
        try:
            # Get all connections subscribed to this subscription
            connections = self.subscription_connections.get(subscription_id, set())
            
            if not connections:
                logger.warning("No connections for subscription", subscription_id=subscription_id)
                return
            
            # Create event message
            event_message = {
                "type": "realtime_event",
                "subscription_id": subscription_id,
                "event": payload.get("eventType"),
                "table": payload.get("table"),
                "data": {
                    "new": payload.get("new"),
                    "old": payload.get("old")
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send to all subscribed connections
            for connection_id in connections:
                try:
                    await self.send_message(connection_id, event_message)
                except Exception as e:
                    logger.error("Failed to send event to connection", 
                                connection_id=connection_id,
                                subscription_id=subscription_id,
                                error=str(e))
            
            logger.debug("Real-time event processed", 
                        subscription_id=subscription_id,
                        connections=len(connections),
                        event=payload.get("eventType"))
            
        except Exception as e:
            logger.error("Failed to handle real-time event", 
                        subscription_id=subscription_id,
                        error=str(e))
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def get_subscription_count(self) -> int:
        """Get number of active subscriptions"""
        return len(self.subscription_connections)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics"""
        return {
            "active_connections": len(self.active_connections),
            "active_subscriptions": len(self.subscription_connections),
            "total_subscriptions": sum(len(conns) for conns in self.connection_subscriptions.values()),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a custom message handler"""
        self.message_handlers[message_type] = handler
        logger.info("Message handler registered", message_type=message_type)

# Global WebSocket manager instance
websocket_manager = WebSocketManager()

