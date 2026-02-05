"""
Real-time Subscription Management Middleware

This module provides middleware for managing real-time subscriptions
and event processing with Supabase real-time features.
"""

from fastapi import Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List, Set
import asyncio
import json
from datetime import datetime
from uuid import uuid4

from ..services.realtime_service import RealTimeService
from ..core.logging import db_operation_logger, get_logger

logger = get_logger(__name__)

class RealTimeMiddleware:
    """
    Middleware for managing real-time subscriptions and WebSocket connections.
    """
    
    def __init__(self):
        """Initialize the real-time middleware."""
        self.realtime_service = RealTimeService()
        self.logger = db_operation_logger
        self._active_connections: Dict[str, WebSocket] = {}
        self._subscription_connections: Dict[str, Set[str]] = {}
        self._connection_subscriptions: Dict[str, Set[str]] = {}
    
    async def handle_websocket_connection(self, websocket: WebSocket, user_id: str) -> str:
        """
        Handle new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            user_id: User ID
            
        Returns:
            Connection ID
        """
        connection_id = str(uuid4())
        
        try:
            await websocket.accept()
            self._active_connections[connection_id] = websocket
            self._connection_subscriptions[connection_id] = set()
            
            # Send connection confirmation
            await websocket.send_text(json.dumps({
                "type": "connection_established",
                "connection_id": connection_id,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }))
            
            logger.info(
                "WebSocket connection established",
                connection_id=connection_id,
                user_id=user_id
            )
            
            return connection_id
            
        except Exception as e:
            logger.error(
                "WebSocket connection failed",
                error=str(e),
                user_id=user_id
            )
            raise
    
    async def handle_subscription_request(self, websocket: WebSocket, connection_id: str,
                                        user_id: str, table_name: str, event_type: str,
                                        filters: Optional[Dict[str, Any]] = None) -> str:
        """
        Handle subscription request.
        
        Args:
            websocket: WebSocket connection
            connection_id: Connection ID
            user_id: User ID
            table_name: Table to subscribe to
            event_type: Event type
            filters: Optional filters
            
        Returns:
            Subscription ID
        """
        try:
            # Create subscription
            subscription_id = self.realtime_service.create_subscription(
                table_name=table_name,
                event_type=event_type,
                user_id=user_id,
                filters=filters
            )
            
            # Track subscription
            self._connection_subscriptions[connection_id].add(subscription_id)
            if subscription_id not in self._subscription_connections:
                self._subscription_connections[subscription_id] = set()
            self._subscription_connections[subscription_id].add(connection_id)
            
            # Send subscription confirmation
            await websocket.send_text(json.dumps({
                "type": "subscription_created",
                "subscription_id": subscription_id,
                "table_name": table_name,
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat()
            }))
            
            logger.info(
                "Real-time subscription created",
                subscription_id=subscription_id,
                connection_id=connection_id,
                user_id=user_id,
                table_name=table_name,
                event_type=event_type
            )
            
            return subscription_id
            
        except Exception as e:
            logger.error(
                "Subscription creation failed",
                error=str(e),
                connection_id=connection_id,
                user_id=user_id,
                table_name=table_name
            )
            raise
    
    async def handle_unsubscription_request(self, websocket: WebSocket, connection_id: str,
                                          subscription_id: str) -> bool:
        """
        Handle unsubscription request.
        
        Args:
            websocket: WebSocket connection
            connection_id: Connection ID
            subscription_id: Subscription ID
            
        Returns:
            True if successful
        """
        try:
            # Cancel subscription
            success = self.realtime_service.cancel_subscription(subscription_id)
            
            if success:
                # Remove from tracking
                self._connection_subscriptions[connection_id].discard(subscription_id)
                self._subscription_connections[subscription_id].discard(connection_id)
                
                # Clean up empty subscription
                if not self._subscription_connections[subscription_id]:
                    del self._subscription_connections[subscription_id]
                
                # Send unsubscription confirmation
                await websocket.send_text(json.dumps({
                    "type": "subscription_cancelled",
                    "subscription_id": subscription_id,
                    "timestamp": datetime.utcnow().isoformat()
                }))
                
                logger.info(
                    "Real-time subscription cancelled",
                    subscription_id=subscription_id,
                    connection_id=connection_id
                )
            
            return success
            
        except Exception as e:
            logger.error(
                "Unsubscription failed",
                error=str(e),
                connection_id=connection_id,
                subscription_id=subscription_id
            )
            return False
    
    async def broadcast_event(self, subscription_id: str, event_data: Dict[str, Any]) -> None:
        """
        Broadcast event to all connections subscribed to a subscription.
        
        Args:
            subscription_id: Subscription ID
            event_data: Event data to broadcast
        """
        try:
            if subscription_id not in self._subscription_connections:
                return
            
            connections = self._subscription_connections[subscription_id].copy()
            
            for connection_id in connections:
                if connection_id in self._active_connections:
                    websocket = self._active_connections[connection_id]
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "realtime_event",
                            "subscription_id": subscription_id,
                            "event_data": event_data,
                            "timestamp": datetime.utcnow().isoformat()
                        }))
                    except Exception as e:
                        logger.error(
                            "Failed to send event to connection",
                            error=str(e),
                            connection_id=connection_id,
                            subscription_id=subscription_id
                        )
                        # Remove failed connection
                        await self._remove_connection(connection_id)
            
            # Log event broadcast
            self.logger.log_real_time_event(
                event_type="BROADCAST",
                table_name=event_data.get("table", "unknown"),
                subscription_id=subscription_id
            )
            
        except Exception as e:
            logger.error(
                "Event broadcast failed",
                error=str(e),
                subscription_id=subscription_id
            )
    
    async def _remove_connection(self, connection_id: str) -> None:
        """
        Remove a connection and clean up its subscriptions.
        
        Args:
            connection_id: Connection ID to remove
        """
        try:
            # Get subscriptions for this connection
            subscriptions = self._connection_subscriptions.get(connection_id, set()).copy()
            
            # Cancel all subscriptions
            for subscription_id in subscriptions:
                self.realtime_service.cancel_subscription(subscription_id)
                self._subscription_connections[subscription_id].discard(connection_id)
                
                # Clean up empty subscription
                if not self._subscription_connections[subscription_id]:
                    del self._subscription_connections[subscription_id]
            
            # Remove connection
            if connection_id in self._active_connections:
                del self._active_connections[connection_id]
            if connection_id in self._connection_subscriptions:
                del self._connection_subscriptions[connection_id]
            
            logger.info(
                "Connection removed",
                connection_id=connection_id,
                cancelled_subscriptions=len(subscriptions)
            )
            
        except Exception as e:
            logger.error(
                "Connection removal failed",
                error=str(e),
                connection_id=connection_id
            )
    
    async def handle_connection_disconnect(self, connection_id: str) -> None:
        """
        Handle connection disconnect.
        
        Args:
            connection_id: Connection ID
        """
        await self._remove_connection(connection_id)
    
    async def process_realtime_event(self, event_data: Dict[str, Any]) -> None:
        """
        Process real-time event from Supabase.
        
        Args:
            event_data: Event data from Supabase
        """
        try:
            # Process event through service
            self.realtime_service.process_realtime_event(event_data)
            
            # Get subscription ID from event (this would be determined by the event)
            subscription_id = event_data.get("subscription_id")
            if subscription_id:
                await self.broadcast_event(subscription_id, event_data)
            
        except Exception as e:
            logger.error(
                "Real-time event processing failed",
                error=str(e),
                event_data=event_data
            )
    
    def get_connection_metrics(self) -> Dict[str, Any]:
        """
        Get connection metrics for monitoring.
        
        Returns:
            Dict containing connection metrics
        """
        return {
            "active_connections": len(self._active_connections),
            "total_subscriptions": len(self._subscription_connections),
            "connections_by_subscription": {
                sub_id: len(conns) for sub_id, conns in self._subscription_connections.items()
            },
            "subscriptions_by_connection": {
                conn_id: len(subs) for conn_id, subs in self._connection_subscriptions.items()
            }
        }
    
    def get_subscription_metrics(self) -> Dict[str, Any]:
        """
        Get subscription metrics for monitoring.
        
        Returns:
            Dict containing subscription metrics
        """
        return self.realtime_service.get_subscription_metrics()

# Global middleware instance
realtime_middleware = RealTimeMiddleware()

async def realtime_websocket_handler(websocket: WebSocket, user_id: str):
    """
    Handle WebSocket connections for real-time features.
    
    Args:
        websocket: WebSocket connection
        user_id: User ID
    """
    connection_id = None
    
    try:
        # Establish connection
        connection_id = await realtime_middleware.handle_websocket_connection(websocket, user_id)
        
        # Handle messages
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                message_type = data.get("type")
                
                if message_type == "subscribe":
                    await realtime_middleware.handle_subscription_request(
                        websocket, connection_id, user_id,
                        data.get("table_name"),
                        data.get("event_type"),
                        data.get("filters")
                    )
                
                elif message_type == "unsubscribe":
                    await realtime_middleware.handle_unsubscription_request(
                        websocket, connection_id,
                        data.get("subscription_id")
                    )
                
                elif message_type == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(
                    "WebSocket message handling error",
                    error=str(e),
                    connection_id=connection_id
                )
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }))
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(
            "WebSocket connection error",
            error=str(e),
            connection_id=connection_id
        )
    finally:
        if connection_id:
            await realtime_middleware.handle_connection_disconnect(connection_id)
