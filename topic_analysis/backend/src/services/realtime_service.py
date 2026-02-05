"""
RealTimeService

This module provides service layer for real-time subscriptions and event handling
using Supabase real-time features.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4

from ..core.logging import db_operation_logger
from ..core.error_handler import safe_database_operation
from .supabase_service import SupabaseService

class RealTimeService:
    """
    Service for managing real-time subscriptions and events.
    
    This service handles real-time subscriptions, event processing,
    and subscription lifecycle management using Supabase real-time features.
    """
    
    def __init__(self):
        """Initialize the real-time service."""
        self.supabase_service = SupabaseService()
        self.logger = db_operation_logger
        self._subscriptions: Dict[str, Any] = {}
    
    def create_subscription(self, table_name: str, event_type: str,
                           user_id: str, filters: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new real-time subscription.
        
        Args:
            table_name: Table to subscribe to
            event_type: Type of events to listen for
            user_id: User creating the subscription
            filters: Optional filters to apply
            
        Returns:
            Subscription ID
        """
        subscription_id = str(uuid4())
        
        subscription_data = {
            "subscription_id": subscription_id,
            "table_name": table_name,
            "event_type": event_type,
            "user_id": user_id,
            "filters": filters,
            "status": "active",
            "created_at": datetime.utcnow().isoformat()
        }
        
        try:
            created_subscription = self.supabase_service.create(
                table_name="realtime_subscriptions",
                data=subscription_data,
                user_id=user_id
            )
            
            # Store subscription reference
            self._subscriptions[subscription_id] = {
                "table_name": table_name,
                "event_type": event_type,
                "user_id": user_id,
                "filters": filters,
                "created_at": datetime.utcnow()
            }
            
            self.logger.log_real_time_event(
                event_type="SUBSCRIPTION_CREATED",
                table_name=table_name,
                subscription_id=subscription_id
            )
            
            return subscription_id
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id=subscription_id,
                error_message=str(e),
                error_type="create_subscription_error"
            )
            raise
    
    def get_subscription(self, subscription_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a subscription by ID.
        
        Args:
            subscription_id: Subscription identifier
            
        Returns:
            Subscription data or None
        """
        try:
            subscriptions = self.supabase_service.read(
                table_name="realtime_subscriptions",
                filters={"subscription_id": subscription_id},
                limit=1
            )
            
            if subscriptions:
                return subscriptions[0]
            return None
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id=subscription_id,
                error_message=str(e),
                error_type="get_subscription_error"
            )
            return None
    
    def get_user_subscriptions(self, user_id: str, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all subscriptions for a user.
        
        Args:
            user_id: User identifier
            active_only: Whether to return only active subscriptions
            
        Returns:
            List of subscription data
        """
        try:
            filters = {"user_id": user_id}
            if active_only:
                filters["status"] = "active"
            
            subscriptions = self.supabase_service.read(
                table_name="realtime_subscriptions",
                filters=filters,
                order_by="created_at",
                order_desc=True
            )
            
            return subscriptions
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id="user_subscriptions",
                error_message=str(e),
                error_type="get_user_subscriptions_error"
            )
            return []
    
    def update_subscription_status(self, subscription_id: str, status: str) -> bool:
        """
        Update subscription status.
        
        Args:
            subscription_id: Subscription identifier
            status: New status
            
        Returns:
            True if update was successful
        """
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            result = self.supabase_service.update(
                table_name="realtime_subscriptions",
                record_id=subscription_id,
                data=update_data
            )
            
            # Update local reference
            if subscription_id in self._subscriptions:
                self._subscriptions[subscription_id]["status"] = status
                self._subscriptions[subscription_id]["updated_at"] = datetime.utcnow()
            
            return result is not None
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id=subscription_id,
                error_message=str(e),
                error_type="update_subscription_status_error"
            )
            return False
    
    def cancel_subscription(self, subscription_id: str) -> bool:
        """
        Cancel a subscription.
        
        Args:
            subscription_id: Subscription identifier
            
        Returns:
            True if cancellation was successful
        """
        try:
            # Update status to cancelled
            success = self.update_subscription_status(subscription_id, "cancelled")
            
            if success:
                # Remove from local references
                if subscription_id in self._subscriptions:
                    del self._subscriptions[subscription_id]
                
                self.logger.log_real_time_event(
                    event_type="SUBSCRIPTION_CANCELLED",
                    table_name="unknown",
                    subscription_id=subscription_id
                )
            
            return success
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id=subscription_id,
                error_message=str(e),
                error_type="cancel_subscription_error"
            )
            return False
    
    def pause_subscription(self, subscription_id: str) -> bool:
        """
        Pause a subscription.
        
        Args:
            subscription_id: Subscription identifier
            
        Returns:
            True if pause was successful
        """
        return self.update_subscription_status(subscription_id, "paused")
    
    def resume_subscription(self, subscription_id: str) -> bool:
        """
        Resume a paused subscription.
        
        Args:
            subscription_id: Subscription identifier
            
        Returns:
            True if resume was successful
        """
        return self.update_subscription_status(subscription_id, "active")
    
    def process_realtime_event(self, event_data: Dict[str, Any]) -> None:
        """
        Process a real-time event.
        
        Args:
            event_data: Event data from Supabase
        """
        try:
            event_type = event_data.get("eventType", "UNKNOWN")
            table_name = event_data.get("table", "unknown")
            record = event_data.get("new", event_data.get("old", {}))
            
            # Log the event
            self.logger.log_real_time_event(
                event_type=event_type,
                table_name=table_name,
                subscription_id="system"
            )
            
            # Process based on event type
            if event_type == "INSERT":
                self._handle_insert_event(table_name, record)
            elif event_type == "UPDATE":
                self._handle_update_event(table_name, record)
            elif event_type == "DELETE":
                self._handle_delete_event(table_name, record)
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id="realtime_event",
                error_message=str(e),
                error_type="process_realtime_event_error"
            )
    
    def _handle_insert_event(self, table_name: str, record: Dict[str, Any]) -> None:
        """Handle INSERT events."""
        self.logger.log_real_time_event(
            event_type="INSERT",
            table_name=table_name,
            subscription_id="system"
        )
    
    def _handle_update_event(self, table_name: str, record: Dict[str, Any]) -> None:
        """Handle UPDATE events."""
        self.logger.log_real_time_event(
            event_type="UPDATE",
            table_name=table_name,
            subscription_id="system"
        )
    
    def _handle_delete_event(self, table_name: str, record: Dict[str, Any]) -> None:
        """Handle DELETE events."""
        self.logger.log_real_time_event(
            event_type="DELETE",
            table_name=table_name,
            subscription_id="system"
        )
    
    def get_subscription_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get subscription metrics for the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dict containing subscription metrics
        """
        try:
            # Get subscriptions from the last N hours
            cutoff_time = datetime.utcnow().replace(hour=datetime.utcnow().hour - hours)
            
            subscriptions = self.supabase_service.read(
                table_name="realtime_subscriptions",
                filters={"created_at": cutoff_time.isoformat()},
                limit=1000
            )
            
            if not subscriptions:
                return {
                    "total_subscriptions": 0,
                    "active_subscriptions": 0,
                    "paused_subscriptions": 0,
                    "cancelled_subscriptions": 0,
                    "subscriptions_by_table": {},
                    "subscriptions_by_user": {}
                }
            
            # Calculate metrics
            total_subscriptions = len(subscriptions)
            active_subscriptions = len([s for s in subscriptions if s.get("status") == "active"])
            paused_subscriptions = len([s for s in subscriptions if s.get("status") == "paused"])
            cancelled_subscriptions = len([s for s in subscriptions if s.get("status") == "cancelled"])
            
            # Group by table
            subscriptions_by_table = {}
            for subscription in subscriptions:
                table_name = subscription.get("table_name", "unknown")
                subscriptions_by_table[table_name] = subscriptions_by_table.get(table_name, 0) + 1
            
            # Group by user
            subscriptions_by_user = {}
            for subscription in subscriptions:
                user_id = subscription.get("user_id", "unknown")
                subscriptions_by_user[user_id] = subscriptions_by_user.get(user_id, 0) + 1
            
            return {
                "total_subscriptions": total_subscriptions,
                "active_subscriptions": active_subscriptions,
                "paused_subscriptions": paused_subscriptions,
                "cancelled_subscriptions": cancelled_subscriptions,
                "subscriptions_by_table": subscriptions_by_table,
                "subscriptions_by_user": subscriptions_by_user
            }
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id="subscription_metrics",
                error_message=str(e),
                error_type="get_subscription_metrics_error"
            )
            return {}
    
    def cleanup_old_subscriptions(self, days: int = 30) -> int:
        """
        Clean up old cancelled subscriptions.
        
        Args:
            days: Number of days to keep cancelled subscriptions
            
        Returns:
            Number of subscriptions cleaned up
        """
        try:
            cutoff_time = datetime.utcnow().replace(day=datetime.utcnow().day - days)
            
            # Get old cancelled subscriptions
            old_subscriptions = self.supabase_service.read(
                table_name="realtime_subscriptions",
                filters={
                    "status": "cancelled",
                    "created_at": cutoff_time.isoformat()
                },
                limit=1000
            )
            
            # Delete old subscriptions
            cleaned_count = 0
            for subscription in old_subscriptions:
                try:
                    self.supabase_service.delete(
                        table_name="realtime_subscriptions",
                        record_id=subscription["id"]
                    )
                    cleaned_count += 1
                except Exception:
                    continue
            
            return cleaned_count
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id="cleanup_subscriptions",
                error_message=str(e),
                error_type="cleanup_subscriptions_error"
            )
            return 0