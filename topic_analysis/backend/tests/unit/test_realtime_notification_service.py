"""
Unit tests for real-time notification service
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json

from src.services.realtime_notification_service import RealtimeNotificationService

class TestRealtimeNotificationService:
    """Test cases for RealtimeNotificationService"""
    
    @pytest.fixture
    def mock_websocket_handler(self):
        """Mock WebSocket handler"""
        handler = Mock()
        handler.broadcast = Mock()
        handler.send_personal_message = Mock()
        return handler
    
    @pytest.fixture
    def mock_realtime_service(self):
        """Mock real-time service"""
        service = Mock()
        service.get_subscription = Mock()
        return service
    
    @pytest.fixture
    def notification_service(self, mock_websocket_handler, mock_realtime_service):
        """RealtimeNotificationService instance with mocked dependencies"""
        return RealtimeNotificationService(mock_websocket_handler, mock_realtime_service)
    
    def test_initialization(self, notification_service, mock_websocket_handler, mock_realtime_service):
        """Test RealtimeNotificationService initialization"""
        assert notification_service.websocket_handler == mock_websocket_handler
        assert notification_service.realtime_service == mock_realtime_service
    
    def test_initialization_without_dependencies(self):
        """Test initialization without provided dependencies"""
        with patch('src.services.realtime_notification_service.WebSocketHandler') as mock_ws_class, \
             patch('src.services.realtime_notification_service.RealTimeService') as mock_rt_class:
            
            mock_ws_handler = Mock()
            mock_rt_service = Mock()
            mock_ws_class.return_value = mock_ws_handler
            mock_rt_class.return_value = mock_rt_service
            
            service = RealtimeNotificationService()
            
            assert service.websocket_handler == mock_ws_handler
            assert service.realtime_service == mock_rt_service
    
    @pytest.mark.asyncio
    async def test_send_notification_to_user(self, notification_service, mock_websocket_handler):
        """Test sending notification to specific user"""
        user_id = "user123"
        message = {
            "type": "trend_update",
            "data": {"trend_id": "trend_456", "status": "completed"}
        }
        
        await notification_service.send_notification_to_user(user_id, message)
        
        expected_channel = f"user_{user_id}"
        expected_message = json.dumps(message)
        mock_websocket_handler.broadcast.assert_called_once_with(
            message=expected_message,
            channel_id=expected_channel
        )
    
    @pytest.mark.asyncio
    async def test_send_global_notification(self, notification_service, mock_websocket_handler):
        """Test sending global notification"""
        message = {
            "type": "system_maintenance",
            "data": {"message": "System will be down for maintenance"}
        }
        
        await notification_service.send_global_notification(message)
        
        expected_message = json.dumps(message)
        mock_websocket_handler.broadcast.assert_called_once_with(
            message=expected_message,
            channel_id="global_notifications"
        )
    
    @pytest.mark.asyncio
    async def test_process_supabase_realtime_event_insert(self, notification_service, mock_websocket_handler):
        """Test processing Supabase real-time INSERT event"""
        payload = {
            "event_type": "INSERT",
            "table": "trend_analyses",
            "record": {
                "id": "trend_123",
                "user_id": "user_456",
                "title": "New Trend Analysis",
                "status": "completed"
            },
            "old_record": None
        }
        
        await notification_service.process_supabase_realtime_event(payload)
        
        # Should send notification to user
        mock_websocket_handler.broadcast.assert_called_once()
        call_args = mock_websocket_handler.broadcast.call_args
        assert call_args[1]["channel_id"] == "user_user_456"
        
        # Verify message content
        message = json.loads(call_args[1]["message"])
        assert message["type"] == "supabase_change_insert"
        assert message["table"] == "trend_analyses"
        assert message["data"]["id"] == "trend_123"
    
    @pytest.mark.asyncio
    async def test_process_supabase_realtime_event_update(self, notification_service, mock_websocket_handler):
        """Test processing Supabase real-time UPDATE event"""
        payload = {
            "event_type": "UPDATE",
            "table": "trend_analyses",
            "record": {
                "id": "trend_123",
                "user_id": "user_456",
                "title": "Updated Trend Analysis",
                "status": "in_progress"
            },
            "old_record": {
                "id": "trend_123",
                "user_id": "user_456",
                "title": "Original Trend Analysis",
                "status": "pending"
            }
        }
        
        await notification_service.process_supabase_realtime_event(payload)
        
        # Should send notification to user
        mock_websocket_handler.broadcast.assert_called_once()
        call_args = mock_websocket_handler.broadcast.call_args
        assert call_args[1]["channel_id"] == "user_user_456"
        
        # Verify message content
        message = json.loads(call_args[1]["message"])
        assert message["type"] == "supabase_change_update"
        assert message["table"] == "trend_analyses"
        assert message["data"]["id"] == "trend_123"
        assert message["old_data"]["status"] == "pending"
    
    @pytest.mark.asyncio
    async def test_process_supabase_realtime_event_delete(self, notification_service, mock_websocket_handler):
        """Test processing Supabase real-time DELETE event"""
        payload = {
            "event_type": "DELETE",
            "table": "trend_analyses",
            "record": None,
            "old_record": {
                "id": "trend_123",
                "user_id": "user_456",
                "title": "Deleted Trend Analysis"
            }
        }
        
        await notification_service.process_supabase_realtime_event(payload)
        
        # Should send notification to user
        mock_websocket_handler.broadcast.assert_called_once()
        call_args = mock_websocket_handler.broadcast.call_args
        assert call_args[1]["channel_id"] == "user_user_456"
        
        # Verify message content
        message = json.loads(call_args[1]["message"])
        assert message["type"] == "supabase_change_delete"
        assert message["table"] == "trend_analyses"
        assert message["data"] is None
        assert message["old_data"]["id"] == "trend_123"
    
    @pytest.mark.asyncio
    async def test_process_supabase_realtime_event_new_user(self, notification_service, mock_websocket_handler):
        """Test processing Supabase real-time event for new user"""
        payload = {
            "event_type": "INSERT",
            "table": "users",
            "record": {
                "id": "user_789",
                "email": "newuser@example.com",
                "name": "New User"
            },
            "old_record": None
        }
        
        await notification_service.process_supabase_realtime_event(payload)
        
        # Should send global notification
        mock_websocket_handler.broadcast.assert_called_once()
        call_args = mock_websocket_handler.broadcast.call_args
        assert call_args[1]["channel_id"] == "global_notifications"
        
        # Verify message content
        message = json.loads(call_args[1]["message"])
        assert message["type"] == "new_user_signup"
        assert message["user_email"] == "newuser@example.com"
    
    @pytest.mark.asyncio
    async def test_process_supabase_realtime_event_no_user_id(self, notification_service, mock_websocket_handler):
        """Test processing event without user_id"""
        payload = {
            "event_type": "UPDATE",
            "table": "trend_analyses",
            "record": {
                "id": "trend_123",
                "title": "Updated Trend Analysis"
                # No user_id field
            },
            "old_record": None
        }
        
        await notification_service.process_supabase_realtime_event(payload)
        
        # Should not send notification
        mock_websocket_handler.broadcast.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_process_supabase_realtime_event_unknown_table(self, notification_service, mock_websocket_handler):
        """Test processing event for unknown table"""
        payload = {
            "event_type": "INSERT",
            "table": "unknown_table",
            "record": {"id": "123"},
            "old_record": None
        }
        
        await notification_service.process_supabase_realtime_event(payload)
        
        # Should not send notification
        mock_websocket_handler.broadcast.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_process_supabase_realtime_event_exception(self, notification_service, mock_websocket_handler):
        """Test processing event with exception"""
        payload = {
            "event_type": "INSERT",
            "table": "trend_analyses",
            "record": {"id": "trend_123", "user_id": "user_456"}
        }
        
        mock_websocket_handler.broadcast.side_effect = Exception("Broadcast failed")
        
        # Should not raise exception
        await notification_service.process_supabase_realtime_event(payload)
    
    @pytest.mark.asyncio
    async def test_send_trend_analysis_notification(self, notification_service, mock_websocket_handler):
        """Test sending trend analysis specific notification"""
        user_id = "user_123"
        trend_data = {
            "id": "trend_456",
            "title": "Market Analysis",
            "status": "completed",
            "insights": ["Insight 1", "Insight 2"]
        }
        
        await notification_service.send_trend_analysis_notification(user_id, trend_data)
        
        mock_websocket_handler.broadcast.assert_called_once()
        call_args = mock_websocket_handler.broadcast.call_args
        assert call_args[1]["channel_id"] == f"user_{user_id}"
        
        message = json.loads(call_args[1]["message"])
        assert message["type"] == "trend_analysis_update"
        assert message["data"] == trend_data
    
    @pytest.mark.asyncio
    async def test_send_system_alert(self, notification_service, mock_websocket_handler):
        """Test sending system alert"""
        alert_data = {
            "level": "warning",
            "message": "High CPU usage detected",
            "timestamp": "2024-12-19T10:30:00Z"
        }
        
        await notification_service.send_system_alert(alert_data)
        
        mock_websocket_handler.broadcast.assert_called_once()
        call_args = mock_websocket_handler.broadcast.call_args
        assert call_args[1]["channel_id"] == "system_alerts"
        
        message = json.loads(call_args[1]["message"])
        assert message["type"] == "system_alert"
        assert message["data"] == alert_data
    
    @pytest.mark.asyncio
    async def test_send_user_specific_alert(self, notification_service, mock_websocket_handler):
        """Test sending user-specific alert"""
        user_id = "user_123"
        alert_data = {
            "level": "info",
            "message": "Your subscription will expire in 7 days",
            "action_required": True
        }
        
        await notification_service.send_user_specific_alert(user_id, alert_data)
        
        mock_websocket_handler.broadcast.assert_called_once()
        call_args = mock_websocket_handler.broadcast.call_args
        assert call_args[1]["channel_id"] == f"user_{user_id}"
        
        message = json.loads(call_args[1]["message"])
        assert message["type"] == "user_alert"
        assert message["data"] == alert_data
    
    @pytest.mark.asyncio
    async def test_send_broadcast_message(self, notification_service, mock_websocket_handler):
        """Test sending broadcast message"""
        message_data = {
            "title": "System Maintenance",
            "content": "The system will be down for maintenance from 2 AM to 4 AM UTC",
            "scheduled_time": "2024-12-20T02:00:00Z"
        }
        
        await notification_service.send_broadcast_message(message_data)
        
        mock_websocket_handler.broadcast.assert_called_once()
        call_args = mock_websocket_handler.broadcast.call_args
        assert call_args[1]["channel_id"] == "broadcast_messages"
        
        message = json.loads(call_args[1]["message"])
        assert message["type"] == "broadcast_message"
        assert message["data"] == message_data
    
    def test_create_notification_message(self, notification_service):
        """Test creating notification message"""
        message_type = "test_notification"
        data = {"key": "value"}
        additional_fields = {"priority": "high"}
        
        message = notification_service._create_notification_message(
            message_type, data, additional_fields
        )
        
        assert message["type"] == message_type
        assert message["data"] == data
        assert message["priority"] == "high"
        assert "timestamp" in message
        assert "id" in message
    
    def test_create_notification_message_minimal(self, notification_service):
        """Test creating notification message with minimal parameters"""
        message_type = "simple_notification"
        data = {"key": "value"}
        
        message = notification_service._create_notification_message(message_type, data)
        
        assert message["type"] == message_type
        assert message["data"] == data
        assert "timestamp" in message
        assert "id" in message
    
    def test_get_user_channel_id(self, notification_service):
        """Test getting user channel ID"""
        user_id = "user_123"
        
        channel_id = notification_service._get_user_channel_id(user_id)
        
        assert channel_id == "user_user_123"
    
    def test_get_user_channel_id_with_prefix(self, notification_service):
        """Test getting user channel ID with existing prefix"""
        user_id = "user_123"
        
        channel_id = notification_service._get_user_channel_id(user_id)
        
        assert channel_id == "user_user_123"
    
    @pytest.mark.asyncio
    async def test_send_notification_with_retry(self, notification_service, mock_websocket_handler):
        """Test sending notification with retry mechanism"""
        user_id = "user_123"
        message = {"type": "test", "data": "test_data"}
        
        # Mock first call fails, second succeeds
        mock_websocket_handler.broadcast.side_effect = [
            Exception("First attempt failed"),
            None
        ]
        
        await notification_service.send_notification_to_user(user_id, message)
        
        # Should have been called twice (retry)
        assert mock_websocket_handler.broadcast.call_count == 2
    
    @pytest.mark.asyncio
    async def test_send_notification_max_retries_exceeded(self, notification_service, mock_websocket_handler):
        """Test sending notification when max retries exceeded"""
        user_id = "user_123"
        message = {"type": "test", "data": "test_data"}
        
        # Mock all calls fail
        mock_websocket_handler.broadcast.side_effect = Exception("Persistent failure")
        
        # Should not raise exception
        await notification_service.send_notification_to_user(user_id, message)
        
        # Should have been called max_retries times
        assert mock_websocket_handler.broadcast.call_count == 3  # Default max_retries

