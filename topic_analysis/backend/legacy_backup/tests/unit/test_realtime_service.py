"""
Unit tests for real-time service
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import uuid

from src.services.realtime_service import RealTimeService

class TestRealTimeService:
    """Test cases for RealTimeService"""
    
    @pytest.fixture
    def realtime_service(self):
        """RealTimeService instance with mocked Supabase client"""
        with patch('src.services.realtime_service.get_supabase_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            service = RealTimeService()
            service.client = mock_client
            return service
    
    def test_initialization(self, realtime_service):
        """Test RealTimeService initialization"""
        assert realtime_service.client is not None
        assert realtime_service.subscriptions == {}
        assert realtime_service.subscription_callbacks == {}
    
    @patch('src.services.realtime_service.create_client')
    def test_create_subscription(self, mock_create_client, realtime_service):
        """Test creating a real-time subscription"""
        mock_channel = Mock()
        mock_channel.on = Mock()
        mock_channel.subscribe = Mock()
        realtime_service.client.channel.return_value = mock_channel
        
        result = realtime_service.create_subscription(
            table_name="users",
            event_types=["INSERT", "UPDATE"],
            filter_conditions={"user_id": "123"}
        )
        
        assert "subscription_id" in result
        assert "websocket_url" in result
        assert "status" in result
        assert result["status"] == "active"
        
        # Verify channel setup
        realtime_service.client.channel.assert_called_once()
        mock_channel.on.assert_called()
        mock_channel.subscribe.assert_called_once()
        
        # Verify subscription tracking
        subscription_id = result["subscription_id"]
        assert subscription_id in realtime_service.subscriptions
        assert realtime_service.subscriptions[subscription_id]["table_name"] == "users"
    
    def test_create_subscription_minimal(self, realtime_service):
        """Test creating subscription with minimal parameters"""
        with patch('src.services.realtime_service.create_client'):
            mock_channel = Mock()
            mock_channel.on = Mock()
            mock_channel.subscribe = Mock()
            realtime_service.client.channel.return_value = mock_channel
            
            result = realtime_service.create_subscription(
                table_name="posts",
                event_types=["INSERT"]
            )
            
            assert "subscription_id" in result
            assert result["status"] == "active"
    
    def test_unsubscribe(self, realtime_service):
        """Test unsubscribing from real-time updates"""
        subscription_id = "sub_123"
        mock_channel = Mock()
        mock_channel.unsubscribe = Mock()
        
        realtime_service.subscriptions[subscription_id] = {
            "channel": mock_channel,
            "table_name": "users",
            "event_types": ["INSERT"],
            "status": "active"
        }
        
        result = realtime_service.unsubscribe(subscription_id)
        
        assert result is True
        mock_channel.unsubscribe.assert_called_once()
        assert subscription_id not in realtime_service.subscriptions
    
    def test_unsubscribe_not_found(self, realtime_service):
        """Test unsubscribing from non-existent subscription"""
        result = realtime_service.unsubscribe("nonexistent")
        
        assert result is False
    
    def test_get_subscription(self, realtime_service):
        """Test getting subscription details"""
        subscription_id = "sub_123"
        subscription_data = {
            "channel": Mock(),
            "table_name": "users",
            "event_types": ["INSERT"],
            "status": "active",
            "created_at": "2024-12-19T10:30:00Z"
        }
        
        realtime_service.subscriptions[subscription_id] = subscription_data
        
        result = realtime_service.get_subscription(subscription_id)
        
        assert result is not None
        assert result["table_name"] == "users"
        assert result["event_types"] == ["INSERT"]
        assert result["status"] == "active"
        assert "channel" not in result  # Channel object should be removed
    
    def test_get_subscription_not_found(self, realtime_service):
        """Test getting non-existent subscription"""
        result = realtime_service.get_subscription("nonexistent")
        
        assert result is None
    
    def test_get_all_subscriptions(self, realtime_service):
        """Test getting all active subscriptions"""
        # Add some test subscriptions
        realtime_service.subscriptions["sub_1"] = {
            "channel": Mock(),
            "table_name": "users",
            "event_types": ["INSERT"],
            "status": "active"
        }
        realtime_service.subscriptions["sub_2"] = {
            "channel": Mock(),
            "table_name": "posts",
            "event_types": ["UPDATE"],
            "status": "active"
        }
        realtime_service.subscriptions["sub_3"] = {
            "channel": Mock(),
            "table_name": "comments",
            "event_types": ["DELETE"],
            "status": "unsubscribed"
        }
        
        result = realtime_service.get_all_subscriptions()
        
        assert len(result) == 2  # Only active subscriptions
        assert all(sub["status"] == "active" for sub in result)
        assert all("channel" not in sub for sub in result)  # Channel objects removed
    
    def test_set_subscription_callback(self, realtime_service):
        """Test setting subscription callback"""
        subscription_id = "sub_123"
        callback = Mock()
        
        realtime_service.subscriptions[subscription_id] = {
            "channel": Mock(),
            "table_name": "users",
            "status": "active"
        }
        
        realtime_service.set_subscription_callback(subscription_id, callback)
        
        assert realtime_service.subscription_callbacks[subscription_id] == callback
    
    def test_cleanup_inactive_subscriptions(self, realtime_service):
        """Test cleaning up inactive subscriptions"""
        # Add test subscriptions
        realtime_service.subscriptions["sub_1"] = {
            "channel": Mock(),
            "table_name": "users",
            "status": "active"
        }
        realtime_service.subscriptions["sub_2"] = {
            "channel": Mock(),
            "table_name": "posts",
            "status": "unsubscribed"
        }
        realtime_service.subscription_callbacks["sub_1"] = Mock()
        realtime_service.subscription_callbacks["sub_2"] = Mock()
        
        result = realtime_service.cleanup_inactive_subscriptions()
        
        assert result == 1  # One inactive subscription cleaned up
        assert "sub_1" in realtime_service.subscriptions
        assert "sub_2" not in realtime_service.subscriptions
        assert "sub_1" in realtime_service.subscription_callbacks
        assert "sub_2" not in realtime_service.subscription_callbacks
    
    def test_get_subscription_statistics(self, realtime_service):
        """Test getting subscription statistics"""
        # Add test subscriptions
        realtime_service.subscriptions["sub_1"] = {
            "table_name": "users",
            "status": "active"
        }
        realtime_service.subscriptions["sub_2"] = {
            "table_name": "posts",
            "status": "active"
        }
        realtime_service.subscriptions["sub_3"] = {
            "table_name": "users",
            "status": "unsubscribed"
        }
        
        result = realtime_service.get_subscription_statistics()
        
        assert result["total_subscriptions"] == 3
        assert result["active_subscriptions"] == 2
        assert result["inactive_subscriptions"] == 1
        assert result["subscriptions_by_table"]["users"] == 2
        assert result["subscriptions_by_table"]["posts"] == 1
    
    def test_test_realtime_connection(self, realtime_service):
        """Test real-time connection test"""
        with patch.object(realtime_service, 'create_subscription') as mock_create, \
             patch.object(realtime_service, 'unsubscribe') as mock_unsubscribe:
            
            mock_create.return_value = {"subscription_id": "test_sub"}
            
            result = realtime_service.test_realtime_connection()
            
            assert result["status"] == "success"
            assert "message" in result
            assert "timestamp" in result
            
            mock_create.assert_called_once()
            mock_unsubscribe.assert_called_once_with("test_sub")
    
    def test_test_realtime_connection_failure(self, realtime_service):
        """Test real-time connection test with failure"""
        with patch.object(realtime_service, 'create_subscription') as mock_create:
            mock_create.side_effect = Exception("Connection failed")
            
            result = realtime_service.test_realtime_connection()
            
            assert result["status"] == "failed"
            assert "Connection failed" in result["message"]
    
    def test_handle_realtime_event(self, realtime_service):
        """Test handling real-time event"""
        subscription_id = "sub_123"
        callback = Mock()
        
        realtime_service.subscription_callbacks[subscription_id] = callback
        
        payload = {
            "eventType": "INSERT",
            "table": "users",
            "new": {"id": 1, "name": "John"},
            "old": None
        }
        
        realtime_service._handle_realtime_event(subscription_id, payload)
        
        callback.assert_called_once_with(payload)
    
    def test_handle_realtime_event_no_callback(self, realtime_service):
        """Test handling real-time event without callback"""
        subscription_id = "sub_123"
        
        payload = {
            "eventType": "INSERT",
            "table": "users",
            "new": {"id": 1, "name": "John"}
        }
        
        # Should not raise exception
        realtime_service._handle_realtime_event(subscription_id, payload)
    
    def test_handle_realtime_event_exception(self, realtime_service):
        """Test handling real-time event with callback exception"""
        subscription_id = "sub_123"
        callback = Mock(side_effect=Exception("Callback error"))
        
        realtime_service.subscription_callbacks[subscription_id] = callback
        
        payload = {
            "eventType": "INSERT",
            "table": "users"
        }
        
        # Should not raise exception
        realtime_service._handle_realtime_event(subscription_id, payload)
    
    def test_find_subscription_for_event(self, realtime_service):
        """Test finding subscription for event"""
        realtime_service.subscriptions["sub_1"] = {
            "table_name": "users",
            "event_types": ["INSERT"],
            "status": "active"
        }
        realtime_service.subscriptions["sub_2"] = {
            "table_name": "posts",
            "event_types": ["UPDATE"],
            "status": "active"
        }
        
        payload = {
            "table": "users",
            "eventType": "INSERT"
        }
        
        result = realtime_service._find_subscription_for_event(payload)
        
        assert result == "sub_1"
    
    def test_find_subscription_for_event_not_found(self, realtime_service):
        """Test finding subscription for event that doesn't match"""
        realtime_service.subscriptions["sub_1"] = {
            "table_name": "users",
            "event_types": ["INSERT"],
            "status": "active"
        }
        
        payload = {
            "table": "posts",
            "eventType": "INSERT"
        }
        
        result = realtime_service._find_subscription_for_event(payload)
        
        assert result is None
    
    def test_get_websocket_url(self, realtime_service):
        """Test getting WebSocket URL"""
        url = realtime_service._get_websocket_url()
        
        assert url.startswith("wss://")
        assert "supabase.co" in url
        assert "/realtime/v1/websocket" in url

