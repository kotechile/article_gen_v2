"""
Unit tests for WebSocket handler
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

from src.services.websocket_handler import WebSocketHandler

class TestWebSocketHandler:
    """Test cases for WebSocketHandler"""
    
    @pytest.fixture
    def websocket_handler(self):
        """WebSocketHandler instance"""
        return WebSocketHandler()
    
    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket connection"""
        websocket = Mock(spec=WebSocket)
        websocket.accept = Mock()
        websocket.send_text = Mock()
        websocket.receive_text = Mock()
        websocket.client = Mock()
        websocket.client.host = "127.0.0.1"
        websocket.client.port = 8080
        return websocket
    
    def test_initialization(self, websocket_handler):
        """Test WebSocketHandler initialization"""
        assert websocket_handler.active_connections == {}
        assert websocket_handler.subscription_callbacks == {}
    
    @pytest.mark.asyncio
    async def test_connect_success(self, websocket_handler, mock_websocket):
        """Test successful WebSocket connection"""
        channel_id = "test_channel"
        
        await websocket_handler.connect(mock_websocket, channel_id)
        
        assert channel_id in websocket_handler.active_connections
        assert mock_websocket in websocket_handler.active_connections[channel_id]
        mock_websocket.accept.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_existing_channel(self, websocket_handler, mock_websocket):
        """Test connecting to existing channel"""
        channel_id = "test_channel"
        mock_websocket2 = Mock(spec=WebSocket)
        mock_websocket2.accept = Mock()
        mock_websocket2.client = Mock()
        mock_websocket2.client.host = "127.0.0.1"
        mock_websocket2.client.port = 8081
        
        # Connect first websocket
        await websocket_handler.connect(mock_websocket, channel_id)
        
        # Connect second websocket to same channel
        await websocket_handler.connect(mock_websocket2, channel_id)
        
        assert len(websocket_handler.active_connections[channel_id]) == 2
        assert mock_websocket in websocket_handler.active_connections[channel_id]
        assert mock_websocket2 in websocket_handler.active_connections[channel_id]
    
    def test_disconnect_existing_connection(self, websocket_handler, mock_websocket):
        """Test disconnecting existing connection"""
        channel_id = "test_channel"
        websocket_handler.active_connections[channel_id] = {mock_websocket}
        
        websocket_handler.disconnect(mock_websocket, channel_id)
        
        assert channel_id not in websocket_handler.active_connections
    
    def test_disconnect_non_existing_connection(self, websocket_handler, mock_websocket):
        """Test disconnecting non-existing connection"""
        channel_id = "test_channel"
        
        # Should not raise exception
        websocket_handler.disconnect(mock_websocket, channel_id)
        
        assert channel_id not in websocket_handler.active_connections
    
    def test_disconnect_connection_not_in_channel(self, websocket_handler, mock_websocket):
        """Test disconnecting connection not in channel"""
        channel_id = "test_channel"
        websocket_handler.active_connections[channel_id] = set()
        
        # Should not raise exception
        websocket_handler.disconnect(mock_websocket, channel_id)
    
    @pytest.mark.asyncio
    async def test_send_personal_message_success(self, websocket_handler, mock_websocket):
        """Test sending personal message successfully"""
        message = "Hello, World!"
        
        await websocket_handler.send_personal_message(message, mock_websocket)
        
        mock_websocket.send_text.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_send_personal_message_failure(self, websocket_handler, mock_websocket):
        """Test sending personal message with failure"""
        message = "Hello, World!"
        mock_websocket.send_text.side_effect = Exception("Send failed")
        
        # Should not raise exception
        await websocket_handler.send_personal_message(message, mock_websocket)
        
        mock_websocket.send_text.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_broadcast_success(self, websocket_handler, mock_websocket):
        """Test broadcasting message successfully"""
        channel_id = "test_channel"
        message = "Broadcast message"
        
        websocket_handler.active_connections[channel_id] = {mock_websocket}
        
        await websocket_handler.broadcast(message, channel_id)
        
        mock_websocket.send_text.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_broadcast_multiple_connections(self, websocket_handler):
        """Test broadcasting to multiple connections"""
        channel_id = "test_channel"
        message = "Broadcast message"
        
        # Create multiple mock websockets
        websocket1 = Mock(spec=WebSocket)
        websocket1.send_text = Mock()
        websocket1.client = Mock()
        websocket1.client.host = "127.0.0.1"
        websocket1.client.port = 8080
        
        websocket2 = Mock(spec=WebSocket)
        websocket2.send_text = Mock()
        websocket2.client = Mock()
        websocket2.client.host = "127.0.0.1"
        websocket2.client.port = 8081
        
        websocket_handler.active_connections[channel_id] = {websocket1, websocket2}
        
        await websocket_handler.broadcast(message, channel_id)
        
        websocket1.send_text.assert_called_once_with(message)
        websocket2.send_text.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_broadcast_with_failures(self, websocket_handler):
        """Test broadcasting with some failures"""
        channel_id = "test_channel"
        message = "Broadcast message"
        
        # Create websockets with different behaviors
        websocket1 = Mock(spec=WebSocket)
        websocket1.send_text = Mock()
        websocket1.client = Mock()
        websocket1.client.host = "127.0.0.1"
        websocket1.client.port = 8080
        
        websocket2 = Mock(spec=WebSocket)
        websocket2.send_text = Mock(side_effect=Exception("Send failed"))
        websocket2.client = Mock()
        websocket2.client.host = "127.0.0.1"
        websocket2.client.port = 8081
        
        websocket_handler.active_connections[channel_id] = {websocket1, websocket2}
        
        await websocket_handler.broadcast(message, channel_id)
        
        websocket1.send_text.assert_called_once_with(message)
        websocket2.send_text.assert_called_once_with(message)
        
        # Failed websocket should be removed
        assert websocket1 in websocket_handler.active_connections[channel_id]
        assert websocket2 not in websocket_handler.active_connections[channel_id]
    
    @pytest.mark.asyncio
    async def test_broadcast_empty_channel(self, websocket_handler):
        """Test broadcasting to empty channel"""
        channel_id = "empty_channel"
        message = "Broadcast message"
        
        # Should not raise exception
        await websocket_handler.broadcast(message, channel_id)
    
    @pytest.mark.asyncio
    async def test_broadcast_nonexistent_channel(self, websocket_handler):
        """Test broadcasting to non-existent channel"""
        channel_id = "nonexistent_channel"
        message = "Broadcast message"
        
        # Should not raise exception
        await websocket_handler.broadcast(message, channel_id)
    
    @pytest.mark.asyncio
    async def test_get_active_channel_count(self, websocket_handler, mock_websocket):
        """Test getting active channel count"""
        channel_id = "test_channel"
        websocket_handler.active_connections[channel_id] = {mock_websocket}
        
        count = await websocket_handler.get_active_channel_count(channel_id)
        
        assert count == 1
    
    @pytest.mark.asyncio
    async def test_get_active_channel_count_empty(self, websocket_handler):
        """Test getting active channel count for empty channel"""
        channel_id = "empty_channel"
        
        count = await websocket_handler.get_active_channel_count(channel_id)
        
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_get_total_connections(self, websocket_handler):
        """Test getting total connections"""
        # Create multiple channels with connections
        websocket1 = Mock(spec=WebSocket)
        websocket2 = Mock(spec=WebSocket)
        websocket3 = Mock(spec=WebSocket)
        
        websocket_handler.active_connections["channel1"] = {websocket1, websocket2}
        websocket_handler.active_connections["channel2"] = {websocket3}
        
        total = await websocket_handler.get_total_connections()
        
        assert total == 3
    
    @pytest.mark.asyncio
    async def test_get_total_connections_empty(self, websocket_handler):
        """Test getting total connections when empty"""
        total = await websocket_handler.get_total_connections()
        
        assert total == 0
    
    def test_get_channel_connections(self, websocket_handler, mock_websocket):
        """Test getting connections for a channel"""
        channel_id = "test_channel"
        websocket_handler.active_connections[channel_id] = {mock_websocket}
        
        connections = websocket_handler.get_channel_connections(channel_id)
        
        assert connections == {mock_websocket}
    
    def test_get_channel_connections_empty(self, websocket_handler):
        """Test getting connections for empty channel"""
        channel_id = "empty_channel"
        
        connections = websocket_handler.get_channel_connections(channel_id)
        
        assert connections == set()
    
    def test_get_all_channels(self, websocket_handler, mock_websocket):
        """Test getting all active channels"""
        websocket_handler.active_connections["channel1"] = {mock_websocket}
        websocket_handler.active_connections["channel2"] = set()
        
        channels = websocket_handler.get_all_channels()
        
        assert "channel1" in channels
        assert "channel2" not in channels  # Empty channels should be excluded
    
    def test_cleanup_empty_channels(self, websocket_handler, mock_websocket):
        """Test cleaning up empty channels"""
        websocket_handler.active_connections["channel1"] = {mock_websocket}
        websocket_handler.active_connections["channel2"] = set()
        websocket_handler.active_connections["channel3"] = set()
        
        websocket_handler.cleanup_empty_channels()
        
        assert "channel1" in websocket_handler.active_connections
        assert "channel2" not in websocket_handler.active_connections
        assert "channel3" not in websocket_handler.active_connections
    
    def test_is_connected(self, websocket_handler, mock_websocket):
        """Test checking if websocket is connected"""
        channel_id = "test_channel"
        websocket_handler.active_connections[channel_id] = {mock_websocket}
        
        assert websocket_handler.is_connected(mock_websocket, channel_id) is True
        assert websocket_handler.is_connected(mock_websocket, "other_channel") is False
    
    def test_is_connected_nonexistent(self, websocket_handler, mock_websocket):
        """Test checking connection for non-existent websocket"""
        channel_id = "test_channel"
        
        assert websocket_handler.is_connected(mock_websocket, channel_id) is False
    
    def test_get_connection_info(self, websocket_handler, mock_websocket):
        """Test getting connection information"""
        channel_id = "test_channel"
        websocket_handler.active_connections[channel_id] = {mock_websocket}
        
        info = websocket_handler.get_connection_info()
        
        assert info["total_connections"] == 1
        assert info["active_channels"] == 1
        assert info["connections_by_channel"]["test_channel"] == 1
    
    def test_get_connection_info_empty(self, websocket_handler):
        """Test getting connection info when empty"""
        info = websocket_handler.get_connection_info()
        
        assert info["total_connections"] == 0
        assert info["active_channels"] == 0
        assert info["connections_by_channel"] == {}
    
    @pytest.mark.asyncio
    async def test_handle_websocket_disconnect(self, websocket_handler, mock_websocket):
        """Test handling WebSocket disconnect"""
        channel_id = "test_channel"
        websocket_handler.active_connections[channel_id] = {mock_websocket}
        
        await websocket_handler.handle_websocket_disconnect(mock_websocket, channel_id)
        
        assert channel_id not in websocket_handler.active_connections
    
    @pytest.mark.asyncio
    async def test_handle_websocket_disconnect_with_callback(self, websocket_handler, mock_websocket):
        """Test handling WebSocket disconnect with callback"""
        channel_id = "test_channel"
        callback = Mock()
        websocket_handler.subscription_callbacks[channel_id] = callback
        websocket_handler.active_connections[channel_id] = {mock_websocket}
        
        await websocket_handler.handle_websocket_disconnect(mock_websocket, channel_id)
        
        assert channel_id not in websocket_handler.active_connections
        assert channel_id not in websocket_handler.subscription_callbacks
        callback.assert_called_once_with(mock_websocket, channel_id)
    
    def test_set_subscription_callback(self, websocket_handler, mock_websocket):
        """Test setting subscription callback"""
        channel_id = "test_channel"
        callback = Mock()
        
        websocket_handler.set_subscription_callback(channel_id, callback)
        
        assert websocket_handler.subscription_callbacks[channel_id] == callback
    
    def test_remove_subscription_callback(self, websocket_handler):
        """Test removing subscription callback"""
        channel_id = "test_channel"
        callback = Mock()
        websocket_handler.subscription_callbacks[channel_id] = callback
        
        websocket_handler.remove_subscription_callback(channel_id)
        
        assert channel_id not in websocket_handler.subscription_callbacks
    
    def test_remove_subscription_callback_nonexistent(self, websocket_handler):
        """Test removing non-existent subscription callback"""
        channel_id = "nonexistent_channel"
        
        # Should not raise exception
        websocket_handler.remove_subscription_callback(channel_id)
    
    def test_get_subscription_callback(self, websocket_handler):
        """Test getting subscription callback"""
        channel_id = "test_channel"
        callback = Mock()
        websocket_handler.subscription_callbacks[channel_id] = callback
        
        result = websocket_handler.get_subscription_callback(channel_id)
        
        assert result == callback
    
    def test_get_subscription_callback_nonexistent(self, websocket_handler):
        """Test getting non-existent subscription callback"""
        channel_id = "nonexistent_channel"
        
        result = websocket_handler.get_subscription_callback(channel_id)
        
        assert result is None

