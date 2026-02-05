"""
Unit tests for Supabase connection manager
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import asyncio

from src.services.supabase_connection_manager import SupabaseConnectionManager
from src.services.supabase_error_handler import SupabaseErrorHandler

class TestSupabaseConnectionManager:
    """Test cases for SupabaseConnectionManager"""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client"""
        client = Mock()
        client.auth = Mock()
        return client
    
    @pytest.fixture
    def connection_manager(self, mock_supabase_client):
        """SupabaseConnectionManager instance with mocked dependencies"""
        with patch('src.services.supabase_connection_manager.get_supabase_client') as mock_get_client, \
             patch('src.services.supabase_connection_manager.SupabaseErrorHandler') as mock_error_handler_class:
            
            mock_get_client.return_value = mock_supabase_client
            mock_error_handler = Mock()
            mock_error_handler_class.return_value = mock_error_handler
            
            manager = SupabaseConnectionManager()
            manager.client = mock_supabase_client
            manager.error_handler = mock_error_handler
            return manager
    
    def test_singleton_pattern(self):
        """Test that SupabaseConnectionManager follows singleton pattern"""
        manager1 = SupabaseConnectionManager()
        manager2 = SupabaseConnectionManager()
        
        assert manager1 is manager2
    
    def test_initialization(self, connection_manager):
        """Test SupabaseConnectionManager initialization"""
        assert connection_manager.client is not None
        assert connection_manager.error_handler is not None
        assert connection_manager.max_retries == 3
        assert connection_manager.retry_delay == 1.0
        assert connection_manager.connection_timeout == 30.0
        assert connection_manager.health_check_interval == 60.0
        assert connection_manager.last_health_check is None
        assert connection_manager.connection_status == "disconnected"
    
    @pytest.mark.asyncio
    async def test_connect_success(self, connection_manager, mock_supabase_client):
        """Test successful connection"""
        # Mock successful health check
        mock_response = Mock()
        mock_response.data = [{"count": 1}]
        mock_supabase_client.table.return_value.select.return_value.execute.return_value = mock_response
        
        result = await connection_manager.connect()
        
        assert result == mock_supabase_client
        assert connection_manager.connection_status == "connected"
        assert connection_manager.last_health_check is not None
    
    @pytest.mark.asyncio
    async def test_connect_already_connected(self, connection_manager, mock_supabase_client):
        """Test connecting when already connected"""
        connection_manager.connection_status = "connected"
        connection_manager.last_health_check = datetime.now()
        
        result = await connection_manager.connect()
        
        assert result == mock_supabase_client
        # Should not call health check again
        mock_supabase_client.table.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_connect_health_check_failure(self, connection_manager, mock_supabase_client):
        """Test connection with health check failure"""
        # Mock health check failure
        mock_supabase_client.table.return_value.select.return_value.execute.side_effect = Exception("Health check failed")
        
        with pytest.raises(Exception, match="Health check failed"):
            await connection_manager.connect()
        
        assert connection_manager.connection_status == "error"
    
    @pytest.mark.asyncio
    async def test_connect_retry_mechanism(self, connection_manager, mock_supabase_client):
        """Test connection retry mechanism"""
        # Mock first two attempts failing, third succeeding
        mock_response = Mock()
        mock_response.data = [{"count": 1}]
        
        mock_supabase_client.table.return_value.select.return_value.execute.side_effect = [
            Exception("First attempt failed"),
            Exception("Second attempt failed"),
            mock_response
        ]
        
        result = await connection_manager.connect()
        
        assert result == mock_supabase_client
        assert connection_manager.connection_status == "connected"
        # Should have made 3 attempts
        assert mock_supabase_client.table.call_count == 3
    
    @pytest.mark.asyncio
    async def test_connect_max_retries_exceeded(self, connection_manager, mock_supabase_client):
        """Test connection when max retries exceeded"""
        # Mock all attempts failing
        mock_supabase_client.table.return_value.select.return_value.execute.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            await connection_manager.connect()
        
        assert connection_manager.connection_status == "error"
        # Should have made max_retries attempts
        assert mock_supabase_client.table.call_count == connection_manager.max_retries
    
    def test_get_client_connected(self, connection_manager, mock_supabase_client):
        """Test getting client when connected"""
        connection_manager.connection_status = "connected"
        connection_manager.client = mock_supabase_client
        
        result = connection_manager.get_client()
        
        assert result == mock_supabase_client
    
    def test_get_client_not_connected(self, connection_manager):
        """Test getting client when not connected"""
        connection_manager.connection_status = "disconnected"
        connection_manager.client = None
        
        with pytest.raises(Exception, match="Supabase client not connected"):
            connection_manager.get_client()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, connection_manager, mock_supabase_client):
        """Test disconnecting"""
        connection_manager.connection_status = "connected"
        connection_manager.client = mock_supabase_client
        
        await connection_manager.disconnect()
        
        assert connection_manager.connection_status == "disconnected"
        assert connection_manager.client is None
    
    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self, connection_manager):
        """Test disconnecting when not connected"""
        connection_manager.connection_status = "disconnected"
        connection_manager.client = None
        
        # Should not raise exception
        await connection_manager.disconnect()
        
        assert connection_manager.connection_status == "disconnected"
    
    @pytest.mark.asyncio
    async def test_refresh_session_success(self, connection_manager, mock_supabase_client):
        """Test successful session refresh"""
        refresh_token = "refresh_token_123"
        mock_session = Mock()
        mock_session.dict.return_value = {"access_token": "new_token", "refresh_token": "new_refresh_token"}
        mock_supabase_client.auth.refresh_session.return_value = mock_session
        
        result = await connection_manager.refresh_session(refresh_token)
        
        assert result == {"access_token": "new_token", "refresh_token": "new_refresh_token"}
        mock_supabase_client.auth.refresh_session.assert_called_once_with(refresh_token)
    
    @pytest.mark.asyncio
    async def test_refresh_session_failure(self, connection_manager, mock_supabase_client):
        """Test session refresh failure"""
        refresh_token = "invalid_token"
        mock_supabase_client.auth.refresh_session.side_effect = Exception("Invalid refresh token")
        
        with pytest.raises(Exception, match="Invalid refresh token"):
            await connection_manager.refresh_session(refresh_token)
    
    @pytest.mark.asyncio
    async def test_refresh_session_no_user(self, connection_manager, mock_supabase_client):
        """Test session refresh with no user in response"""
        refresh_token = "refresh_token_123"
        mock_response = Mock()
        mock_response.user = None
        mock_supabase_client.auth.refresh_session.return_value = mock_response
        
        with pytest.raises(Exception, match="Failed to refresh session: No user in response"):
            await connection_manager.refresh_session(refresh_token)
    
    @pytest.mark.asyncio
    async def test_refresh_session_auto_connect(self, connection_manager, mock_supabase_client):
        """Test session refresh with auto-connect when not connected"""
        connection_manager.connection_status = "disconnected"
        connection_manager.client = None
        
        refresh_token = "refresh_token_123"
        mock_session = Mock()
        mock_session.dict.return_value = {"access_token": "new_token"}
        mock_supabase_client.auth.refresh_session.return_value = mock_session
        
        # Mock connect method
        with patch.object(connection_manager, 'connect') as mock_connect:
            mock_connect.return_value = mock_supabase_client
            
            result = await connection_manager.refresh_session(refresh_token)
            
            assert result == {"access_token": "new_token"}
            mock_connect.assert_called_once()
    
    def test_is_connected_true(self, connection_manager):
        """Test is_connected when connected"""
        connection_manager.connection_status = "connected"
        
        assert connection_manager.is_connected() is True
    
    def test_is_connected_false(self, connection_manager):
        """Test is_connected when not connected"""
        connection_manager.connection_status = "disconnected"
        
        assert connection_manager.is_connected() is False
    
    def test_is_healthy_true(self, connection_manager):
        """Test is_healthy when healthy"""
        connection_manager.connection_status = "connected"
        connection_manager.last_health_check = datetime.now()
        
        assert connection_manager.is_healthy() is True
    
    def test_is_healthy_false(self, connection_manager):
        """Test is_healthy when not healthy"""
        connection_manager.connection_status = "error"
        
        assert connection_manager.is_healthy() is False
    
    def test_is_healthy_stale_check(self, connection_manager):
        """Test is_healthy with stale health check"""
        connection_manager.connection_status = "connected"
        connection_manager.last_health_check = datetime.now() - timedelta(seconds=120)  # 2 minutes ago
        
        assert connection_manager.is_healthy() is False
    
    def test_needs_reconnection_true(self, connection_manager):
        """Test needs_reconnection when reconnection needed"""
        connection_manager.connection_status = "disconnected"
        
        assert connection_manager.needs_reconnection() is True
    
    def test_needs_reconnection_false(self, connection_manager):
        """Test needs_reconnection when reconnection not needed"""
        connection_manager.connection_status = "connected"
        
        assert connection_manager.needs_reconnection() is False
    
    def test_get_connection_info(self, connection_manager, mock_supabase_client):
        """Test getting connection information"""
        connection_manager.connection_status = "connected"
        connection_manager.client = mock_supabase_client
        connection_manager.last_health_check = datetime.now()
        
        info = connection_manager.get_connection_info()
        
        assert info["status"] == "connected"
        assert info["client"] == mock_supabase_client
        assert info["last_health_check"] is not None
        assert info["max_retries"] == 3
        assert info["retry_delay"] == 1.0
    
    def test_get_connection_info_not_connected(self, connection_manager):
        """Test getting connection info when not connected"""
        connection_manager.connection_status = "disconnected"
        connection_manager.client = None
        
        info = connection_manager.get_connection_info()
        
        assert info["status"] == "disconnected"
        assert info["client"] is None
        assert info["last_health_check"] is None
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, connection_manager, mock_supabase_client):
        """Test successful health check"""
        mock_response = Mock()
        mock_response.data = [{"count": 1}]
        mock_supabase_client.table.return_value.select.return_value.execute.return_value = mock_response
        
        result = await connection_manager.health_check()
        
        assert result is True
        assert connection_manager.connection_status == "connected"
        assert connection_manager.last_health_check is not None
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, connection_manager, mock_supabase_client):
        """Test health check failure"""
        mock_supabase_client.table.return_value.select.return_value.execute.side_effect = Exception("Health check failed")
        
        result = await connection_manager.health_check()
        
        assert result is False
        assert connection_manager.connection_status == "error"
    
    @pytest.mark.asyncio
    async def test_health_check_no_data(self, connection_manager, mock_supabase_client):
        """Test health check with no data returned"""
        mock_response = Mock()
        mock_response.data = []
        mock_supabase_client.table.return_value.select.return_value.execute.return_value = mock_response
        
        result = await connection_manager.health_check()
        
        assert result is False
        assert connection_manager.connection_status == "error"
    
    def test_set_connection_status(self, connection_manager):
        """Test setting connection status"""
        connection_manager.set_connection_status("connected")
        
        assert connection_manager.connection_status == "connected"
    
    def test_set_connection_status_with_timestamp(self, connection_manager):
        """Test setting connection status with timestamp"""
        timestamp = datetime.now()
        connection_manager.set_connection_status("connected", timestamp)
        
        assert connection_manager.connection_status == "connected"
        assert connection_manager.last_health_check == timestamp
    
    def test_reset_connection(self, connection_manager, mock_supabase_client):
        """Test resetting connection"""
        connection_manager.connection_status = "error"
        connection_manager.client = mock_supabase_client
        connection_manager.last_health_check = datetime.now()
        
        connection_manager.reset_connection()
        
        assert connection_manager.connection_status == "disconnected"
        assert connection_manager.client is None
        assert connection_manager.last_health_check is None

