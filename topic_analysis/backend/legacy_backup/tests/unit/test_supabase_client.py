"""
Unit tests for Supabase client operations
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import uuid

from src.database.supabase_client import SupabaseClient, ConnectionStatus
from src.models.supabase_client import SupabaseClient as SupabaseClientModel

class TestSupabaseClient:
    """Test cases for SupabaseClient"""
    
    def test_initialization(self):
        """Test SupabaseClient initialization"""
        with patch.dict('os.environ', {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_SERVICE_ROLE_KEY': 'test-key'
        }):
            client = SupabaseClient()
            
            assert client.url == 'https://test.supabase.co'
            assert client.service_role_key == 'test-key'
            assert client.connection_status == ConnectionStatus.INITIALIZED
    
    def test_initialization_missing_url(self):
        """Test initialization with missing URL"""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="Supabase URL and service role key are required"):
                SupabaseClient()
    
    def test_initialization_missing_key(self):
        """Test initialization with missing service role key"""
        with patch.dict('os.environ', {'SUPABASE_URL': 'https://test.supabase.co'}, clear=True):
            with pytest.raises(ValueError, match="Supabase URL and service role key are required"):
                SupabaseClient()
    
    @patch('src.database.supabase_client.create_client')
    def test_get_client(self, mock_create_client):
        """Test getting Supabase client"""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        with patch.dict('os.environ', {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_SERVICE_ROLE_KEY': 'test-key'
        }):
            client = SupabaseClient()
            result = client.get_client()
            
            assert result == mock_client
            mock_create_client.assert_called_once_with('https://test.supabase.co', 'test-key')
    
    @patch('src.database.supabase_client.create_client')
    def test_get_anon_client(self, mock_create_client):
        """Test getting anonymous Supabase client"""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        with patch.dict('os.environ', {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_SERVICE_ROLE_KEY': 'test-key',
            'SUPABASE_ANON_KEY': 'anon-key'
        }):
            client = SupabaseClient()
            result = client.get_anon_client()
            
            assert result == mock_client
            mock_create_client.assert_called_once_with('https://test.supabase.co', 'anon-key')
    
    def test_get_anon_client_missing_key(self):
        """Test getting anonymous client with missing anon key"""
        with patch.dict('os.environ', {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_SERVICE_ROLE_KEY': 'test-key'
        }):
            client = SupabaseClient()
            
            with pytest.raises(ValueError, match="SUPABASE_ANON_KEY is required for anon client"):
                client.get_anon_client()
    
    @patch('src.database.supabase_client.create_client')
    def test_test_connection_success(self, mock_create_client):
        """Test successful connection test"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [{'count': 1}]
        mock_client.table.return_value.select.return_value.execute.return_value = mock_response
        mock_create_client.return_value = mock_client
        
        with patch.dict('os.environ', {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_SERVICE_ROLE_KEY': 'test-key'
        }):
            client = SupabaseClient()
            result = client.test_connection()
            
            assert result['status'] == 'healthy'
            assert result['supabase_status'] == 'connected'
            assert 'response_time_ms' in result
            assert result['response_time_ms'] >= 0
    
    @patch('src.database.supabase_client.create_client')
    def test_test_connection_failure(self, mock_create_client):
        """Test connection test failure"""
        mock_client = Mock()
        mock_client.table.return_value.select.return_value.execute.side_effect = Exception("Connection failed")
        mock_create_client.return_value = mock_client
        
        with patch.dict('os.environ', {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_SERVICE_ROLE_KEY': 'test-key'
        }):
            client = SupabaseClient()
            result = client.test_connection()
            
            assert result['status'] == 'unhealthy'
            assert result['supabase_status'] == 'error'
            assert 'error' in result
    
    def test_update_status(self):
        """Test updating connection status"""
        with patch.dict('os.environ', {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_SERVICE_ROLE_KEY': 'test-key'
        }):
            client = SupabaseClient()
            client.update_status(ConnectionStatus.CONNECTED)
            
            assert client.connection_status == ConnectionStatus.CONNECTED
    
    def test_update_activity(self):
        """Test updating last activity"""
        with patch.dict('os.environ', {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_SERVICE_ROLE_KEY': 'test-key'
        }):
            client = SupabaseClient()
            client.update_activity()
            
            assert client.last_activity is not None
    
    def test_is_healthy(self):
        """Test health check"""
        with patch.dict('os.environ', {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_SERVICE_ROLE_KEY': 'test-key'
        }):
            client = SupabaseClient()
            client.connection_status = ConnectionStatus.CONNECTED
            
            assert client.is_healthy() is True
            
            client.connection_status = ConnectionStatus.ERROR
            assert client.is_healthy() is False
    
    def test_is_error(self):
        """Test error check"""
        with patch.dict('os.environ', {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_SERVICE_ROLE_KEY': 'test-key'
        }):
            client = SupabaseClient()
            client.connection_status = ConnectionStatus.ERROR
            
            assert client.is_error() is True
            
            client.connection_status = ConnectionStatus.CONNECTED
            assert client.is_error() is False
    
    def test_needs_reconnection(self):
        """Test reconnection check"""
        with patch.dict('os.environ', {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_SERVICE_ROLE_KEY': 'test-key'
        }):
            client = SupabaseClient()
            
            client.connection_status = ConnectionStatus.DISCONNECTED
            assert client.needs_reconnection() is True
            
            client.connection_status = ConnectionStatus.ERROR
            assert client.needs_reconnection() is True
            
            client.connection_status = ConnectionStatus.CONNECTED
            assert client.needs_reconnection() is False

class TestSupabaseClientModel:
    """Test cases for SupabaseClientModel"""
    
    def test_model_creation(self):
        """Test creating SupabaseClientModel instance"""
        model = SupabaseClientModel(
            url="https://test.supabase.co",
            service_role_key="test-key",
            anon_key="anon-key",
            connection_status=ConnectionStatus.CONNECTED
        )
        
        assert model.url == "https://test.supabase.co"
        assert model.service_role_key == "test-key"
        assert model.anon_key == "anon-key"
        assert model.connection_status == ConnectionStatus.CONNECTED
        assert model.id is not None
    
    def test_to_dict(self):
        """Test converting model to dictionary"""
        model = SupabaseClientModel(
            url="https://test.supabase.co",
            service_role_key="test-key",
            connection_status=ConnectionStatus.CONNECTED
        )
        
        data = model.to_dict()
        
        assert data["url"] == "https://test.supabase.co"
        assert data["connection_status"] == "connected"
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_update_status(self):
        """Test updating connection status"""
        model = SupabaseClientModel(
            url="https://test.supabase.co",
            service_role_key="test-key",
            connection_status=ConnectionStatus.INITIALIZED
        )
        
        model.update_status(ConnectionStatus.CONNECTED)
        
        assert model.connection_status == ConnectionStatus.CONNECTED
        assert model.updated_at is not None
    
    def test_update_activity(self):
        """Test updating last activity"""
        model = SupabaseClientModel(
            url="https://test.supabase.co",
            service_role_key="test-key"
        )
        
        model.update_activity()
        
        assert model.last_activity is not None
        assert model.updated_at is not None
    
    def test_is_healthy(self):
        """Test health check"""
        model = SupabaseClientModel(
            url="https://test.supabase.co",
            service_role_key="test-key",
            connection_status=ConnectionStatus.CONNECTED
        )
        
        assert model.is_healthy() is True
        
        model.connection_status = ConnectionStatus.ERROR
        assert model.is_healthy() is False
    
    def test_is_error(self):
        """Test error check"""
        model = SupabaseClientModel(
            url="https://test.supabase.co",
            service_role_key="test-key",
            connection_status=ConnectionStatus.ERROR
        )
        
        assert model.is_error() is True
        
        model.connection_status = ConnectionStatus.CONNECTED
        assert model.is_error() is False
    
    def test_needs_reconnection(self):
        """Test reconnection check"""
        model = SupabaseClientModel(
            url="https://test.supabase.co",
            service_role_key="test-key",
            connection_status=ConnectionStatus.DISCONNECTED
        )
        
        assert model.needs_reconnection() is True
        
        model.connection_status = ConnectionStatus.CONNECTED
        assert model.needs_reconnection() is False

