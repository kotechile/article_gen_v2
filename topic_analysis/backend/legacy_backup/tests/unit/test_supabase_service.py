"""
Unit tests for Supabase service
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import uuid

from src.services.supabase_service import SupabaseService
from src.models.database_operation import OperationType, OperationStatus

class TestSupabaseService:
    """Test cases for SupabaseService"""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client"""
        client = Mock()
        client.from_.return_value = Mock()
        return client
    
    @pytest.fixture
    def mock_db_operation_service(self):
        """Mock database operation service"""
        service = Mock()
        service.create_operation = Mock()
        service.update_operation_status = Mock()
        return service
    
    @pytest.fixture
    def supabase_service(self, mock_supabase_client, mock_db_operation_service):
        """SupabaseService instance with mocked dependencies"""
        with patch('src.services.supabase_service.DatabaseOperationService') as mock_db_service_class:
            mock_db_service_class.return_value = mock_db_operation_service
            service = SupabaseService(mock_supabase_client)
            return service
    
    def test_initialization(self, supabase_service, mock_supabase_client):
        """Test SupabaseService initialization"""
        assert supabase_service.client == mock_supabase_client
        assert supabase_service.db_operation_service is not None
    
    def test_initialization_without_client(self):
        """Test initialization without provided client"""
        with patch('src.services.supabase_service.get_supabase_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            service = SupabaseService()
            
            assert service.client == mock_client
            mock_get_client.assert_called_once()
    
    def test_insert_success(self, supabase_service, mock_supabase_client, mock_db_operation_service):
        """Test successful insert operation"""
        table_name = "users"
        data = {"name": "John", "email": "john@example.com"}
        user_id = "user123"
        
        # Mock operation creation
        mock_operation = Mock()
        mock_operation.operation_id = uuid.uuid4()
        mock_db_operation_service.create_operation.return_value = mock_operation
        
        # Mock Supabase response
        mock_response = Mock()
        mock_response.data = [{"id": 1, "name": "John", "email": "john@example.com"}]
        mock_response.error = None
        mock_supabase_client.from_.return_value.insert.return_value.execute.return_value = mock_response
        
        result = supabase_service.insert(table_name, data, user_id)
        
        assert result == [{"id": 1, "name": "John", "email": "john@example.com"}]
        
        # Verify operation logging
        mock_db_operation_service.create_operation.assert_called_once_with(
            operation_type=OperationType.CREATE,
            table_name=table_name,
            query_data=data,
            user_id=user_id
        )
        mock_db_operation_service.update_operation_status.assert_called_once()
    
    def test_insert_failure(self, supabase_service, mock_supabase_client, mock_db_operation_service):
        """Test insert operation failure"""
        table_name = "users"
        data = {"name": "John"}
        
        # Mock operation creation
        mock_operation = Mock()
        mock_operation.operation_id = uuid.uuid4()
        mock_db_operation_service.create_operation.return_value = mock_operation
        
        # Mock Supabase error
        mock_supabase_client.from_.return_value.insert.return_value.execute.side_effect = Exception("Database error")
        
        with pytest.raises(Exception, match="Database error"):
            supabase_service.insert(table_name, data)
        
        # Verify error logging
        mock_db_operation_service.update_operation_status.assert_called_once()
    
    def test_select_success(self, supabase_service, mock_supabase_client, mock_db_operation_service):
        """Test successful select operation"""
        table_name = "users"
        query = {"status": "active"}
        user_id = "user123"
        
        # Mock operation creation
        mock_operation = Mock()
        mock_operation.operation_id = uuid.uuid4()
        mock_db_operation_service.create_operation.return_value = mock_operation
        
        # Mock Supabase response
        mock_response = Mock()
        mock_response.data = [
            {"id": 1, "name": "John", "status": "active"},
            {"id": 2, "name": "Jane", "status": "active"}
        ]
        mock_response.error = None
        mock_supabase_client.from_.return_value.select.return_value.match.return_value.execute.return_value = mock_response
        
        result = supabase_service.select(table_name, query, user_id)
        
        assert len(result) == 2
        assert result[0]["name"] == "John"
        assert result[1]["name"] == "Jane"
        
        # Verify operation logging
        mock_db_operation_service.create_operation.assert_called_once_with(
            operation_type=OperationType.READ,
            table_name=table_name,
            query_data=query,
            user_id=user_id
        )
        mock_db_operation_service.update_operation_status.assert_called_once()
    
    def test_select_failure(self, supabase_service, mock_supabase_client, mock_db_operation_service):
        """Test select operation failure"""
        table_name = "users"
        query = {"status": "active"}
        
        # Mock operation creation
        mock_operation = Mock()
        mock_operation.operation_id = uuid.uuid4()
        mock_db_operation_service.create_operation.return_value = mock_operation
        
        # Mock Supabase error
        mock_supabase_client.from_.return_value.select.return_value.match.return_value.execute.side_effect = Exception("Query failed")
        
        with pytest.raises(Exception, match="Query failed"):
            supabase_service.select(table_name, query)
    
    def test_update_success(self, supabase_service, mock_supabase_client, mock_db_operation_service):
        """Test successful update operation"""
        table_name = "users"
        query = {"id": 1}
        data = {"name": "John Updated"}
        user_id = "user123"
        
        # Mock operation creation
        mock_operation = Mock()
        mock_operation.operation_id = uuid.uuid4()
        mock_db_operation_service.create_operation.return_value = mock_operation
        
        # Mock Supabase response
        mock_response = Mock()
        mock_response.data = [{"id": 1, "name": "John Updated"}]
        mock_response.error = None
        mock_supabase_client.from_.return_value.update.return_value.match.return_value.execute.return_value = mock_response
        
        result = supabase_service.update(table_name, query, data, user_id)
        
        assert result == [{"id": 1, "name": "John Updated"}]
        
        # Verify operation logging
        mock_db_operation_service.create_operation.assert_called_once_with(
            operation_type=OperationType.UPDATE,
            table_name=table_name,
            query_data={"query": query, "data": data},
            user_id=user_id
        )
        mock_db_operation_service.update_operation_status.assert_called_once()
    
    def test_update_failure(self, supabase_service, mock_supabase_client, mock_db_operation_service):
        """Test update operation failure"""
        table_name = "users"
        query = {"id": 1}
        data = {"name": "John Updated"}
        
        # Mock operation creation
        mock_operation = Mock()
        mock_operation.operation_id = uuid.uuid4()
        mock_db_operation_service.create_operation.return_value = mock_operation
        
        # Mock Supabase error
        mock_supabase_client.from_.return_value.update.return_value.match.return_value.execute.side_effect = Exception("Update failed")
        
        with pytest.raises(Exception, match="Update failed"):
            supabase_service.update(table_name, query, data)
    
    def test_delete_success(self, supabase_service, mock_supabase_client, mock_db_operation_service):
        """Test successful delete operation"""
        table_name = "users"
        query = {"id": 1}
        user_id = "user123"
        
        # Mock operation creation
        mock_operation = Mock()
        mock_operation.operation_id = uuid.uuid4()
        mock_db_operation_service.create_operation.return_value = mock_operation
        
        # Mock Supabase response
        mock_response = Mock()
        mock_response.data = [{"id": 1, "name": "John"}]
        mock_response.error = None
        mock_supabase_client.from_.return_value.delete.return_value.match.return_value.execute.return_value = mock_response
        
        result = supabase_service.delete(table_name, query, user_id)
        
        assert result == [{"id": 1, "name": "John"}]
        
        # Verify operation logging
        mock_db_operation_service.create_operation.assert_called_once_with(
            operation_type=OperationType.DELETE,
            table_name=table_name,
            query_data=query,
            user_id=user_id
        )
        mock_db_operation_service.update_operation_status.assert_called_once()
    
    def test_delete_failure(self, supabase_service, mock_supabase_client, mock_db_operation_service):
        """Test delete operation failure"""
        table_name = "users"
        query = {"id": 1}
        
        # Mock operation creation
        mock_operation = Mock()
        mock_operation.operation_id = uuid.uuid4()
        mock_db_operation_service.create_operation.return_value = mock_operation
        
        # Mock Supabase error
        mock_supabase_client.from_.return_value.delete.return_value.match.return_value.execute.side_effect = Exception("Delete failed")
        
        with pytest.raises(Exception, match="Delete failed"):
            supabase_service.delete(table_name, query)
    
    def test_execute_and_log_success(self, supabase_service, mock_db_operation_service):
        """Test successful execute_and_log operation"""
        table_name = "users"
        operation_type = OperationType.CREATE
        query_data = {"name": "John"}
        user_id = "user123"
        
        # Mock operation creation
        mock_operation = Mock()
        mock_operation.operation_id = uuid.uuid4()
        mock_db_operation_service.create_operation.return_value = mock_operation
        
        # Mock query function
        query_func = Mock(return_value={"id": 1, "name": "John"})
        
        result = supabase_service._execute_and_log(
            operation_type, table_name, query_func, user_id, query_data
        )
        
        assert result == {"id": 1, "name": "John"}
        
        # Verify operation logging
        mock_db_operation_service.create_operation.assert_called_once_with(
            operation_type=operation_type,
            table_name=table_name,
            query_data=query_data,
            user_id=user_id
        )
        mock_db_operation_service.update_operation_status.assert_called_once()
    
    def test_execute_and_log_failure(self, supabase_service, mock_db_operation_service):
        """Test execute_and_log operation failure"""
        table_name = "users"
        operation_type = OperationType.CREATE
        query_data = {"name": "John"}
        user_id = "user123"
        
        # Mock operation creation
        mock_operation = Mock()
        mock_operation.operation_id = uuid.uuid4()
        mock_db_operation_service.create_operation.return_value = mock_operation
        
        # Mock query function that raises exception
        query_func = Mock(side_effect=Exception("Query failed"))
        
        with pytest.raises(Exception, match="Query failed"):
            supabase_service._execute_and_log(
                operation_type, table_name, query_func, user_id, query_data
            )
        
        # Verify error logging
        mock_db_operation_service.update_operation_status.assert_called_once()
    
    def test_execute_and_log_without_user_id(self, supabase_service, mock_db_operation_service):
        """Test execute_and_log without user_id"""
        table_name = "users"
        operation_type = OperationType.CREATE
        query_data = {"name": "John"}
        
        # Mock operation creation
        mock_operation = Mock()
        mock_operation.operation_id = uuid.uuid4()
        mock_db_operation_service.create_operation.return_value = mock_operation
        
        # Mock query function
        query_func = Mock(return_value={"id": 1, "name": "John"})
        
        result = supabase_service._execute_and_log(
            operation_type, table_name, query_func, None, query_data
        )
        
        assert result == {"id": 1, "name": "John"}
        
        # Verify operation logging without user_id
        mock_db_operation_service.create_operation.assert_called_once_with(
            operation_type=operation_type,
            table_name=table_name,
            query_data=query_data,
            user_id=None
        )
    
    def test_execute_and_log_without_query_data(self, supabase_service, mock_db_operation_service):
        """Test execute_and_log without query_data"""
        table_name = "users"
        operation_type = OperationType.READ
        
        # Mock operation creation
        mock_operation = Mock()
        mock_operation.operation_id = uuid.uuid4()
        mock_db_operation_service.create_operation.return_value = mock_operation
        
        # Mock query function
        query_func = Mock(return_value=[{"id": 1, "name": "John"}])
        
        result = supabase_service._execute_and_log(
            operation_type, table_name, query_func, "user123", None
        )
        
        assert result == [{"id": 1, "name": "John"}]
        
        # Verify operation logging without query_data
        mock_db_operation_service.create_operation.assert_called_once_with(
            operation_type=operation_type,
            table_name=table_name,
            query_data=None,
            user_id="user123"
        )

