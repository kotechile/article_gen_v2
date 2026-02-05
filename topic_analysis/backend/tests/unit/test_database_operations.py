"""
Unit tests for database operations
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import uuid

from src.models.database_operation import DatabaseOperation, OperationType, OperationStatus
from src.services.database_operation_service import DatabaseOperationService

class TestDatabaseOperation:
    """Test cases for DatabaseOperation model"""
    
    def test_operation_creation(self):
        """Test creating DatabaseOperation instance"""
        operation = DatabaseOperation(
            operation_type=OperationType.CREATE,
            table_name="users",
            query_data={"email": "test@example.com"},
            user_id=str(uuid.uuid4())
        )
        
        assert operation.operation_type == OperationType.CREATE
        assert operation.table_name == "users"
        assert operation.query_data == {"email": "test@example.com"}
        assert operation.status == OperationStatus.PENDING
        assert operation.operation_id is not None
    
    def test_to_dict(self):
        """Test converting operation to dictionary"""
        user_id = str(uuid.uuid4())
        operation = DatabaseOperation(
            operation_type=OperationType.CREATE,
            table_name="users",
            query_data={"email": "test@example.com"},
            user_id=user_id
        )
        
        data = operation.to_dict()
        
        assert data["operation_type"] == "create"
        assert data["table_name"] == "users"
        assert data["query_data"] == {"email": "test@example.com"}
        assert data["status"] == "pending"
        assert data["user_id"] == user_id
        assert "operation_id" in data
        assert "created_at" in data
    
    def test_mark_success(self):
        """Test marking operation as successful"""
        operation = DatabaseOperation(
            operation_type=OperationType.CREATE,
            table_name="users"
        )
        
        operation.mark_success(45.2)
        
        assert operation.status == OperationStatus.SUCCESS
        assert operation.execution_time == 45.2
        assert operation.is_successful() is True
        assert operation.is_failed() is False
    
    def test_mark_failed(self):
        """Test marking operation as failed"""
        operation = DatabaseOperation(
            operation_type=OperationType.CREATE,
            table_name="users"
        )
        
        operation.mark_failed("Connection timeout", 30.0)
        
        assert operation.status == OperationStatus.FAILED
        assert operation.error_message == "Connection timeout"
        assert operation.execution_time == 30.0
        assert operation.is_successful() is False
        assert operation.is_failed() is True
    
    def test_mark_timeout(self):
        """Test marking operation as timed out"""
        operation = DatabaseOperation(
            operation_type=OperationType.CREATE,
            table_name="users"
        )
        
        operation.mark_timeout(60.0)
        
        assert operation.status == OperationStatus.TIMEOUT
        assert operation.execution_time == 60.0
        assert operation.is_timeout() is True
        assert operation.is_successful() is False
    
    def test_get_duration_seconds(self):
        """Test getting duration in seconds"""
        operation = DatabaseOperation(
            operation_type=OperationType.CREATE,
            table_name="users"
        )
        
        operation.execution_time = 1500.0  # 1.5 seconds in milliseconds
        
        assert operation.get_duration_seconds() == 1.5
        assert operation.get_duration_ms() == 1500.0
    
    def test_get_duration_no_execution_time(self):
        """Test getting duration when no execution time"""
        operation = DatabaseOperation(
            operation_type=OperationType.CREATE,
            table_name="users"
        )
        
        assert operation.get_duration_seconds() == 0.0
        assert operation.get_duration_ms() == 0.0

class TestDatabaseOperationService:
    """Test cases for DatabaseOperationService"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock()
        session.add = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        session.query = Mock()
        return session
    
    @pytest.fixture
    def operation_service(self, mock_db_session):
        """DatabaseOperationService instance with mocked session"""
        return DatabaseOperationService(mock_db_session)
    
    def test_create_operation(self, operation_service, mock_db_session):
        """Test creating a database operation"""
        operation = operation_service.create_operation(
            operation_type=OperationType.CREATE,
            table_name="users",
            query_data={"email": "test@example.com"},
            user_id="user123"
        )
        
        assert operation.operation_type == OperationType.CREATE
        assert operation.table_name == "users"
        assert operation.query_data == {"email": "test@example.com"}
        assert operation.user_id == "user123"
        assert operation.status == OperationStatus.PENDING
        
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    def test_create_operation_exception(self, operation_service, mock_db_session):
        """Test creating operation with exception"""
        mock_db_session.commit.side_effect = Exception("Database error")
        
        with pytest.raises(Exception):
            operation_service.create_operation(
                operation_type=OperationType.CREATE,
                table_name="users"
            )
        
        mock_db_session.rollback.assert_called_once()
    
    def test_mark_operation_success(self, operation_service, mock_db_session):
        """Test marking operation as successful"""
        operation_id = str(uuid.uuid4())
        mock_operation = Mock()
        mock_operation.mark_success = Mock()
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_operation
        
        result = operation_service.mark_operation_success(operation_id, 45.2)
        
        assert result is True
        mock_operation.mark_success.assert_called_once_with(45.2)
        mock_db_session.commit.assert_called_once()
    
    def test_mark_operation_success_not_found(self, operation_service, mock_db_session):
        """Test marking non-existent operation as successful"""
        operation_id = str(uuid.uuid4())
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        result = operation_service.mark_operation_success(operation_id, 45.2)
        
        assert result is False
        mock_db_session.commit.assert_not_called()
    
    def test_mark_operation_failed(self, operation_service, mock_db_session):
        """Test marking operation as failed"""
        operation_id = str(uuid.uuid4())
        mock_operation = Mock()
        mock_operation.mark_failed = Mock()
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_operation
        
        result = operation_service.mark_operation_failed(operation_id, "Connection error", 30.0)
        
        assert result is True
        mock_operation.mark_failed.assert_called_once_with("Connection error", 30.0)
        mock_db_session.commit.assert_called_once()
    
    def test_mark_operation_timeout(self, operation_service, mock_db_session):
        """Test marking operation as timed out"""
        operation_id = str(uuid.uuid4())
        mock_operation = Mock()
        mock_operation.mark_timeout = Mock()
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_operation
        
        result = operation_service.mark_operation_timeout(operation_id, 60.0)
        
        assert result is True
        mock_operation.mark_timeout.assert_called_once_with(60.0)
        mock_db_session.commit.assert_called_once()
    
    def test_get_operation(self, operation_service, mock_db_session):
        """Test getting operation by ID"""
        operation_id = str(uuid.uuid4())
        mock_operation = Mock()
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_operation
        
        result = operation_service.get_operation(operation_id)
        
        assert result == mock_operation
    
    def test_get_operations_with_filters(self, operation_service, mock_db_session):
        """Test getting operations with filters"""
        mock_operations = [Mock(), Mock()]
        mock_query = Mock()
        mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = mock_operations
        mock_db_session.query.return_value = mock_query
        
        result = operation_service.get_operations(
            limit=10,
            offset=0,
            operation_type=OperationType.CREATE,
            status=OperationStatus.SUCCESS,
            table_name="users",
            user_id="user123"
        )
        
        assert result == mock_operations
        mock_query.filter.assert_called()
        mock_query.order_by.assert_called_once()
        mock_query.offset.assert_called_once_with(0)
        mock_query.limit.assert_called_once_with(10)
    
    def test_get_operation_count(self, operation_service, mock_db_session):
        """Test getting operation count"""
        mock_db_session.query.return_value.count.return_value = 42
        
        result = operation_service.get_operation_count(
            operation_type=OperationType.CREATE,
            status=OperationStatus.SUCCESS
        )
        
        assert result == 42
    
    def test_get_failed_operations(self, operation_service, mock_db_session):
        """Test getting failed operations"""
        mock_operations = [Mock(), Mock()]
        mock_query = Mock()
        mock_query.filter.return_value.order_by.return_value.all.return_value = mock_operations
        mock_db_session.query.return_value = mock_query
        
        result = operation_service.get_failed_operations(hours=24)
        
        assert result == mock_operations
        mock_query.filter.assert_called()
        mock_query.order_by.assert_called_once()
    
    def test_get_slow_operations(self, operation_service, mock_db_session):
        """Test getting slow operations"""
        mock_operations = [Mock(), Mock()]
        mock_query = Mock()
        mock_query.filter.return_value.order_by.return_value.all.return_value = mock_operations
        mock_db_session.query.return_value = mock_query
        
        result = operation_service.get_slow_operations(threshold_ms=1000)
        
        assert result == mock_operations
        mock_query.filter.assert_called()
        mock_query.order_by.assert_called_once()
    
    def test_get_operation_stats(self, operation_service, mock_db_session):
        """Test getting operation statistics"""
        # Mock different query results
        mock_db_session.query.return_value.filter.return_value.count.side_effect = [100, 90, 8, 2]
        
        # Mock average execution time query
        mock_avg_query = Mock()
        mock_avg_query.with_entities.return_value.all.return_value = [(100.0,), (200.0,), (300.0,)]
        mock_db_session.query.return_value = mock_avg_query
        
        result = operation_service.get_operation_stats(hours=24)
        
        assert result["total_operations"] == 100
        assert result["successful_operations"] == 90
        assert result["failed_operations"] == 8
        assert result["timeout_operations"] == 2
        assert result["success_rate"] == 90.0
        assert result["average_execution_time_ms"] == 200.0
        assert result["time_period_hours"] == 24
    
    def test_cleanup_old_operations(self, operation_service, mock_db_session):
        """Test cleaning up old operations"""
        mock_db_session.query.return_value.filter.return_value.delete.return_value = 25
        
        result = operation_service.cleanup_old_operations(days=30)
        
        assert result == 25
        mock_db_session.commit.assert_called_once()
    
    def test_cleanup_old_operations_exception(self, operation_service, mock_db_session):
        """Test cleanup with exception"""
        mock_db_session.commit.side_effect = Exception("Database error")
        
        result = operation_service.cleanup_old_operations(days=30)
        
        assert result == 0
        mock_db_session.rollback.assert_called_once()

