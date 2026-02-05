"""
Unit tests for migration service
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import uuid

from src.services.migration_service import MigrationService
from src.database.migration import MigrationType, MigrationStatus

class TestMigrationService:
    """Test cases for MigrationService"""
    
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
    def migration_service(self, mock_db_session):
        """MigrationService instance with mocked session"""
        with patch('src.services.migration_service.DatabaseMigration') as mock_migration_class:
            mock_migration_manager = Mock()
            mock_migration_class.return_value = mock_migration_manager
            service = MigrationService(mock_db_session)
            service.migration_manager = mock_migration_manager
            return service
    
    def test_start_migration(self, migration_service, mock_db_session):
        """Test starting a migration"""
        migration_info = {
            "migration_id": "migration_123",
            "status": "started",
            "migration_type": "full",
            "tables": ["users", "posts"],
            "dry_run": False
        }
        
        migration_service.migration_manager.start_migration.return_value = migration_info
        
        result = migration_service.start_migration(
            migration_type=MigrationType.FULL,
            tables=["users", "posts"],
            dry_run=False
        )
        
        assert result == migration_info
        migration_service.migration_manager.start_migration.assert_called_once_with(
            migration_type=MigrationType.FULL,
            tables=["users", "posts"],
            dry_run=False
        )
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    def test_get_migration_status(self, migration_service):
        """Test getting migration status"""
        migration_id = "migration_123"
        status_info = {
            "migration_id": migration_id,
            "status": "in_progress",
            "progress_percentage": 50
        }
        
        migration_service.migration_manager.get_migration_status.return_value = status_info
        
        result = migration_service.get_migration_status(migration_id)
        
        assert result == status_info
        migration_service.migration_manager.get_migration_status.assert_called_once_with(migration_id)
    
    def test_get_migration_status_not_found(self, migration_service):
        """Test getting status for non-existent migration"""
        migration_id = "nonexistent"
        migration_service.migration_manager.get_migration_status.return_value = None
        
        result = migration_service.get_migration_status(migration_id)
        
        assert result is None
    
    def test_execute_migration(self, migration_service, mock_db_session):
        """Test executing a migration"""
        migration_id = "migration_123"
        execution_result = {
            "migration_id": migration_id,
            "status": "completed",
            "progress_percentage": 100
        }
        
        migration_service.migration_manager.execute_migration.return_value = execution_result
        
        result = migration_service.execute_migration(migration_id)
        
        assert result == execution_result
        migration_service.migration_manager.execute_migration.assert_called_once_with(migration_id)
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    def test_execute_migration_failure(self, migration_service, mock_db_session):
        """Test executing migration with failure"""
        migration_id = "migration_123"
        migration_service.migration_manager.execute_migration.side_effect = Exception("Migration failed")
        
        with pytest.raises(Exception, match="Migration failed"):
            migration_service.execute_migration(migration_id)
        
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    def test_rollback_migration(self, migration_service, mock_db_session):
        """Test rolling back a migration"""
        migration_id = "migration_123"
        rollback_result = {
            "migration_id": migration_id,
            "status": "completed",
            "rollback": True
        }
        
        migration_service.migration_manager.rollback_migration.return_value = rollback_result
        
        result = migration_service.rollback_migration(migration_id)
        
        assert result == rollback_result
        migration_service.migration_manager.rollback_migration.assert_called_once_with(migration_id)
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    def test_get_migration_history(self, migration_service, mock_db_session):
        """Test getting migration history"""
        mock_operations = [
            Mock(query_data={"migration_id": "migration_1", "status": "completed"}),
            Mock(query_data={"migration_id": "migration_2", "status": "failed"})
        ]
        
        mock_query = Mock()
        mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = mock_operations
        mock_db_session.query.return_value = mock_query
        
        result = migration_service.get_migration_history(limit=10, offset=0)
        
        assert len(result) == 2
        assert result[0]["migration_id"] == "migration_1"
        assert result[1]["migration_id"] == "migration_2"
    
    def test_validate_migration_data_not_completed(self, migration_service):
        """Test validating migration that's not completed"""
        migration_id = "migration_123"
        status = {
            "migration_id": migration_id,
            "status": "in_progress"
        }
        
        migration_service.migration_manager.get_migration_status.return_value = status
        
        result = migration_service.validate_migration_data(migration_id)
        
        assert result["valid"] is False
        assert "Migration not completed" in result["error"]
    
    def test_validate_migration_data_dry_run(self, migration_service):
        """Test validating dry run migration"""
        migration_id = "migration_123"
        status = {
            "migration_id": migration_id,
            "status": "completed",
            "dry_run": True
        }
        
        migration_service.migration_manager.get_migration_status.return_value = status
        
        result = migration_service.validate_migration_data(migration_id)
        
        assert result["valid"] is True
        assert result["checks"][0]["check"] == "dry_run"
        assert result["checks"][0]["status"] == "skipped"
    
    @patch('src.services.migration_service.text')
    def test_validate_migration_data_full_validation(self, mock_text, migration_service):
        """Test full migration validation"""
        migration_id = "migration_123"
        status = {
            "migration_id": migration_id,
            "status": "completed",
            "dry_run": False,
            "tables": ["users"]
        }
        
        migration_service.migration_manager.get_migration_status.return_value = status
        
        # Mock database query results
        mock_result = Mock()
        mock_result.scalar.return_value = 100  # Row count
        mock_db_session = migration_service.db
        mock_db_session.execute.return_value = mock_result
        
        result = migration_service.validate_migration_data(migration_id)
        
        assert result["valid"] is True
        assert result["migration_id"] == migration_id
        assert len(result["checks"]) > 0
    
    def test_validate_migration_data_exception(self, migration_service):
        """Test validation with exception"""
        migration_id = "migration_123"
        migration_service.migration_manager.get_migration_status.side_effect = Exception("Database error")
        
        result = migration_service.validate_migration_data(migration_id)
        
        assert result["valid"] is False
        assert "Database error" in result["error"]
    
    def test_get_migration_statistics(self, migration_service, mock_db_session):
        """Test getting migration statistics"""
        # Mock query results for different operation types
        mock_db_session.query.return_value.filter.return_value.count.side_effect = [10, 8, 2]
        
        result = migration_service.get_migration_statistics()
        
        assert result["total_migrations"] == 10
        assert result["successful_migrations"] == 8
        assert result["failed_migrations"] == 2
        assert result["success_rate"] == 80.0
    
    def test_get_migration_statistics_exception(self, migration_service, mock_db_session):
        """Test getting statistics with exception"""
        mock_db_session.query.return_value.filter.return_value.count.side_effect = Exception("Database error")
        
        result = migration_service.get_migration_statistics()
        
        assert result["total_migrations"] == 0
        assert result["successful_migrations"] == 0
        assert result["failed_migrations"] == 0
        assert result["success_rate"] == 0
    
    def test_log_migration_operation(self, migration_service, mock_db_session):
        """Test logging migration operation"""
        from src.models.database_operation import OperationType, OperationStatus
        
        migration_service._log_migration_operation(
            operation_type=OperationType.CREATE,
            table_name="migrations",
            query_data={"migration_id": "test_123"},
            status=OperationStatus.SUCCESS
        )
        
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    def test_log_migration_operation_exception(self, migration_service, mock_db_session):
        """Test logging migration operation with exception"""
        from src.models.database_operation import OperationType, OperationStatus
        
        mock_db_session.commit.side_effect = Exception("Database error")
        
        migration_service._log_migration_operation(
            operation_type=OperationType.CREATE,
            table_name="migrations",
            query_data={"migration_id": "test_123"},
            status=OperationStatus.SUCCESS
        )
        
        mock_db_session.rollback.assert_called_once()

