"""
DatabaseOperationService

This module provides service layer for managing database operations,
including operation tracking, status monitoring, and performance metrics.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..core.logging import db_operation_logger
from ..core.error_handler import safe_database_operation
from ..models.database_operation import DatabaseOperation, OperationType, OperationStatus
from .supabase_service import SupabaseService

class DatabaseOperationService:
    """
    Service for managing database operations and their lifecycle.
    
    This service tracks operations, manages their status, and provides
    monitoring capabilities for database performance.
    """
    
    def __init__(self):
        """Initialize the database operation service."""
        self.supabase_service = SupabaseService()
        self.logger = db_operation_logger
    
    def create_operation(self, client_id: str, operation_type: OperationType, 
                       table_name: str, query_data: Optional[Dict[str, Any]] = None,
                       user_id: Optional[str] = None, request_id: Optional[str] = None) -> DatabaseOperation:
        """
        Create a new database operation record.
        
        Args:
            client_id: ID of the Supabase client
            operation_type: Type of operation
            table_name: Target table name
            query_data: Optional query data
            user_id: User performing the operation
            request_id: Request identifier for tracing
            
        Returns:
            DatabaseOperation instance
        """
        operation = DatabaseOperation(
            client_id=client_id,
            operation_type=operation_type,
            table_name=table_name,
            query_data=query_data,
            user_id=user_id,
            request_id=request_id
        )
        
        # Store operation in database
        operation_data = operation.to_dict()
        created_operation = self.supabase_service.create(
            table_name="database_operations",
            data=operation_data,
            user_id=user_id,
            request_id=request_id
        )
        
        return DatabaseOperation(**created_operation)
    
    def get_operation(self, operation_id: str) -> Optional[DatabaseOperation]:
        """
        Get a database operation by ID.
        
        Args:
            operation_id: Operation identifier
            
        Returns:
            DatabaseOperation instance or None
        """
        try:
            operations = self.supabase_service.read(
                table_name="database_operations",
                filters={"operation_id": operation_id},
                limit=1
            )
            
            if operations:
                return DatabaseOperation(**operations[0])
            return None
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id=operation_id,
                error_message=str(e),
                error_type="get_operation_error"
            )
            return None
    
    def update_operation_status(self, operation_id: str, status: OperationStatus,
                              response_data: Optional[Dict[str, Any]] = None,
                              error_message: Optional[str] = None,
                              execution_time_ms: Optional[int] = None) -> bool:
        """
        Update the status of a database operation.
        
        Args:
            operation_id: Operation identifier
            status: New status
            response_data: Optional response data
            error_message: Optional error message
            execution_time_ms: Optional execution time
            
        Returns:
            True if update was successful
        """
        try:
            update_data = {
                "status": status.value,
                "completed_at": datetime.utcnow().isoformat()
            }
            
            if response_data:
                update_data["response_data"] = response_data
            if error_message:
                update_data["error_message"] = error_message
            if execution_time_ms:
                update_data["execution_time_ms"] = execution_time_ms
            
            result = self.supabase_service.update(
                table_name="database_operations",
                record_id=operation_id,
                data=update_data
            )
            
            return result is not None
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id=operation_id,
                error_message=str(e),
                error_type="update_status_error"
            )
            return False
    
    def get_operations_by_user(self, user_id: str, limit: int = 100) -> List[DatabaseOperation]:
        """
        Get operations for a specific user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of operations to return
            
        Returns:
            List of DatabaseOperation instances
        """
        try:
            operations = self.supabase_service.read(
                table_name="database_operations",
                filters={"user_id": user_id},
                limit=limit,
                order_by="created_at",
                order_desc=True
            )
            
            return [DatabaseOperation(**op) for op in operations]
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id="user_operations",
                error_message=str(e),
                error_type="get_user_operations_error"
            )
            return []
    
    def get_operations_by_status(self, status: OperationStatus, limit: int = 100) -> List[DatabaseOperation]:
        """
        Get operations by status.
        
        Args:
            status: Operation status
            limit: Maximum number of operations to return
            
        Returns:
            List of DatabaseOperation instances
        """
        try:
            operations = self.supabase_service.read(
                table_name="database_operations",
                filters={"status": status.value},
                limit=limit,
                order_by="created_at",
                order_desc=True
            )
            
            return [DatabaseOperation(**op) for op in operations]
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id="status_operations",
                error_message=str(e),
                error_type="get_status_operations_error"
            )
            return []
    
    def get_operation_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get operation metrics for the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dict containing operation metrics
        """
        try:
            # Get operations from the last N hours
            cutoff_time = datetime.utcnow().replace(hour=datetime.utcnow().hour - hours)
            
            operations = self.supabase_service.read(
                table_name="database_operations",
                filters={"created_at": cutoff_time.isoformat()},
                limit=1000
            )
            
            if not operations:
                return {
                    "total_operations": 0,
                    "successful_operations": 0,
                    "failed_operations": 0,
                    "average_execution_time": 0,
                    "operations_by_type": {},
                    "operations_by_status": {}
                }
            
            # Calculate metrics
            total_operations = len(operations)
            successful_operations = len([op for op in operations if op.get("status") == "success"])
            failed_operations = len([op for op in operations if op.get("status") in ["error", "timeout"]])
            
            # Calculate average execution time
            execution_times = [op.get("execution_time_ms", 0) for op in operations if op.get("execution_time_ms")]
            average_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
            
            # Group by operation type
            operations_by_type = {}
            for op in operations:
                op_type = op.get("operation_type", "unknown")
                operations_by_type[op_type] = operations_by_type.get(op_type, 0) + 1
            
            # Group by status
            operations_by_status = {}
            for op in operations:
                status = op.get("status", "unknown")
                operations_by_status[status] = operations_by_status.get(status, 0) + 1
            
            return {
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "failed_operations": failed_operations,
                "success_rate": (successful_operations / total_operations) * 100 if total_operations > 0 else 0,
                "average_execution_time": average_execution_time,
                "operations_by_type": operations_by_type,
                "operations_by_status": operations_by_status
            }
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id="metrics",
                error_message=str(e),
                error_type="get_metrics_error"
            )
            return {}
    
    def cleanup_old_operations(self, days: int = 30) -> int:
        """
        Clean up old completed operations.
        
        Args:
            days: Number of days to keep operations
            
        Returns:
            Number of operations cleaned up
        """
        try:
            cutoff_time = datetime.utcnow().replace(day=datetime.utcnow().day - days)
            
            # Get old completed operations
            old_operations = self.supabase_service.read(
                table_name="database_operations",
                filters={
                    "status": ["success", "error", "timeout"],
                    "completed_at": cutoff_time.isoformat()
                },
                limit=1000
            )
            
            # Delete old operations
            deleted_count = 0
            for operation in old_operations:
                try:
                    self.supabase_service.delete(
                        table_name="database_operations",
                        record_id=operation["id"]
                    )
                    deleted_count += 1
                except Exception:
                    continue
            
            return deleted_count
            
        except Exception as e:
            self.logger.log_operation_error(
                operation_id="cleanup",
                error_message=str(e),
                error_type="cleanup_error"
            )
            return 0