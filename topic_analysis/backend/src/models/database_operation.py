"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
DatabaseOperation Model

This module defines the DatabaseOperation data model for tracking all database
operations performed through the Supabase SDK.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid

class OperationType(str, Enum):
    """Database operation types."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    REAL_TIME = "real_time"

class OperationStatus(str, Enum):
    """Operation status values."""
    PENDING = "pending"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"

class DatabaseOperation(BaseModel):
    """
    DatabaseOperation model representing a database operation.
    
    This model tracks all database operations with their status, timing, and results.
    """
    
    # Primary identifier
    operation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique operation identifier")
    
    # Operation details
    client_id: str = Field(..., description="Reference to SupabaseClient")
    operation_type: OperationType = Field(..., description="Type of database operation")
    table_name: str = Field(..., description="Target database table")
    
    # Operation data
    query_data: Optional[Dict[str, Any]] = Field(default=None, description="Query parameters and data")
    response_data: Optional[Dict[str, Any]] = Field(default=None, description="Operation response data")
    
    # Status and timing
    status: OperationStatus = Field(default=OperationStatus.PENDING, description="Operation status")
    error_message: Optional[str] = Field(default=None, description="Error details if operation failed")
    execution_time_ms: Optional[int] = Field(default=None, ge=0, description="Operation execution time in milliseconds")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Operation start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Operation completion timestamp")
    
    # Context
    user_id: Optional[str] = Field(default=None, description="User who initiated the operation")
    request_id: Optional[str] = Field(default=None, description="Request identifier for tracing")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('table_name')
    def validate_table_name(cls, v):
        """Validate table name format."""
        if not v or not v.strip():
            raise ValueError('Table name must be non-empty')
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Table name must contain only alphanumeric characters, underscores, and hyphens')
        return v.strip()
    
    @validator('completed_at')
    def validate_completed_at(cls, v, values):
        """Validate completed_at timestamp."""
        if v and 'created_at' in values and v < values['created_at']:
            raise ValueError('completed_at must be after created_at')
        return v
    
    @validator('execution_time_ms')
    def validate_execution_time(cls, v):
        """Validate execution time."""
        if v is not None and v < 0:
            raise ValueError('execution_time_ms must be non-negative')
        return v
    
    def mark_success(self, response_data: Optional[Dict[str, Any]] = None, execution_time_ms: Optional[int] = None) -> None:
        """Mark operation as successful."""
        self.status = OperationStatus.SUCCESS
        self.completed_at = datetime.utcnow()
        self.response_data = response_data
        self.execution_time_ms = execution_time_ms
        self.error_message = None
    
    def mark_error(self, error_message: str, execution_time_ms: Optional[int] = None) -> None:
        """Mark operation as failed."""
        self.status = OperationStatus.ERROR
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        self.execution_time_ms = execution_time_ms
        self.response_data = None
    
    def mark_timeout(self, execution_time_ms: Optional[int] = None) -> None:
        """Mark operation as timed out."""
        self.status = OperationStatus.TIMEOUT
        self.completed_at = datetime.utcnow()
        self.error_message = "Operation timed out"
        self.execution_time_ms = execution_time_ms
        self.response_data = None
    
    def is_completed(self) -> bool:
        """Check if operation is completed."""
        return self.status in [OperationStatus.SUCCESS, OperationStatus.ERROR, OperationStatus.TIMEOUT]
    
    def is_successful(self) -> bool:
        """Check if operation was successful."""
        return self.status == OperationStatus.SUCCESS
    
    def is_failed(self) -> bool:
        """Check if operation failed."""
        return self.status in [OperationStatus.ERROR, OperationStatus.TIMEOUT]
    
    def get_duration(self) -> Optional[float]:
        """Get operation duration in seconds."""
        if self.completed_at and self.created_at:
            return (self.completed_at - self.created_at).total_seconds()
        return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring."""
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "table_name": self.table_name,
            "status": self.status,
            "execution_time_ms": self.execution_time_ms,
            "duration_seconds": self.get_duration(),
            "is_completed": self.is_completed(),
            "is_successful": self.is_successful(),
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
    
    def get_error_details(self) -> Optional[Dict[str, Any]]:
        """Get error details if operation failed."""
        if not self.is_failed():
            return None
        
        return {
            "operation_id": self.operation_id,
            "status": self.status,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
            "failed_at": self.completed_at.isoformat() if self.completed_at else None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "operation_id": self.operation_id,
            "client_id": self.client_id,
            "operation_type": self.operation_type,
            "table_name": self.table_name,
            "query_data": self.query_data,
            "response_data": self.response_data,
            "status": self.status,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "user_id": self.user_id,
            "request_id": self.request_id
        }
    
    @classmethod
    def create_read_operation(cls, client_id: str, table_name: str, query_data: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None, request_id: Optional[str] = None) -> 'DatabaseOperation':
        """Create a read operation."""
        return cls(
            client_id=client_id,
            operation_type=OperationType.READ,
            table_name=table_name,
            query_data=query_data,
            user_id=user_id,
            request_id=request_id
        )
    
    @classmethod
    def create_create_operation(cls, client_id: str, table_name: str, query_data: Dict[str, Any], user_id: Optional[str] = None, request_id: Optional[str] = None) -> 'DatabaseOperation':
        """Create a create operation."""
        return cls(
            client_id=client_id,
            operation_type=OperationType.CREATE,
            table_name=table_name,
            query_data=query_data,
            user_id=user_id,
            request_id=request_id
        )
    
    @classmethod
    def create_update_operation(cls, client_id: str, table_name: str, query_data: Dict[str, Any], user_id: Optional[str] = None, request_id: Optional[str] = None) -> 'DatabaseOperation':
        """Create an update operation."""
        return cls(
            client_id=client_id,
            operation_type=OperationType.UPDATE,
            table_name=table_name,
            query_data=query_data,
            user_id=user_id,
            request_id=request_id
        )
    
    @classmethod
    def create_delete_operation(cls, client_id: str, table_name: str, query_data: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None, request_id: Optional[str] = None) -> 'DatabaseOperation':
        """Create a delete operation."""
        return cls(
            client_id=client_id,
            operation_type=OperationType.DELETE,
            table_name=table_name,
            query_data=query_data,
            user_id=user_id,
            request_id=request_id
        )
    
    @classmethod
    def create_realtime_operation(cls, client_id: str, table_name: str, query_data: Dict[str, Any], user_id: Optional[str] = None, request_id: Optional[str] = None) -> 'DatabaseOperation':
        """Create a real-time operation."""
        return cls(
            client_id=client_id,
            operation_type=OperationType.REAL_TIME,
            table_name=table_name,
            query_data=query_data,
            user_id=user_id,
            request_id=request_id
        )
    
    def __str__(self) -> str:
        """String representation."""
        return f"DatabaseOperation(id={self.operation_id}, type={self.operation_type}, status={self.status})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"DatabaseOperation("
            f"operation_id='{self.operation_id}', "
            f"operation_type='{self.operation_type}', "
            f"table_name='{self.table_name}', "
            f"status='{self.status}', "
            f"execution_time_ms={self.execution_time_ms}"
            f")"
        )