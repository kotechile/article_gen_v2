"""
Structured Logging Configuration for Database Operations

This module provides structured logging for Supabase database operations,
including request/response logging, error tracking, and performance monitoring.
"""

import os
import sys
import json
import logging
import structlog
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structured logger
    """
    return structlog.get_logger(name)

class DatabaseOperationLogger:
    """
    Specialized logger for database operations with Supabase.
    """
    
    def __init__(self, logger_name: str = "database_operations"):
        self.logger = get_logger(logger_name)
        self.operation_count = 0
        
    def log_operation_start(self, 
                          operation_type: str, 
                          table_name: str, 
                          user_id: Optional[str] = None,
                          request_id: Optional[str] = None,
                          **kwargs) -> str:
        """
        Log the start of a database operation.
        
        Args:
            operation_type: Type of operation (create, read, update, delete, real_time)
            table_name: Target table name
            user_id: User performing the operation
            request_id: Request identifier for tracing
            **kwargs: Additional context
            
        Returns:
            Operation ID for tracking
        """
        self.operation_count += 1
        operation_id = f"op_{self.operation_count}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(
            "Database operation started",
            operation_id=operation_id,
            operation_type=operation_type,
            table_name=table_name,
            user_id=user_id,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )
        
        return operation_id
    
    def log_operation_success(self,
                            operation_id: str,
                            execution_time_ms: float,
                            rows_affected: Optional[int] = None,
                            **kwargs) -> None:
        """
        Log successful completion of a database operation.
        
        Args:
            operation_id: Operation identifier
            execution_time_ms: Execution time in milliseconds
            rows_affected: Number of rows affected
            **kwargs: Additional context
        """
        self.logger.info(
            "Database operation completed successfully",
            operation_id=operation_id,
            execution_time_ms=execution_time_ms,
            rows_affected=rows_affected,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )
    
    def log_operation_error(self,
                          operation_id: str,
                          error_message: str,
                          error_type: str,
                          execution_time_ms: Optional[float] = None,
                          **kwargs) -> None:
        """
        Log failed database operation.
        
        Args:
            operation_id: Operation identifier
            error_message: Error message
            error_type: Type of error (timeout, auth, connection, etc.)
            execution_time_ms: Execution time before failure
            **kwargs: Additional context
        """
        self.logger.error(
            "Database operation failed",
            operation_id=operation_id,
            error_message=error_message,
            error_type=error_type,
            execution_time_ms=execution_time_ms,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )
    
    def log_connection_health(self,
                            status: str,
                            execution_time_ms: Optional[float] = None,
                            error_count: int = 0,
                            **kwargs) -> None:
        """
        Log Supabase connection health status.
        
        Args:
            status: Health status (healthy, unhealthy)
            execution_time_ms: Health check execution time
            error_count: Current error count
            **kwargs: Additional context
        """
        self.logger.info(
            "Database health check",
            status=status,
            execution_time_ms=execution_time_ms,
            error_count=error_count,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )
    
    def log_real_time_event(self,
                          event_type: str,
                          table_name: str,
                          subscription_id: str,
                          **kwargs) -> None:
        """
        Log real-time subscription events.
        
        Args:
            event_type: Type of real-time event (INSERT, UPDATE, DELETE)
            table_name: Table being monitored
            subscription_id: Subscription identifier
            **kwargs: Additional context
        """
        self.logger.info(
            "Real-time event received",
            event_type=event_type,
            table_name=table_name,
            subscription_id=subscription_id,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )

class PerformanceLogger:
    """
    Logger for performance monitoring and optimization.
    """
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = get_logger(logger_name)
    
    def log_slow_query(self,
                      query: str,
                      execution_time_ms: float,
                      threshold_ms: float = 200.0,
                      **kwargs) -> None:
        """
        Log slow database queries for optimization.
        
        Args:
            query: Query that was slow
            execution_time_ms: Actual execution time
            threshold_ms: Performance threshold
            **kwargs: Additional context
        """
        self.logger.warning(
            "Slow query detected",
            query=query,
            execution_time_ms=execution_time_ms,
            threshold_ms=threshold_ms,
            performance_ratio=execution_time_ms / threshold_ms,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )
    
    def log_performance_metrics(self,
                              operation_type: str,
                              execution_time_ms: float,
                              rows_processed: int,
                              **kwargs) -> None:
        """
        Log performance metrics for analysis.
        
        Args:
            operation_type: Type of operation
            execution_time_ms: Execution time
            rows_processed: Number of rows processed
            **kwargs: Additional context
        """
        self.logger.info(
            "Performance metrics",
            operation_type=operation_type,
            execution_time_ms=execution_time_ms,
            rows_processed=rows_processed,
            throughput=rows_processed / (execution_time_ms / 1000) if execution_time_ms > 0 else 0,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )

# Global logger instances
db_operation_logger = DatabaseOperationLogger()
performance_logger = PerformanceLogger()

def setup_logging(log_level: str = "INFO", 
                 log_format: str = "json",
                 log_file: Optional[str] = None) -> None:
    """
    Configure application logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Log format (json, text)
        log_file: Optional log file path
    """
    # Set up basic logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        logging.getLogger().addHandler(file_handler)
    
    # Configure structlog based on format
    if log_format.lower() == "json":
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    else:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

# Initialize logging on module import
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_format=os.getenv("LOG_FORMAT", "json"),
    log_file=os.getenv("LOG_FILE")
)
