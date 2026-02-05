"""
Error Handling Utilities for Supabase Integration

This module provides comprehensive error handling for Supabase database operations,
including timeout management, authentication failures, and service unavailability.
"""

import time
import logging
from typing import Any, Dict, Optional, Callable, Type, Union
from datetime import datetime, timedelta
from functools import wraps
from enum import Enum

from supabase import PostgrestAPIError
from .logging import get_logger

logger = get_logger(__name__)

class ErrorType(Enum):
    """Types of database errors."""
    AUTHENTICATION = "authentication"
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    VALIDATION = "validation"
    PERMISSION = "permission"
    UNKNOWN = "unknown"

class SupabaseError(Exception):
    """Base exception for Supabase-related errors."""
    
    def __init__(self, message: str, error_type: ErrorType, status_code: Optional[int] = None):
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        super().__init__(self.message)

class DatabaseTimeoutError(SupabaseError):
    """Raised when database operations exceed timeout limit."""
    
    def __init__(self, message: str = "Database operation timed out after 60 seconds"):
        super().__init__(message, ErrorType.TIMEOUT, 408)

class DatabaseAuthenticationError(SupabaseError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed - redirect to login"):
        super().__init__(message, ErrorType.AUTHENTICATION, 401)

class DatabaseConnectionError(SupabaseError):
    """Raised when database connection fails."""
    
    def __init__(self, message: str = "Database service unavailable - fail fast with clear error"):
        super().__init__(message, ErrorType.CONNECTION, 503)

class DatabaseRateLimitError(SupabaseError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded - please retry later"):
        super().__init__(message, ErrorType.RATE_LIMIT, 429)

class ErrorHandler:
    """
    Centralized error handling for Supabase operations.
    """
    
    def __init__(self, max_retries: int = 3, timeout_seconds: int = 60):
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.error_counts: Dict[str, int] = {}
        
    def handle_supabase_error(self, error: Exception, operation_id: str) -> SupabaseError:
        """
        Convert Supabase exceptions to our custom error types.
        
        Args:
            error: Original Supabase exception
            operation_id: Operation identifier for logging
            
        Returns:
            Appropriate SupabaseError instance
        """
        if isinstance(error, PostgrestAPIError):
            return self._handle_api_error(error, operation_id)
        else:
            logger.error(
                "Unexpected error in database operation",
                operation_id=operation_id,
                error_type="unknown",
                error_message=str(error),
                timestamp=datetime.utcnow().isoformat()
            )
            return SupabaseError(
                f"Unexpected error: {str(error)}",
                ErrorType.UNKNOWN
            )
    
    def _handle_api_error(self, error: PostgrestAPIError, operation_id: str) -> SupabaseError:
        """Handle Supabase API errors."""
        status_code = getattr(error, 'status_code', None)
        
        if status_code == 401:
            logger.error(
                "Authentication failed",
                operation_id=operation_id,
                error_type="authentication",
                status_code=status_code,
                timestamp=datetime.utcnow().isoformat()
            )
            return DatabaseAuthenticationError()
            
        elif status_code == 408:
            logger.error(
                "Request timeout",
                operation_id=operation_id,
                error_type="timeout",
                status_code=status_code,
                timestamp=datetime.utcnow().isoformat()
            )
            return DatabaseTimeoutError()
            
        elif status_code == 429:
            logger.error(
                "Rate limit exceeded",
                operation_id=operation_id,
                error_type="rate_limit",
                status_code=status_code,
                timestamp=datetime.utcnow().isoformat()
            )
            return DatabaseRateLimitError()
            
        elif status_code in [500, 502, 503, 504]:
            logger.error(
                "Service unavailable",
                operation_id=operation_id,
                error_type="connection",
                status_code=status_code,
                timestamp=datetime.utcnow().isoformat()
            )
            return DatabaseConnectionError()
            
        else:
            logger.error(
                "API error",
                operation_id=operation_id,
                error_type="unknown",
                status_code=status_code,
                error_message=str(error),
                timestamp=datetime.utcnow().isoformat()
            )
            return SupabaseError(
                f"API error: {str(error)}",
                ErrorType.UNKNOWN,
                status_code
            )

def timeout_handler(timeout_seconds: int = 60):
    """
    Decorator to handle operation timeouts.
    
    Args:
        timeout_seconds: Maximum execution time in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                if execution_time > timeout_seconds:
                    raise DatabaseTimeoutError(
                        f"Operation timed out after {execution_time:.2f} seconds"
                    )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                if execution_time > timeout_seconds:
                    raise DatabaseTimeoutError(
                        f"Operation timed out after {execution_time:.2f} seconds"
                    )
                else:
                    raise
                    
        return wrapper
    return decorator

def retry_on_failure(max_retries: int = 3, 
                    backoff_factor: float = 1.0,
                    retry_on: tuple = (DatabaseConnectionError, DatabaseRateLimitError)):
    """
    Decorator to retry operations on specific failures.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff factor
        retry_on: Exception types to retry on
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except retry_on as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        wait_time = backoff_factor * (2 ** attempt)
                        logger.warning(
                            "Retrying operation after failure",
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            wait_time=wait_time,
                            error_type=type(e).__name__,
                            timestamp=datetime.utcnow().isoformat()
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            "Max retries exceeded",
                            max_retries=max_retries,
                            error_type=type(e).__name__,
                            timestamp=datetime.utcnow().isoformat()
                        )
                        raise
                        
                except Exception as e:
                    # Don't retry on non-retryable errors
                    raise
                    
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator

def safe_database_operation(operation_func: Callable, 
                          operation_id: str,
                          *args, 
                          **kwargs) -> Any:
    """
    Safely execute database operations with comprehensive error handling.
    
    Args:
        operation_func: Function to execute
        operation_id: Operation identifier for logging
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
        
    Raises:
        SupabaseError: For database-related errors
    """
    error_handler = ErrorHandler()
    start_time = time.time()
    
    try:
        result = operation_func(*args, **kwargs)
        execution_time = (time.time() - start_time) * 1000
        
        logger.info(
            "Database operation completed successfully",
            operation_id=operation_id,
            execution_time_ms=execution_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
        return result
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        
        # Convert to our error types
        if isinstance(e, SupabaseError):
            raise
        else:
            supabase_error = error_handler.handle_supabase_error(e, operation_id)
            logger.error(
                "Database operation failed",
                operation_id=operation_id,
                execution_time_ms=execution_time,
                error_type=supabase_error.error_type.value,
                error_message=supabase_error.message,
                timestamp=datetime.utcnow().isoformat()
            )
            raise supabase_error

def handle_authentication_error(error: Exception) -> None:
    """
    Handle authentication errors by redirecting to login.
    
    Args:
        error: Authentication error
    """
    logger.error(
        "Authentication error - redirecting to login",
        error_type="authentication",
        error_message=str(error),
        timestamp=datetime.utcnow().isoformat()
    )
    # In a real application, this would trigger a redirect
    # For now, we just log the error
    raise DatabaseAuthenticationError()

def handle_service_unavailable(error: Exception) -> None:
    """
    Handle service unavailability with fail-fast behavior.
    
    Args:
        error: Service unavailability error
    """
    logger.error(
        "Service unavailable - failing fast",
        error_type="connection",
        error_message=str(error),
        timestamp=datetime.utcnow().isoformat()
    )
    raise DatabaseConnectionError()

def handle_timeout_error(error: Exception) -> None:
    """
    Handle timeout errors with clear messaging.
    
    Args:
        error: Timeout error
    """
    logger.error(
        "Operation timeout - failing with clear error",
        error_type="timeout",
        error_message=str(error),
        timestamp=datetime.utcnow().isoformat()
    )
    raise DatabaseTimeoutError()

# Global error handler instance
error_handler = ErrorHandler()
