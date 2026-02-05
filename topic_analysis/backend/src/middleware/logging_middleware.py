"""
Logging Middleware

This module provides request/response logging middleware for database operations
and API endpoints with comprehensive monitoring and performance tracking.
"""

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import time
import json
from datetime import datetime
from uuid import uuid4

from ..core.logging import db_operation_logger, get_logger

logger = get_logger(__name__)

class LoggingMiddleware:
    """
    Middleware for logging requests and responses with performance monitoring.
    """
    
    def __init__(self):
        """Initialize the logging middleware."""
        self.logger = db_operation_logger
    
    async def log_request(self, request: Request) -> str:
        """
        Log incoming request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Request ID for tracking
        """
        request_id = str(uuid4())
        
        # Extract request information
        request_info = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("User-Agent"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Log request
        self.logger.log_operation_start(
            operation_type="api_request",
            table_name="api",
            user_id=request_info.get("headers", {}).get("Authorization", "anonymous"),
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request_info["client_ip"]
        )
        
        # Store request ID in request state
        request.state.request_id = request_id
        request.state.start_time = time.time()
        
        return request_id
    
    async def log_response(self, request: Request, response: Response) -> None:
        """
        Log outgoing response.
        
        Args:
            request: FastAPI request object
            response: FastAPI response object
        """
        try:
            request_id = getattr(request.state, 'request_id', 'unknown')
            start_time = getattr(request.state, 'start_time', time.time())
            execution_time = (time.time() - start_time) * 1000
            
            # Extract response information
            response_info = {
                "request_id": request_id,
                "status_code": response.status_code,
                "execution_time_ms": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Log response
            if response.status_code < 400:
                self.logger.log_operation_success(
                    operation_id=request_id,
                    execution_time_ms=execution_time,
                    status_code=response.status_code
                )
            else:
                self.logger.log_operation_error(
                    operation_id=request_id,
                    error_message=f"HTTP {response.status_code}",
                    error_type="http_error",
                    execution_time_ms=execution_time
                )
            
            # Log performance metrics
            if execution_time > 200:  # Log slow requests
                logger.warning(
                    "Slow request detected",
                    request_id=request_id,
                    method=request.method,
                    path=request.url.path,
                    execution_time_ms=execution_time,
                    status_code=response.status_code
                )
            
        except Exception as e:
            logger.error(
                "Failed to log response",
                error=str(e),
                request_id=getattr(request.state, 'request_id', 'unknown')
            )
    
    async def log_database_operation(self, operation_type: str, table_name: str,
                                    user_id: Optional[str] = None, request_id: Optional[str] = None,
                                    **kwargs) -> str:
        """
        Log database operation.
        
        Args:
            operation_type: Type of database operation
            table_name: Target table name
            user_id: User performing the operation
            request_id: Request identifier
            **kwargs: Additional context
            
        Returns:
            Operation ID
        """
        operation_id = self.logger.log_operation_start(
            operation_type=operation_type,
            table_name=table_name,
            user_id=user_id,
            request_id=request_id,
            **kwargs
        )
        
        return operation_id
    
    async def log_database_success(self, operation_id: str, execution_time_ms: float,
                                 rows_affected: Optional[int] = None, **kwargs) -> None:
        """
        Log successful database operation.
        
        Args:
            operation_id: Operation identifier
            execution_time_ms: Execution time in milliseconds
            rows_affected: Number of rows affected
            **kwargs: Additional context
        """
        self.logger.log_operation_success(
            operation_id=operation_id,
            execution_time_ms=execution_time_ms,
            rows_affected=rows_affected,
            **kwargs
        )
    
    async def log_database_error(self, operation_id: str, error_message: str,
                                error_type: str, execution_time_ms: Optional[float] = None,
                                **kwargs) -> None:
        """
        Log failed database operation.
        
        Args:
            operation_id: Operation identifier
            error_message: Error message
            error_type: Type of error
            execution_time_ms: Execution time before failure
            **kwargs: Additional context
        """
        self.logger.log_operation_error(
            operation_id=operation_id,
            error_message=error_message,
            error_type=error_type,
            execution_time_ms=execution_time_ms,
            **kwargs
        )
    
    async def log_real_time_event(self, event_type: str, table_name: str,
                                subscription_id: str, **kwargs) -> None:
        """
        Log real-time event.
        
        Args:
            event_type: Type of real-time event
            table_name: Table being monitored
            subscription_id: Subscription identifier
            **kwargs: Additional context
        """
        self.logger.log_real_time_event(
            event_type=event_type,
            table_name=table_name,
            subscription_id=subscription_id,
            **kwargs
        )
    
    async def log_connection_health(self, status: str, execution_time_ms: Optional[float] = None,
                                  error_count: int = 0, **kwargs) -> None:
        """
        Log connection health status.
        
        Args:
            status: Health status
            execution_time_ms: Health check execution time
            error_count: Current error count
            **kwargs: Additional context
        """
        self.logger.log_connection_health(
            status=status,
            execution_time_ms=execution_time_ms,
            error_count=error_count,
            **kwargs
        )
    
    def get_request_metrics(self, request: Request) -> Dict[str, Any]:
        """
        Get request metrics for monitoring.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Dict containing request metrics
        """
        request_id = getattr(request.state, 'request_id', 'unknown')
        start_time = getattr(request.state, 'start_time', time.time())
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "execution_time_ms": execution_time,
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("User-Agent"),
            "timestamp": datetime.utcnow().isoformat()
        }

# Global middleware instance
logging_middleware = LoggingMiddleware()

async def log_request_middleware(request: Request, call_next):
    """
    FastAPI middleware for logging requests.
    
    Args:
        request: FastAPI request object
        call_next: Next middleware/handler
        
    Returns:
        Response object
    """
    # Log request
    request_id = await logging_middleware.log_request(request)
    
    try:
        # Process request
        response = await call_next(request)
        
        # Log response
        await logging_middleware.log_response(request, response)
        
        return response
        
    except Exception as e:
        # Log error
        await logging_middleware.log_database_error(
            operation_id=request_id,
            error_message=str(e),
            error_type="request_processing_error"
        )
        
        # Return error response
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": str(e),
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
