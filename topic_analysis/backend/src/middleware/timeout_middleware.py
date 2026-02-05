"""
Timeout Middleware

This module provides timeout handling middleware for database queries
with a 60-second limit and proper error handling.
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import asyncio
import time
from datetime import datetime

from ..core.error_handler import DatabaseTimeoutError
from ..core.logging import db_operation_logger, get_logger

logger = get_logger(__name__)

class TimeoutMiddleware:
    """
    Middleware for handling request timeouts with a 60-second limit.
    """
    
    def __init__(self, timeout_seconds: int = 60):
        """
        Initialize the timeout middleware.
        
        Args:
            timeout_seconds: Maximum request timeout in seconds
        """
        self.timeout_seconds = timeout_seconds
        self.logger = db_operation_logger
    
    async def handle_timeout(self, request: Request, call_next) -> JSONResponse:
        """
        Handle request timeout with proper error handling.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/handler
            
        Returns:
            Response object
        """
        request_id = getattr(request.state, 'request_id', 'unknown')
        start_time = time.time()
        
        try:
            # Create timeout task
            timeout_task = asyncio.create_task(
                self._timeout_handler(request_id, self.timeout_seconds)
            )
            
            # Create request task
            request_task = asyncio.create_task(call_next(request))
            
            # Wait for either completion or timeout
            done, pending = await asyncio.wait(
                [timeout_task, request_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Check if timeout occurred
            if timeout_task in done:
                # Timeout occurred
                if request_task in pending:
                    request_task.cancel()
                    try:
                        await request_task
                    except asyncio.CancelledError:
                        pass
                
                execution_time = (time.time() - start_time) * 1000
                
                # Log timeout
                self.logger.log_operation_error(
                    operation_id=request_id,
                    error_message=f"Request timed out after {self.timeout_seconds} seconds",
                    error_type="timeout",
                    execution_time_ms=execution_time
                )
                
                return JSONResponse(
                    status_code=408,
                    content={
                        "error": "Request Timeout",
                        "message": f"Request timed out after {self.timeout_seconds} seconds",
                        "request_id": request_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "details": {
                            "timeout_seconds": self.timeout_seconds,
                            "execution_time_ms": execution_time
                        }
                    }
                )
            
            # Request completed successfully
            if request_task in done:
                if timeout_task in pending:
                    timeout_task.cancel()
                    try:
                        await timeout_task
                    except asyncio.CancelledError:
                        pass
                
                response = request_task.result()
                execution_time = (time.time() - start_time) * 1000
                
                # Log successful completion
                self.logger.log_operation_success(
                    operation_id=request_id,
                    execution_time_ms=execution_time
                )
                
                return response
            
            # This should not happen
            raise Exception("Unexpected timeout handling result")
            
        except asyncio.CancelledError:
            # Request was cancelled
            execution_time = (time.time() - start_time) * 1000
            
            self.logger.log_operation_error(
                operation_id=request_id,
                error_message="Request was cancelled",
                error_type="cancelled",
                execution_time_ms=execution_time
            )
            
            return JSONResponse(
                status_code=408,
                content={
                    "error": "Request Cancelled",
                    "message": "Request was cancelled due to timeout",
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            self.logger.log_operation_error(
                operation_id=request_id,
                error_message=str(e),
                error_type="timeout_error",
                execution_time_ms=execution_time
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "Timeout handling failed",
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    async def _timeout_handler(self, request_id: str, timeout_seconds: int) -> None:
        """
        Handle timeout by waiting for the specified duration.
        
        Args:
            request_id: Request identifier
            timeout_seconds: Timeout duration in seconds
        """
        await asyncio.sleep(timeout_seconds)
        logger.warning(
            "Request timeout reached",
            request_id=request_id,
            timeout_seconds=timeout_seconds
        )
    
    def get_timeout_metrics(self, request: Request) -> Dict[str, Any]:
        """
        Get timeout metrics for monitoring.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Dict containing timeout metrics
        """
        request_id = getattr(request.state, 'request_id', 'unknown')
        start_time = getattr(request.state, 'start_time', time.time())
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "request_id": request_id,
            "timeout_seconds": self.timeout_seconds,
            "execution_time_ms": execution_time,
            "remaining_time_ms": max(0, (self.timeout_seconds * 1000) - execution_time),
            "is_near_timeout": execution_time > (self.timeout_seconds * 1000 * 0.8)  # 80% of timeout
        }
    
    def check_timeout_status(self, request: Request) -> Dict[str, Any]:
        """
        Check if request is approaching timeout.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Dict containing timeout status
        """
        start_time = getattr(request.state, 'start_time', time.time())
        execution_time = (time.time() - start_time) * 1000
        remaining_time = (self.timeout_seconds * 1000) - execution_time
        
        return {
            "execution_time_ms": execution_time,
            "remaining_time_ms": max(0, remaining_time),
            "timeout_percentage": min(100, (execution_time / (self.timeout_seconds * 1000)) * 100),
            "is_near_timeout": execution_time > (self.timeout_seconds * 1000 * 0.8),
            "is_critical": execution_time > (self.timeout_seconds * 1000 * 0.95)
        }

# Global middleware instance
timeout_middleware = TimeoutMiddleware(timeout_seconds=60)

async def timeout_handler_middleware(request: Request, call_next):
    """
    FastAPI middleware for timeout handling.
    
    Args:
        request: FastAPI request object
        call_next: Next middleware/handler
        
    Returns:
        Response object
    """
    return await timeout_middleware.handle_timeout(request, call_next)
