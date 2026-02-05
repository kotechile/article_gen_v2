"""
Error Handling Middleware

This module provides comprehensive error handling middleware for Supabase exceptions,
including timeout management, authentication failures, and service unavailability.
"""

from fastapi import Request, HTTPException, Response
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import time
from datetime import datetime

from ..core.error_handler import (
    DatabaseTimeoutError,
    DatabaseAuthenticationError,
    DatabaseConnectionError,
    DatabaseRateLimitError,
    SupabaseError
)
from ..core.logging import db_operation_logger, get_logger

logger = get_logger(__name__)

class ErrorHandlingMiddleware:
    """
    Middleware for handling Supabase exceptions and providing consistent error responses.
    """
    
    def __init__(self):
        """Initialize the error handling middleware."""
        self.logger = db_operation_logger
    
    async def handle_supabase_error(self, request: Request, exc: Exception) -> JSONResponse:
        """
        Handle Supabase-related errors.
        
        Args:
            request: FastAPI request object
            exc: Exception that occurred
            
        Returns:
            JSONResponse with error details
        """
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        try:
            if isinstance(exc, DatabaseTimeoutError):
                return await self._handle_timeout_error(request, exc, request_id)
            elif isinstance(exc, DatabaseAuthenticationError):
                return await self._handle_auth_error(request, exc, request_id)
            elif isinstance(exc, DatabaseConnectionError):
                return await self._handle_connection_error(request, exc, request_id)
            elif isinstance(exc, DatabaseRateLimitError):
                return await self._handle_rate_limit_error(request, exc, request_id)
            elif isinstance(exc, SupabaseError):
                return await self._handle_supabase_error(request, exc, request_id)
            else:
                return await self._handle_generic_error(request, exc, request_id)
                
        except Exception as e:
            logger.error(
                "Error in error handling middleware",
                error=str(e),
                original_error=str(exc),
                request_id=request_id
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    async def _handle_timeout_error(self, request: Request, exc: DatabaseTimeoutError, 
                                   request_id: str) -> JSONResponse:
        """Handle timeout errors."""
        self.logger.log_operation_error(
            operation_id=request_id,
            error_message=str(exc),
            error_type="timeout",
            execution_time_ms=60000  # 60 second timeout
        )
        
        return JSONResponse(
            status_code=408,
            content={
                "error": "Request Timeout",
                "message": "Database operation timed out after 60 seconds",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "error_type": "timeout",
                    "timeout_seconds": 60
                }
            }
        )
    
    async def _handle_auth_error(self, request: Request, exc: DatabaseAuthenticationError,
                                request_id: str) -> JSONResponse:
        """Handle authentication errors."""
        self.logger.log_operation_error(
            operation_id=request_id,
            error_message=str(exc),
            error_type="authentication",
            execution_time_ms=None
        )
        
        return JSONResponse(
            status_code=401,
            content={
                "error": "Unauthorized",
                "message": "Authentication failed - redirect to login",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "error_type": "authentication",
                    "redirect_url": "/login"
                }
            }
        )
    
    async def _handle_connection_error(self, request: Request, exc: DatabaseConnectionError,
                                     request_id: str) -> JSONResponse:
        """Handle connection errors."""
        self.logger.log_operation_error(
            operation_id=request_id,
            error_message=str(exc),
            error_type="connection",
            execution_time_ms=None
        )
        
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service Unavailable",
                "message": "Database service unavailable - fail fast with clear error",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "error_type": "connection",
                    "retry_after": 30
                }
            }
        )
    
    async def _handle_rate_limit_error(self, request: Request, exc: DatabaseRateLimitError,
                                     request_id: str) -> JSONResponse:
        """Handle rate limit errors."""
        self.logger.log_operation_error(
            operation_id=request_id,
            error_message=str(exc),
            error_type="rate_limit",
            execution_time_ms=None
        )
        
        return JSONResponse(
            status_code=429,
            content={
                "error": "Too Many Requests",
                "message": "Rate limit exceeded - please retry later",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "error_type": "rate_limit",
                    "retry_after": 60
                }
            }
        )
    
    async def _handle_supabase_error(self, request: Request, exc: SupabaseError,
                                   request_id: str) -> JSONResponse:
        """Handle general Supabase errors."""
        self.logger.log_operation_error(
            operation_id=request_id,
            error_message=str(exc),
            error_type=exc.error_type.value,
            execution_time_ms=None
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Database Error",
                "message": str(exc),
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "error_type": exc.error_type.value,
                    "status_code": exc.status_code
                }
            }
        )
    
    async def _handle_generic_error(self, request: Request, exc: Exception,
                                   request_id: str) -> JSONResponse:
        """Handle generic errors."""
        self.logger.log_operation_error(
            operation_id=request_id,
            error_message=str(exc),
            error_type="generic",
            execution_time_ms=None
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "error_type": "generic",
                    "error_class": type(exc).__name__
                }
            }
        )
    
    async def handle_http_exception(self, request: Request, exc: HTTPException) -> JSONResponse:
        """
        Handle HTTP exceptions.
        
        Args:
            request: FastAPI request object
            exc: HTTPException that occurred
            
        Returns:
            JSONResponse with error details
        """
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Log the error
        self.logger.log_operation_error(
            operation_id=request_id,
            error_message=str(exc.detail),
            error_type="http_error",
            execution_time_ms=None
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP Error",
                "message": str(exc.detail),
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "error_type": "http_error",
                    "status_code": exc.status_code
                }
            }
        )
    
    async def handle_validation_error(self, request: Request, exc: Exception) -> JSONResponse:
        """
        Handle validation errors.
        
        Args:
            request: FastAPI request object
            exc: Validation exception
            
        Returns:
            JSONResponse with error details
        """
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Log the error
        self.logger.log_operation_error(
            operation_id=request_id,
            error_message=str(exc),
            error_type="validation_error",
            execution_time_ms=None
        )
        
        return JSONResponse(
            status_code=400,
            content={
                "error": "Validation Error",
                "message": "Request validation failed",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "error_type": "validation_error",
                    "validation_details": str(exc)
                }
            }
        )

# Global middleware instance
error_middleware = ErrorHandlingMiddleware()

async def error_handler_middleware(request: Request, call_next):
    """
    FastAPI middleware for error handling.
    
    Args:
        request: FastAPI request object
        call_next: Next middleware/handler
        
    Returns:
        Response object
    """
    try:
        response = await call_next(request)
        return response
        
    except DatabaseTimeoutError as exc:
        return await error_middleware.handle_supabase_error(request, exc)
    except DatabaseAuthenticationError as exc:
        return await error_middleware.handle_supabase_error(request, exc)
    except DatabaseConnectionError as exc:
        return await error_middleware.handle_supabase_error(request, exc)
    except DatabaseRateLimitError as exc:
        return await error_middleware.handle_supabase_error(request, exc)
    except SupabaseError as exc:
        return await error_middleware.handle_supabase_error(request, exc)
    except HTTPException as exc:
        return await error_middleware.handle_http_exception(request, exc)
    except Exception as exc:
        return await error_middleware.handle_supabase_error(request, exc)
