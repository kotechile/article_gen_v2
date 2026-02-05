"""
Error handling middleware for research topics API
This module provides specific error handling for research topics operations
"""

import logging
from typing import Dict, Any
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from uuid import UUID

logger = logging.getLogger(__name__)

class ResearchTopicsErrorHandler:
    """Error handler for research topics operations"""
    
    def __init__(self):
        self.logger = logger
    
    async def handle_validation_error(self, request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle validation errors"""
        self.logger.warning(f"Validation error in {request.url}: {exc.errors()}")
        
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Invalid input data",
                "details": exc.errors(),
                "timestamp": "2025-01-27T10:00:00Z",
                "request_id": str(request.state.get("request_id", "unknown"))
            }
        )
    
    async def handle_value_error(self, request: Request, exc: ValueError) -> JSONResponse:
        """Handle value errors (business logic errors)"""
        self.logger.warning(f"Value error in {request.url}: {str(exc)}")
        
        return JSONResponse(
            status_code=400,
            content={
                "error": "business_logic_error",
                "message": str(exc),
                "timestamp": "2025-01-27T10:00:00Z",
                "request_id": str(request.state.get("request_id", "unknown"))
            }
        )
    
    async def handle_not_found_error(self, request: Request, exc: HTTPException) -> JSONResponse:
        """Handle not found errors"""
        self.logger.info(f"Resource not found in {request.url}: {exc.detail}")
        
        return JSONResponse(
            status_code=404,
            content={
                "error": "not_found",
                "message": exc.detail,
                "timestamp": "2025-01-27T10:00:00Z",
                "request_id": str(request.state.get("request_id", "unknown"))
            }
        )
    
    async def handle_unauthorized_error(self, request: Request, exc: HTTPException) -> JSONResponse:
        """Handle unauthorized errors"""
        self.logger.warning(f"Unauthorized access in {request.url}: {exc.detail}")
        
        return JSONResponse(
            status_code=401,
            content={
                "error": "unauthorized",
                "message": "Authentication required",
                "timestamp": "2025-01-27T10:00:00Z",
                "request_id": str(request.state.get("request_id", "unknown"))
            }
        )
    
    async def handle_forbidden_error(self, request: Request, exc: HTTPException) -> JSONResponse:
        """Handle forbidden errors"""
        self.logger.warning(f"Forbidden access in {request.url}: {exc.detail}")
        
        return JSONResponse(
            status_code=403,
            content={
                "error": "forbidden",
                "message": "Insufficient permissions",
                "timestamp": "2025-01-27T10:00:00Z",
                "request_id": str(request.state.get("request_id", "unknown"))
            }
        )
    
    async def handle_conflict_error(self, request: Request, exc: HTTPException) -> JSONResponse:
        """Handle conflict errors (duplicate resources, concurrent updates)"""
        self.logger.warning(f"Conflict in {request.url}: {exc.detail}")
        
        return JSONResponse(
            status_code=409,
            content={
                "error": "conflict",
                "message": exc.detail,
                "timestamp": "2025-01-27T10:00:00Z",
                "request_id": str(request.state.get("request_id", "unknown"))
            }
        )
    
    async def handle_database_error(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle database errors"""
        self.logger.error(f"Database error in {request.url}: {str(exc)}")
        
        return JSONResponse(
            status_code=503,
            content={
                "error": "database_error",
                "message": "Database operation failed",
                "timestamp": "2025-01-27T10:00:00Z",
                "request_id": str(request.state.get("request_id", "unknown"))
            }
        )
    
    async def handle_generic_error(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle generic errors"""
        self.logger.error(f"Unexpected error in {request.url}: {str(exc)}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
                "timestamp": "2025-01-27T10:00:00Z",
                "request_id": str(request.state.get("request_id", "unknown"))
            }
        )
    
    def is_validation_error(self, exc: Exception) -> bool:
        """Check if exception is a validation error"""
        return isinstance(exc, (RequestValidationError, ValidationError))
    
    def is_value_error(self, exc: Exception) -> bool:
        """Check if exception is a value error"""
        return isinstance(exc, ValueError)
    
    def is_http_exception(self, exc: Exception) -> bool:
        """Check if exception is an HTTP exception"""
        return isinstance(exc, HTTPException)
    
    def get_http_status_code(self, exc: HTTPException) -> int:
        """Get HTTP status code from HTTP exception"""
        return exc.status_code
    
    async def handle_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """Main exception handler"""
        # Add request ID if not present
        if not hasattr(request.state, "request_id"):
            request.state.request_id = str(UUID.uuid4())
        
        # Route to appropriate handler based on exception type
        if self.is_validation_error(exc):
            return await self.handle_validation_error(request, exc)
        elif self.is_value_error(exc):
            return await self.handle_value_error(request, exc)
        elif self.is_http_exception(exc):
            status_code = self.get_http_status_code(exc)
            if status_code == 404:
                return await self.handle_not_found_error(request, exc)
            elif status_code == 401:
                return await self.handle_unauthorized_error(request, exc)
            elif status_code == 403:
                return await self.handle_forbidden_error(request, exc)
            elif status_code == 409:
                return await self.handle_conflict_error(request, exc)
            else:
                return await self.handle_generic_error(request, exc)
        else:
            return await self.handle_generic_error(request, exc)

# Global error handler instance
research_topics_error_handler = ResearchTopicsErrorHandler()
