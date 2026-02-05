"""
Error Handling Middleware

Provides centralized error handling and logging for all API endpoints.
"""

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import Dict, Any, Optional
import logging
import traceback
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class ErrorHandler:
    """Centralized error handling for the application"""
    
    @staticmethod
    def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        """Handle HTTP exceptions"""
        error_id = str(uuid.uuid4())
        
        logger.warning(
            f"HTTP Exception [{error_id}]: {exc.status_code} - {exc.detail}",
            extra={
                "error_id": error_id,
                "status_code": exc.status_code,
                "detail": exc.detail,
                "path": str(request.url),
                "method": request.method
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP Exception",
                "message": exc.detail,
                "error_id": error_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @staticmethod
    def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle request validation errors"""
        error_id = str(uuid.uuid4())
        
        logger.warning(
            f"Validation Error [{error_id}]: {exc.errors()}",
            extra={
                "error_id": error_id,
                "errors": exc.errors(),
                "path": str(request.url),
                "method": request.method
            }
        )
        
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation Error",
                "message": "Request validation failed",
                "details": exc.errors(),
                "error_id": error_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @staticmethod
    def handle_general_exception(request: Request, exc: Exception) -> JSONResponse:
        """Handle general exceptions"""
        error_id = str(uuid.uuid4())
        
        logger.error(
            f"Unhandled Exception [{error_id}]: {str(exc)}",
            extra={
                "error_id": error_id,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "path": str(request.url),
                "method": request.method
            }
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "error_id": error_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @staticmethod
    def handle_starlette_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Handle Starlette HTTP exceptions"""
        error_id = str(uuid.uuid4())
        
        logger.warning(
            f"Starlette HTTP Exception [{error_id}]: {exc.status_code} - {exc.detail}",
            extra={
                "error_id": error_id,
                "status_code": exc.status_code,
                "detail": exc.detail,
                "path": str(request.url),
                "method": request.method
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP Exception",
                "message": exc.detail,
                "error_id": error_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

class LoggingMiddleware:
    """Middleware for request/response logging"""
    
    @staticmethod
    async def log_request(request: Request) -> None:
        """Log incoming request"""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        logger.info(
            f"Request [{request_id}]: {request.method} {request.url}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": str(request.url),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @staticmethod
    async def log_response(request: Request, response: Response) -> None:
        """Log outgoing response"""
        request_id = getattr(request.state, "request_id", "unknown")
        
        logger.info(
            f"Response [{request_id}]: {response.status_code}",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "method": request.method,
                "path": str(request.url),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

class DatabaseErrorHandler:
    """Handle database-related errors"""
    
    @staticmethod
    def handle_connection_error(error: Exception) -> HTTPException:
        """Handle database connection errors"""
        logger.error(f"Database connection error: {str(error)}")
        return HTTPException(
            status_code=503,
            detail="Database service temporarily unavailable"
        )
    
    @staticmethod
    def handle_query_error(error: Exception) -> HTTPException:
        """Handle database query errors"""
        logger.error(f"Database query error: {str(error)}")
        return HTTPException(
            status_code=500,
            detail="Database query failed"
        )
    
    @staticmethod
    def handle_constraint_error(error: Exception) -> HTTPException:
        """Handle database constraint errors"""
        logger.error(f"Database constraint error: {str(error)}")
        return HTTPException(
            status_code=400,
            detail="Data constraint violation"
        )

class FileProcessingErrorHandler:
    """Handle file processing errors"""
    
    @staticmethod
    def handle_parsing_error(error: Exception) -> HTTPException:
        """Handle file parsing errors"""
        logger.error(f"File parsing error: {str(error)}")
        return HTTPException(
            status_code=400,
            detail="File parsing failed - invalid format"
        )
    
    @staticmethod
    def handle_validation_error(error: Exception) -> HTTPException:
        """Handle file validation errors"""
        logger.error(f"File validation error: {str(error)}")
        return HTTPException(
            status_code=400,
            detail="File validation failed"
        )
    
    @staticmethod
    def handle_processing_error(error: Exception) -> HTTPException:
        """Handle file processing errors"""
        logger.error(f"File processing error: {str(error)}")
        return HTTPException(
            status_code=500,
            detail="File processing failed"
        )

class AnalysisErrorHandler:
    """Handle analysis-related errors"""
    
    @staticmethod
    def handle_analysis_error(error: Exception) -> HTTPException:
        """Handle analysis errors"""
        logger.error(f"Analysis error: {str(error)}")
        return HTTPException(
            status_code=500,
            detail="Analysis processing failed"
        )
    
    @staticmethod
    def handle_insufficient_data_error() -> HTTPException:
        """Handle insufficient data errors"""
        logger.warning("Insufficient data for analysis")
        return HTTPException(
            status_code=400,
            detail="Insufficient data for analysis"
        )
    
    @staticmethod
    def handle_timeout_error() -> HTTPException:
        """Handle analysis timeout errors"""
        logger.error("Analysis processing timeout")
        return HTTPException(
            status_code=504,
            detail="Analysis processing timeout"
        )