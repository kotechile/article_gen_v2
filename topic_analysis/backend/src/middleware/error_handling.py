"""
Error Handling Middleware
Comprehensive error handling and recovery mechanisms
"""

import traceback
import uuid
from typing import Dict, Any, Optional, Union
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import structlog

from ..core.config import settings

logger = structlog.get_logger()

class ErrorHandler:
    """Centralized error handling"""
    
    def __init__(self):
        self.error_codes = {
            "VALIDATION_ERROR": "E001",
            "AUTHENTICATION_ERROR": "E002",
            "AUTHORIZATION_ERROR": "E003",
            "NOT_FOUND_ERROR": "E004",
            "RATE_LIMIT_ERROR": "E005",
            "DATABASE_ERROR": "E006",
            "EXTERNAL_API_ERROR": "E007",
            "INTERNAL_SERVER_ERROR": "E008",
            "SERVICE_UNAVAILABLE": "E009",
            "TIMEOUT_ERROR": "E010"
        }
        
        self.error_messages = {
            "E001": "Validation error occurred",
            "E002": "Authentication failed",
            "E003": "Access denied",
            "E004": "Resource not found",
            "E005": "Rate limit exceeded",
            "E006": "Database operation failed",
            "E007": "External service error",
            "E008": "Internal server error",
            "E009": "Service temporarily unavailable",
            "E010": "Request timeout"
        }
    
    def get_error_code(self, error_type: str) -> str:
        """Get error code for error type"""
        return self.error_codes.get(error_type, "E008")
    
    def get_error_message(self, error_code: str) -> str:
        """Get error message for error code"""
        return self.error_messages.get(error_code, "An unexpected error occurred")
    
    def create_error_response(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create standardized error response"""
        response = {
            "error": {
                "code": error_code,
                "message": message,
                "request_id": request_id,
                "timestamp": structlog.get_logger().info("timestamp")
            }
        }
        
        if details:
            response["error"]["details"] = details
        
        if settings.DEBUG_MODE:
            response["error"]["debug"] = {
                "traceback": traceback.format_exc()
            }
        
        return response

class ErrorHandlingMiddleware:
    """Main error handling middleware"""
    
    def __init__(self):
        self.error_handler = ErrorHandler()
        self.circuit_breaker = CircuitBreaker()
        self.retry_handler = RetryHandler()
    
    async def __call__(self, request: Request, call_next):
        """Process request with error handling"""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        try:
            # Check circuit breaker
            if not self.circuit_breaker.is_available():
                return self._handle_circuit_breaker_error(request_id)
            
            # Process request
            response = await call_next(request)
            
            # Reset circuit breaker on success
            self.circuit_breaker.record_success()
            
            return response
            
        except HTTPException as e:
            return self._handle_http_exception(e, request_id)
        
        except RequestValidationError as e:
            return self._handle_validation_error(e, request_id)
        
        except StarletteHTTPException as e:
            return self._handle_starlette_exception(e, request_id)
        
        except Exception as e:
            return self._handle_generic_error(e, request_id)
    
    def _handle_circuit_breaker_error(self, request_id: str) -> JSONResponse:
        """Handle circuit breaker error"""
        error_code = self.error_handler.get_error_code("SERVICE_UNAVAILABLE")
        message = "Service temporarily unavailable due to high error rate"
        
        response_data = self.error_handler.create_error_response(
            error_code, message, request_id=request_id
        )
        
        logger.warning("Circuit breaker triggered", request_id=request_id)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response_data
        )
    
    def _handle_http_exception(self, e: HTTPException, request_id: str) -> JSONResponse:
        """Handle FastAPI HTTP exceptions"""
        error_code = self._get_http_error_code(e.status_code)
        message = e.detail
        
        response_data = self.error_handler.create_error_response(
            error_code, message, request_id=request_id
        )
        
        logger.warning(
            "HTTP exception",
            status_code=e.status_code,
            detail=e.detail,
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=e.status_code,
            content=response_data
        )
    
    def _handle_validation_error(self, e: RequestValidationError, request_id: str) -> JSONResponse:
        """Handle request validation errors"""
        error_code = self.error_handler.get_error_code("VALIDATION_ERROR")
        message = "Request validation failed"
        
        details = {
            "validation_errors": e.errors()
        }
        
        response_data = self.error_handler.create_error_response(
            error_code, message, details, request_id
        )
        
        logger.warning(
            "Validation error",
            errors=e.errors(),
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=response_data
        )
    
    def _handle_starlette_exception(self, e: StarletteHTTPException, request_id: str) -> JSONResponse:
        """Handle Starlette HTTP exceptions"""
        error_code = self._get_http_error_code(e.status_code)
        message = e.detail
        
        response_data = self.error_handler.create_error_response(
            error_code, message, request_id=request_id
        )
        
        logger.warning(
            "Starlette exception",
            status_code=e.status_code,
            detail=e.detail,
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=e.status_code,
            content=response_data
        )
    
    def _handle_generic_error(self, e: Exception, request_id: str) -> JSONResponse:
        """Handle generic exceptions"""
        error_code = self.error_handler.get_error_code("INTERNAL_SERVER_ERROR")
        message = "An unexpected error occurred"
        
        # Record error in circuit breaker
        self.circuit_breaker.record_failure()
        
        # Log error details
        logger.error(
            "Unhandled exception",
            error_type=type(e).__name__,
            error_message=str(e),
            traceback=traceback.format_exc(),
            request_id=request_id
        )
        
        response_data = self.error_handler.create_error_response(
            error_code, message, request_id=request_id
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response_data
        )
    
    def _get_http_error_code(self, status_code: int) -> str:
        """Get error code for HTTP status code"""
        error_mapping = {
            400: "E001",  # Bad Request
            401: "E002",  # Unauthorized
            403: "E003",  # Forbidden
            404: "E004",  # Not Found
            422: "E001",  # Unprocessable Entity
            429: "E005",  # Too Many Requests
            500: "E008",  # Internal Server Error
            502: "E007",  # Bad Gateway
            503: "E009",  # Service Unavailable
            504: "E010"   # Gateway Timeout
        }
        
        return error_mapping.get(status_code, "E008")

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def is_available(self) -> bool:
        """Check if circuit breaker allows requests"""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                return True
            return False
        
        if self.state == "HALF_OPEN":
            return True
        
        return False
    
    def record_success(self):
        """Record successful request"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = structlog.get_logger().info("timestamp")
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                "Circuit breaker opened",
                failure_count=self.failure_count,
                threshold=self.failure_threshold
            )
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if not self.last_failure_time:
            return True
        
        current_time = structlog.get_logger().info("timestamp")
        return (current_time - self.last_failure_time) >= self.recovery_timeout

class RetryHandler:
    """Retry mechanism for failed requests"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    async def retry_async(self, func, *args, **kwargs):
        """Retry async function with exponential backoff"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    break
                
                # Calculate delay with exponential backoff
                delay = self.base_delay * (2 ** attempt)
                
                logger.warning(
                    "Retry attempt",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    delay=delay,
                    error=str(e)
                )
                
                await asyncio.sleep(delay)
        
        # Re-raise last exception if all retries failed
        raise last_exception

class ErrorRecovery:
    """Error recovery mechanisms"""
    
    def __init__(self):
        self.recovery_strategies = {
            "database_error": self._recover_database_error,
            "external_api_error": self._recover_external_api_error,
            "timeout_error": self._recover_timeout_error,
            "rate_limit_error": self._recover_rate_limit_error
        }
    
    async def attempt_recovery(self, error_type: str, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt to recover from error"""
        strategy = self.recovery_strategies.get(error_type)
        if not strategy:
            return False
        
        try:
            return await strategy(error, context)
        except Exception as e:
            logger.error("Recovery strategy failed", error=str(e))
            return False
    
    async def _recover_database_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover from database errors"""
        # Implement database connection retry logic
        logger.info("Attempting database recovery")
        return False
    
    async def _recover_external_api_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover from external API errors"""
        # Implement fallback to cached data or alternative APIs
        logger.info("Attempting external API recovery")
        return False
    
    async def _recover_timeout_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover from timeout errors"""
        # Implement timeout recovery logic
        logger.info("Attempting timeout recovery")
        return False
    
    async def _recover_rate_limit_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover from rate limit errors"""
        # Implement rate limit recovery logic
        logger.info("Attempting rate limit recovery")
        return False

# Global instances
error_handling_middleware = ErrorHandlingMiddleware()
circuit_breaker = CircuitBreaker()
retry_handler = RetryHandler()
error_recovery = ErrorRecovery()

# Main error handling middleware
async def error_handling_middleware(request: Request, call_next):
    """Main error handling middleware"""
    return await error_handling_middleware(request, call_next)

# Utility functions
def handle_validation_error(errors: list) -> Dict[str, Any]:
    """Handle validation errors"""
    return {
        "error": {
            "code": "E001",
            "message": "Validation error",
            "details": {
                "validation_errors": errors
            }
        }
    }

def handle_authentication_error(message: str = "Authentication failed") -> Dict[str, Any]:
    """Handle authentication errors"""
    return {
        "error": {
            "code": "E002",
            "message": message
        }
    }

def handle_authorization_error(message: str = "Access denied") -> Dict[str, Any]:
    """Handle authorization errors"""
    return {
        "error": {
            "code": "E003",
            "message": message
        }
    }

def handle_not_found_error(resource: str) -> Dict[str, Any]:
    """Handle not found errors"""
    return {
        "error": {
            "code": "E004",
            "message": f"{resource} not found"
        }
    }

def handle_rate_limit_error(retry_after: int = 60) -> Dict[str, Any]:
    """Handle rate limit errors"""
    return {
        "error": {
            "code": "E005",
            "message": "Rate limit exceeded",
            "details": {
                "retry_after": retry_after
            }
        }
    }

def handle_database_error(operation: str) -> Dict[str, Any]:
    """Handle database errors"""
    return {
        "error": {
            "code": "E006",
            "message": f"Database {operation} failed"
        }
    }

def handle_external_api_error(service: str) -> Dict[str, Any]:
    """Handle external API errors"""
    return {
        "error": {
            "code": "E007",
            "message": f"External service {service} error"
        }
    }

def handle_internal_error(message: str = "Internal server error") -> Dict[str, Any]:
    """Handle internal errors"""
    return {
        "error": {
            "code": "E008",
            "message": message
        }
    }

def handle_service_unavailable_error(service: str) -> Dict[str, Any]:
    """Handle service unavailable errors"""
    return {
        "error": {
            "code": "E009",
            "message": f"Service {service} temporarily unavailable"
        }
    }

def handle_timeout_error(operation: str) -> Dict[str, Any]:
    """Handle timeout errors"""
    return {
        "error": {
            "code": "E010",
            "message": f"Operation {operation} timed out"
        }
    }