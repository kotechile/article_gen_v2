"""
SupabaseErrorHandler for handling Supabase-specific errors
"""
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import structlog
from supabase.exceptions import APIError, AuthError
from fastapi import HTTPException

logger = structlog.get_logger()

class SupabaseErrorHandler:
    """Service for handling Supabase-specific errors"""
    
    def __init__(self):
        self.error_mappings = {
            "PGRST116": {
                "message": "Row Level Security policy violation",
                "status_code": 403,
                "category": "authorization"
            },
            "PGRST301": {
                "message": "JWT token is invalid or expired",
                "status_code": 401,
                "category": "authentication"
            },
            "PGRST302": {
                "message": "JWT token is missing",
                "status_code": 401,
                "category": "authentication"
            },
            "23505": {
                "message": "Unique constraint violation",
                "status_code": 409,
                "category": "constraint"
            },
            "23503": {
                "message": "Foreign key constraint violation",
                "status_code": 409,
                "category": "constraint"
            },
            "23502": {
                "message": "Not null constraint violation",
                "status_code": 400,
                "category": "validation"
            },
            "42P01": {
                "message": "Table does not exist",
                "status_code": 404,
                "category": "not_found"
            },
            "42703": {
                "message": "Column does not exist",
                "status_code": 400,
                "category": "validation"
            }
        }
    
    def handle_api_error(self, error: APIError, context: str = "") -> HTTPException:
        """
        Handle Supabase API errors
        
        Args:
            error: APIError from Supabase
            context: Additional context for logging
            
        Returns:
            HTTPException with appropriate status code and message
        """
        try:
            logger.error("Supabase API error", 
                        error=str(error),
                        context=context,
                        error_code=getattr(error, 'code', None))
            
            # Extract error details
            error_code = getattr(error, 'code', None)
            error_message = str(error)
            
            # Check if we have a specific mapping for this error
            if error_code and error_code in self.error_mappings:
                mapping = self.error_mappings[error_code]
                return HTTPException(
                    status_code=mapping["status_code"],
                    detail={
                        "error": mapping["category"],
                        "message": mapping["message"],
                        "supabase_error_code": error_code,
                        "context": context,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            
            # Handle common error patterns
            if "permission denied" in error_message.lower():
                return HTTPException(
                    status_code=403,
                    detail={
                        "error": "authorization",
                        "message": "Permission denied",
                        "supabase_error": error_message,
                        "context": context,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            
            if "not found" in error_message.lower():
                return HTTPException(
                    status_code=404,
                    detail={
                        "error": "not_found",
                        "message": "Resource not found",
                        "supabase_error": error_message,
                        "context": context,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            
            if "timeout" in error_message.lower():
                return HTTPException(
                    status_code=504,
                    detail={
                        "error": "timeout",
                        "message": "Request timeout",
                        "supabase_error": error_message,
                        "context": context,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            
            # Default error handling
            return HTTPException(
                status_code=500,
                detail={
                    "error": "supabase_api_error",
                    "message": "Supabase API error occurred",
                    "supabase_error": error_message,
                    "supabase_error_code": error_code,
                    "context": context,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error("Failed to handle API error", 
                        original_error=str(error),
                        handling_error=str(e),
                        context=context)
            
            return HTTPException(
                status_code=500,
                detail={
                    "error": "error_handling_failed",
                    "message": "Failed to handle Supabase error",
                    "original_error": str(error),
                    "handling_error": str(e),
                    "context": context,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    def handle_auth_error(self, error: AuthError, context: str = "") -> HTTPException:
        """
        Handle Supabase authentication errors
        
        Args:
            error: AuthError from Supabase
            context: Additional context for logging
            
        Returns:
            HTTPException with appropriate status code and message
        """
        try:
            logger.error("Supabase auth error", 
                        error=str(error),
                        context=context)
            
            error_message = str(error)
            
            # Handle specific auth error patterns
            if "invalid" in error_message.lower() and "token" in error_message.lower():
                return HTTPException(
                    status_code=401,
                    detail={
                        "error": "authentication",
                        "message": "Invalid authentication token",
                        "supabase_error": error_message,
                        "context": context,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            
            if "expired" in error_message.lower():
                return HTTPException(
                    status_code=401,
                    detail={
                        "error": "authentication",
                        "message": "Authentication token expired",
                        "supabase_error": error_message,
                        "context": context,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            
            if "missing" in error_message.lower():
                return HTTPException(
                    status_code=401,
                    detail={
                        "error": "authentication",
                        "message": "Authentication token missing",
                        "supabase_error": error_message,
                        "context": context,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            
            # Default auth error handling
            return HTTPException(
                status_code=401,
                detail={
                    "error": "authentication",
                    "message": "Authentication failed",
                    "supabase_error": error_message,
                    "context": context,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error("Failed to handle auth error", 
                        original_error=str(error),
                        handling_error=str(e),
                        context=context)
            
            return HTTPException(
                status_code=500,
                detail={
                    "error": "error_handling_failed",
                    "message": "Failed to handle Supabase auth error",
                    "original_error": str(error),
                    "handling_error": str(e),
                    "context": context,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    def handle_generic_error(self, error: Exception, context: str = "") -> HTTPException:
        """
        Handle generic errors that might occur with Supabase
        
        Args:
            error: Generic exception
            context: Additional context for logging
            
        Returns:
            HTTPException with appropriate status code and message
        """
        try:
            logger.error("Generic error in Supabase context", 
                        error=str(error),
                        context=context,
                        error_type=type(error).__name__)
            
            error_message = str(error)
            error_type = type(error).__name__
            
            # Handle connection errors
            if "connection" in error_message.lower() or "ConnectionError" in error_type:
                return HTTPException(
                    status_code=503,
                    detail={
                        "error": "connection",
                        "message": "Database connection failed",
                        "error_type": error_type,
                        "error_message": error_message,
                        "context": context,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            
            # Handle timeout errors
            if "timeout" in error_message.lower() or "TimeoutError" in error_type:
                return HTTPException(
                    status_code=504,
                    detail={
                        "error": "timeout",
                        "message": "Request timeout",
                        "error_type": error_type,
                        "error_message": error_message,
                        "context": context,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            
            # Handle validation errors
            if "validation" in error_message.lower() or "ValueError" in error_type:
                return HTTPException(
                    status_code=400,
                    detail={
                        "error": "validation",
                        "message": "Validation error",
                        "error_type": error_type,
                        "error_message": error_message,
                        "context": context,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            
            # Default generic error handling
            return HTTPException(
                status_code=500,
                detail={
                    "error": "internal_error",
                    "message": "Internal server error",
                    "error_type": error_type,
                    "error_message": error_message,
                    "context": context,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error("Failed to handle generic error", 
                        original_error=str(error),
                        handling_error=str(e),
                        context=context)
            
            return HTTPException(
                status_code=500,
                detail={
                    "error": "error_handling_failed",
                    "message": "Failed to handle error",
                    "original_error": str(error),
                    "handling_error": str(e),
                    "context": context,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    def safe_execute(self, operation: Callable, context: str = "", *args, **kwargs) -> Any:
        """
        Safely execute a Supabase operation with error handling
        
        Args:
            operation: Function to execute
            context: Context for error logging
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            HTTPException: If the operation fails
        """
        try:
            return operation(*args, **kwargs)
        except APIError as e:
            raise self.handle_api_error(e, context)
        except AuthError as e:
            raise self.handle_auth_error(e, context)
        except Exception as e:
            raise self.handle_generic_error(e, context)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics
        
        Returns:
            Dict containing error statistics
        """
        try:
            # This would typically come from a database or logging system
            # For now, return a placeholder structure
            return {
                "total_errors": 0,
                "error_types": {
                    "api_errors": 0,
                    "auth_errors": 0,
                    "connection_errors": 0,
                    "timeout_errors": 0,
                    "validation_errors": 0,
                    "other_errors": 0
                },
                "error_codes": {},
                "last_updated": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error("Failed to get error statistics", error=str(e))
            return {
                "total_errors": 0,
                "error_types": {},
                "error_codes": {},
                "last_updated": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    def add_error_mapping(self, error_code: str, message: str, status_code: int, category: str):
        """
        Add a custom error mapping
        
        Args:
            error_code: Supabase error code
            message: Human-readable error message
            status_code: HTTP status code
            category: Error category
        """
        self.error_mappings[error_code] = {
            "message": message,
            "status_code": status_code,
            "category": category
        }
        
        logger.info("Added custom error mapping", 
                   error_code=error_code,
                   status_code=status_code,
                   category=category)

