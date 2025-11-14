"""
Logging middleware for Content Generator V2.

This module provides request/response logging middleware
for monitoring and debugging.
"""

import logging
import time
from datetime import datetime
from flask import request, g


logger = logging.getLogger(__name__)


class LoggingMiddleware:
    """Logging middleware for request/response logging."""
    
    @staticmethod
    def before_request():
        """Log request details."""
        g.start_time = time.time()
        g.request_id = f"req_{int(time.time() * 1000)}"
        
        logger.info(
            f"Request started: {g.request_id} - {request.method} {request.path} "
            f"from {request.remote_addr}"
        )
        
        # Log request body for POST requests (excluding sensitive data)
        if request.method == 'POST' and request.is_json:
            data = request.get_json()
            # Remove sensitive fields
            safe_data = {k: v for k, v in data.items() 
                        if k not in ['api_key', 'llm_key', 'password']}
            logger.debug(f"Request body: {safe_data}")
    
    @staticmethod
    def after_request(response):
        """Log response details."""
        if hasattr(g, 'start_time'):
            duration = time.time() - g.start_time
            
            logger.info(
                f"Request completed: {g.request_id} - {response.status_code} "
                f"in {duration:.3f}s"
            )
            
            # Log response for errors
            if response.status_code >= 400:
                logger.warning(
                    f"Error response: {g.request_id} - {response.status_code} "
                    f"for {request.method} {request.path}"
                )
        
        return response
    
    @staticmethod
    def log_error(error: Exception, context: dict = None):
        """
        Log error with context.
        
        Args:
            error: Exception to log
            context: Additional context
        """
        request_id = getattr(g, 'request_id', 'unknown')
        
        logger.error(
            f"Error in request {request_id}: {str(error)}",
            exc_info=True,
            extra={
                'request_id': request_id,
                'context': context or {}
            }
        )
    
    @staticmethod
    def log_performance(operation: str, duration: float, details: dict = None):
        """
        Log performance metrics.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            details: Additional details
        """
        request_id = getattr(g, 'request_id', 'unknown')
        
        logger.info(
            f"Performance: {operation} took {duration:.3f}s",
            extra={
                'request_id': request_id,
                'operation': operation,
                'duration': duration,
                'details': details or {}
            }
        )
