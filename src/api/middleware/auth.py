"""
Authentication middleware for Content Generator V2.

This module provides API key authentication middleware
for protecting endpoints.
"""

import logging
from functools import wraps
from flask import request, jsonify, g

from ...core.models.errors import ErrorResponse, AuthenticationError


logger = logging.getLogger(__name__)


class AuthMiddleware:
    """Authentication middleware for API key validation."""
    
    @staticmethod
    def before_request():
        """Process request before handling."""
        # Skip auth for health check endpoints
        if request.endpoint in ['health.health_check', 'health.detailed_health_check', 
                               'health.readiness_check', 'health.liveness_check']:
            return None
        
        # Get API key from header
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="API key is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401
        
        # Validate API key
        if not AuthMiddleware.validate_api_key(api_key):
            return jsonify(ErrorResponse(
                error="invalid_api_key",
                message="Invalid API key",
                error_code="INVALID_API_KEY",
                status=401
            ).dict()), 401
        
        # Store API key in request context
        g.api_key = api_key
        
        return None
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """
        Validate API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid, False otherwise
        """
        from ...utils.config import get_config
        
        config = get_config()
        valid_keys = config.API_KEYS
        
        # If no API keys are configured, allow all requests (development mode)
        # In production, API_KEYS should always be configured
        if not valid_keys:
            logger.warning("No API keys configured - allowing all requests (development mode)")
            return True
        
        return api_key in valid_keys
    
    @staticmethod
    def require_api_key(f):
        """
        Decorator to require API key authentication.
        
        Args:
            f: Function to decorate
            
        Returns:
            Decorated function
        """
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check if API key is in request context
            if not hasattr(g, 'api_key') or not g.api_key:
                return jsonify(ErrorResponse(
                    error="authentication_required",
                    message="API key is required",
                    error_code="AUTHENTICATION_REQUIRED",
                    status=401
                ).dict()), 401
            
            return f(*args, **kwargs)
        
        return decorated_function


def require_api_key(f):
    """
    Decorator to require API key authentication.
    
    This is a convenience function that can be imported directly.
    """
    return AuthMiddleware.require_api_key(f)
