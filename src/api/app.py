"""
Main Flask application for Content Generator V2.

This module creates and configures the Flask application
with all necessary middleware, blueprints, and error handlers.
"""

import logging
import os
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from .endpoints import research_bp, health_bp
from .middleware.auth import AuthMiddleware
from .middleware.logging import LoggingMiddleware
from .middleware.error_handler import ErrorHandler
from ..core.models.errors import ErrorResponse
from ..utils.config import get_config
from ..utils.logging import setup_logging


def create_app(config_name: str = None) -> Flask:
    """
    Create and configure Flask application.
    
    Args:
        config_name: Configuration name (development, production, testing)
        
    Returns:
        Configured Flask application
    """
    # Create Flask app
    app = Flask(__name__)
    
    # Load configuration
    config = get_config(config_name)
    app.config.from_object(config)
    
    # Setup logging
    setup_logging(app.config)
    
    # Initialize extensions
    CORS(app, origins=app.config.get('CORS_ORIGINS', ['*']))
    
    # Initialize rate limiter (compatible with Flask-Limiter v3+)
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[],  # Use per-endpoint limits to avoid parsing issues
        storage_uri=app.config.get('RATELIMIT_STORAGE_URL', 'memory://')
    )
    limiter.init_app(app)
    
    # Register middleware
    app.before_request(AuthMiddleware.before_request)
    app.after_request(LoggingMiddleware.after_request)
    
    # Register blueprints
    app.register_blueprint(research_bp)
    app.register_blueprint(health_bp)
    
    # Register error handlers
    ErrorHandler.register_handlers(app)
    
    # Add custom error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify(ErrorResponse(
            error="not_found",
            message="The requested resource was not found",
            status=404
        ).dict()), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify(ErrorResponse(
            error="method_not_allowed",
            message="The method is not allowed for the requested URL",
            status=405
        ).dict()), 405
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify(ErrorResponse(
            error="internal_server_error",
            message="An internal server error occurred",
            error_code="INTERNAL_SERVER_ERROR",
            status=500
        ).dict()), 500
    
    # Add request logging
    @app.before_request
    def log_request():
        if app.config.get('LOG_REQUESTS', True):
            logger = logging.getLogger(__name__)
            logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")
    
    # Add response logging
    @app.after_request
    def log_response(response):
        if app.config.get('LOG_REQUESTS', True):
            logger = logging.getLogger(__name__)
            logger.info(f"Response: {response.status_code} for {request.method} {request.path}")
        return response
    
    # Health check endpoint
    @app.route('/')
    def root():
        return jsonify({
            "service": "content-generator-v2",
            "version": "2.0.0",
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "endpoints": {
                "health": "/api/v1/health",
                "research": "/api/v1/research",
                "docs": "/api/v1/docs"
            }
        })
    
    # API documentation endpoint
    @app.route('/api/v1/docs')
    def api_docs():
        return jsonify({
            "title": "Content Generator V2 API",
            "version": "2.0.0",
            "description": "A clean and reliable content generation system",
            "endpoints": {
                "research": {
                    "create": {
                        "method": "POST",
                        "path": "/api/v1/research",
                        "description": "Create a new research task"
                    },
                    "status": {
                        "method": "GET",
                        "path": "/api/v1/research/{task_id}",
                        "description": "Get task status and progress"
                    },
                    "result": {
                        "method": "GET",
                        "path": "/api/v1/research/{task_id}/result",
                        "description": "Get completed task result"
                    },
                    "cancel": {
                        "method": "POST",
                        "path": "/api/v1/research/{task_id}/cancel",
                        "description": "Cancel a running task"
                    }
                },
                "health": {
                    "basic": {
                        "method": "GET",
                        "path": "/api/v1/health",
                        "description": "Basic health check"
                    },
                    "detailed": {
                        "method": "GET",
                        "path": "/api/v1/health/detailed",
                        "description": "Detailed health check"
                    },
                    "ready": {
                        "method": "GET",
                        "path": "/api/v1/health/ready",
                        "description": "Readiness check"
                    },
                    "live": {
                        "method": "GET",
                        "path": "/api/v1/health/live",
                        "description": "Liveness check"
                    }
                }
            },
            "authentication": {
                "type": "API Key",
                "header": "X-API-Key",
                "description": "API key authentication required for all endpoints except health checks"
            },
            "rate_limiting": {
                "default": "1000 requests per hour",
                "research_creation": "10 requests per minute",
                "status_checking": "1000 requests per hour"
            }
        })
    
    logger = logging.getLogger(__name__)
    logger.info(f"Flask application created with config: {config_name}")
    
    return app


def run_app(host: str = '0.0.0.0', port: int = 5001, debug: bool = False):
    """
    Run the Flask application.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    app = create_app()
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Content Generator V2 on {host}:{port}")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_app()
