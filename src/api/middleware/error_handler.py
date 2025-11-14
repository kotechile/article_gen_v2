"""
Error handling middleware for Content Generator V2.

This module provides centralized error handling and
response formatting for the API.
"""

import logging
from flask import jsonify, request, g

from ...core.models.errors import (
    ErrorResponse,
    ValidationError,
    LLMError,
    RAGError,
    SearchError,
    ArticleGenerationError,
    TaskError,
    ConfigurationError,
    RateLimitError,
    AuthenticationError,
    ExternalServiceError
)


logger = logging.getLogger(__name__)


class ErrorHandler:
    """Centralized error handling for the API."""
    
    @staticmethod
    def register_handlers(app):
        """Register error handlers with Flask app."""
        
        @app.errorhandler(ValidationError)
        def handle_validation_error(error):
            return ErrorHandler.handle_validation_error(error)
        
        @app.errorhandler(LLMError)
        def handle_llm_error(error):
            return ErrorHandler.handle_llm_error(error)
        
        @app.errorhandler(RAGError)
        def handle_rag_error(error):
            return ErrorHandler.handle_rag_error(error)
        
        @app.errorhandler(SearchError)
        def handle_search_error(error):
            return ErrorHandler.handle_search_error(error)
        
        @app.errorhandler(ArticleGenerationError)
        def handle_article_generation_error(error):
            return ErrorHandler.handle_article_generation_error(error)
        
        @app.errorhandler(TaskError)
        def handle_task_error(error):
            return ErrorHandler.handle_task_error(error)
        
        @app.errorhandler(ConfigurationError)
        def handle_configuration_error(error):
            return ErrorHandler.handle_configuration_error(error)
        
        @app.errorhandler(RateLimitError)
        def handle_rate_limit_error(error):
            return ErrorHandler.handle_rate_limit_error(error)
        
        @app.errorhandler(AuthenticationError)
        def handle_authentication_error(error):
            return ErrorHandler.handle_authentication_error(error)
        
        @app.errorhandler(ExternalServiceError)
        def handle_external_service_error(error):
            return ErrorHandler.handle_external_service_error(error)
        
        @app.errorhandler(Exception)
        def handle_generic_error(error):
            return ErrorHandler.handle_generic_error(error)
    
    @staticmethod
    def handle_validation_error(error: ValidationError):
        """Handle validation errors."""
        logger.warning(f"Validation error: {error.message}")
        
        return jsonify(ErrorResponse(
            error="validation_error",
            message=error.message,
            status=400,
            field=error.field,
            value=error.value
        ).dict()), 400
    
    @staticmethod
    def handle_llm_error(error: LLMError):
        """Handle LLM errors."""
        logger.error(f"LLM error: {error.message}")
        
        status = 503 if error.retryable else 400
        
        return jsonify(ErrorResponse(
            error="llm_error",
            message=error.message,
            status=status,
            details={
                "provider": error.provider,
                "model": error.model,
                "retryable": error.retryable
            }
        ).dict()), status
    
    @staticmethod
    def handle_rag_error(error: RAGError):
        """Handle RAG errors."""
        logger.error(f"RAG error: {error.message}")
        
        return jsonify(ErrorResponse(
            error="rag_error",
            message=error.message,
            status=503,
            details={
                "endpoint": error.endpoint,
                "collection": error.collection
            }
        ).dict()), 503
    
    @staticmethod
    def handle_search_error(error: SearchError):
        """Handle search errors."""
        logger.error(f"Search error: {error.message}")
        
        return jsonify(ErrorResponse(
            error="search_error",
            message=error.message,
            status=503,
            details={
                "query": error.query,
                "provider": error.provider
            }
        ).dict()), 503
    
    @staticmethod
    def handle_article_generation_error(error: ArticleGenerationError):
        """Handle article generation errors."""
        logger.error(f"Article generation error: {error.message}")
        
        return jsonify(ErrorResponse(
            error="article_generation_error",
            message=error.message,
            status=500,
            details={
                "stage": error.stage,
                "task_id": error.task_id
            }
        ).dict()), 500
    
    @staticmethod
    def handle_task_error(error: TaskError):
        """Handle task errors."""
        logger.error(f"Task error: {error.message}")
        
        return jsonify(ErrorResponse(
            error="task_error",
            message=error.message,
            status=500,
            details={
                "task_id": error.task_id,
                "retry_count": error.retry_count
            }
        ).dict()), 500
    
    @staticmethod
    def handle_configuration_error(error: ConfigurationError):
        """Handle configuration errors."""
        logger.error(f"Configuration error: {error.message}")
        
        return jsonify(ErrorResponse(
            error="configuration_error",
            message=error.message,
            status=500,
            details={
                "config_key": error.config_key
            }
        ).dict()), 500
    
    @staticmethod
    def handle_rate_limit_error(error: RateLimitError):
        """Handle rate limit errors."""
        logger.warning(f"Rate limit error: {error.message}")
        
        return jsonify(ErrorResponse(
            error="rate_limit_error",
            message=error.message,
            status=429,
            details={
                "retry_after": error.retry_after,
                "limit": error.limit
            }
        ).dict()), 429
    
    @staticmethod
    def handle_authentication_error(error: AuthenticationError):
        """Handle authentication errors."""
        logger.warning(f"Authentication error: {error.message}")
        
        return jsonify(ErrorResponse(
            error="authentication_error",
            message=error.message,
            status=401,
            details={
                "api_key": error.api_key[:8] + "..." if error.api_key else None
            }
        ).dict()), 401
    
    @staticmethod
    def handle_external_service_error(error: ExternalServiceError):
        """Handle external service errors."""
        logger.error(f"External service error: {error.message}")
        
        return jsonify(ErrorResponse(
            error="external_service_error",
            message=error.message,
            status=503,
            details={
                "service": error.service,
                "status_code": error.status_code
            }
        ).dict()), 503
    
    @staticmethod
    def handle_generic_error(error: Exception):
        """Handle generic errors."""
        request_id = getattr(g, 'request_id', 'unknown')
        
        logger.error(
            f"Unhandled error in request {request_id}: {str(error)}",
            exc_info=True
        )
        
        return jsonify(ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred",
            error_code="INTERNAL_SERVER_ERROR",
            status=500,
            details={
                "request_id": request_id,
                "error_type": type(error).__name__
            }
        ).dict()), 500
