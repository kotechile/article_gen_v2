"""
Error models and exception classes.

This module defines custom exception classes and error models
for the Content Generator V2 system.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ContentGeneratorError(Exception):
    """Base exception for Content Generator V2."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(ContentGeneratorError):
    """Validation error."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.field = field
        self.value = value
        super().__init__(message, "VALIDATION_ERROR", {"field": field, "value": value})


class LLMError(ContentGeneratorError):
    """LLM-related error."""
    
    def __init__(self, message: str, provider: str = None, model: str = None, retryable: bool = True):
        self.provider = provider
        self.model = model
        self.retryable = retryable
        super().__init__(
            message, 
            "LLM_ERROR", 
            {"provider": provider, "model": model, "retryable": retryable}
        )


class RAGError(ContentGeneratorError):
    """RAG-related error."""
    
    def __init__(self, message: str, endpoint: str = None, collection: str = None):
        self.endpoint = endpoint
        self.collection = collection
        super().__init__(
            message,
            "RAG_ERROR",
            {"endpoint": endpoint, "collection": collection}
        )


class SearchError(ContentGeneratorError):
    """Search-related error."""
    
    def __init__(self, message: str, query: str = None, provider: str = None):
        self.query = query
        self.provider = provider
        super().__init__(
            message,
            "SEARCH_ERROR",
            {"query": query, "provider": provider}
        )


class ArticleGenerationError(ContentGeneratorError):
    """Article generation error."""
    
    def __init__(self, message: str, stage: str = None, task_id: str = None):
        self.stage = stage
        self.task_id = task_id
        super().__init__(
            message,
            "ARTICLE_GENERATION_ERROR",
            {"stage": stage, "task_id": task_id}
        )


class TaskError(ContentGeneratorError):
    """Task processing error."""
    
    def __init__(self, message: str, task_id: str = None, retry_count: int = 0):
        self.task_id = task_id
        self.retry_count = retry_count
        super().__init__(
            message,
            "TASK_ERROR",
            {"task_id": task_id, "retry_count": retry_count}
        )


class ConfigurationError(ContentGeneratorError):
    """Configuration error."""
    
    def __init__(self, message: str, config_key: str = None):
        self.config_key = config_key
        super().__init__(
            message,
            "CONFIGURATION_ERROR",
            {"config_key": config_key}
        )


class RateLimitError(ContentGeneratorError):
    """Rate limit error."""
    
    def __init__(self, message: str, retry_after: int = None, limit: int = None):
        self.retry_after = retry_after
        self.limit = limit
        super().__init__(
            message,
            "RATE_LIMIT_ERROR",
            {"retry_after": retry_after, "limit": limit}
        )


class AuthenticationError(ContentGeneratorError):
    """Authentication error."""
    
    def __init__(self, message: str, api_key: str = None):
        self.api_key = api_key
        super().__init__(
            message,
            "AUTHENTICATION_ERROR",
            {"api_key": api_key}
        )


class ExternalServiceError(ContentGeneratorError):
    """External service error."""
    
    def __init__(self, message: str, service: str = None, status_code: int = None):
        self.service = service
        self.status_code = status_code
        super().__init__(
            message,
            "EXTERNAL_SERVICE_ERROR",
            {"service": service, "status_code": status_code}
        )


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    status: int = Field(..., description="HTTP status code")
    
    # Optional Details
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    field: Optional[str] = Field(None, description="Field that caused error")
    value: Optional[Any] = Field(None, description="Value that caused error")
    
    # Request Information
    request_id: Optional[str] = Field(None, description="Request ID")
    task_id: Optional[str] = Field(None, description="Task ID")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @classmethod
    def from_exception(cls, exc: ContentGeneratorError, status: int = 500) -> 'ErrorResponse':
        """Create error response from exception."""
        return cls(
            error=exc.__class__.__name__,
            message=exc.message,
            error_code=exc.error_code or "UNKNOWN_ERROR",
            status=status,
            details=exc.details
        )


class ValidationErrorResponse(BaseModel):
    """Validation error response model."""
    
    error: str = Field(default="validation_error", description="Error type")
    message: str = Field(default="Validation failed", description="Error message")
    status: int = Field(default=400, description="HTTP status code")
    
    validation_errors: List[Dict[str, Any]] = Field(default_factory=list, description="Validation errors")
    
    # Request Information
    request_id: Optional[str] = Field(None, description="Request ID")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def add_validation_error(self, field: str, message: str, value: Any = None):
        """Add a validation error."""
        error = {
            "field": field,
            "message": message
        }
        if value is not None:
            error["value"] = value
        
        self.validation_errors.append(error)
    
    def has_errors(self) -> bool:
        """Check if there are validation errors."""
        return len(self.validation_errors) > 0
