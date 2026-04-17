"""
LLM-related data models and schemas.

This module defines the data structures for LLM configuration,
requests, and responses.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator, model_validator


class LLMProvider(str, Enum):
    """LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    # Alias for compatibility with configs that refer to Gemini directly.
    GEMINI = "gemini"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    MOONSHOT = "moonshot"
    COHERE = "cohere"
    MISTRAL = "mistral"
    OLLAMA = "ollama"


class LLMModel(BaseModel):
    """LLM model configuration."""
    
    provider: LLMProvider = Field(..., description="LLM provider")
    model_name: str = Field(..., description="Model name")
    api_key: str = Field(..., description="API key")
    base_url: Optional[str] = Field(None, description="Base URL for API")
    
    # Model Parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature")
    max_tokens: Optional[int] = Field(None, ge=1, le=100000, description="Maximum tokens")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    
    # Advanced Parameters
    stop_sequences: List[str] = Field(default_factory=list, description="Stop sequences")
    timeout: int = Field(default=60, ge=1, le=300, description="Request timeout in seconds")
    retry_count: int = Field(default=3, ge=0, le=10, description="Number of retries")
    retry_delay: float = Field(default=1.0, ge=0.0, le=60.0, description="Retry delay in seconds")
    
    class Config:
        use_enum_values = True


class LLMConfig(BaseModel):
    """LLM configuration for requests."""
    
    model: LLMModel = Field(..., description="LLM model configuration")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    user_prompt: str = Field(..., description="User prompt")
    
    # Request Parameters
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Override temperature")
    max_tokens: Optional[int] = Field(None, ge=1, le=100000, description="Override max tokens")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Override top-p")
    
    # Context
    context: Optional[str] = Field(None, description="Additional context")
    examples: List[Dict[str, str]] = Field(default_factory=list, description="Few-shot examples")
    
    # Metadata
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    user_id: Optional[str] = Field(None, description="User ID")
    
    class Config:
        use_enum_values = True


class LLMResponse(BaseModel):
    """LLM response model."""
    
    # Content
    content: str = Field(..., description="Generated content")
    finish_reason: str = Field(..., description="Reason for completion")
    
    # Usage Statistics
    prompt_tokens: int = Field(..., ge=0, description="Prompt tokens used")
    completion_tokens: int = Field(..., ge=0, description="Completion tokens used")
    total_tokens: int = Field(..., ge=0, description="Total tokens used")
    
    # Model Information
    model: str = Field(..., description="Model used")
    provider: LLMProvider = Field(..., description="Provider used")
    
    # Request Information
    request_id: Optional[str] = Field(None, description="Request ID")
    response_time: float = Field(..., ge=0.0, description="Response time in seconds")
    
    # Quality Metrics
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quality score")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def get_cost_estimate(self, cost_per_token: float = 0.0001) -> float:
        """Estimate cost based on token usage."""
        return self.total_tokens * cost_per_token
    
    def get_tokens_per_second(self) -> float:
        """Calculate tokens per second."""
        if self.response_time > 0:
            return self.completion_tokens / self.response_time
        return 0.0


class LLMError(BaseModel):
    """
    LLM error model (serializable).

    Historical note: older code constructs this with `message`/`retryable`;
    newer code expects `error_type`/`error_message`/`is_retryable`. This class
    accepts both to avoid runtime validation errors.
    """

    # New-style fields (preferred)
    error_type: Optional[str] = Field(None, description="Error type")
    error_message: Optional[str] = Field(None, description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")

    # Back-compat fields (accepted as input)
    message: Optional[str] = Field(None, description="Back-compat: error message")
    retryable: Optional[bool] = Field(None, description="Back-compat: whether error is retryable")

    # Request Information
    request_id: Optional[str] = Field(None, description="Request ID")
    model: Optional[str] = Field(None, description="Model that failed")
    provider: Optional[LLMProvider] = Field(None, description="Provider that failed")

    # Error Details
    retry_count: int = Field(default=0, description="Number of retries attempted")
    is_retryable: bool = Field(default=True, description="Whether error is retryable")

    # Timestamp
    occurred_at: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_fields(cls, data: Any):
        if not isinstance(data, dict):
            return data

        # Populate error_message from legacy `message` if needed.
        if not data.get("error_message") and data.get("message"):
            data["error_message"] = data.get("message")

        # Populate error_type if missing.
        if not data.get("error_type"):
            data["error_type"] = "LLMError"

        # Populate is_retryable from legacy `retryable` if provided.
        if data.get("retryable") is not None and data.get("is_retryable") is None:
            data["is_retryable"] = bool(data.get("retryable"))

        return data

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class LLMBatchRequest(BaseModel):
    """Batch LLM request model."""
    
    requests: List[LLMConfig] = Field(..., min_items=1, max_items=100, description="Batch requests")
    batch_id: str = Field(..., description="Batch ID")
    
    # Batch Parameters
    max_concurrent: int = Field(default=10, ge=1, le=50, description="Max concurrent requests")
    timeout: int = Field(default=300, ge=1, le=1800, description="Batch timeout in seconds")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Batch creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LLMBatchResponse(BaseModel):
    """Batch LLM response model."""
    
    batch_id: str = Field(..., description="Batch ID")
    responses: List[LLMResponse] = Field(..., description="Batch responses")
    errors: List[LLMError] = Field(default_factory=list, description="Batch errors")
    
    # Statistics
    total_requests: int = Field(..., description="Total requests")
    successful_requests: int = Field(..., description="Successful requests")
    failed_requests: int = Field(..., description="Failed requests")
    
    # Performance
    total_time: float = Field(..., ge=0.0, description="Total batch time")
    average_response_time: float = Field(..., ge=0.0, description="Average response time")
    
    # Timestamps
    started_at: datetime = Field(..., description="Batch start timestamp")
    completed_at: datetime = Field(..., description="Batch completion timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests > 0:
            return self.successful_requests / self.total_requests
        return 0.0
