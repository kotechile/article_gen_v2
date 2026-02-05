from sqlalchemy import Column, String, Text, Boolean, Float, Integer, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy import TypeDecorator, String as SQLString
import uuid
from datetime import datetime
from src.core.database import Base

class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses PostgreSQL's UUID type, otherwise uses CHAR(36), storing as stringified hex values.
    """
    impl = SQLString
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(UUID())
        else:
            return dialect.type_descriptor(SQLString(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.UUID(value))
            else:
                return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            return value

class LLMProvider(Base):
    """Model for storing LLM provider configurations"""
    __tablename__ = "llm_providers"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)  # e.g., "OpenAI GPT-4", "Anthropic Claude"
    provider_type = Column(String(50), nullable=False)  # e.g., "openai", "anthropic", "google", "local"
    model_name = Column(String(100), nullable=False)  # e.g., "gpt-4", "claude-3-sonnet", "gemini-pro"
    
    # API Configuration
    api_key_env_var = Column(String(100), nullable=True)  # Environment variable name for API key
    base_url = Column(String(500), nullable=True)  # Custom API endpoint
    api_version = Column(String(50), nullable=True)  # API version if needed
    
    # Model Parameters
    max_tokens = Column(Integer, default=2000)
    temperature = Column(Float, default=0.7)
    top_p = Column(Float, default=1.0)
    frequency_penalty = Column(Float, default=0.0)
    presence_penalty = Column(Float, default=0.0)
    
    # Cost and Performance
    cost_per_1k_tokens = Column(Float, default=0.0)  # Cost per 1000 tokens
    max_requests_per_minute = Column(Integer, default=60)
    average_response_time_ms = Column(Integer, default=2000)
    
    # Status and Configuration
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    priority = Column(Integer, default=0)  # Higher number = higher priority
    
    # Custom configuration (JSON)
    custom_config = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_used = Column(DateTime, nullable=True)
    
    # Usage statistics
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)
    
    def __repr__(self):
        return f"<LLMProvider(name='{self.name}', provider_type='{self.provider_type}')>"

class LLMConfiguration(Base):
    """Model for storing global LLM configuration settings"""
    __tablename__ = "llm_configurations"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    
    # Global Settings
    default_provider_id = Column(GUID(), nullable=True)
    fallback_provider_id = Column(GUID(), nullable=True)
    
    # Feature Flags
    enable_llm_analysis = Column(Boolean, default=True)
    enable_auto_fallback = Column(Boolean, default=True)
    enable_cost_tracking = Column(Boolean, default=True)
    enable_usage_analytics = Column(Boolean, default=True)
    
    # Rate Limiting
    global_rate_limit_per_minute = Column(Integer, default=100)
    user_rate_limit_per_minute = Column(Integer, default=10)
    
    # Cost Management
    daily_cost_limit = Column(Float, default=50.0)
    monthly_cost_limit = Column(Float, default=1000.0)
    cost_alert_threshold = Column(Float, default=0.8)  # Alert at 80% of limit
    
    # Quality Control
    min_confidence_score = Column(Float, default=0.7)
    enable_quality_checks = Column(Boolean, default=True)
    auto_retry_failed_requests = Column(Boolean, default=True)
    max_retry_attempts = Column(Integer, default=3)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(GUID(), nullable=True)  # User who updated the config

class LLMUsageLog(Base):
    """Model for logging LLM usage for analytics and billing"""
    __tablename__ = "llm_usage_logs"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    provider_id = Column(GUID(), nullable=False, index=True)
    user_id = Column(GUID(), nullable=True, index=True)
    
    # Request Details
    topic = Column(String(255), nullable=False)
    request_timestamp = Column(DateTime, default=datetime.utcnow)
    response_time_ms = Column(Integer, nullable=False)
    
    # Token Usage
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    
    # Cost and Quality
    cost = Column(Float, default=0.0)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    quality_score = Column(Float, nullable=True)  # User rating or automated quality score
    
    # Request Metadata
    request_metadata = Column(JSON, nullable=True)  # Additional request details
    response_metadata = Column(JSON, nullable=True)  # Response details

class LLMProviderTest(Base):
    """Model for storing LLM provider test results"""
    __tablename__ = "llm_provider_tests"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    provider_id = Column(String(36), nullable=False, index=True)
    
    # Test Details
    test_topic = Column(String(255), nullable=False)
    test_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Test Results
    success = Column(Boolean, default=False)
    response_time_ms = Column(Integer, nullable=True)
    quality_score = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Generated Content
    generated_related_areas = Column(JSON, nullable=True)
    generated_affiliate_programs = Column(JSON, nullable=True)
    
    # Test Metadata
    test_parameters = Column(JSON, nullable=True)
    test_notes = Column(Text, nullable=True)
