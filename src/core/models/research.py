"""
Research-related data models and schemas.

This module defines the core data structures for research requests,
responses, and task management.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4


class ResearchDepth(str, Enum):
    """Research depth levels."""
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    DEEP = "deep"


class ResearchTone(str, Enum):
    """Article tone options."""
    ACADEMIC = "academic"
    JOURNALISTIC = "journalistic"
    CASUAL = "casual"
    TECHNICAL = "technical"
    PERSUASIVE = "persuasive"


class ResearchStatus(str, Enum):
    """Research task status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchRequest(BaseModel):
    """Request model for research endpoint."""
    
    brief: str = Field(..., min_length=10, max_length=3000, description="Research brief or topic")
    keywords: str = Field(..., min_length=1, max_length=500, description="Comma-separated keywords")
    
    # LLM Configuration
    provider: str = Field(..., description="LLM provider (e.g., 'openai', 'anthropic')")
    model: str = Field(..., description="Model name (e.g., 'gpt-4', 'claude-3.5-sonnet')")
    api_key: str = Field(..., min_length=1, description="LLM API key")
    
    # Research Parameters
    depth: ResearchDepth = Field(default=ResearchDepth.STANDARD, description="Research depth")
    tone: ResearchTone = Field(default=ResearchTone.JOURNALISTIC, description="Article tone")
    target_word_count: int = Field(default=2000, ge=500, le=10000, description="Target word count")
    
    # Optional Features
    claims_research_enabled: bool = Field(default=True, description="Enable claims research")
    rag_enabled: bool = Field(default=True, description="Enable RAG evidence collection")
    include_in_text_citations: bool = Field(default=True, description="Include in-text citation references like [^1], [^2] in the content")
    
    # RAG Configuration (optional)
    rag_collection: Optional[str] = Field(None, description="RAG collection name")
    rag_endpoint: Optional[str] = Field(None, description="RAG endpoint URL")
    rag_llm_provider: Optional[str] = Field(None, description="RAG LLM provider")
    
    @validator('rag_endpoint')
    def validate_rag_endpoint(cls, v, values):
        """Validate RAG endpoint if RAG is enabled."""
        if values.get('rag_enabled') and values.get('rag_collection') and not v:
            raise ValueError('rag_endpoint is required when rag_collection is provided')
        return v
    
    @validator('rag_llm_provider')
    def validate_rag_llm_provider(cls, v, values):
        """Validate RAG LLM provider if RAG is enabled."""
        if values.get('rag_enabled') and values.get('rag_collection') and not v:
            raise ValueError('rag_llm_provider is required when rag_collection is provided')
        return v
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ResearchResponse(BaseModel):
    """Response model for research endpoint."""
    
    research_id: str = Field(..., description="Unique research task ID")
    status: ResearchStatus = Field(..., description="Current task status")
    brief: str = Field(..., description="Research brief")
    model: str = Field(..., description="LLM model used")
    depth: ResearchDepth = Field(..., description="Research depth")
    tone: ResearchTone = Field(..., description="Article tone")
    target_word_count: int = Field(..., description="Target word count")
    created_at: datetime = Field(..., description="Task creation timestamp")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ResearchProgress(BaseModel):
    """Progress tracking for research tasks."""
    
    task_id: str = Field(..., description="Task ID")
    status: ResearchStatus = Field(..., description="Current status")
    progress_percent: int = Field(0, ge=0, le=100, description="Progress percentage")
    current_step: str = Field(..., description="Current processing step")
    message: str = Field(..., description="Progress message")
    stage: str = Field(..., description="Processing stage")
    eta: Optional[datetime] = Field(None, description="Estimated time to completion")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Progress timestamp")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ResearchTask(BaseModel):
    """Complete research task model."""
    
    task_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique task ID")
    brief: str = Field(..., description="Research brief")
    keywords: List[str] = Field(..., description="Keywords list")
    
    # LLM Configuration
    provider: str = Field(..., description="LLM provider")
    model: str = Field(..., description="LLM model")
    api_key: str = Field(..., description="LLM API key")
    
    # Research Parameters
    depth: ResearchDepth = Field(..., description="Research depth")
    tone: ResearchTone = Field(..., description="Article tone")
    target_word_count: int = Field(..., description="Target word count")
    
    # Task Management
    status: ResearchStatus = Field(default=ResearchStatus.PENDING, description="Task status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    
    # Results
    claims: List[str] = Field(default_factory=list, description="Extracted claims")
    evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Collected evidence")
    article: Optional[Dict[str, Any]] = Field(None, description="Generated article")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Article citations")
    
    # Error Handling
    error: Optional[str] = Field(None, description="Error message if failed")
    retry_count: int = Field(default=0, description="Number of retries")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def update_status(self, status: ResearchStatus, message: str = None):
        """Update task status and timestamp."""
        self.status = status
        self.updated_at = datetime.utcnow()
        
        if status == ResearchStatus.IN_PROGRESS and not self.started_at:
            self.started_at = datetime.utcnow()
        elif status in [ResearchStatus.COMPLETED, ResearchStatus.FAILED]:
            self.completed_at = datetime.utcnow()
    
    def add_error(self, error: str):
        """Add error and update status."""
        self.error = error
        self.update_status(ResearchStatus.FAILED)
    
    def increment_retry(self):
        """Increment retry count."""
        self.retry_count += 1
        self.updated_at = datetime.utcnow()
