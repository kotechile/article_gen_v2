"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
Research Topic model for dataflow persistence
This model represents the main research subject that users investigate
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field, validator
from enum import Enum

class ResearchTopicStatus(str, Enum):
    """Research topic status enumeration"""
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class ResearchTopicBase(BaseModel):
    """Base research topic model with common fields"""
    title: str = Field(..., min_length=1, max_length=255, description="Research topic title")
    description: Optional[str] = Field(None, description="Detailed description of the research topic")
    status: ResearchTopicStatus = Field(ResearchTopicStatus.ACTIVE, description="Research topic status")
    
    @validator('title')
    def validate_title(cls, v):
        if not v or not v.strip():
            raise ValueError('Title cannot be empty or whitespace only')
        return v.strip()
    
    @validator('description')
    def validate_description(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        return v

class ResearchTopicCreate(ResearchTopicBase):
    """Model for creating a new research topic"""
    pass

class ResearchTopicUpdate(BaseModel):
    """Model for updating an existing research topic"""
    title: Optional[str] = Field(None, min_length=1, max_length=255, description="Research topic title")
    description: Optional[str] = Field(None, description="Detailed description of the research topic")
    status: Optional[ResearchTopicStatus] = Field(None, description="Research topic status")
    version: int = Field(..., ge=1, description="Version number for optimistic concurrency control")
    
    @validator('title')
    def validate_title(cls, v):
        if v is not None and (not v or not v.strip()):
            raise ValueError('Title cannot be empty or whitespace only')
        return v.strip() if v else v
    
    @validator('description')
    def validate_description(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        return v

class ResearchTopic(ResearchTopicBase):
    """Complete research topic model with all fields"""
    id: UUID = Field(..., description="Unique identifier")
    user_id: UUID = Field(..., description="User who owns this research topic")
    version: int = Field(..., ge=1, description="Version number for optimistic concurrency control")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }

class ResearchTopicResponse(ResearchTopic):
    """Research topic response model for API responses"""
    pass

class ResearchTopicListResponse(BaseModel):
    """Response model for listing research topics"""
    items: List[ResearchTopicResponse] = Field(..., description="List of research topics")
    total: int = Field(..., ge=0, description="Total number of research topics")
    page: int = Field(..., ge=1, description="Current page number")
    size: int = Field(..., ge=1, le=100, description="Page size")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")

class ResearchTopicWithSubtopics(ResearchTopic):
    """Research topic model with associated subtopics"""
    subtopics: List[dict] = Field(default_factory=list, description="Associated subtopics")

class ResearchTopicComplete(ResearchTopic):
    """Complete research topic model with all related data"""
    subtopics: List[dict] = Field(default_factory=list, description="Associated subtopics")
    trend_analyses: List[dict] = Field(default_factory=list, description="Associated trend analyses")
    content_ideas: List[dict] = Field(default_factory=list, description="Associated content ideas")

class ResearchTopicStats(BaseModel):
    """Research topic statistics model"""
    total_topics: int = Field(..., ge=0, description="Total number of research topics")
    active_topics: int = Field(..., ge=0, description="Number of active research topics")
    completed_topics: int = Field(..., ge=0, description="Number of completed research topics")
    archived_topics: int = Field(..., ge=0, description="Number of archived research topics")
    total_subtopics: int = Field(..., ge=0, description="Total number of subtopics across all topics")
    total_analyses: int = Field(..., ge=0, description="Total number of trend analyses")
    total_content_ideas: int = Field(..., ge=0, description="Total number of content ideas")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
