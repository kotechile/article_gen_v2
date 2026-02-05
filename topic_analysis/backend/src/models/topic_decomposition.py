"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
Enhanced Topic Decomposition model for dataflow persistence
This model represents the decomposition of research topics into subtopics
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field, validator

class SubtopicItem(BaseModel):
    """Individual subtopic item within a topic decomposition"""
    name: str = Field(..., min_length=1, max_length=255, description="Subtopic name")
    description: str = Field(..., min_length=1, description="Subtopic description")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Subtopic name cannot be empty or whitespace only')
        return v.strip()
    
    @validator('description')
    def validate_description(cls, v):
        if not v or not v.strip():
            raise ValueError('Subtopic description cannot be empty or whitespace only')
        return v.strip()

class TopicDecompositionBase(BaseModel):
    """Base topic decomposition model with common fields"""
    search_query: str = Field(..., min_length=1, max_length=500, description="Search query used for decomposition")
    subtopics: List[SubtopicItem] = Field(..., min_items=1, description="List of subtopics")
    original_topic_included: bool = Field(True, description="Whether the original topic is included as a subtopic")
    
    @validator('search_query')
    def validate_search_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Search query cannot be empty or whitespace only')
        return v.strip()
    
    @validator('subtopics')
    def validate_subtopics(cls, v):
        if not v:
            raise ValueError('At least one subtopic is required')
        
        # Check for duplicate subtopic names
        names = [subtopic.name for subtopic in v]
        if len(names) != len(set(names)):
            raise ValueError('Subtopic names must be unique')
        
        return v

class TopicDecompositionCreate(TopicDecompositionBase):
    """Model for creating a new topic decomposition"""
    research_topic_id: UUID = Field(..., description="ID of the research topic this decomposition belongs to")

class TopicDecompositionUpdate(BaseModel):
    """Model for updating an existing topic decomposition"""
    search_query: Optional[str] = Field(None, min_length=1, max_length=500, description="Search query used for decomposition")
    subtopics: Optional[List[SubtopicItem]] = Field(None, min_items=1, description="List of subtopics")
    original_topic_included: Optional[bool] = Field(None, description="Whether the original topic is included as a subtopic")
    
    @validator('search_query')
    def validate_search_query(cls, v):
        if v is not None and (not v or not v.strip()):
            raise ValueError('Search query cannot be empty or whitespace only')
        return v.strip() if v else v
    
    @validator('subtopics')
    def validate_subtopics(cls, v):
        if v is not None:
            if not v:
                raise ValueError('At least one subtopic is required')
            
            # Check for duplicate subtopic names
            names = [subtopic.name for subtopic in v]
            if len(names) != len(set(names)):
                raise ValueError('Subtopic names must be unique')
        
        return v

class TopicDecomposition(TopicDecompositionBase):
    """Complete topic decomposition model with all fields"""
    id: UUID = Field(..., description="Unique identifier")
    research_topic_id: UUID = Field(..., description="ID of the research topic this decomposition belongs to")
    user_id: UUID = Field(..., description="User who owns this decomposition")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }

class TopicDecompositionResponse(TopicDecomposition):
    """Topic decomposition response model for API responses"""
    pass

class TopicDecompositionListResponse(BaseModel):
    """Response model for listing topic decompositions"""
    items: List[TopicDecompositionResponse] = Field(..., description="List of topic decompositions")
    total: int = Field(..., ge=0, description="Total number of topic decompositions")
    page: int = Field(..., ge=1, description="Current page number")
    size: int = Field(..., ge=1, le=100, description="Page size")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")

class TopicDecompositionWithAnalyses(TopicDecomposition):
    """Topic decomposition model with associated trend analyses"""
    trend_analyses: List[dict] = Field(default_factory=list, description="Associated trend analyses")

class TopicDecompositionStats(BaseModel):
    """Topic decomposition statistics model"""
    total_decompositions: int = Field(..., ge=0, description="Total number of topic decompositions")
    total_subtopics: int = Field(..., ge=0, description="Total number of subtopics across all decompositions")
    average_subtopics_per_decomposition: float = Field(..., ge=0, description="Average number of subtopics per decomposition")
    most_common_subtopic_names: List[Dict[str, Any]] = Field(default_factory=list, description="Most common subtopic names")
    last_decomposition: Optional[datetime] = Field(None, description="Last decomposition timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SubtopicAnalysis(BaseModel):
    """Model for analyzing subtopic patterns"""
    subtopic_name: str = Field(..., description="Name of the subtopic")
    frequency: int = Field(..., ge=0, description="How often this subtopic appears")
    research_topics: List[UUID] = Field(default_factory=list, description="Research topics that contain this subtopic")
    trend_analyses_count: int = Field(..., ge=0, description="Number of trend analyses for this subtopic")
    content_ideas_count: int = Field(..., ge=0, description="Number of content ideas for this subtopic")
    
    class Config:
        json_encoders = {
            UUID: lambda v: str(v)
        }
