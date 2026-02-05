from typing import Optional, List
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field

class ContentTopicBase(BaseModel):
    """Base Pydantic model for Content Topic data"""
    title: str
    description: Optional[str] = None
    
    # Keyword relationships
    primary_keyword_id: Optional[UUID] = None
    supporting_keyword_ids: List[UUID] = []
    
    # Metrics
    estimated_profitability_score: Optional[float] = None
    total_search_volume: Optional[int] = None
    average_cpc: Optional[float] = None
    average_difficulty: Optional[float] = None
    
    # Intent classification
    intent_type: Optional[str] = None
    
    # Content planning
    content_type: Optional[str] = None
    target_word_count: Optional[int] = None
    priority_score: Optional[float] = None
    
    # Status tracking
    status: str = "suggested"

class ContentTopicCreate(ContentTopicBase):
    """Schema for creating a new content topic"""
    research_topic_id: UUID
    user_id: UUID

class ContentTopicUpdate(BaseModel):
    """Schema for updating a content topic"""
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    content_type: Optional[str] = None
    target_word_count: Optional[int] = None
    priority_score: Optional[float] = None
    supporting_keyword_ids: Optional[List[UUID]] = None

class ContentTopicResponse(ContentTopicBase):
    """Schema for content topic response"""
    id: UUID
    research_topic_id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class ContentTopicListResponse(BaseModel):
    """Schema for list of content topics response"""
    total: int
    items: List[ContentTopicResponse]
