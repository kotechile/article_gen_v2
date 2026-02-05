from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field

class KeywordBase(BaseModel):
    """Base Pydantic model for Keyword data"""
    seed_keyword: str
    keyword: str
    
    # Core DataForSEO metrics
    search_volume: Optional[int] = None
    cpc: Optional[float] = None
    competition: Optional[float] = None
    competition_level: Optional[str] = None
    difficulty: Optional[int] = None
    keyword_difficulty: Optional[int] = None
    
    # Intent
    main_intent: Optional[str] = None
    intent_type: Optional[str] = None
    
    # Additional DataForSEO fields
    low_top_of_page_bid: Optional[float] = None
    high_top_of_page_bid: Optional[float] = None
    categories: Optional[List[int]] = []
    monthly_searches: Optional[List[Dict[str, Any]]] = []
    
    # Keyword properties
    core_keyword: Optional[str] = None
    synonym_clustering_algorithm: Optional[str] = None
    detected_language: Optional[str] = None
    is_another_language: bool = False
    
    # Trend data
    monthly_trend: Optional[int] = None
    quarterly_trend: Optional[int] = None
    yearly_trend: Optional[int] = None
    
    # Profitability
    profitability_score: Optional[float] = None
    
    # Metadata
    source: str = "dataforseo_keyword_ideas"
    depth: int = 0

class KeywordCreate(KeywordBase):
    """Schema for creating a new keyword"""
    research_topic_id: UUID
    subtopic_id: Optional[UUID] = None
    user_id: UUID

class KeywordUpdate(BaseModel):
    """Schema for updating a keyword"""
    profitability_score: Optional[float] = None
    subtopic_id: Optional[UUID] = None
    search_volume: Optional[int] = None
    cpc: Optional[float] = None
    # Add other fields as needed for specific update operations

class KeywordResponse(KeywordBase):
    """Schema for keyword response"""
    id: UUID
    research_topic_id: UUID
    subtopic_id: Optional[UUID] = None
    user_id: UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class KeywordListResponse(BaseModel):
    """Schema for list of keywords response"""
    total: int
    items: List[KeywordResponse]
