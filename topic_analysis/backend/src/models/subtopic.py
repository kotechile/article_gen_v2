"""
Subtopic models for API validation and documentation
"""

from typing import List, Optional, Any, Dict, Union
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field

class SubtopicBase(BaseModel):
    name: str = Field(..., min_length=1, description="Subtopic name")
    trend_direction: Optional[str] = Field(None, description="Trend direction: up, down, stable")
    trend_score: Optional[float] = Field(None, ge=0, le=100, description="Trend score (0-100)")
    seo_difficulty: Optional[float] = Field(None, ge=0, le=100, description="SEO Difficulty (0-100)")
    search_volume: Optional[int] = Field(None, ge=0, description="Search volume")
    cpc: Optional[float] = Field(None, ge=0, description="Cost Per Click")
    affiliate_offer_count: Optional[int] = Field(None, ge=0, description="Number of affiliate offers found")
    keywords: Optional[List[Union[str, Dict[str, Any]]]] = Field(default=[], description="Related keywords")
    monetization_data: Optional[Dict[str, Any]] = Field(default=None, description="Detailed monetization data including offers")
    trend_analysis: Optional[Dict[str, Any]] = Field(default=None, description="Detailed trend analysis data")

class SubtopicCreate(SubtopicBase):
    """Schema for creating a subtopic"""
    pass

class SubtopicUpdate(SubtopicBase):
    """Schema for updating a subtopic"""
    name: Optional[str] = None

class SubtopicResponse(SubtopicBase):
    """Schema for subtopic response"""
    id: UUID
    research_topic_id: UUID
    user_id: UUID
    viability_score: Optional[float] = Field(None, ge=0, le=100, description="Calculated viability score")
    interest_over_time: Optional[List[Dict[str, Any]]] = Field(default=[], description="Trend interest data")
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class SubtopicListResponse(BaseModel):
    items: List[SubtopicResponse]
    total: int
