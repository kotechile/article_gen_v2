"""
Schemas for Trends Analysis API
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class TrendData(BaseModel):
    id: str
    topic: str = ""
    subtopic: str = ""
    trend_score: int = Field(ge=0, le=100)
    opportunity_score: int = Field(ge=0, le=100)
    search_volume: int = Field(ge=0)
    competition: str = Field(pattern="^(low|medium|high)$")
    trend_direction: str = Field(pattern="^(up|down|stable)$")
    forecast_data: Dict[str, Any] = {}
    selected: bool = False

class TrendAnalysisRequest(BaseModel):
    topics: List[str] = Field(min_items=0)
    subtopics: List[str] = Field(min_items=0)
    geo: str = Field(default="US", max_length=10)
    time_range: str = Field(default="12m", pattern="^(1m|3m|6m|12m|5y)$")
    affiliate_research_id: Optional[str] = None

class TrendAnalysisResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: str

class AffiliateResearchResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: str

class ContentIdeaRequest(BaseModel):
    selected_trends: List[TrendData]
    affiliate_programs: List[Dict[str, Any]]
    content_types: List[str] = Field(default=["blog_post", "video", "social_media"])
    max_ideas: int = Field(default=10, ge=1, le=50)

class ContentIdeaResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: str

class TrendAnalysisSummary(BaseModel):
    total_topics: int
    high_opportunity_trends: int
    trending_up: int
    low_competition: int
