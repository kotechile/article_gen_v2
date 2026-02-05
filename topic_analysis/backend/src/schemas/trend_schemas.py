"""
Trend Analysis API schemas
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class TrendDirection(str, Enum):
    """Trend direction enumeration"""
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    VOLATILE = "volatile"

class AnalysisStatus(str, Enum):
    """Analysis status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TrendingTopicsRequest(BaseModel):
    """Request schema for discovering trending topics"""
    search_term: str = Field(..., min_length=1, max_length=200, description="Search term to analyze")
    selected_programs: List[str] = Field(..., description="Selected affiliate program IDs")
    user_id: Optional[str] = Field(None, description="User ID")

class TrendingTopic(BaseModel):
    """Schema for individual trending topic"""
    id: str = Field(..., description="Topic ID")
    topic: str = Field(..., description="Trending topic title")
    search_volume: int = Field(..., ge=0, description="Monthly search volume")
    trend_direction: TrendDirection = Field(..., description="Trend direction")
    competition: str = Field(..., description="Competition level")
    opportunity_score: int = Field(..., ge=0, le=100, description="Opportunity score (0-100)")
    related_keywords: List[str] = Field(..., description="Related keywords")
    content_ideas: List[str] = Field(..., description="Content ideas for this topic")
    trend_angle: str = Field(..., description="Trend angle/category")
    target_audience: str = Field(..., description="Target audience")
    seasonality: str = Field(..., description="Seasonal pattern")
    difficulty: str = Field(..., description="Content creation difficulty")

class TrendingTopicsResponse(BaseModel):
    """Response schema for trending topics discovery"""
    success: bool = Field(..., description="Success status")
    message: str = Field(..., description="Response message")
    data: Dict[str, Any] = Field(..., description="Trending topics data")

class TrendAnalysisHistoryResponse(BaseModel):
    """Response schema for trend analysis history"""
    success: bool = Field(..., description="Success status")
    message: str = Field(..., description="Response message")
    data: Dict[str, Any] = Field(..., description="History data")

class TrendAnalysisRequest(BaseModel):
    """Request schema for trend analysis"""
    search_term: str = Field(..., min_length=1, max_length=200, description="Search term")
    analysis_type: str = Field(..., description="Type of analysis")
    filters: Optional[Dict[str, Any]] = Field(None, description="Analysis filters")
    user_id: Optional[str] = Field(None, description="User ID")

class TrendAnalysisResponse(BaseModel):
    """Response schema for trend analysis"""
    id: str = Field(..., description="Analysis ID")
    search_term: str = Field(..., description="Search term")
    analysis_type: str = Field(..., description="Analysis type")
    trending_topics: List[TrendingTopic] = Field(..., description="Trending topics")
    created_at: datetime = Field(..., description="Creation timestamp")
    status: AnalysisStatus = Field(..., description="Analysis status")

class TrendAnalysisListResponse(BaseModel):
    """Response schema for trend analysis list"""
    analyses: List[TrendAnalysisResponse] = Field(..., description="List of analyses")
    total: int = Field(..., description="Total number of analyses")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")

class TrendAnalysisUpdate(BaseModel):
    """Request schema for updating trend analysis"""
    status: Optional[AnalysisStatus] = Field(None, description="New status")
    search_term: Optional[str] = Field(None, min_length=1, max_length=200, description="Updated search term")
    analysis_type: Optional[str] = Field(None, description="Updated analysis type")
    filters: Optional[Dict[str, Any]] = Field(None, description="Updated filters")

class TrendingKeywordsResponse(BaseModel):
    """Response schema for trending keywords"""
    success: bool = Field(..., description="Success status")
    message: str = Field(..., description="Response message")
    data: Dict[str, Any] = Field(..., description="Trending keywords data")

class TrendMetricsResponse(BaseModel):
    """Response schema for trend metrics"""
    total_searches: int = Field(..., description="Total searches performed")
    trending_categories: List[Dict[str, Any]] = Field(..., description="Trending categories")
    popular_keywords: List[str] = Field(..., description="Popular keywords")
    average_opportunity_score: float = Field(..., description="Average opportunity score")
    top_trending_topics: List[Dict[str, Any]] = Field(..., description="Top trending topics")
    last_updated: datetime = Field(..., description="Last update timestamp")

class TrendComparisonRequest(BaseModel):
    """Request schema for trend comparison"""
    search_terms: List[str] = Field(..., min_items=2, max_items=5, description="Search terms to compare")
    user_id: Optional[str] = Field(None, description="User ID")

class TrendComparisonResponse(BaseModel):
    """Response schema for trend comparison"""
    comparisons: List[Dict[str, Any]] = Field(..., description="Trend comparisons")
    winner: str = Field(..., description="Best performing search term")
    insights: List[str] = Field(..., description="Comparison insights")
    recommendations: List[str] = Field(..., description="Recommendations")

class SeasonalTrendRequest(BaseModel):
    """Request schema for seasonal trend analysis"""
    search_term: str = Field(..., min_length=1, max_length=200, description="Search term")
    months: Optional[List[int]] = Field(None, description="Specific months to analyze")
    user_id: Optional[str] = Field(None, description="User ID")

class SeasonalTrendResponse(BaseModel):
    """Response schema for seasonal trend analysis"""
    search_term: str = Field(..., description="Search term")
    seasonal_patterns: Dict[str, Any] = Field(..., description="Seasonal patterns")
    peak_months: List[int] = Field(..., description="Peak months")
    low_months: List[int] = Field(..., description="Low months")
    recommendations: List[str] = Field(..., description="Seasonal recommendations")
    created_at: datetime = Field(..., description="Creation timestamp")

class TrendForecastResponse(BaseModel):
    """Response schema for trend forecasting"""
    search_term: str = Field(..., description="Search term")
    forecast_data: Dict[str, Any] = Field(..., description="Forecast data")
    confidence_interval: Dict[str, float] = Field(..., description="Confidence interval")
    predictions: List[Dict[str, Any]] = Field(..., description="Future predictions")
    accuracy_score: float = Field(..., ge=0.0, le=1.0, description="Forecast accuracy score")
    created_at: datetime = Field(..., description="Creation timestamp")