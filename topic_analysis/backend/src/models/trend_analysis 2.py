"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
Enhanced Trend Analysis model for dataflow persistence
This model represents trend analysis results linked to specific subtopics
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field, validator
from enum import Enum

class TrendAnalysisStatus(str, Enum):
    """Trend analysis status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TrendAnalysisBase(BaseModel):
    """Base trend analysis model with common fields"""
    analysis_name: str = Field(..., min_length=1, max_length=255, description="Name of the trend analysis")
    description: Optional[str] = Field(None, description="Description of the trend analysis")
    subtopic_name: str = Field(..., min_length=1, max_length=255, description="Name of the subtopic being analyzed")
    keywords: List[str] = Field(default_factory=list, description="Keywords used in the analysis")
    timeframe: str = Field(..., min_length=1, max_length=50, description="Time period for the analysis")
    geo: Optional[str] = Field(None, max_length=100, description="Geographic region for the analysis")
    status: TrendAnalysisStatus = Field(TrendAnalysisStatus.PENDING, description="Analysis status")
    
    @validator('analysis_name')
    def validate_analysis_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Analysis name cannot be empty or whitespace only')
        return v.strip()
    
    @validator('subtopic_name')
    def validate_subtopic_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Subtopic name cannot be empty or whitespace only')
        return v.strip()
    
    @validator('description')
    def validate_description(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        return v
    
    @validator('keywords')
    def validate_keywords(cls, v):
        if v:
            # Remove empty keywords and strip whitespace
            cleaned_keywords = [kw.strip() for kw in v if kw and kw.strip()]
            if len(cleaned_keywords) != len(set(cleaned_keywords)):
                raise ValueError('Keywords must be unique')
            return cleaned_keywords
        return v
    
    @validator('timeframe')
    def validate_timeframe(cls, v):
        if not v or not v.strip():
            raise ValueError('Timeframe cannot be empty or whitespace only')
        return v.strip()
    
    @validator('geo')
    def validate_geo(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        return v.strip() if v else v

class TrendAnalysisCreate(TrendAnalysisBase):
    """Model for creating a new trend analysis"""
    topic_decomposition_id: UUID = Field(..., description="ID of the topic decomposition this analysis belongs to")

class TrendAnalysisUpdate(BaseModel):
    """Model for updating an existing trend analysis"""
    analysis_name: Optional[str] = Field(None, min_length=1, max_length=255, description="Name of the trend analysis")
    description: Optional[str] = Field(None, description="Description of the trend analysis")
    subtopic_name: Optional[str] = Field(None, min_length=1, max_length=255, description="Name of the subtopic being analyzed")
    keywords: Optional[List[str]] = Field(None, description="Keywords used in the analysis")
    timeframe: Optional[str] = Field(None, min_length=1, max_length=50, description="Time period for the analysis")
    geo: Optional[str] = Field(None, max_length=100, description="Geographic region for the analysis")
    status: Optional[TrendAnalysisStatus] = Field(None, description="Analysis status")
    trend_data: Optional[Dict[str, Any]] = Field(None, description="Trend analysis data")
    
    @validator('analysis_name')
    def validate_analysis_name(cls, v):
        if v is not None and (not v or not v.strip()):
            raise ValueError('Analysis name cannot be empty or whitespace only')
        return v.strip() if v else v
    
    @validator('subtopic_name')
    def validate_subtopic_name(cls, v):
        if v is not None and (not v or not v.strip()):
            raise ValueError('Subtopic name cannot be empty or whitespace only')
        return v.strip() if v else v
    
    @validator('description')
    def validate_description(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        return v
    
    @validator('keywords')
    def validate_keywords(cls, v):
        if v is not None:
            # Remove empty keywords and strip whitespace
            cleaned_keywords = [kw.strip() for kw in v if kw and kw.strip()]
            if len(cleaned_keywords) != len(set(cleaned_keywords)):
                raise ValueError('Keywords must be unique')
            return cleaned_keywords
        return v
    
    @validator('timeframe')
    def validate_timeframe(cls, v):
        if v is not None and (not v or not v.strip()):
            raise ValueError('Timeframe cannot be empty or whitespace only')
        return v.strip() if v else v
    
    @validator('geo')
    def validate_geo(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        return v.strip() if v else v

class TrendAnalysis(TrendAnalysisBase):
    """Complete trend analysis model with all fields"""
    id: UUID = Field(..., description="Unique identifier")
    user_id: UUID = Field(..., description="User who owns this analysis")
    topic_decomposition_id: UUID = Field(..., description="ID of the topic decomposition this analysis belongs to")
    trend_data: Optional[Dict[str, Any]] = Field(None, description="Trend analysis data")
    error_message: Optional[str] = Field(None, description="Error message if analysis failed")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }

class TrendAnalysisResponse(TrendAnalysis):
    """Trend analysis response model for API responses"""
    pass

class TrendAnalysisListResponse(BaseModel):
    """Response model for listing trend analyses"""
    items: List[TrendAnalysisResponse] = Field(..., description="List of trend analyses")
    total: int = Field(..., ge=0, description="Total number of trend analyses")
    page: int = Field(..., ge=1, description="Current page number")
    size: int = Field(..., ge=1, le=100, description="Page size")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")

class TrendAnalysisWithContentIdeas(TrendAnalysis):
    """Trend analysis model with associated content ideas"""
    content_ideas: List[dict] = Field(default_factory=list, description="Associated content ideas")

class TrendAnalysisStats(BaseModel):
    """Trend analysis statistics model"""
    total_analyses: int = Field(..., ge=0, description="Total number of trend analyses")
    completed_analyses: int = Field(..., ge=0, description="Number of completed analyses")
    failed_analyses: int = Field(..., ge=0, description="Number of failed analyses")
    pending_analyses: int = Field(..., ge=0, description="Number of pending analyses")
    in_progress_analyses: int = Field(..., ge=0, description="Number of in-progress analyses")
    average_completion_time: Optional[float] = Field(None, description="Average completion time in minutes")
    most_analyzed_subtopics: List[Dict[str, Any]] = Field(default_factory=list, description="Most analyzed subtopics")
    last_analysis: Optional[datetime] = Field(None, description="Last analysis timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class TrendData(BaseModel):
    """Model for trend data structure"""
    search_volume: Optional[int] = Field(None, ge=0, description="Search volume for the trend")
    trend_score: Optional[float] = Field(None, ge=0, le=100, description="Trend score (0-100)")
    interest_over_time: Optional[List[Dict[str, Any]]] = Field(None, description="Interest over time data")
    related_queries: Optional[List[str]] = Field(None, description="Related search queries")
    rising_queries: Optional[List[str]] = Field(None, description="Rising search queries")
    top_queries: Optional[List[str]] = Field(None, description="Top search queries")
    geo_data: Optional[List[Dict[str, Any]]] = Field(None, description="Geographic trend data")
    category_data: Optional[List[Dict[str, Any]]] = Field(None, description="Category trend data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class TrendAnalysisResult(BaseModel):
    """Model for trend analysis results"""
    analysis_id: UUID = Field(..., description="ID of the trend analysis")
    subtopic_name: str = Field(..., description="Name of the analyzed subtopic")
    trend_data: TrendData = Field(..., description="Trend analysis data")
    insights: List[str] = Field(default_factory=list, description="Key insights from the analysis")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations based on the analysis")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Confidence score for the analysis")
    generated_at: datetime = Field(..., description="When the analysis was generated")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
