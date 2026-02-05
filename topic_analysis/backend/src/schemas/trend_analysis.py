"""
Trend Analysis Pydantic Schemas
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class TrendAnalysisSource(str, Enum):
    GOOGLE_TRENDS = "google_trends"
    CSV_UPLOAD = "csv_upload"
    SEMRUSH = "semrush"
    AHREFS = "ahrefs"
    UBERSUGGEST = "ubersuggest"
    FALLBACK = "fallback"

class TrendAnalysisStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class TrendAnalysisCreate(BaseModel):
    """Schema for creating a new trend analysis"""
    workflow_session_id: str = Field(..., description="Workflow session ID")
    analysis_name: str = Field(..., min_length=1, max_length=255, description="Name of the analysis")
    keywords: List[str] = Field(..., min_items=1, max_items=100, description="List of keywords to analyze")
    timeframe: str = Field(default="12m", description="Time period for analysis")
    geo: str = Field(default="US", description="Geographic location")
    category: Optional[int] = Field(None, description="Google Trends category ID")
    description: Optional[str] = Field(None, max_length=1000, description="Analysis description")
    topic_decomposition_id: Optional[str] = Field(None, description="Topic decomposition ID")

    @validator('timeframe')
    def validate_timeframe(cls, v):
        valid_timeframes = ['1h', '4h', '1d', '7d', '30d', '90d', '12m', '5y', 'all']
        if v not in valid_timeframes:
            raise ValueError(f'Invalid timeframe. Must be one of: {", ".join(valid_timeframes)}')
        return v

    @validator('geo')
    def validate_geo(cls, v):
        if not v or len(v) < 2:
            raise ValueError('Geo must be at least 2 characters')
        return v.upper()

    @validator('keywords')
    def validate_keywords(cls, v):
        if not v:
            raise ValueError('At least one keyword is required')
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in v:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword.strip())
        return unique_keywords

class TrendAnalysisUpdate(BaseModel):
    """Schema for updating a trend analysis"""
    analysis_name: Optional[str] = Field(None, min_length=1, max_length=255, description="Name of the analysis")
    description: Optional[str] = Field(None, max_length=1000, description="Analysis description")
    timeframe: Optional[str] = Field(None, description="Time period for analysis")
    geo: Optional[str] = Field(None, description="Geographic location")
    category: Optional[int] = Field(None, description="Google Trends category ID")

    @validator('timeframe')
    def validate_timeframe(cls, v):
        if v is not None:
            valid_timeframes = ['1h', '4h', '1d', '7d', '30d', '90d', '12m', '5y', 'all']
            if v not in valid_timeframes:
                raise ValueError(f'Invalid timeframe. Must be one of: {", ".join(valid_timeframes)}')
        return v

    @validator('geo')
    def validate_geo(cls, v):
        if v is not None:
            if not v or len(v) < 2:
                raise ValueError('Geo must be at least 2 characters')
            return v.upper()
        return v

class TrendAnalysisStart(BaseModel):
    """Schema for starting trend analysis processing"""
    source: str = Field(..., description="Source for trend analysis")

    @validator('source')
    def validate_source(cls, v):
        valid_sources = [source.value for source in TrendAnalysisSource]
        if v not in valid_sources:
            raise ValueError(f'Invalid source. Must be one of: {", ".join(valid_sources)}')
        return v

class TrendData(BaseModel):
    """Schema for trend data"""
    keyword: str = Field(..., description="Keyword")
    search_volume: int = Field(0, description="Search volume")
    trend_score: Optional[int] = Field(None, description="Trend score")
    growth_rate: Optional[float] = Field(None, description="Growth rate percentage")
    seasonality: Optional[str] = Field(None, description="Seasonality pattern")
    related_queries: Optional[List[str]] = Field(None, description="Related queries")
    time_series: Optional[List[Dict[str, Any]]] = Field(None, description="Time series data")
    source: str = Field(..., description="Data source")
    raw_data: Optional[Dict[str, Any]] = Field(None, description="Raw data from source")

class AnalysisSummary(BaseModel):
    """Schema for analysis summary"""
    total_keywords: int = Field(..., description="Total keywords analyzed")
    total_volume: int = Field(..., description="Total search volume")
    average_volume: float = Field(..., description="Average search volume")
    growing_keywords: int = Field(..., description="Number of growing keywords")
    declining_keywords: int = Field(..., description="Number of declining keywords")
    competition_levels: Dict[str, int] = Field(..., description="Competition level distribution")

class AnalysisResults(BaseModel):
    """Schema for analysis results"""
    summary: AnalysisSummary = Field(..., description="Analysis summary")
    top_keywords: List[TrendData] = Field(..., description="Top performing keywords")
    growing_keywords: List[TrendData] = Field(..., description="Growing keywords")
    declining_keywords: List[TrendData] = Field(..., description="Declining keywords")
    time_series_analysis: Dict[str, Any] = Field(..., description="Time series analysis")
    processed_at: str = Field(..., description="Processing timestamp")

class Insight(BaseModel):
    """Schema for individual insight"""
    keyword: str = Field(..., description="Keyword")
    search_volume: int = Field(..., description="Search volume")
    trend_score: Optional[int] = Field(None, description="Trend score")
    growth_rate: Optional[float] = Field(None, description="Growth rate")
    reason: str = Field(..., description="Reason for insight")
    pattern: Optional[str] = Field(None, description="Pattern type")
    peak_months: Optional[List[int]] = Field(None, description="Peak months for seasonal patterns")

class Insights(BaseModel):
    """Schema for insights"""
    insights: List[str] = Field(..., description="General insights")
    top_trending: List[Insight] = Field(..., description="Top trending keywords")
    growth_opportunities: List[Insight] = Field(..., description="Growth opportunities")
    seasonal_patterns: List[Insight] = Field(..., description="Seasonal patterns")
    competition_level: str = Field(..., description="Overall competition level")
    generated_at: str = Field(..., description="Generation timestamp")

class TrendAnalysisResponse(BaseModel):
    """Schema for trend analysis response"""
    id: str = Field(..., description="Analysis ID")
    user_id: str = Field(..., description="User ID")
    workflow_session_id: str = Field(..., description="Workflow session ID")
    topic_decomposition_id: Optional[str] = Field(None, description="Topic decomposition ID")
    analysis_name: str = Field(..., description="Analysis name")
    description: Optional[str] = Field(None, description="Analysis description")
    keywords: List[str] = Field(..., description="Keywords analyzed")
    timeframe: str = Field(..., description="Analysis timeframe")
    geo: str = Field(..., description="Geographic location")
    category: Optional[int] = Field(None, description="Google Trends category")
    trend_data: Dict[str, Any] = Field(..., description="Raw trend data")
    analysis_results: Dict[str, Any] = Field(..., description="Processed analysis results")
    insights: Dict[str, Any] = Field(..., description="Generated insights")
    source: str = Field(..., description="Data source")
    status: str = Field(..., description="Analysis status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    api_calls_made: int = Field(..., description="Number of API calls made")
    cache_hit: bool = Field(..., description="Whether result was from cache")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")

    class Config:
        from_attributes = True

class TrendAnalysisSummary(BaseModel):
    """Schema for trend analysis summary (list view)"""
    id: str = Field(..., description="Analysis ID")
    analysis_name: str = Field(..., description="Analysis name")
    description: Optional[str] = Field(None, description="Analysis description")
    keywords_count: int = Field(..., description="Number of keywords analyzed")
    timeframe: str = Field(..., description="Analysis timeframe")
    geo: str = Field(..., description="Geographic location")
    source: str = Field(..., description="Data source")
    status: str = Field(..., description="Analysis status")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    created_at: str = Field(..., description="Creation timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")

    class Config:
        from_attributes = True

class CSVUploadRequest(BaseModel):
    """Schema for CSV upload request"""
    tool_type: str = Field(default="google_trends", description="Type of tool (google_trends, semrush, ahrefs, ubersuggest)")
    workflow_session_id: Optional[str] = Field(None, description="Workflow session ID")

    @validator('tool_type')
    def validate_tool_type(cls, v):
        valid_tools = ['google_trends', 'semrush', 'ahrefs', 'ubersuggest']
        if v not in valid_tools:
            raise ValueError(f'Invalid tool type. Must be one of: {", ".join(valid_tools)}')
        return v

class CSVUploadResponse(BaseModel):
    """Schema for CSV upload response"""
    success: bool = Field(..., description="Upload success status")
    filename: str = Field(..., description="Uploaded filename")
    rows_processed: int = Field(..., description="Number of rows processed")
    trends_extracted: int = Field(..., description="Number of trends extracted")
    tool_type: str = Field(..., description="Tool type")
    trend_data: Dict[str, Any] = Field(..., description="Extracted trend data")
    processed_at: str = Field(..., description="Processing timestamp")

class CSVValidationResponse(BaseModel):
    """Schema for CSV validation response"""
    valid: bool = Field(..., description="Whether CSV structure is valid")
    columns: List[str] = Field(..., description="Detected columns")
    required_columns: List[str] = Field(..., description="Required columns")
    missing_columns: List[str] = Field(..., description="Missing required columns")
    mapped_columns: Dict[str, str] = Field(..., description="Column name mappings")
    sample_rows: List[Dict[str, Any]] = Field(..., description="Sample data rows")

class TrendAnalysisStats(BaseModel):
    """Schema for trend analysis statistics"""
    total_analyses: int = Field(..., description="Total number of analyses")
    completed_analyses: int = Field(..., description="Number of completed analyses")
    failed_analyses: int = Field(..., description="Number of failed analyses")
    pending_analyses: int = Field(..., description="Number of pending analyses")
    average_processing_time_ms: float = Field(..., description="Average processing time")
    total_api_calls: int = Field(..., description="Total API calls made")
    cache_hit_rate: float = Field(..., description="Cache hit rate")

class TrendAnalysisFilters(BaseModel):
    """Schema for trend analysis filters"""
    status: Optional[str] = Field(None, description="Filter by status")
    source: Optional[str] = Field(None, description="Filter by source")
    timeframe: Optional[str] = Field(None, description="Filter by timeframe")
    geo: Optional[str] = Field(None, description="Filter by geographic location")
    created_after: Optional[str] = Field(None, description="Filter by creation date (ISO format)")
    created_before: Optional[str] = Field(None, description="Filter by creation date (ISO format)")
    workflow_session_id: Optional[str] = Field(None, description="Filter by workflow session")

    @validator('status')
    def validate_status(cls, v):
        if v is not None:
            valid_statuses = [status.value for status in TrendAnalysisStatus]
            if v not in valid_statuses:
                raise ValueError(f'Invalid status. Must be one of: {", ".join(valid_statuses)}')
        return v

    @validator('source')
    def validate_source(cls, v):
        if v is not None:
            valid_sources = [source.value for source in TrendAnalysisSource]
            if v not in valid_sources:
                raise ValueError(f'Invalid source. Must be one of: {", ".join(valid_sources)}')
        return v

    @validator('timeframe')
    def validate_timeframe(cls, v):
        if v is not None:
            valid_timeframes = ['1h', '4h', '1d', '7d', '30d', '90d', '12m', '5y', 'all']
            if v not in valid_timeframes:
                raise ValueError(f'Invalid timeframe. Must be one of: {", ".join(valid_timeframes)}')
        return v
