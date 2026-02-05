"""
Analytics Schemas
Pydantic models for analytics-related requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class UserAnalyticsRequest(BaseModel):
    """Request model for user analytics"""
    user_id: str = Field(..., description="User ID")
    start_date: Optional[datetime] = Field(None, description="Start date for analytics")
    end_date: Optional[datetime] = Field(None, description="End date for analytics")

class PlatformAnalyticsRequest(BaseModel):
    """Request model for platform analytics"""
    start_date: Optional[datetime] = Field(None, description="Start date for analytics")
    end_date: Optional[datetime] = Field(None, description="End date for analytics")

class ContentPerformanceRequest(BaseModel):
    """Request model for content performance analytics"""
    user_id: str = Field(..., description="User ID")
    content_ids: Optional[List[str]] = Field(None, description="List of content IDs to analyze")
    start_date: Optional[datetime] = Field(None, description="Start date for analytics")
    end_date: Optional[datetime] = Field(None, description="End date for analytics")

class TrendAnalyticsRequest(BaseModel):
    """Request model for trend analytics"""
    user_id: str = Field(..., description="User ID")
    start_date: Optional[datetime] = Field(None, description="Start date for analytics")
    end_date: Optional[datetime] = Field(None, description="End date for analytics")

class AnalyticsExportRequest(BaseModel):
    """Request model for analytics export"""
    user_id: str = Field(..., description="User ID")
    report_type: str = Field(..., description="Type of report (user, content, trend, platform)")
    format: str = Field("json", description="Export format (json, csv, pdf)")
    start_date: Optional[datetime] = Field(None, description="Start date for report")
    end_date: Optional[datetime] = Field(None, description="End date for report")

class UserAnalytics(BaseModel):
    """Model for user analytics data"""
    activity_data: Dict[str, Any] = Field(..., description="User activity data")
    content_data: Dict[str, Any] = Field(..., description="Content performance data")
    trend_data: Dict[str, Any] = Field(..., description="Trend analysis data")
    period: Dict[str, str] = Field(..., description="Analytics period")

class PlatformAnalytics(BaseModel):
    """Model for platform analytics data"""
    platform_stats: Dict[str, Any] = Field(..., description="Platform statistics")
    user_growth: Dict[str, Any] = Field(..., description="User growth data")
    content_creation: Dict[str, Any] = Field(..., description="Content creation statistics")
    period: Dict[str, str] = Field(..., description="Analytics period")

class ContentPerformanceAnalytics(BaseModel):
    """Model for content performance analytics"""
    performance_metrics: List[Dict[str, Any]] = Field(..., description="Performance metrics")
    engagement_data: Dict[str, Any] = Field(..., description="Engagement data")
    content_ids: Optional[List[str]] = Field(None, description="Content IDs analyzed")
    period: Dict[str, str] = Field(..., description="Analytics period")

class TrendAnalytics(BaseModel):
    """Model for trend analytics data"""
    trend_analyses: List[Dict[str, Any]] = Field(..., description="Trend analyses")
    accuracy_data: Dict[str, Any] = Field(..., description="Trend accuracy data")
    period: Dict[str, str] = Field(..., description="Analytics period")

class AnalyticsInsight(BaseModel):
    """Model for analytics insight"""
    type: str = Field(..., description="Insight type")
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Insight description")
    value: Optional[Any] = Field(None, description="Insight value")
    trend: Optional[str] = Field(None, description="Trend direction")
    recommendation: Optional[str] = Field(None, description="Recommendation")

class AnalyticsRecommendation(BaseModel):
    """Model for analytics recommendation"""
    type: str = Field(..., description="Recommendation type")
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Recommendation description")
    priority: str = Field(..., description="Recommendation priority")
    action: Optional[str] = Field(None, description="Recommended action")

class AnalyticsExport(BaseModel):
    """Model for analytics export"""
    file_type: str = Field(..., description="Export file type")
    file_size: int = Field(..., description="Export file size")
    download_url: str = Field(..., description="Download URL")
    generated_at: str = Field(..., description="Generation timestamp")

