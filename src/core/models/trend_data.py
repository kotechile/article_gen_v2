"""
TrendData model for DataForSEO API integration.

Represents keyword popularity trends over time and regional data from DataForSEO Trends API.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class TrendingStatus(str, Enum):
    """Enum for trending status."""
    TRENDING = "TRENDING"
    STABLE = "STABLE"
    DECLINING = "DECLINING"


class TimeSeriesPoint(BaseModel):
    """Time series data point for trend analysis."""
    date: str = Field(..., description="Date of the data point in ISO 8601 format")
    value: float = Field(..., ge=0, le=100, description="Search interest value (0-100 scale)")
    
    @validator('date')
    def validate_date(cls, v):
        """Validate ISO 8601 date format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError('Date must be in ISO 8601 format')


class Demographics(BaseModel):
    """Demographic breakdown for trend data."""
    age_groups: Optional[List[Dict[str, Any]]] = Field(None, description="Age group data")
    gender: Optional[Dict[str, float]] = Field(None, description="Gender distribution")


class GeographicData(BaseModel):
    """Geographic data point for trend analysis."""
    location_code: int = Field(..., description="DataForSEO location code")
    location_name: str = Field(..., description="Human-readable location name")
    interest_value: float = Field(..., ge=0, le=100, description="Search interest value (0-100 scale)")
    region_type: str = Field(..., description="Type of region (country, state, city, etc.)")
    
    @validator('location_name')
    def validate_location_name(cls, v):
        """Validate location name is not empty and trimmed."""
        if not v or not v.strip():
            raise ValueError('Location name must not be empty')
        return v.strip()
    
    @validator('region_type')
    def validate_region_type(cls, v):
        """Validate region type is not empty and trimmed."""
        if not v or not v.strip():
            raise ValueError('Region type must not be empty')
        return v.strip()


class TrendData(BaseModel):
    """Trend data model for keyword popularity trends."""
    
    keyword: str = Field(..., min_length=1, max_length=100, description="The keyword being analyzed")
    location: str = Field(..., min_length=1, description="Geographic location")
    time_series: List[TimeSeriesPoint] = Field(default_factory=list, description="Array of time-series data points")
    demographics: Optional[Demographics] = Field(None, description="Demographic breakdown")
    geographic_data: Optional[List[GeographicData]] = Field(None, description="Geographic breakdown by region")
    related_queries: Optional[List[str]] = Field(None, description="Related trending queries")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the data was fetched")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When the data was last updated")
    
    # Calculated fields for frontend compatibility
    average_interest: Optional[float] = Field(None, description="Average interest over time period")
    peak_interest: Optional[float] = Field(None, description="Peak interest value")
    
    @validator('keyword')
    def validate_keyword(cls, v):
        """Validate keyword is not empty and trimmed."""
        if not v or not v.strip():
            raise ValueError('Keyword must not be empty')
        return v.strip()
    
    @validator('location')
    def validate_location(cls, v):
        """Validate location is not empty and trimmed."""
        if not v or not v.strip():
            raise ValueError('Location must not be empty')
        return v.strip()
    
    @validator('time_series')
    def validate_time_series(cls, v):
        """Validate time series data points."""
        return v or []
    
    @validator('related_queries')
    def validate_related_queries(cls, v):
        """Validate related queries are trimmed strings."""
        if v is not None:
            return [query.strip() for query in v if query.strip()]
        return v
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "keyword": "weight loss",
                "location": "United States",
                "time_series": [
                    {"date": "2024-01-01", "value": 75},
                    {"date": "2024-02-01", "value": 82}
                ],
                "demographics": {
                    "age_groups": [
                        {"age_range": "25-34", "percentage": 45},
                        {"age_range": "35-44", "percentage": 35}
                    ]
                },
                "related_queries": ["weight loss tips", "diet plans"],
                "created_at": "2025-01-14T10:00:00Z",
                "updated_at": "2025-01-14T10:00:00Z"
            }
        }


class TrendDataResponse(BaseModel):
    """Response model for trend analysis API."""
    success: bool = Field(..., description="Whether the request was successful")
    data: List[TrendData] = Field(..., description="List of trend data")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "data": [
                    {
                        "keyword": "weight loss",
                        "location": "United States",
                        "time_series": [
                            {"date": "2024-01-01", "value": 75},
                            {"date": "2024-02-01", "value": 82}
                        ],
                        "demographics": {
                            "age_groups": [
                                {"age_range": "25-34", "percentage": 45},
                                {"age_range": "35-44", "percentage": 35}
                            ]
                        },
                        "related_queries": ["weight loss tips", "diet plans"],
                        "created_at": "2025-01-14T10:00:00Z",
                        "updated_at": "2025-01-14T10:00:00Z"
                    }
                ],
                "metadata": {
                    "total_subtopics": 1,
                    "analysis_date": "2025-01-14T10:00:00Z",
                    "cache_status": "fresh"
                }
            }
        }


class TrendComparisonResponse(BaseModel):
    """Response model for trend comparison API."""
    success: bool = Field(..., description="Whether the request was successful")
    data: List[TrendData] = Field(..., description="List of trend data for comparison")
    comparison_metrics: Dict[str, Any] = Field(..., description="Comparison metrics")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "data": [
                    {
                        "keyword": "weight loss",
                        "location": "United States",
                        "time_series": [
                            {"date": "2024-01-01", "value": 75},
                            {"date": "2024-02-01", "value": 82}
                        ],
                        "created_at": "2025-01-14T10:00:00Z",
                        "updated_at": "2025-01-14T10:00:00Z"
                    }
                ],
                "comparison_metrics": {
                    "top_performer": "keto diet",
                    "growth_leader": "weight loss",
                    "average_trend": 75.5
                }
            }
        }


class SubtopicSuggestion(BaseModel):
    """Model for subtopic suggestions."""
    topic: str = Field(..., min_length=1, max_length=200, description="The suggested topic")
    trending_status: TrendingStatus = Field(..., description="Current trending status")
    growth_potential: float = Field(..., ge=0, le=100, description="Growth potential score (0-100)")
    related_queries: Optional[List[str]] = Field(None, description="Related queries")
    search_volume: Optional[int] = Field(None, ge=0, description="Search volume")
    
    @validator('topic')
    def validate_topic(cls, v):
        """Validate topic is not empty and trimmed."""
        if not v or not v.strip():
            raise ValueError('Topic must not be empty')
        return v.strip()
    
    @validator('related_queries')
    def validate_related_queries(cls, v):
        """Validate related queries are trimmed strings."""
        if v is not None:
            return [query.strip() for query in v if query.strip()]
        return v


class SubtopicSuggestionsResponse(BaseModel):
    """Response model for subtopic suggestions API."""
    success: bool = Field(..., description="Whether the request was successful")
    suggestions: List[SubtopicSuggestion] = Field(..., description="List of subtopic suggestions")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "suggestions": [
                    {
                        "topic": "intermittent fasting",
                        "trending_status": "TRENDING",
                        "growth_potential": 85,
                        "related_queries": ["16:8 fasting", "OMAD diet"],
                        "search_volume": 25000
                    }
                ]
            }
        }
