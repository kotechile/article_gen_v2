"""
KeywordData model for DataForSEO API integration.

Represents keyword research data from DataForSEO Labs API.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class CompetitionLevel(str, Enum):
    """Enum for competition level."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class IntentType(str, Enum):
    """Enum for commercial intent."""
    INFORMATIONAL = "INFORMATIONAL"
    COMMERCIAL = "COMMERCIAL"
    TRANSACTIONAL = "TRANSACTIONAL"


class TrendDirection(str, Enum):
    """Enum for trend direction."""
    RISING = "RISING"
    FALLING = "FALLING"
    STABLE = "STABLE"


class KeywordStatus(str, Enum):
    """Enum for keyword status."""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    ERROR = "ERROR"


class KeywordSource(str, Enum):
    """Enum for keyword data source."""
    DATAFORSEO = "DATAFORSEO"
    SEMRUSH = "SEMRUSH"
    AHREFS = "AHREFS"
    UBERSUGGEST = "UBERSUGGEST"
    MANUAL = "MANUAL"


class SearchVolumeTrend(BaseModel):
    """Search volume trend data point."""
    month: str = Field(..., description="Month in YYYY-MM format")
    volume: int = Field(..., ge=0, description="Search volume for the month")
    
    @validator('month')
    def validate_month(cls, v):
        """Validate month format."""
        try:
            # Validate YYYY-MM format
            year, month = v.split('-')
            int(year)
            int(month)
            if not (1 <= int(month) <= 12):
                raise ValueError('Month must be between 01 and 12')
            return v
        except (ValueError, IndexError):
            raise ValueError('Month must be in YYYY-MM format')


class MonthlyDataPoint(BaseModel):
    """Monthly data point for trend analysis."""
    month: str = Field(..., description="Month in YYYY-MM-DD format")
    volume: int = Field(..., ge=0, description="Search volume for the month")
    
    @validator('month')
    def validate_month(cls, v):
        """Validate month format."""
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError('Month must be in YYYY-MM-DD format')


class Trends(BaseModel):
    """Trend data for keyword analysis."""
    monthly_data: Optional[List[MonthlyDataPoint]] = Field(None, description="Monthly search volume data")
    trend_direction: Optional[TrendDirection] = Field(None, description="Overall trend direction")
    trend_percentage: Optional[float] = Field(None, description="Percentage change over 12 months")
    
    @validator('trend_percentage')
    def validate_trend_percentage(cls, v):
        """Validate trend percentage is a valid number."""
        if v is not None and not isinstance(v, (int, float)):
            raise ValueError('Trend percentage must be a valid number')
        return v


class KeywordData(BaseModel):
    """Keyword data model for keyword research."""
    
    keyword: str = Field(..., min_length=1, max_length=100, description="The keyword being analyzed")
    search_volume: Optional[int] = Field(None, ge=0, description="Monthly search volume")
    keyword_difficulty: Optional[float] = Field(None, ge=0, le=100, description="Difficulty score (0-100)")
    cpc: Optional[float] = Field(None, ge=0, description="Cost per click in USD")
    competition: Optional[float] = Field(None, ge=0, le=1, description="Competition level (0-1)")
    competition_level: Optional[CompetitionLevel] = Field(None, description="Competition level category")
    trends: Optional[Trends] = Field(None, description="12-month trend data")
    related_keywords: Optional[List[str]] = Field(None, description="Related keyword suggestions")
    intent: Optional[IntentType] = Field(None, description="Commercial intent")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the data was fetched")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When the data was last updated")
    
    @validator('keyword')
    def validate_keyword(cls, v):
        """Validate keyword is not empty and trimmed."""
        if not v or not v.strip():
            raise ValueError('Keyword must not be empty')
        return v.strip()
    
    @validator('related_keywords')
    def validate_related_keywords(cls, v):
        """Validate related keywords are trimmed strings."""
        if v is not None:
            return [keyword.strip() for keyword in v if keyword.strip()]
        return v
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "keyword": "weight loss tips",
                "search_volume": 50000,
                "keyword_difficulty": 35,
                "cpc": 2.50,
                "competition": 0.65,
                "competition_level": "MEDIUM",
                "trends": {
                    "monthly_data": [
                        {"month": "2024-01-01", "volume": 45000},
                        {"month": "2024-02-01", "volume": 48000}
                    ],
                    "trend_direction": "RISING",
                    "trend_percentage": 15.5
                },
                "related_keywords": ["weight loss diet", "fat burning tips"],
                "intent": "COMMERCIAL",
                "created_at": "2025-01-14T10:00:00Z",
                "updated_at": "2025-01-14T10:00:00Z"
            }
        }


class KeywordResearchResponse(BaseModel):
    """Response model for keyword research API."""
    success: bool = Field(..., description="Whether the request was successful")
    data: List[KeywordData] = Field(..., description="List of keyword data")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "data": [
                    {
                        "keyword": "weight loss tips",
                        "search_volume": 50000,
                        "keyword_difficulty": 35,
                        "cpc": 2.50,
                        "competition": 0.65,
                        "competition_level": "MEDIUM",
                        "trends": {
                            "trend_direction": "RISING",
                            "trend_percentage": 15.5
                        },
                        "intent": "COMMERCIAL",
                        "created_at": "2025-01-14T10:00:00Z",
                        "updated_at": "2025-01-14T10:00:00Z"
                    }
                ],
                "metadata": {
                    "total_keywords": 150,
                    "filtered_keywords": 75,
                    "average_difficulty": 35.2
                }
            }
        }


class PriorityFactors(BaseModel):
    """Priority factors for keyword prioritization."""
    cpc_weight: float = Field(..., ge=0, le=1, description="Weight for CPC factor (0-1)")
    volume_weight: float = Field(..., ge=0, le=1, description="Weight for volume factor (0-1)")
    trend_weight: float = Field(..., ge=0, le=1, description="Weight for trend factor (0-1)")
    
    @validator('cpc_weight', 'volume_weight', 'trend_weight')
    def validate_weights(cls, v):
        """Validate weights are within valid range."""
        if not 0 <= v <= 1:
            raise ValueError('Weights must be between 0 and 1')
        return v
    
    @validator('cpc_weight', 'volume_weight', 'trend_weight')
    def validate_weight_sum(cls, v, values):
        """Validate that weights sum to approximately 1."""
        if 'cpc_weight' in values and 'volume_weight' in values and 'trend_weight' in values:
            total = values['cpc_weight'] + values['volume_weight'] + values['trend_weight']
            if not 0.9 <= total <= 1.1:  # Allow small floating point errors
                raise ValueError('Weights must sum to approximately 1.0')
        return v


class PrioritizedKeyword(KeywordData):
    """Keyword data with priority scoring."""
    priority_score: float = Field(..., ge=0, le=100, description="Calculated priority score")
    rank: int = Field(..., ge=1, description="Priority rank (1 = highest)")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "keyword": "weight loss tips",
                "search_volume": 50000,
                "cpc": 2.50,
                "trends": {"trend_percentage": 15.5},
                "priority_score": 85.5,
                "rank": 1
            }
        }


class KeywordPrioritizationResponse(BaseModel):
    """Response model for keyword prioritization API."""
    success: bool = Field(..., description="Whether the request was successful")
    prioritized_keywords: List[PrioritizedKeyword] = Field(..., description="List of prioritized keywords")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "prioritized_keywords": [
                    {
                        "keyword": "weight loss tips",
                        "search_volume": 50000,
                        "cpc": 2.50,
                        "trends": {"trend_percentage": 15.5},
                        "priority_score": 85.5,
                        "rank": 1
                    }
                ]
            }
        }