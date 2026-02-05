"""
SubtopicData model for DataForSEO API integration.

Represents subtopics from affiliate research that serve as input for trend analysis.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
from .trend_data import TrendData, TrendingStatus


class SubtopicSource(str, Enum):
    """Enum for subtopic source."""
    AFFILIATE_RESEARCH = "affiliate_research"
    USER_INPUT = "user_input"
    API_SUGGESTION = "api_suggestion"


class SubtopicData(BaseModel):
    """Subtopic data model for trend analysis input."""
    
    id: str = Field(..., min_length=1, description="Unique identifier")
    topic: str = Field(..., min_length=1, max_length=200, description="The subtopic name")
    source: SubtopicSource = Field(..., description="Source of the subtopic")
    trend_data: Optional[TrendData] = Field(None, description="Associated trend data")
    related_subtopics: Optional[List[str]] = Field(None, description="Related subtopic suggestions")
    trending_status: Optional[TrendingStatus] = Field(None, description="Current trending status")
    growth_potential: Optional[float] = Field(None, ge=0, le=100, description="Growth potential score (0-100)")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the subtopic was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When the subtopic was last updated")
    
    @validator('id')
    def validate_id(cls, v):
        """Validate ID is not empty and trimmed."""
        if not v or not v.strip():
            raise ValueError('ID must not be empty')
        return v.strip()
    
    @validator('topic')
    def validate_topic(cls, v):
        """Validate topic is not empty and trimmed."""
        if not v or not v.strip():
            raise ValueError('Topic must not be empty')
        return v.strip()
    
    @validator('related_subtopics')
    def validate_related_subtopics(cls, v):
        """Validate related subtopics are trimmed strings."""
        if v is not None:
            return [subtopic.strip() for subtopic in v if subtopic.strip()]
        return v
    
    @validator('growth_potential')
    def validate_growth_potential(cls, v):
        """Validate growth potential is within valid range."""
        if v is not None and not (0 <= v <= 100):
            raise ValueError('Growth potential must be between 0 and 100')
        return v
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "id": "subtopic_001",
                "topic": "weight loss",
                "source": "affiliate_research",
                "trending_status": "TRENDING",
                "growth_potential": 85,
                "related_subtopics": ["keto diet", "intermittent fasting"],
                "created_at": "2025-01-14T10:00:00Z",
                "updated_at": "2025-01-14T10:00:00Z"
            }
        }


class SubtopicDataResponse(BaseModel):
    """Response model for subtopic data API."""
    success: bool = Field(..., description="Whether the request was successful")
    data: List[SubtopicData] = Field(..., description="List of subtopic data")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "data": [
                    {
                        "id": "subtopic_001",
                        "topic": "weight loss",
                        "source": "affiliate_research",
                        "trending_status": "TRENDING",
                        "growth_potential": 85,
                        "related_subtopics": ["keto diet", "intermittent fasting"],
                        "created_at": "2025-01-14T10:00:00Z",
                        "updated_at": "2025-01-14T10:00:00Z"
                    }
                ],
                "metadata": {
                    "total_subtopics": 1,
                    "analysis_date": "2025-01-14T10:00:00Z"
                }
            }
        }
