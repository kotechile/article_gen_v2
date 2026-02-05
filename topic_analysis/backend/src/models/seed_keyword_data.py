"""
SeedKeywordData model for DataForSEO API integration.

Represents seed keywords from Idea Burst module for keyword research expansion.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
from .keyword_data import IntentType


class SeedKeywordSource(str, Enum):
    """Enum for seed keyword source."""
    IDEA_BURST = "idea_burst"
    USER_INPUT = "user_input"


class KeywordFilters(BaseModel):
    """Filters applied to keyword research."""
    min_volume: Optional[int] = Field(None, ge=0, description="Minimum search volume")
    max_difficulty: Optional[float] = Field(None, ge=0, le=100, description="Maximum keyword difficulty")
    intent_types: Optional[List[IntentType]] = Field(None, description="Allowed intent types")
    
    @validator('min_volume')
    def validate_min_volume(cls, v):
        """Validate minimum volume is non-negative."""
        if v is not None and v < 0:
            raise ValueError('Minimum volume must be non-negative')
        return v
    
    @validator('max_difficulty')
    def validate_max_difficulty(cls, v):
        """Validate maximum difficulty is within valid range."""
        if v is not None and not (0 <= v <= 100):
            raise ValueError('Maximum difficulty must be between 0 and 100')
        return v


class SeedKeywordData(BaseModel):
    """Seed keyword data model for keyword research expansion."""
    
    id: str = Field(..., min_length=1, description="Unique identifier")
    keyword: str = Field(..., min_length=1, max_length=100, description="The seed keyword")
    source: SeedKeywordSource = Field(..., description="Source of the keyword")
    max_difficulty: Optional[float] = Field(None, ge=0, le=100, description="Maximum keyword difficulty threshold")
    generated_keywords: Optional[List[str]] = Field(None, description="Generated keyword suggestions")
    filters: Optional[KeywordFilters] = Field(None, description="Applied filters")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the keyword was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When the keyword was last updated")
    
    @validator('id')
    def validate_id(cls, v):
        """Validate ID is not empty and trimmed."""
        if not v or not v.strip():
            raise ValueError('ID must not be empty')
        return v.strip()
    
    @validator('keyword')
    def validate_keyword(cls, v):
        """Validate keyword is not empty and trimmed."""
        if not v or not v.strip():
            raise ValueError('Keyword must not be empty')
        return v.strip()
    
    @validator('max_difficulty')
    def validate_max_difficulty(cls, v):
        """Validate maximum difficulty is within valid range."""
        if v is not None and not (0 <= v <= 100):
            raise ValueError('Maximum difficulty must be between 0 and 100')
        return v
    
    @validator('generated_keywords')
    def validate_generated_keywords(cls, v):
        """Validate generated keywords are trimmed strings."""
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
                "id": "seed_001",
                "keyword": "weight loss tips",
                "source": "idea_burst",
                "max_difficulty": 50,
                "generated_keywords": ["weight loss diet", "fat burning tips"],
                "filters": {
                    "min_volume": 1000,
                    "max_difficulty": 50,
                    "intent_types": ["COMMERCIAL", "TRANSACTIONAL"]
                },
                "created_at": "2025-01-14T10:00:00Z",
                "updated_at": "2025-01-14T10:00:00Z"
            }
        }


class SeedKeywordDataResponse(BaseModel):
    """Response model for seed keyword data API."""
    success: bool = Field(..., description="Whether the request was successful")
    data: List[SeedKeywordData] = Field(..., description="List of seed keyword data")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "data": [
                    {
                        "id": "seed_001",
                        "keyword": "weight loss tips",
                        "source": "idea_burst",
                        "max_difficulty": 50,
                        "generated_keywords": ["weight loss diet", "fat burning tips"],
                        "filters": {
                            "min_volume": 1000,
                            "max_difficulty": 50,
                            "intent_types": ["COMMERCIAL", "TRANSACTIONAL"]
                        },
                        "created_at": "2025-01-14T10:00:00Z",
                        "updated_at": "2025-01-14T10:00:00Z"
                    }
                ],
                "metadata": {
                    "total_seed_keywords": 1,
                    "analysis_date": "2025-01-14T10:00:00Z"
                }
            }
        }
