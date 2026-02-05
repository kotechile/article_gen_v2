"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
SearchVolumeIndicator model for Google Autocomplete integration
Represents metrics derived from autocomplete data that suggest topic popularity
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum

class IndicatorType(str, Enum):
    """Type of search volume indicator"""
    HIGH_SEARCH_VOLUME = "high_search_volume"
    TRENDING = "trending"
    COMMERCIAL_INTENT = "commercial_intent"
    LOW_COMPETITION = "low_competition"

class SearchVolumeIndicator(BaseModel):
    """
    Represents metrics derived from autocomplete data that suggest topic popularity
    
    Fields:
        indicator_type: Type of indicator (e.g., 'high_search_volume', 'trending', 'commercial_intent')
        confidence_level: Confidence in the indicator (0.0-1.0)
        description: Human-readable description of the indicator
        source_data: Raw data that led to this indicator
    """
    
    indicator_type: IndicatorType = Field(..., description="Type of indicator")
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Confidence in the indicator (0.0-1.0)")
    description: str = Field(..., min_length=1, description="Human-readable description of the indicator")
    source_data: List[str] = Field(..., description="Raw data that led to this indicator")
    
    @validator('indicator_type')
    def validate_indicator_type(cls, v):
        """Validate indicator type is valid"""
        if v not in IndicatorType:
            raise ValueError(f'Invalid indicator type: {v}')
        return v
    
    @validator('confidence_level')
    def validate_confidence_level(cls, v):
        """Validate confidence level is between 0.0 and 1.0"""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence level must be between 0.0 and 1.0')
        return float(v)
    
    @validator('description')
    def validate_description(cls, v):
        """Validate description is not empty"""
        if not v or not v.strip():
            raise ValueError('Description cannot be empty')
        return v.strip()
    
    @validator('source_data')
    def validate_source_data(cls, v):
        """Validate source data is not empty"""
        if not v:
            raise ValueError('Source data cannot be empty')
        return [item.strip() for item in v if item and item.strip()]
    
    def is_high_confidence(self) -> bool:
        """Check if indicator has high confidence"""
        return self.confidence_level >= 0.8
    
    def is_medium_confidence(self) -> bool:
        """Check if indicator has medium confidence"""
        return 0.5 <= self.confidence_level < 0.8
    
    def is_low_confidence(self) -> bool:
        """Check if indicator has low confidence"""
        return self.confidence_level < 0.5
    
    def get_confidence_level_description(self) -> str:
        """Get human-readable confidence level description"""
        if self.is_high_confidence():
            return "High confidence"
        elif self.is_medium_confidence():
            return "Medium confidence"
        else:
            return "Low confidence"
    
    def add_source_data(self, data: str) -> None:
        """Add new source data"""
        if data and data.strip():
            clean_data = data.strip()
            if clean_data not in self.source_data:
                self.source_data.append(clean_data)
    
    def remove_source_data(self, data: str) -> None:
        """Remove source data"""
        if data in self.source_data:
            self.source_data.remove(data)
    
    def update_confidence_level(self, new_level: float) -> None:
        """Update confidence level"""
        if not 0.0 <= new_level <= 1.0:
            raise ValueError('Confidence level must be between 0.0 and 1.0')
        self.confidence_level = new_level
    
    def get_source_data_count(self) -> int:
        """Get the number of source data items"""
        return len(self.source_data)
    
    def is_reliable(self) -> bool:
        """Check if indicator is reliable based on confidence and source data"""
        return self.is_high_confidence() and self.get_source_data_count() >= 3
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            'indicator_type': self.indicator_type.value,
            'confidence_level': self.confidence_level,
            'description': self.description,
            'source_data': self.source_data,
            'is_reliable': self.is_reliable(),
            'confidence_description': self.get_confidence_level_description()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SearchVolumeIndicator':
        """Create from dictionary representation"""
        return cls(
            indicator_type=IndicatorType(data['indicator_type']),
            confidence_level=data['confidence_level'],
            description=data['description'],
            source_data=data['source_data']
        )
    
    @classmethod
    def create_high_search_volume(cls, source_data: List[str], confidence: float = 0.9) -> 'SearchVolumeIndicator':
        """Create high search volume indicator"""
        return cls(
            indicator_type=IndicatorType.HIGH_SEARCH_VOLUME,
            confidence_level=confidence,
            description="High search volume detected from autocomplete data",
            source_data=source_data
        )
    
    @classmethod
    def create_trending(cls, source_data: List[str], confidence: float = 0.8) -> 'SearchVolumeIndicator':
        """Create trending indicator"""
        return cls(
            indicator_type=IndicatorType.TRENDING,
            confidence_level=confidence,
            description="Trending topic detected from autocomplete data",
            source_data=source_data
        )
    
    @classmethod
    def create_commercial_intent(cls, source_data: List[str], confidence: float = 0.7) -> 'SearchVolumeIndicator':
        """Create commercial intent indicator"""
        return cls(
            indicator_type=IndicatorType.COMMERCIAL_INTENT,
            confidence_level=confidence,
            description="Commercial intent detected from autocomplete data",
            source_data=source_data
        )
    
    @classmethod
    def create_low_competition(cls, source_data: List[str], confidence: float = 0.6) -> 'SearchVolumeIndicator':
        """Create low competition indicator"""
        return cls(
            indicator_type=IndicatorType.LOW_COMPETITION,
            confidence_level=confidence,
            description="Low competition detected from autocomplete data",
            source_data=source_data
        )
    
    def __str__(self) -> str:
        """String representation"""
        return f"SearchVolumeIndicator(type={self.indicator_type.value}, confidence={self.confidence_level:.2f})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"SearchVolumeIndicator(indicator_type={self.indicator_type.value}, "
                f"confidence_level={self.confidence_level}, description='{self.description}', "
                f"source_data={self.source_data})")

class SearchVolumeIndicatorCreate(BaseModel):
    """Model for creating a new search volume indicator"""
    indicator_type: IndicatorType
    confidence_level: float = Field(..., ge=0.0, le=1.0)
    description: str = Field(..., min_length=1)
    source_data: List[str] = Field(..., min_items=1)

class SearchVolumeIndicatorUpdate(BaseModel):
    """Model for updating a search volume indicator"""
    confidence_level: Optional[float] = Field(None, ge=0.0, le=1.0)
    description: Optional[str] = Field(None, min_length=1)
    source_data: Optional[List[str]] = None

class SearchVolumeIndicatorResponse(BaseModel):
    """Response model for search volume indicator"""
    success: bool = Field(..., description="Whether the operation was successful")
    indicator: Optional[SearchVolumeIndicator] = Field(None, description="The search volume indicator")
    message: str = Field(..., description="Human-readable message")

