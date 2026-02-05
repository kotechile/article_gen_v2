"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
EnhancedSubtopic model for Google Autocomplete integration
Represents a topic subdivision enhanced with autocomplete validation and relevance scoring
"""

from datetime import datetime
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

class SubtopicSource(str, Enum):
    """Source of the subtopic data"""
    LLM = "llm"
    AUTOCOMPLETE = "autocomplete"
    HYBRID = "hybrid"

class EnhancedSubtopic(BaseModel):
    """
    Represents a topic subdivision enhanced with autocomplete validation and relevance scoring
    
    Fields:
        id: Unique identifier
        title: The subtopic title/name
        search_volume_indicators: Metrics indicating search popularity
        autocomplete_suggestions: Related search suggestions from Google
        relevance_score: Calculated relevance score (0.0-1.0)
        source: Data source: 'llm', 'autocomplete', or 'hybrid'
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    
    id: str = Field(..., description="Unique identifier for the subtopic")
    title: str = Field(..., min_length=3, max_length=100, description="The subtopic title/name")
    search_volume_indicators: List[str] = Field(..., description="Metrics indicating search popularity")
    autocomplete_suggestions: List[Union[str, Dict[str, Any]]] = Field(..., description="Related search suggestions from Google")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Calculated relevance score (0.0-1.0)")
    source: SubtopicSource = Field(..., description="Data source: 'llm', 'autocomplete', or 'hybrid'")
    rationale: Optional[str] = Field(None, description="Why this subtopic is profitable")
    seed_keywords: List[Union[str, Dict[str, Any]]] = Field(default_factory=list, description="Seed keywords for this subtopic")
    target_audience: Optional[str] = Field(None, description="Target audience for this subtopic")
    search_volume: Optional[int] = Field(None, description="Estimated monthly search volume")
    cpc: Optional[float] = Field(None, description="Estimated CPC")
    keyword_difficulty: Optional[int] = Field(None, description="SEO Difficulty")
    trend_analysis: Optional[Dict[str, Any]] = Field(None, description="Structured trend analysis data")
    monetization_data: Optional[Dict[str, Any]] = Field(None, description="Structured monetization/offers data")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    @validator('title')
    def validate_title(cls, v):
        """Validate title is not empty and properly formatted"""
        if not v or not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()
    
    @validator('search_volume_indicators')
    def validate_search_volume_indicators(cls, v):
        """Validate search volume indicators are not empty"""
        if not v:
            raise ValueError('Search volume indicators cannot be empty')
        return v
    
    @validator('autocomplete_suggestions')
    def validate_autocomplete_suggestions(cls, v):
        """Validate autocomplete suggestions are non-empty strings or valid dicts"""
        if not v:
            return []
        
        cleaned = []
        for suggestion in v:
            if isinstance(suggestion, str):
                if suggestion and suggestion.strip():
                    cleaned.append(suggestion.strip())
            elif isinstance(suggestion, dict):
                # Check if it has a keyword field
                if suggestion.get('keyword'):
                    cleaned.append(suggestion)
        return cleaned
    
    @validator('relevance_score')
    def validate_relevance_score(cls, v):
        """Validate relevance score is within bounds"""
        if not isinstance(v, (int, float)):
            raise ValueError('Relevance score must be a number')
        return float(v)
    
    def update_relevance_score(self, new_score: float) -> None:
        """Update the relevance score and timestamp"""
        if not 0.0 <= new_score <= 1.0:
            raise ValueError('Relevance score must be between 0.0 and 1.0')
        self.relevance_score = new_score
        self.updated_at = datetime.utcnow()
    
    def add_autocomplete_suggestion(self, suggestion: str) -> None:
        """Add a new autocomplete suggestion"""
        if suggestion and suggestion.strip():
            clean_suggestion = suggestion.strip()
            if clean_suggestion not in self.autocomplete_suggestions:
                self.autocomplete_suggestions.append(clean_suggestion)
                self.updated_at = datetime.utcnow()
    
    def add_search_volume_indicator(self, indicator: str) -> None:
        """Add a new search volume indicator"""
        if indicator and indicator.strip():
            clean_indicator = indicator.strip()
            if clean_indicator not in self.search_volume_indicators:
                self.search_volume_indicators.append(clean_indicator)
                self.updated_at = datetime.utcnow()
    
    def calculate_hybrid_relevance_score(self, llm_score: float, autocomplete_score: float) -> float:
        """
        Calculate hybrid relevance score combining LLM and autocomplete scores
        
        Args:
            llm_score: Relevance score from LLM analysis
            autocomplete_score: Relevance score from autocomplete data
            
        Returns:
            Combined relevance score
        """
        if not 0.0 <= llm_score <= 1.0:
            raise ValueError('LLM score must be between 0.0 and 1.0')
        if not 0.0 <= autocomplete_score <= 1.0:
            raise ValueError('Autocomplete score must be between 0.0 and 1.0')
        
        # Weighted combination: 60% LLM, 40% autocomplete
        hybrid_score = (0.6 * llm_score) + (0.4 * autocomplete_score)
        return min(1.0, max(0.0, hybrid_score))
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'title': self.title,
            'search_volume_indicators': self.search_volume_indicators,
            'autocomplete_suggestions': self.autocomplete_suggestions,
            'relevance_score': self.relevance_score,
            'source': self.source.value,
            'rationale': self.rationale,
            'seed_keywords': self.seed_keywords,
            'target_audience': self.target_audience,
            'search_volume': self.search_volume,
            'cpc': self.cpc,
            'keyword_difficulty': self.keyword_difficulty,
            'trend_analysis': self.trend_analysis,
            'monetization_data': self.monetization_data,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EnhancedSubtopic':
        """Create from dictionary representation"""
        return cls(
            id=data['id'],
            title=data['title'],
            search_volume_indicators=data['search_volume_indicators'],
            autocomplete_suggestions=data['autocomplete_suggestions'],
            relevance_score=data['relevance_score'],
            source=SubtopicSource(data['source']),
            rationale=data.get('rationale'),
            seed_keywords=data.get('seed_keywords', []),
            target_audience=data.get('target_audience'),
            search_volume=data.get('search_volume'),
            cpc=data.get('cpc'),
            keyword_difficulty=data.get('keyword_difficulty'),
            trend_analysis=data.get('trend_analysis'),
            monetization_data=data.get('monetization_data'),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at'])
        )
    
    def __str__(self) -> str:
        """String representation"""
        return f"EnhancedSubtopic(id={self.id}, title='{self.title}', source={self.source.value}, score={self.relevance_score})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"EnhancedSubtopic(id='{self.id}', title='{self.title}', "
                f"search_volume_indicators={self.search_volume_indicators}, "
                f"autocomplete_suggestions={self.autocomplete_suggestions}, "
                f"relevance_score={self.relevance_score}, source={self.source.value}, "
                f"created_at={self.created_at}, updated_at={self.updated_at})")

class EnhancedSubtopicCreate(BaseModel):
    """Model for creating a new enhanced subtopic"""
    title: str = Field(..., min_length=3, max_length=100)
    search_volume_indicators: List[str] = Field(default_factory=list)
    autocomplete_suggestions: List[str] = Field(default_factory=list)
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    source: SubtopicSource = Field(default=SubtopicSource.LLM)

class EnhancedSubtopicUpdate(BaseModel):
    """Model for updating an enhanced subtopic"""
    title: Optional[str] = Field(None, min_length=3, max_length=100)
    search_volume_indicators: Optional[List[str]] = None
    autocomplete_suggestions: Optional[List[str]] = None
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    source: Optional[SubtopicSource] = None

class EnhancedSubtopicResponse(BaseModel):
    """Response model for enhanced subtopic"""
    success: bool = Field(..., description="Whether the operation was successful")
    subtopic: Optional[EnhancedSubtopic] = Field(None, description="The enhanced subtopic")
    message: str = Field(..., description="Human-readable message")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

