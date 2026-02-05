"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
AutocompleteResult model for Google Autocomplete integration
Represents the response from Google Autocomplete API for a specific query
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, validator

class AutocompleteResult(BaseModel):
    """
    Represents the response from Google Autocomplete API for a specific query
    
    Fields:
        query: Original search query
        suggestions: List of autocomplete suggestions
        total_suggestions: Count of suggestions returned
        processing_time: Time taken to fetch suggestions (seconds)
        timestamp: When the request was made
        success: Whether the request was successful
        error_message: Error details if request failed
    """
    
    query: str = Field(..., min_length=1, max_length=200, description="Original search query")
    suggestions: List[str] = Field(..., description="List of autocomplete suggestions")
    total_suggestions: int = Field(..., ge=0, description="Count of suggestions returned")
    processing_time: float = Field(..., ge=0, description="Time taken to fetch suggestions (seconds)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the request was made")
    success: bool = Field(..., description="Whether the request was successful")
    is_fallback: bool = Field(False, description="Whether the result contains fallback data")
    error_message: Optional[str] = Field(None, description="Error details if request failed")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query is not empty and properly formatted"""
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()
    
    @validator('suggestions')
    def validate_suggestions(cls, v):
        """Validate suggestions are non-empty strings"""
        if not v:
            return []
        return [suggestion.strip() for suggestion in v if suggestion and suggestion.strip()]
    
    @validator('total_suggestions')
    def validate_total_suggestions(cls, v, values):
        """Validate total_suggestions matches suggestions length"""
        if 'suggestions' in values:
            suggestions = values['suggestions']
            if v != len(suggestions):
                raise ValueError(f'total_suggestions ({v}) must equal length of suggestions ({len(suggestions)})')
        return v
    
    @validator('processing_time')
    def validate_processing_time(cls, v):
        """Validate processing time is positive"""
        if v < 0:
            raise ValueError('Processing time must be non-negative')
        return float(v)
    
    @validator('success')
    def validate_success_consistency(cls, v, values):
        """Validate success field is consistent with other fields"""
        if v and 'error_message' in values and values['error_message']:
            raise ValueError('Cannot have success=True with error_message present')
        if not v and 'error_message' in values and not values['error_message']:
            raise ValueError('Must have error_message when success=False')
        return v
    
    def add_suggestion(self, suggestion: str) -> None:
        """Add a new suggestion"""
        if suggestion and suggestion.strip():
            clean_suggestion = suggestion.strip()
            if clean_suggestion not in self.suggestions:
                self.suggestions.append(clean_suggestion)
                self.total_suggestions = len(self.suggestions)
    
    def remove_suggestion(self, suggestion: str) -> None:
        """Remove a suggestion"""
        if suggestion in self.suggestions:
            self.suggestions.remove(suggestion)
            self.total_suggestions = len(self.suggestions)
    
    def filter_suggestions(self, min_length: int = 3) -> None:
        """Filter suggestions by minimum length"""
        self.suggestions = [s for s in self.suggestions if len(s) >= min_length]
        self.total_suggestions = len(self.suggestions)
    
    def deduplicate_suggestions(self) -> None:
        """Remove duplicate suggestions"""
        seen = set()
        unique_suggestions = []
        for suggestion in self.suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
        self.suggestions = unique_suggestions
        self.total_suggestions = len(self.suggestions)
    
    def get_top_suggestions(self, limit: int = 10) -> List[str]:
        """Get top N suggestions"""
        return self.suggestions[:limit]
    
    def is_empty(self) -> bool:
        """Check if result has no suggestions"""
        return self.total_suggestions == 0
    
    def is_successful(self) -> bool:
        """Check if request was successful"""
        return self.success and not self.error_message
    
    def set_error(self, error_message: str) -> None:
        """Set error state"""
        self.success = False
        self.error_message = error_message
        self.suggestions = []
        self.total_suggestions = 0
    
    def clear_error(self) -> None:
        """Clear error state"""
        self.success = True
        self.error_message = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            'query': self.query,
            'suggestions': self.suggestions,
            'total_suggestions': self.total_suggestions,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'is_fallback': self.is_fallback,
            'error_message': self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AutocompleteResult':
        """Create from dictionary representation"""
        return cls(
            query=data['query'],
            suggestions=data['suggestions'],
            total_suggestions=data['total_suggestions'],
            processing_time=data['processing_time'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            success=data['success'],
            is_fallback=data.get('is_fallback', False),
            error_message=data.get('error_message')
        )
    
    @classmethod
    def create_success(cls, query: str, suggestions: List[str], processing_time: float, is_fallback: bool = False) -> 'AutocompleteResult':
        """Create a successful result"""
        return cls(
            query=query,
            suggestions=suggestions,
            total_suggestions=len(suggestions),
            processing_time=processing_time,
            success=True,
            is_fallback=is_fallback
        )
    
    @classmethod
    def create_error(cls, query: str, error_message: str, processing_time: float = 0.0) -> 'AutocompleteResult':
        """Create an error result"""
        return cls(
            query=query,
            suggestions=[],
            total_suggestions=0,
            processing_time=processing_time,
            success=False,
            error_message=error_message
        )
    
    def __str__(self) -> str:
        """String representation"""
        status = "SUCCESS" if self.success else "ERROR"
        return f"AutocompleteResult(query='{self.query}', {self.total_suggestions} suggestions, {status})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"AutocompleteResult(query='{self.query}', suggestions={self.suggestions}, "
                f"total_suggestions={self.total_suggestions}, processing_time={self.processing_time}, "
                f"timestamp={self.timestamp}, success={self.success}, "
                f"error_message='{self.error_message}')")

class AutocompleteResultCreate(BaseModel):
    """Model for creating a new autocomplete result"""
    query: str = Field(..., min_length=1, max_length=200)
    suggestions: List[str] = Field(default_factory=list)
    processing_time: float = Field(default=0.0, ge=0)

class AutocompleteResultUpdate(BaseModel):
    """Model for updating an autocomplete result"""
    suggestions: Optional[List[str]] = None
    processing_time: Optional[float] = Field(None, ge=0)
    success: Optional[bool] = None
    error_message: Optional[str] = None

class AutocompleteResultResponse(BaseModel):
    """Response model for autocomplete result"""
    success: bool = Field(..., description="Whether the operation was successful")
    result: Optional[AutocompleteResult] = Field(None, description="The autocomplete result")
    message: str = Field(..., description="Human-readable message")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

