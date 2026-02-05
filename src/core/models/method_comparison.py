"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
MethodComparison model for Google Autocomplete integration
Represents side-by-side analysis of different decomposition approaches
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from .enhanced_subtopic import EnhancedSubtopic

class MethodResult(BaseModel):
    """Represents results from a specific decomposition method"""
    subtopics: List[str] = Field(..., description="List of subtopics for this method")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    method: str = Field(..., description="Method name")
    
    @validator('subtopics')
    def validate_subtopics(cls, v):
        """Validate subtopics are non-empty strings"""
        if not v:
            return []
        return [subtopic.strip() for subtopic in v if subtopic and subtopic.strip()]
    
    @validator('processing_time')
    def validate_processing_time(cls, v):
        """Validate processing time is non-negative"""
        if v < 0:
            raise ValueError('Processing time must be non-negative')
        return float(v)
    
    @validator('method')
    def validate_method(cls, v):
        """Validate method name is not empty"""
        if not v or not v.strip():
            raise ValueError('Method name cannot be empty')
        return v.strip()
    
    def add_subtopic(self, subtopic: str) -> None:
        """Add a new subtopic"""
        if subtopic and subtopic.strip():
            clean_subtopic = subtopic.strip()
            if clean_subtopic not in self.subtopics:
                self.subtopics.append(clean_subtopic)
    
    def remove_subtopic(self, subtopic: str) -> None:
        """Remove a subtopic"""
        if subtopic in self.subtopics:
            self.subtopics.remove(subtopic)
    
    def get_subtopic_count(self) -> int:
        """Get the number of subtopics"""
        return len(self.subtopics)
    
    def is_empty(self) -> bool:
        """Check if method has no subtopics"""
        return len(self.subtopics) == 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            'subtopics': self.subtopics,
            'processing_time': self.processing_time,
            'method': self.method
        }

class MethodComparison(BaseModel):
    """
    Represents side-by-side analysis of different decomposition approaches
    
    Fields:
        id: Unique identifier
        original_query: The topic being analyzed
        llm_only_results: Results from LLM-only approach
        autocomplete_only_results: Results from autocomplete-only approach
        hybrid_results: Results from hybrid approach
        comparison_metrics: Performance and quality metrics
        created_at: Creation timestamp
    """
    
    id: str = Field(..., description="Unique identifier")
    original_query: str = Field(..., min_length=1, max_length=200, description="The topic being analyzed")
    llm_only_results: MethodResult = Field(..., description="Results from LLM-only approach")
    autocomplete_only_results: MethodResult = Field(..., description="Results from autocomplete-only approach")
    hybrid_results: MethodResult = Field(..., description="Results from hybrid approach")
    comparison_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance and quality metrics")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    @validator('original_query')
    def validate_original_query(cls, v):
        """Validate original query is not empty"""
        if not v or not v.strip():
            raise ValueError('Original query cannot be empty')
        return v.strip()
    
    @validator('llm_only_results', 'autocomplete_only_results', 'hybrid_results')
    def validate_method_results(cls, v):
        """Validate method results are valid"""
        if not isinstance(v, MethodResult):
            raise ValueError('Method results must be MethodResult instances')
        return v
    
    def get_fastest_method(self) -> str:
        """Get the method with the fastest processing time"""
        methods = {
            'LLM Only': self.llm_only_results.processing_time,
            'Autocomplete Only': self.autocomplete_only_results.processing_time,
            'Hybrid': self.hybrid_results.processing_time
        }
        return min(methods, key=methods.get)
    
    def get_most_comprehensive_method(self) -> str:
        """Get the method with the most subtopics"""
        methods = {
            'LLM Only': len(self.llm_only_results.subtopics),
            'Autocomplete Only': len(self.autocomplete_only_results.subtopics),
            'Hybrid': len(self.hybrid_results.subtopics)
        }
        return max(methods, key=methods.get)
    
    def calculate_efficiency_score(self, method: str) -> float:
        """
        Calculate efficiency score (subtopics per second)
        
        Args:
            method: Method name ('LLM Only', 'Autocomplete Only', 'Hybrid')
            
        Returns:
            Efficiency score (subtopics per second)
        """
        method_results = {
            'LLM Only': self.llm_only_results,
            'Autocomplete Only': self.autocomplete_only_results,
            'Hybrid': self.hybrid_results
        }
        
        if method not in method_results:
            raise ValueError(f'Unknown method: {method}')
        
        result = method_results[method]
        if result.processing_time == 0:
            return float('inf') if result.subtopics else 0.0
        
        return len(result.subtopics) / result.processing_time
    
    def get_recommendation(self) -> str:
        """Get recommendation based on comparison metrics"""
        # Calculate efficiency scores
        llm_efficiency = self.calculate_efficiency_score('LLM Only')
        autocomplete_efficiency = self.calculate_efficiency_score('Autocomplete Only')
        hybrid_efficiency = self.calculate_efficiency_score('Hybrid')
        
        # Get method with best balance of speed and comprehensiveness
        if hybrid_efficiency > max(llm_efficiency, autocomplete_efficiency):
            return "Hybrid approach provides the best balance of intelligence and real-world relevance"
        elif llm_efficiency > autocomplete_efficiency:
            return "LLM-only approach provides the most intelligent and comprehensive results"
        else:
            return "Autocomplete-only approach provides the fastest and most relevant results"
    
    def update_metrics(self) -> None:
        """Update comparison metrics"""
        self.comparison_metrics = {
            'llm_processing_time': self.llm_only_results.processing_time,
            'autocomplete_processing_time': self.autocomplete_only_results.processing_time,
            'hybrid_processing_time': self.hybrid_results.processing_time,
            'llm_subtopic_count': len(self.llm_only_results.subtopics),
            'autocomplete_subtopic_count': len(self.autocomplete_only_results.subtopics),
            'hybrid_subtopic_count': len(self.hybrid_results.subtopics),
            'fastest_method': self.get_fastest_method(),
            'most_comprehensive_method': self.get_most_comprehensive_method(),
            'llm_efficiency': self.calculate_efficiency_score('LLM Only'),
            'autocomplete_efficiency': self.calculate_efficiency_score('Autocomplete Only'),
            'hybrid_efficiency': self.calculate_efficiency_score('Hybrid'),
            'recommendation': self.get_recommendation()
        }
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'original_query': self.original_query,
            'llm_only': self.llm_only_results.to_dict(),
            'autocomplete_only': self.autocomplete_only_results.to_dict(),
            'hybrid': self.hybrid_results.to_dict(),
            'comparison_metrics': self.comparison_metrics,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MethodComparison':
        """Create from dictionary representation"""
        return cls(
            id=data['id'],
            original_query=data['original_query'],
            llm_only_results=MethodResult(**data['llm_only']),
            autocomplete_only_results=MethodResult(**data['autocomplete_only']),
            hybrid_results=MethodResult(**data['hybrid']),
            comparison_metrics=data.get('comparison_metrics', {}),
            created_at=datetime.fromisoformat(data['created_at'])
        )
    
    def __str__(self) -> str:
        """String representation"""
        return f"MethodComparison(id={self.id}, query='{self.original_query}', created={self.created_at})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"MethodComparison(id='{self.id}', original_query='{self.original_query}', "
                f"llm_only={self.llm_only_results}, autocomplete_only={self.autocomplete_only_results}, "
                f"hybrid={self.hybrid_results}, comparison_metrics={self.comparison_metrics}, "
                f"created_at={self.created_at})")

class MethodComparisonCreate(BaseModel):
    """Model for creating a new method comparison"""
    original_query: str = Field(..., min_length=1, max_length=200)
    llm_only_results: MethodResult
    autocomplete_only_results: MethodResult
    hybrid_results: MethodResult

class MethodComparisonUpdate(BaseModel):
    """Model for updating a method comparison"""
    llm_only_results: Optional[MethodResult] = None
    autocomplete_only_results: Optional[MethodResult] = None
    hybrid_results: Optional[MethodResult] = None
    comparison_metrics: Optional[Dict[str, Any]] = None

class MethodComparisonResponse(BaseModel):
    """Response model for method comparison"""
    success: bool = Field(..., description="Whether the operation was successful")
    comparison: Optional[MethodComparison] = Field(None, description="The method comparison")
    message: str = Field(..., description="Human-readable message")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

