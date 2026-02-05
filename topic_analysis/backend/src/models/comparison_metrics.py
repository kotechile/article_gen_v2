"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
ComparisonMetrics model for Google Autocomplete integration
Represents performance and quality metrics for method comparison
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator

class ComparisonMetrics(BaseModel):
    """
    Represents performance and quality metrics for method comparison
    
    Fields:
        llm_processing_time: Time for LLM-only approach
        autocomplete_processing_time: Time for autocomplete-only approach
        hybrid_processing_time: Time for hybrid approach
        llm_relevance_avg: Average relevance score for LLM results
        autocomplete_relevance_avg: Average relevance score for autocomplete results
        hybrid_relevance_avg: Average relevance score for hybrid results
        total_suggestions_found: Total unique suggestions across all methods
    """
    
    llm_processing_time: float = Field(..., ge=0, description="Time for LLM-only approach")
    autocomplete_processing_time: float = Field(..., ge=0, description="Time for autocomplete-only approach")
    hybrid_processing_time: float = Field(..., ge=0, description="Time for hybrid approach")
    llm_relevance_avg: float = Field(..., ge=0.0, le=1.0, description="Average relevance score for LLM results")
    autocomplete_relevance_avg: float = Field(..., ge=0.0, le=1.0, description="Average relevance score for autocomplete results")
    hybrid_relevance_avg: float = Field(..., ge=0.0, le=1.0, description="Average relevance score for hybrid results")
    total_suggestions_found: int = Field(..., ge=0, description="Total unique suggestions across all methods")
    
    @validator('llm_processing_time', 'autocomplete_processing_time', 'hybrid_processing_time')
    def validate_processing_times(cls, v):
        """Validate processing times are non-negative"""
        if v < 0:
            raise ValueError('Processing times must be non-negative')
        return float(v)
    
    @validator('llm_relevance_avg', 'autocomplete_relevance_avg', 'hybrid_relevance_avg')
    def validate_relevance_scores(cls, v):
        """Validate relevance scores are between 0.0 and 1.0"""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Relevance scores must be between 0.0 and 1.0')
        return float(v)
    
    @validator('total_suggestions_found')
    def validate_total_suggestions(cls, v):
        """Validate total suggestions is non-negative"""
        if v < 0:
            raise ValueError('Total suggestions must be non-negative')
        return int(v)
    
    def get_fastest_method(self) -> str:
        """Get the method with the fastest processing time"""
        times = {
            'LLM Only': self.llm_processing_time,
            'Autocomplete Only': self.autocomplete_processing_time,
            'Hybrid': self.hybrid_processing_time
        }
        return min(times, key=times.get)
    
    def get_highest_relevance_method(self) -> str:
        """Get the method with the highest average relevance score"""
        relevances = {
            'LLM Only': self.llm_relevance_avg,
            'Autocomplete Only': self.autocomplete_relevance_avg,
            'Hybrid': self.hybrid_relevance_avg
        }
        return max(relevances, key=relevances.get)
    
    def calculate_efficiency_score(self, method: str) -> float:
        """
        Calculate efficiency score (relevance per second)
        
        Args:
            method: Method name ('LLM Only', 'Autocomplete Only', 'Hybrid')
            
        Returns:
            Efficiency score (relevance per second)
        """
        method_data = {
            'LLM Only': (self.llm_relevance_avg, self.llm_processing_time),
            'Autocomplete Only': (self.autocomplete_relevance_avg, self.autocomplete_processing_time),
            'Hybrid': (self.hybrid_relevance_avg, self.hybrid_processing_time)
        }
        
        if method not in method_data:
            raise ValueError(f'Unknown method: {method}')
        
        relevance, processing_time = method_data[method]
        if processing_time == 0:
            return float('inf') if relevance > 0 else 0.0
        
        return relevance / processing_time
    
    def get_most_efficient_method(self) -> str:
        """Get the method with the highest efficiency score"""
        efficiencies = {
            'LLM Only': self.calculate_efficiency_score('LLM Only'),
            'Autocomplete Only': self.calculate_efficiency_score('Autocomplete Only'),
            'Hybrid': self.calculate_efficiency_score('Hybrid')
        }
        return max(efficiencies, key=efficiencies.get)
    
    def calculate_quality_score(self, method: str) -> float:
        """
        Calculate quality score based on relevance and processing time
        
        Args:
            method: Method name ('LLM Only', 'Autocomplete Only', 'Hybrid')
            
        Returns:
            Quality score (0.0-1.0)
        """
        method_data = {
            'LLM Only': (self.llm_relevance_avg, self.llm_processing_time),
            'Autocomplete Only': (self.autocomplete_relevance_avg, self.autocomplete_processing_time),
            'Hybrid': (self.hybrid_relevance_avg, self.hybrid_processing_time)
        }
        
        if method not in method_data:
            raise ValueError(f'Unknown method: {method}')
        
        relevance, processing_time = method_data[method]
        
        # Quality score combines relevance and speed
        # Higher relevance = higher quality
        # Lower processing time = higher quality
        # Normalize processing time (assume max reasonable time is 5 seconds)
        normalized_time = min(processing_time / 5.0, 1.0)
        time_score = 1.0 - normalized_time
        
        # Weighted combination: 70% relevance, 30% speed
        quality_score = (0.7 * relevance) + (0.3 * time_score)
        return min(1.0, max(0.0, quality_score))
    
    def get_highest_quality_method(self) -> str:
        """Get the method with the highest quality score"""
        qualities = {
            'LLM Only': self.calculate_quality_score('LLM Only'),
            'Autocomplete Only': self.calculate_quality_score('Autocomplete Only'),
            'Hybrid': self.calculate_quality_score('Hybrid')
        }
        return max(qualities, key=qualities.get)
    
    def get_recommendation(self) -> str:
        """Get recommendation based on metrics analysis"""
        fastest = self.get_fastest_method()
        highest_relevance = self.get_highest_relevance_method()
        most_efficient = self.get_most_efficient_method()
        highest_quality = self.get_highest_quality_method()
        
        # If hybrid is best in multiple categories, recommend it
        hybrid_scores = sum([
            highest_quality == 'Hybrid',
            most_efficient == 'Hybrid',
            highest_relevance == 'Hybrid'
        ])
        
        if hybrid_scores >= 2:
            return "Hybrid approach provides the best balance of intelligence and real-world relevance"
        elif highest_quality == 'LLM Only':
            return "LLM-only approach provides the most intelligent and comprehensive results"
        elif fastest == 'Autocomplete Only' and highest_relevance == 'Autocomplete Only':
            return "Autocomplete-only approach provides the fastest and most relevant results"
        else:
            return f"Based on analysis, {highest_quality} method is recommended"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        return {
            'fastest_method': self.get_fastest_method(),
            'highest_relevance_method': self.get_highest_relevance_method(),
            'most_efficient_method': self.get_most_efficient_method(),
            'highest_quality_method': self.get_highest_quality_method(),
            'recommendation': self.get_recommendation(),
            'efficiency_scores': {
                'LLM Only': self.calculate_efficiency_score('LLM Only'),
                'Autocomplete Only': self.calculate_efficiency_score('Autocomplete Only'),
                'Hybrid': self.calculate_efficiency_score('Hybrid')
            },
            'quality_scores': {
                'LLM Only': self.calculate_quality_score('LLM Only'),
                'Autocomplete Only': self.calculate_quality_score('Autocomplete Only'),
                'Hybrid': self.calculate_quality_score('Hybrid')
            }
        }
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            'llm_processing_time': self.llm_processing_time,
            'autocomplete_processing_time': self.autocomplete_processing_time,
            'hybrid_processing_time': self.hybrid_processing_time,
            'llm_relevance_avg': self.llm_relevance_avg,
            'autocomplete_relevance_avg': self.autocomplete_relevance_avg,
            'hybrid_relevance_avg': self.hybrid_relevance_avg,
            'total_suggestions_found': self.total_suggestions_found,
            'summary': self.get_summary()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ComparisonMetrics':
        """Create from dictionary representation"""
        return cls(
            llm_processing_time=data['llm_processing_time'],
            autocomplete_processing_time=data['autocomplete_processing_time'],
            hybrid_processing_time=data['hybrid_processing_time'],
            llm_relevance_avg=data['llm_relevance_avg'],
            autocomplete_relevance_avg=data['autocomplete_relevance_avg'],
            hybrid_relevance_avg=data['hybrid_relevance_avg'],
            total_suggestions_found=data['total_suggestions_found']
        )
    
    def __str__(self) -> str:
        """String representation"""
        return (f"ComparisonMetrics(fastest={self.get_fastest_method()}, "
                f"highest_relevance={self.get_highest_relevance_method()}, "
                f"total_suggestions={self.total_suggestions_found})")
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"ComparisonMetrics(llm_processing_time={self.llm_processing_time}, "
                f"autocomplete_processing_time={self.autocomplete_processing_time}, "
                f"hybrid_processing_time={self.hybrid_processing_time}, "
                f"llm_relevance_avg={self.llm_relevance_avg}, "
                f"autocomplete_relevance_avg={self.autocomplete_relevance_avg}, "
                f"hybrid_relevance_avg={self.hybrid_relevance_avg}, "
                f"total_suggestions_found={self.total_suggestions_found})")

class ComparisonMetricsCreate(BaseModel):
    """Model for creating new comparison metrics"""
    llm_processing_time: float = Field(..., ge=0)
    autocomplete_processing_time: float = Field(..., ge=0)
    hybrid_processing_time: float = Field(..., ge=0)
    llm_relevance_avg: float = Field(..., ge=0.0, le=1.0)
    autocomplete_relevance_avg: float = Field(..., ge=0.0, le=1.0)
    hybrid_relevance_avg: float = Field(..., ge=0.0, le=1.0)
    total_suggestions_found: int = Field(..., ge=0)

class ComparisonMetricsUpdate(BaseModel):
    """Model for updating comparison metrics"""
    llm_processing_time: Optional[float] = Field(None, ge=0)
    autocomplete_processing_time: Optional[float] = Field(None, ge=0)
    hybrid_processing_time: Optional[float] = Field(None, ge=0)
    llm_relevance_avg: Optional[float] = Field(None, ge=0.0, le=1.0)
    autocomplete_relevance_avg: Optional[float] = Field(None, ge=0.0, le=1.0)
    hybrid_relevance_avg: Optional[float] = Field(None, ge=0.0, le=1.0)
    total_suggestions_found: Optional[int] = Field(None, ge=0)

class ComparisonMetricsResponse(BaseModel):
    """Response model for comparison metrics"""
    success: bool = Field(..., description="Whether the operation was successful")
    metrics: Optional[ComparisonMetrics] = Field(None, description="The comparison metrics")
    message: str = Field(..., description="Human-readable message")

