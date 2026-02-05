"""
DataForSEO API integration models.

This package contains all the data models for the DataForSEO API integration feature.
"""

from .trend_data import (
    TrendData,
    TrendDataResponse,
    TrendComparisonResponse,
    SubtopicSuggestion,
    SubtopicSuggestionsResponse,
    TimeSeriesPoint,
    Demographics,
    TrendingStatus
)

from .keyword_data import (
    KeywordData,
    KeywordResearchResponse,
    KeywordPrioritizationResponse,
    PrioritizedKeyword,
    PriorityFactors,
    CompetitionLevel,
    IntentType,
    TrendDirection,
    Trends,
    MonthlyDataPoint
)

from .subtopic_data import (
    SubtopicData,
    SubtopicDataResponse,
    SubtopicSource
)

from .seed_keyword_data import (
    SeedKeywordData,
    SeedKeywordDataResponse,
    SeedKeywordSource,
    KeywordFilters
)

from .api_credentials import (
    APICredentials,
    APICredentialsResponse,
    APICredentialStatus
)

__all__ = [
    # Trend data models
    "TrendData",
    "TrendDataResponse", 
    "TrendComparisonResponse",
    "SubtopicSuggestion",
    "SubtopicSuggestionsResponse",
    "TimeSeriesPoint",
    "Demographics",
    "TrendingStatus",
    
    # Keyword data models
    "KeywordData",
    "KeywordResearchResponse",
    "KeywordPrioritizationResponse",
    "PrioritizedKeyword",
    "PriorityFactors",
    "CompetitionLevel",
    "IntentType",
    "TrendDirection",
    "Trends",
    "MonthlyDataPoint",
    
    # Subtopic data models
    "SubtopicData",
    "SubtopicDataResponse",
    "SubtopicSource",
    
    # Seed keyword data models
    "SeedKeywordData",
    "SeedKeywordDataResponse",
    "SeedKeywordSource",
    "KeywordFilters",
    
    # API credentials models
    "APICredentials",
    "APICredentialsResponse",
    "APICredentialStatus"
]