"""
Data models and schemas for Content Generator V2.

This module contains all the data models, validation schemas, and
type definitions used throughout the system.
"""

from .research import (
    ResearchRequest,
    ResearchResponse,
    ResearchTask,
    ResearchProgress,
    ResearchDepth,
    ResearchTone,
    ResearchStatus
)

from .article import (
    Article,
    ArticleSection,
    ArticleMetadata,
    ArticleFormat,
    ArticleTone,
    Citation,
    CitationType
)

from .evidence import (
    Evidence,
    EvidenceSource,
    EvidenceQuality,
    EvidenceRanking,
    Claim,
    ClaimType
)

from .llm import (
    LLMProvider,
    LLMModel,
    LLMConfig,
    LLMResponse
)

from .errors import (
    ContentGeneratorError,
    ValidationError,
    LLMError,
    RAGError,
    SearchError,
    ArticleGenerationError
)

__all__ = [
    # Research models
    'ResearchRequest',
    'ResearchResponse', 
    'ResearchTask',
    'ResearchProgress',
    'ResearchDepth',
    'ResearchTone',
    'ResearchStatus',
    
    # Article models
    'Article',
    'ArticleSection',
    'ArticleMetadata',
    'ArticleFormat',
    'ArticleTone',
    'Citation',
    'CitationType',
    
    # Evidence models
    'Evidence',
    'EvidenceSource',
    'EvidenceQuality',
    'EvidenceRanking',
    'Claim',
    'ClaimType',
    
    # LLM models
    'LLMProvider',
    'LLMModel',
    'LLMConfig',
    'LLMResponse',
    
    # Error models
    'ContentGeneratorError',
    'ValidationError',
    'LLMError',
    'RAGError',
    'SearchError',
    'ArticleGenerationError'
]
