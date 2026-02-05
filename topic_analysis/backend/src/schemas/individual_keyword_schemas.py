"""
Individual Keyword Optimization Schemas
Pydantic models for individual keyword optimization API
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class KeywordType(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    LONG_TAIL = "long_tail"
    SEMANTIC = "semantic"
    QUESTION_BASED = "question_based"
    COMPARISON = "comparison"
    TECHNICAL = "technical"
    BEGINNER = "beginner"
    ADVANCED = "advanced"

class SearchIntent(str, Enum):
    INFORMATIONAL = "informational"
    NAVIGATIONAL = "navigational"
    TRANSACTIONAL = "transactional"
    COMMERCIAL = "commercial"

class CompetitionLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class SourceTool(str, Enum):
    AHREFS = "ahrefs"
    SEMRUSH = "semrush"
    UBERSUGGEST = "ubersuggest"
    DATAFORSEO = "dataforseo"
    MANUAL = "manual"

class OptimizationType(str, Enum):
    AHREFS_UPLOAD = "ahrefs_upload"
    SEMRUSH_UPLOAD = "semrush_upload"
    MANUAL_ENTRY = "manual_entry"
    LLM_GENERATION = "llm_generation"

class IndividualKeywordData(BaseModel):
    """Individual keyword data from external tools"""
    keyword: str = Field(..., min_length=1, max_length=500, description="The keyword phrase")
    search_volume: int = Field(0, ge=0, description="Monthly search volume")
    keyword_difficulty: int = Field(0, ge=0, le=100, description="Keyword difficulty score (0-100)")
    cpc: float = Field(0.0, ge=0, description="Cost per click")
    traffic_potential: int = Field(0, ge=0, description="Traffic potential")
    clicks: int = Field(0, ge=0, description="Total clicks")
    impressions: int = Field(0, ge=0, description="Total impressions")
    ctr: float = Field(0.0, ge=0, le=100, description="Click-through rate")
    position: float = Field(0.0, ge=0, description="Average position")
    keyword_type: KeywordType = Field(KeywordType.PRIMARY, description="Type of keyword")
    search_intent: SearchIntent = Field(SearchIntent.INFORMATIONAL, description="Search intent")
    competition_level: CompetitionLevel = Field(CompetitionLevel.MEDIUM, description="Competition level")
    trend_score: float = Field(0.0, ge=0, le=100, description="Trend score (0-100)")
    opportunity_score: float = Field(0.0, ge=0, le=100, description="Opportunity score (0-100)")
    row_number: Optional[int] = Field(None, description="Row number in source file")

class IndividualKeywordOptimized(BaseModel):
    """Optimized individual keyword with LLM enhancements"""
    id: Optional[str] = Field(None, description="Keyword ID")
    content_idea_id: str = Field(..., description="Content idea ID")
    user_id: str = Field(..., description="User ID")
    
    # Basic keyword data
    keyword: str = Field(..., description="The keyword phrase")
    keyword_type: KeywordType = Field(..., description="Type of keyword")
    
    # Ahrefs metrics
    search_volume: int = Field(0, description="Monthly search volume")
    keyword_difficulty: int = Field(0, description="Keyword difficulty score")
    cpc: float = Field(0.0, description="Cost per click")
    traffic_potential: int = Field(0, description="Traffic potential")
    clicks: int = Field(0, description="Total clicks")
    impressions: int = Field(0, description="Total impressions")
    ctr: float = Field(0.0, description="Click-through rate")
    position: float = Field(0.0, description="Average position")
    
    # SEO metrics
    search_intent: SearchIntent = Field(..., description="Search intent")
    competition_level: CompetitionLevel = Field(..., description="Competition level")
    trend_score: float = Field(0.0, description="Trend score")
    opportunity_score: float = Field(0.0, description="Opportunity score")
    
    # Content optimization
    content_suggestions: List[str] = Field(default_factory=list, description="Content suggestions")
    heading_suggestions: List[str] = Field(default_factory=list, description="Heading suggestions")
    internal_link_suggestions: List[str] = Field(default_factory=list, description="Internal link suggestions")
    related_questions: List[str] = Field(default_factory=list, description="Related questions")
    
    # LLM optimization
    llm_optimized_title: Optional[str] = Field(None, description="LLM-optimized title")
    llm_optimized_description: Optional[str] = Field(None, description="LLM-optimized description")
    llm_keyword_variations: List[str] = Field(default_factory=list, description="LLM keyword variations")
    llm_content_angle: Optional[str] = Field(None, description="LLM content angle")
    llm_target_audience: Optional[str] = Field(None, description="LLM target audience")
    
    # Affiliate integration
    affiliate_potential_score: float = Field(0.0, ge=0, le=100, description="Affiliate potential score")
    suggested_affiliate_networks: List[str] = Field(default_factory=list, description="Suggested affiliate networks")
    monetization_opportunities: List[str] = Field(default_factory=list, description="Monetization opportunities")
    
    # Source tracking
    source_tool: SourceTool = Field(SourceTool.AHREFS, description="Source tool")
    source_file_name: Optional[str] = Field(None, description="Source file name")
    source_row_number: Optional[int] = Field(None, description="Source row number")
    
    # Quality metrics
    relevance_score: float = Field(0.0, ge=0, le=100, description="Relevance score")
    optimization_score: float = Field(0.0, ge=0, le=100, description="Optimization score")
    priority_score: float = Field(0.0, ge=0, le=100, description="Priority score")
    
    # Status flags
    is_optimized: bool = Field(False, description="Whether keyword is optimized")
    is_used_in_content: bool = Field(False, description="Whether keyword is used in content")
    is_archived: bool = Field(False, description="Whether keyword is archived")
    
    # Timestamps
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Update timestamp")
    optimized_at: Optional[datetime] = Field(None, description="Optimization timestamp")

class IndividualKeywordUploadRequest(BaseModel):
    """Request schema for individual keyword upload"""
    keywords: List[IndividualKeywordData] = Field(..., min_items=1, description="List of keywords to upload")
    content_idea_id: str = Field(..., description="Content idea ID")
    user_id: str = Field(..., description="User ID")
    source_tool: SourceTool = Field(SourceTool.AHREFS, description="Source tool")
    source_file_name: Optional[str] = Field(None, description="Source file name")
    
    @validator('keywords')
    def validate_keywords(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one keyword is required')
        if len(v) > 1000:
            raise ValueError('Maximum 1000 keywords allowed per upload')
        return v

class IndividualKeywordOptimizationRequest(BaseModel):
    """Request schema for keyword optimization"""
    keywords: List[IndividualKeywordData] = Field(..., min_items=1, description="Keywords to optimize")
    content_idea_id: str = Field(..., description="Content idea ID")
    user_id: str = Field(..., description="User ID")
    session_id: str = Field(..., description="Optimization session ID")
    optimization_parameters: Optional[Dict[str, Any]] = Field(None, description="Optimization parameters")

class IndividualKeywordResponse(BaseModel):
    """Response schema for individual keyword"""
    success: bool = Field(..., description="Success status")
    keyword: Optional[IndividualKeywordOptimized] = Field(None, description="Keyword data")
    message: Optional[str] = Field(None, description="Response message")

class OptimizationSessionData(BaseModel):
    """Optimization session data"""
    id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    content_idea_id: str = Field(..., description="Content idea ID")
    session_name: str = Field(..., description="Session name")
    session_description: Optional[str] = Field(None, description="Session description")
    optimization_type: OptimizationType = Field(..., description="Optimization type")
    source_file_name: Optional[str] = Field(None, description="Source file name")
    source_file_size: Optional[int] = Field(None, description="Source file size")
    keywords_processed: int = Field(0, description="Keywords processed")
    keywords_optimized: int = Field(0, description="Keywords optimized")
    optimization_summary: Dict[str, Any] = Field(default_factory=dict, description="Optimization summary")
    top_performing_keywords: List[Dict[str, Any]] = Field(default_factory=list, description="Top performing keywords")
    optimization_recommendations: List[str] = Field(default_factory=list, description="Optimization recommendations")
    overall_optimization_score: float = Field(0.0, ge=0, le=100, description="Overall optimization score")
    seo_improvement_score: float = Field(0.0, ge=0, le=100, description="SEO improvement score")
    content_potential_score: float = Field(0.0, ge=0, le=100, description="Content potential score")
    status: str = Field("pending", description="Session status")
    is_active: bool = Field(True, description="Whether session is active")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Update timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")

class OptimizationSessionResponse(BaseModel):
    """Response schema for optimization session"""
    success: bool = Field(..., description="Success status")
    session: Optional[OptimizationSessionData] = Field(None, description="Session data")
    optimized_keywords: List[IndividualKeywordOptimized] = Field(default_factory=list, description="Optimized keywords")
    message: Optional[str] = Field(None, description="Response message")

class ContentIdeaEnhancementData(BaseModel):
    """Enhanced content idea data"""
    id: str = Field(..., description="Content idea ID")
    user_id: str = Field(..., description="User ID")
    title: str = Field(..., description="Content title")
    description: Optional[str] = Field(None, description="Content description")
    content_type: str = Field(..., description="Content type")
    status: str = Field(..., description="Content status")
    priority: str = Field(..., description="Content priority")
    
    # Enhanced keyword data
    enhanced_keywords_data: List[Dict[str, Any]] = Field(default_factory=list, description="Enhanced keywords data")
    seo_optimized_title: Optional[str] = Field(None, description="SEO-optimized title")
    seo_optimized_description: Optional[str] = Field(None, description="SEO-optimized description")
    primary_keywords_optimized: List[str] = Field(default_factory=list, description="Primary keywords optimized")
    keyword_metrics_summary: Dict[str, Any] = Field(default_factory=dict, description="Keyword metrics summary")
    affiliate_networks_suggested: List[str] = Field(default_factory=list, description="Suggested affiliate networks")
    
    # Content generation
    content_generation_prompt: Optional[str] = Field(None, description="Content generation prompt")
    content_generation_parameters: Dict[str, Any] = Field(default_factory=dict, description="Content generation parameters")
    
    # Enhancement status
    is_enhanced: bool = Field(False, description="Whether content is enhanced")
    enhancement_timestamp: Optional[datetime] = Field(None, description="Enhancement timestamp")
    
    # Timestamps
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Update timestamp")

class ContentIdeaEnhancementResponse(BaseModel):
    """Response schema for content idea enhancement"""
    success: bool = Field(..., description="Success status")
    content_idea: Optional[ContentIdeaEnhancementData] = Field(None, description="Content idea data")
    keywords: List[IndividualKeywordOptimized] = Field(default_factory=list, description="Keywords")
    optimization_sessions: List[OptimizationSessionData] = Field(default_factory=list, description="Optimization sessions")
    enhancement_summary: Dict[str, Any] = Field(default_factory=dict, description="Enhancement summary")
    message: Optional[str] = Field(None, description="Response message")

class KeywordAnalyticsData(BaseModel):
    """Keyword analytics data"""
    total_keywords: int = Field(0, description="Total keywords")
    optimized_keywords: int = Field(0, description="Optimized keywords")
    high_priority_keywords: int = Field(0, description="High priority keywords")
    optimization_rate: float = Field(0.0, ge=0, le=100, description="Optimization rate percentage")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Average metrics")
    keyword_types: Dict[str, int] = Field(default_factory=dict, description="Keyword types distribution")
    top_keywords: List[Dict[str, Any]] = Field(default_factory=list, description="Top performing keywords")
    insights: List[str] = Field(default_factory=list, description="Analytics insights")

class KeywordAnalyticsResponse(BaseModel):
    """Response schema for keyword analytics"""
    success: bool = Field(..., description="Success status")
    analytics: Optional[KeywordAnalyticsData] = Field(None, description="Analytics data")
    message: Optional[str] = Field(None, description="Response message")

class BulkKeywordUpdateRequest(BaseModel):
    """Request schema for bulk keyword updates"""
    keyword_ids: List[str] = Field(..., min_items=1, description="Keyword IDs to update")
    updates: Dict[str, Any] = Field(..., description="Updates to apply")
    user_id: str = Field(..., description="User ID")
    
    @validator('keyword_ids')
    def validate_keyword_ids(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one keyword ID is required')
        if len(v) > 100:
            raise ValueError('Maximum 100 keywords allowed per bulk update')
        return v

class BulkKeywordUpdateResponse(BaseModel):
    """Response schema for bulk keyword updates"""
    success: bool = Field(..., description="Success status")
    updated_count: int = Field(0, description="Number of keywords updated")
    failed_count: int = Field(0, description="Number of keywords that failed to update")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    message: Optional[str] = Field(None, description="Response message")

