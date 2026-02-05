"""
Content Generation Pydantic Schemas
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ContentType(str, Enum):
    BLOG_POST = "blog_post"
    ARTICLE = "article"
    GUIDE = "guide"
    REVIEW = "review"
    TUTORIAL = "tutorial"
    NEWS = "news"
    OPINION = "opinion"
    COMPARISON = "comparison"
    LANDING_PAGE = "landing_page"
    PRODUCT_PAGE = "product_page"

class ContentStatus(str, Enum):
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class ContentPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class ContentGenerationRequest(BaseModel):
    """Schema for content generation request"""
    workflow_session_id: str = Field(..., description="Workflow session ID")
    trend_analysis_id: Optional[str] = Field(None, description="Trend analysis ID")
    topic_decomposition_id: Optional[str] = Field(None, description="Topic decomposition ID")
    keyword_clusters: Optional[List[str]] = Field(None, description="List of keyword cluster IDs")
    content_types: Optional[List[str]] = Field(None, description="List of content types to generate")
    target_audience: Optional[str] = Field(None, description="Target audience for content")
    max_ideas: int = Field(default=10, ge=1, le=50, description="Maximum number of ideas to generate")

    @validator('content_types')
    def validate_content_types(cls, v):
        if v is not None:
            valid_types = [content_type.value for content_type in ContentType]
            for content_type in v:
                if content_type not in valid_types:
                    raise ValueError(f'Invalid content type: {content_type}. Must be one of: {", ".join(valid_types)}')
        return v

class ContentGenerationResponse(BaseModel):
    """Schema for content generation response"""
    content_ideas: List[Dict[str, Any]] = Field(..., description="Generated content ideas")
    generation_metadata: Dict[str, Any] = Field(..., description="Generation metadata")

class ContentIdeaCreate(BaseModel):
    """Schema for creating a content idea"""
    workflow_session_id: str = Field(..., description="Workflow session ID")
    trend_analysis_id: Optional[str] = Field(None, description="Trend analysis ID")
    topic_decomposition_id: Optional[str] = Field(None, description="Topic decomposition ID")
    title: str = Field(..., min_length=1, max_length=500, description="Content title")
    description: Optional[str] = Field(None, max_length=1000, description="Content description")
    content_type: str = Field(default=ContentType.BLOG_POST.value, description="Content type")
    target_audience: Optional[str] = Field(None, description="Target audience")
    primary_keyword: str = Field(..., min_length=1, max_length=255, description="Primary keyword")
    secondary_keywords: Optional[List[str]] = Field(None, description="Secondary keywords")
    content_angle: Optional[str] = Field(None, description="Content angle")
    key_points: Optional[List[str]] = Field(None, description="Key points to cover")
    monetization_strategy: Optional[str] = Field(None, description="Monetization strategy")

    @validator('content_type')
    def validate_content_type(cls, v):
        valid_types = [content_type.value for content_type in ContentType]
        if v not in valid_types:
            raise ValueError(f'Invalid content type: {v}. Must be one of: {", ".join(valid_types)}')
        return v

class ContentIdeaUpdate(BaseModel):
    """Schema for updating a content idea"""
    title: Optional[str] = Field(None, min_length=1, max_length=500, description="Content title")
    description: Optional[str] = Field(None, max_length=1000, description="Content description")
    content_type: Optional[str] = Field(None, description="Content type")
    status: Optional[str] = Field(None, description="Content status")
    priority: Optional[str] = Field(None, description="Content priority")
    target_audience: Optional[str] = Field(None, description="Target audience")
    content_angle: Optional[str] = Field(None, description="Content angle")
    key_points: Optional[List[str]] = Field(None, description="Key points to cover")
    secondary_keywords: Optional[List[str]] = Field(None, description="Secondary keywords")
    enhanced_keywords: Optional[List[str]] = Field(None, description="Enhanced keywords")
    monetization_strategy: Optional[str] = Field(None, description="Monetization strategy")
    tags: Optional[List[str]] = Field(None, description="Content tags")
    categories: Optional[List[str]] = Field(None, description="Content categories")
    user_notes: Optional[str] = Field(None, description="User notes")

    @validator('content_type')
    def validate_content_type(cls, v):
        if v is not None:
            valid_types = [content_type.value for content_type in ContentType]
            if v not in valid_types:
                raise ValueError(f'Invalid content type: {v}. Must be one of: {", ".join(valid_types)}')
        return v

    @validator('status')
    def validate_status(cls, v):
        if v is not None:
            valid_statuses = [status.value for status in ContentStatus]
            if v not in valid_statuses:
                raise ValueError(f'Invalid status: {v}. Must be one of: {", ".join(valid_statuses)}')
        return v

    @validator('priority')
    def validate_priority(cls, v):
        if v is not None:
            valid_priorities = [priority.value for priority in ContentPriority]
            if v not in valid_priorities:
                raise ValueError(f'Invalid priority: {v}. Must be one of: {", ".join(valid_priorities)}')
        return v

class ContentIdeaResponse(BaseModel):
    """Schema for content idea response"""
    id: str = Field(..., description="Content idea ID")
    user_id: str = Field(..., description="User ID")
    workflow_session_id: str = Field(..., description="Workflow session ID")
    trend_analysis_id: Optional[str] = Field(None, description="Trend analysis ID")
    topic_decomposition_id: Optional[str] = Field(None, description="Topic decomposition ID")
    title: str = Field(..., description="Content title")
    description: Optional[str] = Field(None, description="Content description")
    content_type: str = Field(..., description="Content type")
    status: str = Field(..., description="Content status")
    priority: str = Field(..., description="Content priority")
    target_audience: Optional[str] = Field(None, description="Target audience")
    content_angle: Optional[str] = Field(None, description="Content angle")
    key_points: List[str] = Field(..., description="Key points to cover")
    content_outline: List[str] = Field(..., description="Content outline")
    primary_keyword: str = Field(..., description="Primary keyword")
    secondary_keywords: List[str] = Field(..., description="Secondary keywords")
    enhanced_keywords: List[str] = Field(..., description="Enhanced keywords")
    keyword_difficulty: Optional[int] = Field(None, description="Keyword difficulty")
    search_volume: Optional[int] = Field(None, description="Search volume")
    cpc: Optional[str] = Field(None, description="Cost per click")
    affiliate_offers: List[str] = Field(..., description="Affiliate offer IDs")
    affiliate_links: List[str] = Field(..., description="Affiliate links")
    monetization_strategy: Optional[str] = Field(None, description="Monetization strategy")
    expected_revenue: Optional[str] = Field(None, description="Expected revenue")
    generation_prompt: Optional[str] = Field(None, description="Generation prompt")
    generation_model: Optional[str] = Field(None, description="Generation model")
    generation_parameters: Dict[str, Any] = Field(..., description="Generation parameters")
    generation_time_ms: Optional[int] = Field(None, description="Generation time in milliseconds")
    readability_score: Optional[int] = Field(None, description="Readability score")
    seo_score: Optional[int] = Field(None, description="SEO score")
    engagement_score: Optional[int] = Field(None, description="Engagement score")
    quality_notes: Optional[str] = Field(None, description="Quality notes")
    target_publish_date: Optional[str] = Field(None, description="Target publish date")
    actual_publish_date: Optional[str] = Field(None, description="Actual publish date")
    publish_url: Optional[str] = Field(None, description="Publish URL")
    word_count: Optional[int] = Field(None, description="Word count")
    reading_time_minutes: Optional[int] = Field(None, description="Reading time in minutes")
    tags: List[str] = Field(..., description="Content tags")
    categories: List[str] = Field(..., description="Content categories")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True

class KeywordClusteringRequest(BaseModel):
    """Schema for keyword clustering request"""
    workflow_session_id: str = Field(..., description="Workflow session ID")
    external_tool_result_id: Optional[str] = Field(None, description="External tool result ID")
    keywords_data: Optional[List[Dict[str, Any]]] = Field(None, description="Keywords data for clustering")
    cluster_method: str = Field(default="kmeans", description="Clustering method")
    n_clusters: Optional[int] = Field(None, ge=2, le=50, description="Number of clusters")
    min_cluster_size: int = Field(default=3, ge=2, description="Minimum cluster size")
    max_clusters: int = Field(default=20, ge=2, le=50, description="Maximum clusters")

    @validator('cluster_method')
    def validate_cluster_method(cls, v):
        valid_methods = ['kmeans', 'dbscan', 'agglomerative']
        if v not in valid_methods:
            raise ValueError(f'Invalid cluster method: {v}. Must be one of: {", ".join(valid_methods)}')
        return v

    @validator('keywords_data')
    def validate_keywords_data(cls, v):
        if v is not None and len(v) < 4:
            raise ValueError('At least 4 keywords are required for clustering')
        return v

class KeywordClusteringResponse(BaseModel):
    """Schema for keyword clustering response"""
    clusters: List[Dict[str, Any]] = Field(..., description="Generated keyword clusters")
    clustering_metadata: Dict[str, Any] = Field(..., description="Clustering metadata")

class ExternalToolRequest(BaseModel):
    """Schema for external tool request"""
    workflow_session_id: str = Field(..., description="Workflow session ID")
    tool_name: str = Field(..., description="External tool name")
    query_type: str = Field(..., description="Query type")
    query_parameters: Dict[str, Any] = Field(..., description="Query parameters")
    seed_keywords: List[str] = Field(..., min_items=1, description="Seed keywords")
    trend_analysis_id: Optional[str] = Field(None, description="Trend analysis ID")

    @validator('tool_name')
    def validate_tool_name(cls, v):
        valid_tools = ['semrush', 'ahrefs', 'ubersuggest']
        if v not in valid_tools:
            raise ValueError(f'Invalid tool name: {v}. Must be one of: {", ".join(valid_tools)}')
        return v

    @validator('query_type')
    def validate_query_type(cls, v):
        valid_types = ['keyword_research', 'competitor_analysis', 'content_ideas', 'trend_analysis']
        if v not in valid_types:
            raise ValueError(f'Invalid query type: {v}. Must be one of: {", ".join(valid_types)}')
        return v

    @validator('seed_keywords')
    def validate_seed_keywords(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one seed keyword is required')
        return v

class ExternalToolResponse(BaseModel):
    """Schema for external tool response"""
    id: str = Field(..., description="External tool result ID")
    user_id: str = Field(..., description="User ID")
    workflow_session_id: str = Field(..., description="Workflow session ID")
    trend_analysis_id: Optional[str] = Field(None, description="Trend analysis ID")
    tool_name: str = Field(..., description="Tool name")
    tool_version: Optional[str] = Field(None, description="Tool version")
    api_key_id: Optional[str] = Field(None, description="API key ID")
    query_type: str = Field(..., description="Query type")
    query_parameters: Dict[str, Any] = Field(..., description="Query parameters")
    seed_keywords: List[str] = Field(..., description="Seed keywords")
    raw_results: Dict[str, Any] = Field(..., description="Raw results")
    processed_results: Dict[str, Any] = Field(..., description="Processed results")
    keywords_data: List[Dict[str, Any]] = Field(..., description="Keywords data")
    clusters_data: List[Dict[str, Any]] = Field(..., description="Clusters data")
    total_keywords: int = Field(..., description="Total keywords")
    total_clusters: int = Field(..., description="Total clusters")
    avg_search_volume: Optional[float] = Field(None, description="Average search volume")
    avg_keyword_difficulty: Optional[float] = Field(None, description="Average keyword difficulty")
    avg_cpc: Optional[float] = Field(None, description="Average CPC")
    total_search_volume: Optional[int] = Field(None, description="Total search volume")
    data_quality_score: Optional[float] = Field(None, description="Data quality score")
    completeness_score: Optional[float] = Field(None, description="Completeness score")
    relevance_score: Optional[float] = Field(None, description="Relevance score")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    api_calls_made: int = Field(..., description="API calls made")
    rate_limit_hit: bool = Field(..., description="Rate limit hit")
    error_count: int = Field(..., description="Error count")
    warning_count: int = Field(..., description="Warning count")
    status: str = Field(..., description="Processing status")
    is_processed: bool = Field(..., description="Is processed")
    is_used_for_content: bool = Field(..., description="Is used for content")
    is_archived: bool = Field(..., description="Is archived")
    error_message: Optional[str] = Field(None, description="Error message")
    error_details: Dict[str, Any] = Field(..., description="Error details")
    retry_count: int = Field(..., description="Retry count")
    last_retry_at: Optional[str] = Field(None, description="Last retry timestamp")
    processing_notes: Optional[str] = Field(None, description="Processing notes")
    user_notes: Optional[str] = Field(None, description="User notes")
    tags: List[str] = Field(..., description="Tags")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    processed_at: Optional[str] = Field(None, description="Processing timestamp")

    class Config:
        from_attributes = True

class ContentIdeaSummary(BaseModel):
    """Schema for content idea summary (list view)"""
    id: str = Field(..., description="Content idea ID")
    title: str = Field(..., description="Content title")
    description: Optional[str] = Field(None, description="Content description")
    content_type: str = Field(..., description="Content type")
    status: str = Field(..., description="Content status")
    priority: str = Field(..., description="Content priority")
    primary_keyword: str = Field(..., description="Primary keyword")
    keyword_difficulty: Optional[int] = Field(None, description="Keyword difficulty")
    search_volume: Optional[int] = Field(None, description="Search volume")
    affiliate_offers_count: int = Field(..., description="Number of affiliate offers")
    word_count: Optional[int] = Field(None, description="Word count")
    reading_time_minutes: Optional[int] = Field(None, description="Reading time in minutes")
    seo_score: Optional[int] = Field(None, description="SEO score")
    created_at: str = Field(..., description="Creation timestamp")
    target_publish_date: Optional[str] = Field(None, description="Target publish date")

    class Config:
        from_attributes = True

class KeywordClusterSummary(BaseModel):
    """Schema for keyword cluster summary"""
    id: str = Field(..., description="Cluster ID")
    cluster_name: str = Field(..., description="Cluster name")
    cluster_type: str = Field(..., description="Cluster type")
    cluster_size: int = Field(..., description="Cluster size")
    primary_keyword: str = Field(..., description="Primary keyword")
    avg_search_volume: Optional[int] = Field(None, description="Average search volume")
    avg_keyword_difficulty: Optional[float] = Field(None, description="Average keyword difficulty")
    competition_level: Optional[str] = Field(None, description="Competition level")
    cluster_quality_score: Optional[float] = Field(None, description="Cluster quality score")
    is_active: bool = Field(..., description="Is active")
    is_processed: bool = Field(..., description="Is processed")
    is_used_for_content: bool = Field(..., description="Is used for content")
    created_at: str = Field(..., description="Creation timestamp")

    class Config:
        from_attributes = True

class ContentMetrics(BaseModel):
    """Schema for content metrics"""
    word_count: int = Field(..., description="Word count")
    reading_time_minutes: int = Field(..., description="Reading time in minutes")
    keyword_count: int = Field(..., description="Keyword count")
    affiliate_offer_count: int = Field(..., description="Affiliate offer count")
    quality_score: float = Field(..., description="Quality score")
    monetization_potential: str = Field(..., description="Monetization potential")
    seo_potential: str = Field(..., description="SEO potential")
    is_high_priority: bool = Field(..., description="Is high priority")
    is_published: bool = Field(..., description="Is published")

class ClusterMetrics(BaseModel):
    """Schema for cluster metrics"""
    total_keywords: int = Field(..., description="Total keywords")
    total_clusters: int = Field(..., description="Total clusters")
    avg_search_volume: Optional[float] = Field(None, description="Average search volume")
    avg_keyword_difficulty: Optional[float] = Field(None, description="Average keyword difficulty")
    avg_cpc: Optional[float] = Field(None, description="Average CPC")
    total_search_volume: Optional[int] = Field(None, description="Total search volume")
    keywords_per_cluster: float = Field(..., description="Keywords per cluster")

class ContentGenerationStats(BaseModel):
    """Schema for content generation statistics"""
    total_ideas: int = Field(..., description="Total content ideas")
    ideas_by_type: Dict[str, int] = Field(..., description="Ideas by content type")
    ideas_by_status: Dict[str, int] = Field(..., description="Ideas by status")
    ideas_by_priority: Dict[str, int] = Field(..., description="Ideas by priority")
    avg_quality_score: float = Field(..., description="Average quality score")
    total_keywords: int = Field(..., description="Total keywords used")
    total_affiliate_offers: int = Field(..., description="Total affiliate offers")
    published_ideas: int = Field(..., description="Published ideas count")
    high_priority_ideas: int = Field(..., description="High priority ideas count")

class ContentFilters(BaseModel):
    """Schema for content filtering"""
    workflow_session_id: Optional[str] = Field(None, description="Filter by workflow session")
    content_type: Optional[str] = Field(None, description="Filter by content type")
    status: Optional[str] = Field(None, description="Filter by status")
    priority: Optional[str] = Field(None, description="Filter by priority")
    target_audience: Optional[str] = Field(None, description="Filter by target audience")
    created_after: Optional[str] = Field(None, description="Filter by creation date (ISO format)")
    created_before: Optional[str] = Field(None, description="Filter by creation date (ISO format)")
    min_quality_score: Optional[float] = Field(None, ge=0, le=100, description="Minimum quality score")
    has_affiliate_offers: Optional[bool] = Field(None, description="Has affiliate offers")
    is_published: Optional[bool] = Field(None, description="Is published")

    @validator('content_type')
    def validate_content_type(cls, v):
        if v is not None:
            valid_types = [content_type.value for content_type in ContentType]
            if v not in valid_types:
                raise ValueError(f'Invalid content type: {v}. Must be one of: {", ".join(valid_types)}')
        return v

    @validator('status')
    def validate_status(cls, v):
        if v is not None:
            valid_statuses = [status.value for status in ContentStatus]
            if v not in valid_statuses:
                raise ValueError(f'Invalid status: {v}. Must be one of: {", ".join(valid_statuses)}')
        return v

    @validator('priority')
    def validate_priority(cls, v):
        if v is not None:
            valid_priorities = [priority.value for priority in ContentPriority]
            if v not in valid_priorities:
                raise ValueError(f'Invalid priority: {v}. Must be one of: {", ".join(valid_priorities)}')
        return v
