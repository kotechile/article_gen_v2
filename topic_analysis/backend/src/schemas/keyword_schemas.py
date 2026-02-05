"""
Keyword Management API schemas
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class KeywordStatus(str, Enum):
    """Keyword status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    ARCHIVED = "archived"

class KeywordDifficulty(str, Enum):
    """Keyword difficulty enumeration"""
    VERY_EASY = "very_easy"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"

class KeywordUploadRequest(BaseModel):
    """Request schema for keyword upload"""
    keywords: List[str] = Field(..., min_items=1, description="List of keywords to upload")
    source: Optional[str] = Field("manual", description="Source of keywords")
    category: Optional[str] = Field(None, max_length=100, description="Keyword category")
    tags: Optional[List[str]] = Field(None, description="Keyword tags")
    priority: Optional[int] = Field(1, ge=1, le=5, description="Keyword priority (1-5)")
    target_url: Optional[str] = Field(None, description="Target URL for keywords")
    notes: Optional[str] = Field(None, description="Additional notes")
    
    @validator('keywords')
    def validate_keywords(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one keyword is required')
        if len(v) > 1000:
            raise ValueError('Maximum 1000 keywords allowed per upload')
        return v

class KeywordDataResponse(BaseModel):
    """Response schema for individual keyword data"""
    id: int = Field(..., description="Keyword ID")
    user_id: int = Field(..., description="User ID")
    keyword: str = Field(..., description="Keyword text")
    status: KeywordStatus = Field(..., description="Keyword status")
    source: str = Field(..., description="Keyword source")
    category: Optional[str] = Field(None, description="Keyword category")
    tags: List[str] = Field(..., description="Keyword tags")
    priority: int = Field(..., description="Keyword priority")
    target_url: Optional[str] = Field(None, description="Target URL")
    notes: Optional[str] = Field(None, description="Additional notes")
    
    # SEO metrics
    search_volume: Optional[int] = Field(None, ge=0, description="Monthly search volume")
    difficulty: Optional[KeywordDifficulty] = Field(None, description="Keyword difficulty")
    cpc: Optional[float] = Field(None, ge=0, description="Cost per click")
    competition: Optional[float] = Field(None, ge=0, le=1, description="Competition level")
    trend: Optional[str] = Field(None, description="Trend direction")
    serp_features: Optional[List[str]] = Field(None, description="SERP features")
    
    # Performance metrics
    current_rank: Optional[int] = Field(None, ge=0, description="Current ranking position")
    best_rank: Optional[int] = Field(None, ge=0, description="Best ranking position")
    clicks: Optional[int] = Field(None, ge=0, description="Total clicks")
    impressions: Optional[int] = Field(None, ge=0, description="Total impressions")
    ctr: Optional[float] = Field(None, ge=0, le=1, description="Click-through rate")
    position: Optional[float] = Field(None, ge=0, description="Average position")
    
    # Cluster information
    cluster_id: Optional[int] = Field(None, description="Cluster ID")
    cluster_name: Optional[str] = Field(None, description="Cluster name")
    cluster_theme: Optional[str] = Field(None, description="Cluster theme")
    
    # Timestamps
    created_at: datetime = Field(..., description="Creation date")
    updated_at: datetime = Field(..., description="Last update date")
    last_analyzed: Optional[datetime] = Field(None, description="Last analysis date")

class KeywordUploadResponse(BaseModel):
    """Response schema for keyword upload"""
    uploaded_count: int = Field(..., description="Number of keywords uploaded")
    skipped_count: int = Field(..., description="Number of keywords skipped")
    error_count: int = Field(..., description="Number of keywords with errors")
    errors: List[str] = Field(..., description="Upload errors")
    warnings: List[str] = Field(..., description="Upload warnings")
    created_keywords: List[KeywordDataResponse] = Field(..., description="Created keywords")
    created_at: datetime = Field(..., description="Upload date")

class KeywordCrawlRequest(BaseModel):
    """Request schema for keyword crawling"""
    seed_keywords: List[str] = Field(..., min_items=1, description="Seed keywords to start crawling")
    max_keywords: Optional[int] = Field(100, ge=1, le=1000, description="Maximum keywords to crawl")
    depth: Optional[int] = Field(2, ge=1, le=5, description="Crawl depth")
    language: Optional[str] = Field("en", max_length=5, description="Language code")
    country: Optional[str] = Field("US", max_length=2, description="Country code")
    include_related: Optional[bool] = Field(True, description="Include related keywords")
    include_questions: Optional[bool] = Field(True, description="Include question keywords")
    include_long_tail: Optional[bool] = Field(True, description="Include long-tail keywords")
    min_search_volume: Optional[int] = Field(0, ge=0, description="Minimum search volume")
    max_search_volume: Optional[int] = Field(None, ge=0, description="Maximum search volume")
    category: Optional[str] = Field(None, description="Category filter")
    
    @validator('seed_keywords')
    def validate_seed_keywords(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one seed keyword is required')
        return v

class KeywordCrawlResponse(BaseModel):
    """Response schema for keyword crawling results"""
    crawl_id: str = Field(..., description="Crawl ID")
    seed_keywords: List[str] = Field(..., description="Seed keywords")
    total_keywords_found: int = Field(..., description="Total keywords found")
    keywords_crawled: int = Field(..., description="Keywords successfully crawled")
    keywords_skipped: int = Field(..., description="Keywords skipped")
    errors: List[str] = Field(..., description="Crawl errors")
    warnings: List[str] = Field(..., description="Crawl warnings")
    crawled_keywords: List[KeywordDataResponse] = Field(..., description="Crawled keywords")
    crawl_metadata: Dict[str, Any] = Field(..., description="Crawl metadata")
    created_at: datetime = Field(..., description="Crawl start date")
    completed_at: Optional[datetime] = Field(None, description="Crawl completion date")

class KeywordListResponse(BaseModel):
    """Response schema for keyword list"""
    keywords: List[KeywordDataResponse] = Field(..., description="List of keywords")
    total: int = Field(..., description="Total number of keywords")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")

class KeywordDataListResponse(BaseModel):
    """Response schema for keyword data list (alias for KeywordListResponse)"""
    keywords: List[KeywordDataResponse] = Field(..., description="List of keywords")
    total: int = Field(..., description="Total number of keywords")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")

class KeywordUpdateRequest(BaseModel):
    """Request schema for updating keyword"""
    status: Optional[KeywordStatus] = Field(None, description="New status")
    category: Optional[str] = Field(None, max_length=100, description="Updated category")
    tags: Optional[List[str]] = Field(None, description="Updated tags")
    priority: Optional[int] = Field(None, ge=1, le=5, description="Updated priority")
    target_url: Optional[str] = Field(None, description="Updated target URL")
    notes: Optional[str] = Field(None, description="Updated notes")

class KeywordAnalysisRequest(BaseModel):
    """Request schema for keyword analysis"""
    keywords: List[str] = Field(..., min_items=1, max_items=100, description="Keywords to analyze")
    country: Optional[str] = Field("US", max_length=2, description="Country code")
    language: Optional[str] = Field("en", max_length=5, description="Language code")
    include_serp_features: Optional[bool] = Field(True, description="Include SERP features")
    include_related_keywords: Optional[bool] = Field(True, description="Include related keywords")
    include_competitor_analysis: Optional[bool] = Field(True, description="Include competitor analysis")
    include_trend_analysis: Optional[bool] = Field(True, description="Include trend analysis")

class KeywordAnalysisResponse(BaseModel):
    """Response schema for keyword analysis results"""
    keywords: List[str] = Field(..., description="Analyzed keywords")
    country: str = Field(..., description="Country code")
    language: str = Field(..., description="Language code")
    analysis_data: List[Dict[str, Any]] = Field(..., description="Analysis data for each keyword")
    related_keywords: List[Dict[str, Any]] = Field(..., description="Related keywords")
    competitor_analysis: Optional[Dict[str, Any]] = Field(None, description="Competitor analysis")
    trend_analysis: Optional[Dict[str, Any]] = Field(None, description="Trend analysis")
    serp_features: Optional[Dict[str, Any]] = Field(None, description="SERP features analysis")
    insights: List[str] = Field(..., description="Analysis insights")
    recommendations: List[str] = Field(..., description="Actionable recommendations")
    created_at: datetime = Field(..., description="Analysis creation date")

class KeywordClusterResponse(BaseModel):
    """Response schema for keyword cluster"""
    id: int = Field(..., description="Cluster ID")
    user_id: int = Field(..., description="User ID")
    name: str = Field(..., description="Cluster name")
    theme: str = Field(..., description="Cluster theme")
    keywords: List[str] = Field(..., description="Keywords in cluster")
    keyword_count: int = Field(..., description="Number of keywords")
    primary_keyword: str = Field(..., description="Primary keyword")
    search_volume: int = Field(..., description="Total search volume")
    difficulty: KeywordDifficulty = Field(..., description="Average difficulty")
    priority: int = Field(..., description="Cluster priority")
    status: KeywordStatus = Field(..., description="Cluster status")
    created_at: datetime = Field(..., description="Cluster creation date")
    updated_at: datetime = Field(..., description="Last update date")

class KeywordClusterListResponse(BaseModel):
    """Response schema for keyword cluster list"""
    clusters: List[KeywordClusterResponse] = Field(..., description="List of clusters")
    total: int = Field(..., description="Total number of clusters")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")

class KeywordClusterCreateRequest(BaseModel):
    """Request schema for creating keyword cluster"""
    name: str = Field(..., min_length=1, max_length=200, description="Cluster name")
    theme: str = Field(..., min_length=1, max_length=200, description="Cluster theme")
    keywords: List[str] = Field(..., min_items=2, description="Keywords to include")
    priority: Optional[int] = Field(1, ge=1, le=5, description="Cluster priority")
    status: Optional[KeywordStatus] = Field(KeywordStatus.ACTIVE, description="Cluster status")

class KeywordClusterUpdateRequest(BaseModel):
    """Request schema for updating keyword cluster"""
    name: Optional[str] = Field(None, min_length=1, max_length=200, description="Updated name")
    theme: Optional[str] = Field(None, min_length=1, max_length=200, description="Updated theme")
    keywords: Optional[List[str]] = Field(None, min_items=2, description="Updated keywords")
    priority: Optional[int] = Field(None, ge=1, le=5, description="Updated priority")
    status: Optional[KeywordStatus] = Field(None, description="Updated status")

class KeywordExportRequest(BaseModel):
    """Request schema for keyword export"""
    format: str = Field("csv", description="Export format (csv, xlsx, json)")
    filters: Optional[Dict[str, Any]] = Field(None, description="Export filters")
    fields: Optional[List[str]] = Field(None, description="Fields to include")
    cluster_id: Optional[int] = Field(None, description="Export specific cluster")
    status: Optional[KeywordStatus] = Field(None, description="Filter by status")
    category: Optional[str] = Field(None, description="Filter by category")

class KeywordExportResponse(BaseModel):
    """Response schema for keyword export"""
    export_id: str = Field(..., description="Export ID")
    format: str = Field(..., description="Export format")
    file_url: str = Field(..., description="Download URL")
    file_size: int = Field(..., description="File size in bytes")
    record_count: int = Field(..., description="Number of records exported")
    created_at: datetime = Field(..., description="Export creation date")
    expires_at: datetime = Field(..., description="Download expiration date")

class KeywordStatsResponse(BaseModel):
    """Response schema for keyword statistics"""
    total_keywords: int = Field(..., description="Total keywords")
    active_keywords: int = Field(..., description="Active keywords")
    pending_keywords: int = Field(..., description="Pending keywords")
    archived_keywords: int = Field(..., description="Archived keywords")
    total_clusters: int = Field(..., description="Total clusters")
    average_difficulty: float = Field(..., description="Average difficulty")
    total_search_volume: int = Field(..., description="Total search volume")
    top_categories: List[Dict[str, Any]] = Field(..., description="Top categories")
    top_keywords: List[Dict[str, Any]] = Field(..., description="Top keywords by volume")
    last_updated: datetime = Field(..., description="Last database update")
