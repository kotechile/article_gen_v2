"""
Affiliate Research API schemas
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ResearchStatus(str, Enum):
    """Research status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AffiliateResearchRequest(BaseModel):
    """Request schema for creating affiliate research"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query for affiliate programs")
    category: Optional[str] = Field(None, max_length=100, description="Category filter")
    country: Optional[str] = Field(None, max_length=2, description="Country code (ISO 2-letter)")
    min_commission: Optional[float] = Field(None, ge=0, le=100, description="Minimum commission rate (%)")
    max_commission: Optional[float] = Field(None, ge=0, le=100, description="Maximum commission rate (%)")
    payment_frequency: Optional[str] = Field(None, description="Payment frequency filter")
    cookie_duration: Optional[int] = Field(None, ge=1, description="Cookie duration in days")
    network_preferences: Optional[List[str]] = Field(None, description="Preferred affiliate networks")
    exclude_networks: Optional[List[str]] = Field(None, description="Networks to exclude")
    
    @validator('max_commission')
    def validate_commission_range(cls, v, values):
        if v is not None and 'min_commission' in values and values['min_commission'] is not None:
            if v < values['min_commission']:
                raise ValueError('max_commission must be greater than or equal to min_commission')
        return v

class AffiliateProgramResponse(BaseModel):
    """Response schema for individual affiliate program"""
    id: str = Field(..., description="Program ID")
    name: str = Field(..., description="Program name")
    network: str = Field(..., description="Affiliate network")
    category: str = Field(..., description="Program category")
    commission_rate: float = Field(..., description="Commission rate (%)")
    commission_type: str = Field(..., description="Commission type (percentage, fixed, etc.)")
    payment_frequency: str = Field(..., description="Payment frequency")
    cookie_duration: int = Field(..., description="Cookie duration in days")
    description: str = Field(..., description="Program description")
    url: str = Field(..., description="Program URL")
    logo_url: Optional[str] = Field(None, description="Program logo URL")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Program rating")
    review_count: Optional[int] = Field(None, ge=0, description="Number of reviews")
    approval_difficulty: Optional[str] = Field(None, description="Approval difficulty level")
    minimum_payout: Optional[float] = Field(None, description="Minimum payout amount")
    currency: Optional[str] = Field(None, description="Currency code")
    countries: Optional[List[str]] = Field(None, description="Available countries")
    tracking_methods: Optional[List[str]] = Field(None, description="Available tracking methods")
    promotional_materials: Optional[bool] = Field(None, description="Promotional materials available")
    real_time_reporting: Optional[bool] = Field(None, description="Real-time reporting available")
    mobile_tracking: Optional[bool] = Field(None, description="Mobile tracking supported")
    deep_linking: Optional[bool] = Field(None, description="Deep linking supported")
    sub_affiliate_support: Optional[bool] = Field(None, description="Sub-affiliate support")
    created_at: datetime = Field(..., description="Program creation date")
    updated_at: datetime = Field(..., description="Last update date")

class AffiliateResearchResponse(BaseModel):
    """Response schema for affiliate research results"""
    id: int = Field(..., description="Research ID")
    user_id: int = Field(..., description="User ID")
    query: str = Field(..., description="Search query")
    status: ResearchStatus = Field(..., description="Research status")
    total_programs: int = Field(..., description="Total programs found")
    programs: List[AffiliateProgramResponse] = Field(..., description="Found programs")
    filters_applied: Dict[str, Any] = Field(..., description="Applied filters")
    search_metadata: Dict[str, Any] = Field(..., description="Search metadata")
    created_at: datetime = Field(..., description="Research creation date")
    updated_at: datetime = Field(..., description="Last update date")
    completed_at: Optional[datetime] = Field(None, description="Completion date")
    error_message: Optional[str] = Field(None, description="Error message if failed")

class AffiliateResearchListResponse(BaseModel):
    """Response schema for affiliate research list"""
    researches: List[AffiliateResearchResponse] = Field(..., description="List of researches")
    total: int = Field(..., description="Total number of researches")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")

class AffiliateResearchUpdate(BaseModel):
    """Request schema for updating affiliate research"""
    status: Optional[ResearchStatus] = Field(None, description="New status")
    query: Optional[str] = Field(None, min_length=1, max_length=500, description="Updated query")
    category: Optional[str] = Field(None, max_length=100, description="Updated category")
    country: Optional[str] = Field(None, max_length=2, description="Updated country code")
    min_commission: Optional[float] = Field(None, ge=0, le=100, description="Updated minimum commission")
    max_commission: Optional[float] = Field(None, ge=0, le=100, description="Updated maximum commission")
    payment_frequency: Optional[str] = Field(None, description="Updated payment frequency")
    cookie_duration: Optional[int] = Field(None, ge=1, description="Updated cookie duration")
    network_preferences: Optional[List[str]] = Field(None, description="Updated network preferences")
    exclude_networks: Optional[List[str]] = Field(None, description="Updated excluded networks")

class AffiliateNetworkResponse(BaseModel):
    """Response schema for affiliate network information"""
    name: str = Field(..., description="Network name")
    url: str = Field(..., description="Network URL")
    description: str = Field(..., description="Network description")
    logo_url: Optional[str] = Field(None, description="Network logo URL")
    supported_countries: List[str] = Field(..., description="Supported countries")
    payment_methods: List[str] = Field(..., description="Supported payment methods")
    minimum_payout: Optional[float] = Field(None, description="Minimum payout amount")
    currency: Optional[str] = Field(None, description="Default currency")
    tracking_methods: List[str] = Field(..., description="Supported tracking methods")
    api_available: bool = Field(..., description="API availability")
    real_time_reporting: bool = Field(..., description="Real-time reporting support")
    mobile_tracking: bool = Field(..., description="Mobile tracking support")
    deep_linking: bool = Field(..., description="Deep linking support")
    sub_affiliate_support: bool = Field(..., description="Sub-affiliate support")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Network rating")
    review_count: Optional[int] = Field(None, ge=0, description="Number of reviews")
    created_at: datetime = Field(..., description="Network creation date")
    updated_at: datetime = Field(..., description="Last update date")

class AffiliateCategoryResponse(BaseModel):
    """Response schema for affiliate category information"""
    name: str = Field(..., description="Category name")
    description: str = Field(..., description="Category description")
    program_count: int = Field(..., description="Number of programs in category")
    average_commission: float = Field(..., description="Average commission rate")
    top_networks: List[str] = Field(..., description="Top networks in category")
    created_at: datetime = Field(..., description="Category creation date")
    updated_at: datetime = Field(..., description="Last update date")

class AffiliateStatsResponse(BaseModel):
    """Response schema for affiliate statistics"""
    total_programs: int = Field(..., description="Total programs in database")
    total_networks: int = Field(..., description="Total networks in database")
    total_categories: int = Field(..., description="Total categories in database")
    average_commission: float = Field(..., description="Average commission rate")
    top_categories: List[Dict[str, Any]] = Field(..., description="Top categories by program count")
    top_networks: List[Dict[str, Any]] = Field(..., description="Top networks by program count")
    recent_searches: List[Dict[str, Any]] = Field(..., description="Recent search queries")
    last_updated: datetime = Field(..., description="Last database update")

class AffiliateSearchFilters(BaseModel):
    """Request schema for affiliate search filters"""
    category: Optional[str] = Field(None, description="Category filter")
    country: Optional[str] = Field(None, description="Country filter")
    min_commission: Optional[float] = Field(None, ge=0, le=100, description="Minimum commission")
    max_commission: Optional[float] = Field(None, ge=0, le=100, description="Maximum commission")
    payment_frequency: Optional[str] = Field(None, description="Payment frequency filter")
    cookie_duration_min: Optional[int] = Field(None, ge=1, description="Minimum cookie duration")
    cookie_duration_max: Optional[int] = Field(None, ge=1, description="Maximum cookie duration")
    network: Optional[str] = Field(None, description="Network filter")
    rating_min: Optional[float] = Field(None, ge=0, le=5, description="Minimum rating")
    has_promotional_materials: Optional[bool] = Field(None, description="Has promotional materials")
    has_real_time_reporting: Optional[bool] = Field(None, description="Has real-time reporting")
    has_mobile_tracking: Optional[bool] = Field(None, description="Has mobile tracking")
    has_deep_linking: Optional[bool] = Field(None, description="Has deep linking")
    has_sub_affiliate_support: Optional[bool] = Field(None, description="Has sub-affiliate support")
    sort_by: Optional[str] = Field("relevance", description="Sort field")
    sort_order: Optional[str] = Field("desc", description="Sort order (asc/desc)")
    page: Optional[int] = Field(1, ge=1, description="Page number")
    per_page: Optional[int] = Field(20, ge=1, le=100, description="Items per page")

# Additional schemas for the new API endpoints
class AffiliateSearchRequest(BaseModel):
    """Request schema for affiliate search"""
    search_term: str = Field(..., min_length=1, max_length=200, description="Search term")
    niche: Optional[str] = Field(None, max_length=100, description="Niche/category")
    budget_range: Optional[str] = Field(None, max_length=50, description="Budget range")
    user_id: Optional[str] = Field(None, description="User ID")

class AffiliateSearchResponse(BaseModel):
    """Response schema for affiliate search"""
    success: bool = Field(..., description="Success status")
    message: str = Field(..., description="Response message")
    data: Dict[str, Any] = Field(..., description="Search results data")

class ContentIdeasRequest(BaseModel):
    """Request schema for content ideas generation"""
    selected_programs: List[str] = Field(..., description="Selected affiliate program IDs")
    user_id: Optional[str] = Field(None, description="User ID")

class ContentIdeasResponse(BaseModel):
    """Response schema for content ideas"""
    success: bool = Field(..., description="Success status")
    message: str = Field(..., description="Response message")
    data: Dict[str, Any] = Field(..., description="Content ideas data")

class ResearchHistoryResponse(BaseModel):
    """Response schema for research history"""
    success: bool = Field(..., description="Success status")
    message: str = Field(..., description="Response message")
    data: Dict[str, Any] = Field(..., description="Research history data")
