"""
This model file is disabled for Supabase-only architecture.
All database operations go through Supabase SDK.
"""

"""
Enhanced Content Idea model for dataflow persistence
This model represents content ideas generated from trend analyses
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field, validator
from enum import Enum

class ContentType(str, Enum):
    """Content type enumeration"""
    BLOG_POST = "blog_post"
    ARTICLE = "article"
    VIDEO = "video"
    PODCAST = "podcast"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    NEWSLETTER = "newsletter"
    WHITEPAPER = "whitepaper"
    CASE_STUDY = "case_study"
    GUIDE = "guide"
    TUTORIAL = "tutorial"
    WEBINAR = "webinar"
    PRESENTATION = "presentation"
    INFOGRAPHIC = "infographic"
    CHECKLIST = "checklist"
    TEMPLATE = "template"
    OTHER = "other"

class IdeaType(str, Enum):
    """Idea type enumeration"""
    TRENDING = "trending"
    EVERGREEN = "evergreen"
    SEASONAL = "seasonal"
    BREAKING = "breaking"
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    PROMOTIONAL = "promotional"
    THOUGHT_LEADERSHIP = "thought_leadership"
    CASE_STUDY = "case_study"
    HOW_TO = "how_to"
    LIFESTYLE = "lifestyle"
    NEWS = "news"
    REVIEW = "review"
    COMPARISON = "comparison"
    OTHER = "other"

class ContentStatus(str, Enum):
    """Content status enumeration"""
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    CANCELLED = "cancelled"

class ContentIdeaBase(BaseModel):
    """Base content idea model with common fields"""
    title: str = Field(..., min_length=1, max_length=500, description="Content idea title")
    description: str = Field(..., min_length=1, description="Content idea description")
    content_type: ContentType = Field(..., description="Type of content")
    idea_type: IdeaType = Field(..., description="Type of idea")
    primary_keyword: str = Field(..., min_length=1, max_length=255, description="Primary keyword for SEO")
    secondary_keywords: List[str] = Field(default_factory=list, description="Secondary keywords")
    target_audience: Optional[str] = Field(None, max_length=500, description="Target audience description")
    key_points: List[str] = Field(default_factory=list, description="Key points to cover")
    tags: List[str] = Field(default_factory=list, description="Content tags")
    estimated_read_time: Optional[int] = Field(None, ge=1, description="Estimated read time in minutes")
    difficulty_level: Optional[str] = Field(None, max_length=50, description="Content difficulty level")
    difficulty_level: Optional[str] = Field(None, max_length=50, description="Content difficulty level")
    status: ContentStatus = Field(ContentStatus.DRAFT, description="Content status")
    markdown_outline: Optional[str] = Field(None, description="Structured content outline (H2/H3)")
    
    @validator('title')
    def validate_title(cls, v):
        if not v or not v.strip():
            raise ValueError('Title cannot be empty or whitespace only')
        return v.strip()
    
    @validator('description')
    def validate_description(cls, v):
        if not v or not v.strip():
            raise ValueError('Description cannot be empty or whitespace only')
        return v.strip()
    
    @validator('primary_keyword')
    def validate_primary_keyword(cls, v):
        if not v or not v.strip():
            raise ValueError('Primary keyword cannot be empty or whitespace only')
        return v.strip()
    
    @validator('secondary_keywords')
    def validate_secondary_keywords(cls, v):
        if v:
            # Remove empty keywords and strip whitespace
            cleaned_keywords = [kw.strip() for kw in v if kw and kw.strip()]
            if len(cleaned_keywords) != len(set(cleaned_keywords)):
                raise ValueError('Secondary keywords must be unique')
            return cleaned_keywords
        return v
    
    @validator('target_audience')
    def validate_target_audience(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        return v.strip() if v else v
    
    @validator('key_points')
    def validate_key_points(cls, v):
        if v:
            # Remove empty points and strip whitespace
            cleaned_points = [point.strip() for point in v if point and point.strip()]
            return cleaned_points
        return v
    
    @validator('tags')
    def validate_tags(cls, v):
        if v:
            # Remove empty tags and strip whitespace
            cleaned_tags = [tag.strip() for tag in v if tag and tag.strip()]
            if len(cleaned_tags) != len(set(cleaned_tags)):
                raise ValueError('Tags must be unique')
            return cleaned_tags
        return v
    
    @validator('difficulty_level')
    def validate_difficulty_level(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        return v.strip() if v else v

class ContentIdeaCreate(ContentIdeaBase):
    """Model for creating a new content idea"""
    trend_analysis_id: UUID = Field(..., description="ID of the trend analysis this idea is based on")
    research_topic_id: UUID = Field(..., description="ID of the research topic this idea belongs to")

class ContentIdeaUpdate(BaseModel):
    """Model for updating an existing content idea"""
    title: Optional[str] = Field(None, min_length=1, max_length=500, description="Content idea title")
    description: Optional[str] = Field(None, min_length=1, description="Content idea description")
    content_type: Optional[ContentType] = Field(None, description="Type of content")
    idea_type: Optional[IdeaType] = Field(None, description="Type of idea")
    primary_keyword: Optional[str] = Field(None, min_length=1, max_length=255, description="Primary keyword for SEO")
    secondary_keywords: Optional[List[str]] = Field(None, description="Secondary keywords")
    target_audience: Optional[str] = Field(None, max_length=500, description="Target audience description")
    key_points: Optional[List[str]] = Field(None, description="Key points to cover")
    tags: Optional[List[str]] = Field(None, description="Content tags")
    estimated_read_time: Optional[int] = Field(None, ge=1, description="Estimated read time in minutes")
    difficulty_level: Optional[str] = Field(None, max_length=50, description="Content difficulty level")
    status: Optional[ContentStatus] = Field(None, description="Content status")
    
    @validator('title')
    def validate_title(cls, v):
        if v is not None and (not v or not v.strip()):
            raise ValueError('Title cannot be empty or whitespace only')
        return v.strip() if v else v
    
    @validator('description')
    def validate_description(cls, v):
        if v is not None and (not v or not v.strip()):
            raise ValueError('Description cannot be empty or whitespace only')
        return v.strip() if v else v
    
    @validator('primary_keyword')
    def validate_primary_keyword(cls, v):
        if v is not None and (not v or not v.strip()):
            raise ValueError('Primary keyword cannot be empty or whitespace only')
        return v.strip() if v else v
    
    @validator('secondary_keywords')
    def validate_secondary_keywords(cls, v):
        if v is not None:
            # Remove empty keywords and strip whitespace
            cleaned_keywords = [kw.strip() for kw in v if kw and kw.strip()]
            if len(cleaned_keywords) != len(set(cleaned_keywords)):
                raise ValueError('Secondary keywords must be unique')
            return cleaned_keywords
        return v
    
    @validator('target_audience')
    def validate_target_audience(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        return v.strip() if v else v
    
    @validator('key_points')
    def validate_key_points(cls, v):
        if v is not None:
            # Remove empty points and strip whitespace
            cleaned_points = [point.strip() for point in v if point and point.strip()]
            return cleaned_points
        return v
    
    @validator('tags')
    def validate_tags(cls, v):
        if v is not None:
            # Remove empty tags and strip whitespace
            cleaned_tags = [tag.strip() for tag in v if tag and tag.strip()]
            if len(cleaned_tags) != len(set(cleaned_tags)):
                raise ValueError('Tags must be unique')
            return cleaned_tags
        return v
    
    @validator('difficulty_level')
    def validate_difficulty_level(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        return v.strip() if v else v

class ContentIdea(ContentIdeaBase):
    """Complete content idea model with all fields"""
    id: UUID = Field(..., description="Unique identifier")
    user_id: UUID = Field(..., description="User who owns this content idea")
    trend_analysis_id: UUID = Field(..., description="ID of the trend analysis this idea is based on")
    research_topic_id: UUID = Field(..., description="ID of the research topic this idea belongs to")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }

class ContentIdeaResponse(ContentIdea):
    """Content idea response model for API responses"""
    pass

class ContentIdeaListResponse(BaseModel):
    """Response model for listing content ideas"""
    items: List[ContentIdeaResponse] = Field(..., description="List of content ideas")
    total: int = Field(..., ge=0, description="Total number of content ideas")
    page: int = Field(..., ge=1, description="Current page number")
    size: int = Field(..., ge=1, le=100, description="Page size")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")

class ContentIdeaWithTrendAnalysis(ContentIdea):
    """Content idea model with associated trend analysis"""
    trend_analysis: Optional[dict] = Field(None, description="Associated trend analysis")

class ContentIdeaStats(BaseModel):
    """Content idea statistics model"""
    total_ideas: int = Field(..., ge=0, description="Total number of content ideas")
    draft_ideas: int = Field(..., ge=0, description="Number of draft ideas")
    published_ideas: int = Field(..., ge=0, description="Number of published ideas")
    ideas_by_type: Dict[str, int] = Field(default_factory=dict, description="Ideas count by content type")
    ideas_by_idea_type: Dict[str, int] = Field(default_factory=dict, description="Ideas count by idea type")
    average_read_time: Optional[float] = Field(None, description="Average estimated read time")
    most_used_keywords: List[Dict[str, Any]] = Field(default_factory=list, description="Most used keywords")
    most_used_tags: List[Dict[str, Any]] = Field(default_factory=list, description="Most used tags")
    last_idea_created: Optional[datetime] = Field(None, description="Last idea creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ContentIdeaFilter(BaseModel):
    """Model for filtering content ideas"""
    content_type: Optional[ContentType] = Field(None, description="Filter by content type")
    idea_type: Optional[IdeaType] = Field(None, description="Filter by idea type")
    status: Optional[ContentStatus] = Field(None, description="Filter by status")
    primary_keyword: Optional[str] = Field(None, description="Filter by primary keyword")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    difficulty_level: Optional[str] = Field(None, description="Filter by difficulty level")
    created_after: Optional[datetime] = Field(None, description="Filter by creation date (after)")
    created_before: Optional[datetime] = Field(None, description="Filter by creation date (before)")
    research_topic_id: Optional[UUID] = Field(None, description="Filter by research topic")
    trend_analysis_id: Optional[UUID] = Field(None, description="Filter by trend analysis")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }

class ContentIdeaSearch(BaseModel):
    """Model for searching content ideas"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    search_fields: List[str] = Field(default=["title", "description", "primary_keyword"], description="Fields to search in")
    filters: Optional[ContentIdeaFilter] = Field(None, description="Additional filters")
    sort_by: Optional[str] = Field("created_at", description="Field to sort by")
    sort_order: Optional[str] = Field("desc", description="Sort order (asc/desc)")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Search query cannot be empty or whitespace only')
        return v.strip()
    
    @validator('search_fields')
    def validate_search_fields(cls, v):
        allowed_fields = ["title", "description", "primary_keyword", "secondary_keywords", "key_points", "tags"]
        for field in v:
            if field not in allowed_fields:
                raise ValueError(f'Invalid search field: {field}')
        return v
    
    @validator('sort_by')
    def validate_sort_by(cls, v):
        allowed_fields = ["created_at", "updated_at", "title", "content_type", "idea_type", "status"]
        if v not in allowed_fields:
            raise ValueError(f'Invalid sort field: {v}')
        return v
    
    @validator('sort_order')
    def validate_sort_order(cls, v):
        if v not in ["asc", "desc"]:
            raise ValueError('Sort order must be "asc" or "desc"')
        return v
