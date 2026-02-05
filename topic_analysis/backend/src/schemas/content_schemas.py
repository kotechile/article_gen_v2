"""
Content Generation API schemas
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ContentType(str, Enum):
    """Content type enumeration"""
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    PRODUCT_REVIEW = "product_review"
    HOW_TO_GUIDE = "how_to_guide"
    LANDING_PAGE = "landing_page"
    EMAIL = "email"
    SOCIAL_MEDIA = "social_media"
    PRESS_RELEASE = "press_release"

class ContentStatus(str, Enum):
    """Content status enumeration"""
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class ContentGenerationRequest(BaseModel):
    """Request schema for content generation"""
    title: str = Field(..., min_length=1, max_length=200, description="Content title")
    content_type: ContentType = Field(..., description="Type of content")
    keywords: List[str] = Field(..., min_items=1, description="Target keywords")
    target_audience: Optional[str] = Field(None, description="Target audience")
    tone: Optional[str] = Field("professional", description="Content tone")
    length: Optional[str] = Field("medium", description="Content length (short, medium, long)")
    language: Optional[str] = Field("en", max_length=5, description="Content language")
    include_images: Optional[bool] = Field(True, description="Include image suggestions")
    include_meta_description: Optional[bool] = Field(True, description="Include meta description")
    include_headings: Optional[bool] = Field(True, description="Include heading structure")
    include_call_to_action: Optional[bool] = Field(True, description="Include call-to-action")
    competitor_urls: Optional[List[str]] = Field(None, description="Competitor URLs for analysis")
    additional_instructions: Optional[str] = Field(None, description="Additional instructions")
    
    @validator('keywords')
    def validate_keywords(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one keyword is required')
        return v

class ContentIdeaResponse(BaseModel):
    """Response schema for content idea"""
    id: int = Field(..., description="Idea ID")
    user_id: int = Field(..., description="User ID")
    title: str = Field(..., description="Content title")
    content_type: ContentType = Field(..., description="Content type")
    keywords: List[str] = Field(..., description="Target keywords")
    target_audience: Optional[str] = Field(None, description="Target audience")
    tone: str = Field(..., description="Content tone")
    length: str = Field(..., description="Content length")
    language: str = Field(..., description="Content language")
    status: ContentStatus = Field(..., description="Content status")
    
    # Generated content
    outline: Optional[str] = Field(None, description="Content outline")
    introduction: Optional[str] = Field(None, description="Introduction paragraph")
    main_content: Optional[str] = Field(None, description="Main content")
    conclusion: Optional[str] = Field(None, description="Conclusion paragraph")
    meta_description: Optional[str] = Field(None, description="Meta description")
    headings: Optional[List[str]] = Field(None, description="Content headings")
    call_to_action: Optional[str] = Field(None, description="Call-to-action")
    
    # SEO metrics
    word_count: Optional[int] = Field(None, ge=0, description="Word count")
    readability_score: Optional[float] = Field(None, ge=0, le=100, description="Readability score")
    keyword_density: Optional[Dict[str, float]] = Field(None, description="Keyword density")
    seo_score: Optional[float] = Field(None, ge=0, le=100, description="SEO score")
    
    # Additional data
    image_suggestions: Optional[List[str]] = Field(None, description="Image suggestions")
    related_topics: Optional[List[str]] = Field(None, description="Related topics")
    competitor_analysis: Optional[Dict[str, Any]] = Field(None, description="Competitor analysis")
    additional_instructions: Optional[str] = Field(None, description="Additional instructions")
    
    # Timestamps
    created_at: datetime = Field(..., description="Creation date")
    updated_at: datetime = Field(..., description="Last update date")
    generated_at: Optional[datetime] = Field(None, description="Generation completion date")

class ContentIdeasResponse(BaseModel):
    """Response schema for content ideas (alias for ContentIdeaResponse)"""
    id: int = Field(..., description="Idea ID")
    user_id: int = Field(..., description="User ID")
    title: str = Field(..., description="Content title")
    content_type: ContentType = Field(..., description="Content type")
    keywords: List[str] = Field(..., description="Target keywords")
    target_audience: Optional[str] = Field(None, description="Target audience")
    tone: str = Field(..., description="Content tone")
    length: str = Field(..., description="Content length")
    language: str = Field(..., description="Content language")
    status: ContentStatus = Field(..., description="Content status")
    
    # Generated content
    outline: Optional[str] = Field(None, description="Content outline")
    introduction: Optional[str] = Field(None, description="Introduction paragraph")
    main_content: Optional[str] = Field(None, description="Main content")
    conclusion: Optional[str] = Field(None, description="Conclusion paragraph")
    meta_description: Optional[str] = Field(None, description="Meta description")
    headings: Optional[List[str]] = Field(None, description="Content headings")
    call_to_action: Optional[str] = Field(None, description="Call-to-action")
    
    # SEO metrics
    word_count: Optional[int] = Field(None, ge=0, description="Word count")
    readability_score: Optional[float] = Field(None, ge=0, le=100, description="Readability score")
    keyword_density: Optional[Dict[str, float]] = Field(None, description="Keyword density")
    seo_score: Optional[float] = Field(None, ge=0, le=100, description="SEO score")
    
    # Additional data
    image_suggestions: Optional[List[str]] = Field(None, description="Image suggestions")
    related_topics: Optional[List[str]] = Field(None, description="Related topics")
    competitor_analysis: Optional[Dict[str, Any]] = Field(None, description="Competitor analysis")
    additional_instructions: Optional[str] = Field(None, description="Additional instructions")
    
    # Timestamps
    created_at: datetime = Field(..., description="Creation date")
    updated_at: datetime = Field(..., description="Last update date")
    generated_at: Optional[datetime] = Field(None, description="Generation completion date")

class ContentOutlineResponse(BaseModel):
    """Response schema for content outline"""
    id: int = Field(..., description="Outline ID")
    content_id: int = Field(..., description="Content ID")
    title: str = Field(..., description="Content title")
    outline: str = Field(..., description="Content outline")
    headings: List[str] = Field(..., description="Content headings")
    subheadings: Optional[Dict[str, List[str]]] = Field(None, description="Subheadings by section")
    estimated_word_count: int = Field(..., description="Estimated word count")
    estimated_reading_time: int = Field(..., description="Estimated reading time in minutes")
    seo_suggestions: List[str] = Field(..., description="SEO suggestions")
    content_angles: List[str] = Field(..., description="Suggested content angles")
    created_at: datetime = Field(..., description="Outline creation date")
    updated_at: datetime = Field(..., description="Last update date")

class ContentIdeaListResponse(BaseModel):
    """Response schema for content idea list"""
    ideas: List[ContentIdeaResponse] = Field(..., description="List of content ideas")
    total: int = Field(..., description="Total number of ideas")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")

class ContentIdeasListResponse(BaseModel):
    """Response schema for content ideas list (alias for ContentIdeaListResponse)"""
    ideas: List[ContentIdeasResponse] = Field(..., description="List of content ideas")
    total: int = Field(..., description="Total number of ideas")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")

class ContentIdeaUpdateRequest(BaseModel):
    """Request schema for updating content idea"""
    title: Optional[str] = Field(None, min_length=1, max_length=200, description="Updated title")
    content_type: Optional[ContentType] = Field(None, description="Updated content type")
    keywords: Optional[List[str]] = Field(None, description="Updated keywords")
    target_audience: Optional[str] = Field(None, description="Updated target audience")
    tone: Optional[str] = Field(None, description="Updated tone")
    length: Optional[str] = Field(None, description="Updated length")
    language: Optional[str] = Field(None, max_length=5, description="Updated language")
    status: Optional[ContentStatus] = Field(None, description="Updated status")
    additional_instructions: Optional[str] = Field(None, description="Updated instructions")

class ContentUpdateRequest(BaseModel):
    """Request schema for updating content (alias for ContentIdeaUpdateRequest)"""
    title: Optional[str] = Field(None, min_length=1, max_length=200, description="Updated title")
    content_type: Optional[ContentType] = Field(None, description="Updated content type")
    keywords: Optional[List[str]] = Field(None, description="Updated keywords")
    target_audience: Optional[str] = Field(None, description="Updated target audience")
    tone: Optional[str] = Field(None, description="Updated tone")
    length: Optional[str] = Field(None, description="Updated length")
    language: Optional[str] = Field(None, max_length=5, description="Updated language")
    status: Optional[ContentStatus] = Field(None, description="Updated status")
    additional_instructions: Optional[str] = Field(None, description="Updated instructions")

class ContentGenerationResponse(BaseModel):
    """Response schema for content generation results"""
    idea_id: int = Field(..., description="Content idea ID")
    generated_content: ContentIdeaResponse = Field(..., description="Generated content")
    generation_metadata: Dict[str, Any] = Field(..., description="Generation metadata")
    processing_time: float = Field(..., description="Processing time in seconds")
    tokens_used: Optional[int] = Field(None, description="Tokens used for generation")
    model_used: Optional[str] = Field(None, description="AI model used")
    created_at: datetime = Field(..., description="Generation date")

class ContentBatchGenerationRequest(BaseModel):
    """Request schema for batch content generation"""
    ideas: List[ContentGenerationRequest] = Field(..., min_items=1, max_items=10, description="Content ideas to generate")
    batch_name: Optional[str] = Field(None, description="Batch name")
    priority: Optional[int] = Field(1, ge=1, le=5, description="Batch priority")
    
    @validator('ideas')
    def validate_ideas(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one content idea is required')
        if len(v) > 10:
            raise ValueError('Maximum 10 content ideas allowed per batch')
        return v

class ContentBatchGenerationResponse(BaseModel):
    """Response schema for batch content generation results"""
    batch_id: str = Field(..., description="Batch ID")
    batch_name: Optional[str] = Field(None, description="Batch name")
    total_ideas: int = Field(..., description="Total ideas in batch")
    completed_ideas: int = Field(..., description="Completed ideas")
    failed_ideas: int = Field(..., description="Failed ideas")
    status: str = Field(..., description="Batch status")
    results: List[ContentGenerationResponse] = Field(..., description="Generation results")
    created_at: datetime = Field(..., description="Batch creation date")
    completed_at: Optional[datetime] = Field(None, description="Batch completion date")

class ContentTemplateResponse(BaseModel):
    """Response schema for content template"""
    id: int = Field(..., description="Template ID")
    name: str = Field(..., description="Template name")
    content_type: ContentType = Field(..., description="Content type")
    description: str = Field(..., description="Template description")
    template: str = Field(..., description="Template content")
    variables: List[str] = Field(..., description="Template variables")
    is_public: bool = Field(..., description="Is template public")
    usage_count: int = Field(..., description="Usage count")
    created_at: datetime = Field(..., description="Template creation date")
    updated_at: datetime = Field(..., description="Last update date")

class ContentTemplateListResponse(BaseModel):
    """Response schema for content template list"""
    templates: List[ContentTemplateResponse] = Field(..., description="List of templates")
    total: int = Field(..., description="Total number of templates")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")

class ContentStatsResponse(BaseModel):
    """Response schema for content statistics"""
    total_ideas: int = Field(..., description="Total content ideas")
    completed_ideas: int = Field(..., description="Completed ideas")
    published_ideas: int = Field(..., description="Published ideas")
    total_word_count: int = Field(..., description="Total word count")
    average_word_count: float = Field(..., description="Average word count")
    average_readability_score: float = Field(..., description="Average readability score")
    average_seo_score: float = Field(..., description="Average SEO score")
    top_content_types: List[Dict[str, Any]] = Field(..., description="Top content types")
    top_keywords: List[Dict[str, Any]] = Field(..., description="Top keywords used")
    last_updated: datetime = Field(..., description="Last database update")
