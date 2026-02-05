"""
ContentIdea model for storing generated content ideas with enhanced keywords and affiliate offers
"""

import uuid
from sqlalchemy import Column, String, DateTime, JSONB, ForeignKey, Enum, Boolean, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from ..core.database import Base

class ContentType(PyEnum):
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

class ContentStatus(PyEnum):
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class ContentPriority(PyEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class ContentIdea(Base):
    """
    ContentIdea model for storing generated content ideas with enhanced keywords and affiliate offers
    """
    __tablename__ = "content_ideas"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    workflow_session_id = Column(UUID(as_uuid=True), ForeignKey("workflow_sessions.id", ondelete="CASCADE"), nullable=False)
    trend_analysis_id = Column(UUID(as_uuid=True), ForeignKey("trend_analyses.id", ondelete="SET NULL"), nullable=True)
    topic_decomposition_id = Column(UUID(as_uuid=True), ForeignKey("topic_decompositions.id", ondelete="SET NULL"), nullable=True)
    
    # Content metadata
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    content_type = Column(Enum(ContentType), nullable=False, default=ContentType.BLOG_POST)
    status = Column(Enum(ContentStatus), nullable=False, default=ContentStatus.DRAFT)
    priority = Column(Enum(ContentPriority), nullable=False, default=ContentPriority.MEDIUM)
    
    # Content structure
    target_audience = Column(String(255), nullable=True)
    content_angle = Column(String(255), nullable=True)
    key_points = Column(JSONB, nullable=False, default=[])  # Array of key points
    content_outline = Column(JSONB, nullable=False, default=[])  # Structured outline
    
    # SEO and Keywords
    primary_keyword = Column(String(255), nullable=False)
    secondary_keywords = Column(JSONB, nullable=False, default=[])  # Array of secondary keywords
    enhanced_keywords = Column(JSONB, nullable=False, default=[])  # Keywords from external tools
    keyword_difficulty = Column(Integer, nullable=True)  # 0-100 scale
    search_volume = Column(Integer, nullable=True)
    cpc = Column(String(20), nullable=True)  # Cost per click
    
    # Affiliate integration
    affiliate_offers = Column(JSONB, nullable=False, default=[])  # Array of affiliate offer IDs
    affiliate_links = Column(JSONB, nullable=False, default=[])  # Array of affiliate links
    monetization_strategy = Column(Text, nullable=True)
    expected_revenue = Column(String(20), nullable=True)
    
    # Content generation metadata
    generation_prompt = Column(Text, nullable=True)
    generation_model = Column(String(100), nullable=True)
    generation_parameters = Column(JSONB, nullable=False, default={})
    generation_time_ms = Column(Integer, nullable=True)
    
    # Quality metrics
    readability_score = Column(Integer, nullable=True)  # 0-100 scale
    seo_score = Column(Integer, nullable=True)  # 0-100 scale
    engagement_score = Column(Integer, nullable=True)  # 0-100 scale
    quality_notes = Column(Text, nullable=True)
    
    # Publishing metadata
    target_publish_date = Column(DateTime(timezone=True), nullable=True)
    actual_publish_date = Column(DateTime(timezone=True), nullable=True)
    publish_url = Column(String(500), nullable=True)
    word_count = Column(Integer, nullable=True)
    reading_time_minutes = Column(Integer, nullable=True)
    
    # Tags and categories
    tags = Column(JSONB, nullable=False, default=[])  # Array of tags
    categories = Column(JSONB, nullable=False, default=[])  # Array of categories
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="content_ideas")
    workflow_session = relationship("WorkflowSessions", back_populates="content_ideas")
    trend_analysis = relationship("TrendAnalysis", back_populates="content_ideas")
    topic_decomposition = relationship("TopicDecomposition", back_populates="content_ideas")

    def __repr__(self):
        return f"<ContentIdea(id={self.id}, title='{self.title[:50]}...', status='{self.status.value}')>"

    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "workflow_session_id": str(self.workflow_session_id),
            "trend_analysis_id": str(self.trend_analysis_id) if self.trend_analysis_id else None,
            "topic_decomposition_id": str(self.topic_decomposition_id) if self.topic_decomposition_id else None,
            "title": self.title,
            "description": self.description,
            "content_type": self.content_type.value,
            "status": self.status.value,
            "priority": self.priority.value,
            "target_audience": self.target_audience,
            "content_angle": self.content_angle,
            "key_points": self.key_points,
            "content_outline": self.content_outline,
            "primary_keyword": self.primary_keyword,
            "secondary_keywords": self.secondary_keywords,
            "enhanced_keywords": self.enhanced_keywords,
            "keyword_difficulty": self.keyword_difficulty,
            "search_volume": self.search_volume,
            "cpc": self.cpc,
            "affiliate_offers": self.affiliate_offers,
            "affiliate_links": self.affiliate_links,
            "monetization_strategy": self.monetization_strategy,
            "expected_revenue": self.expected_revenue,
            "generation_prompt": self.generation_prompt,
            "generation_model": self.generation_model,
            "generation_parameters": self.generation_parameters,
            "generation_time_ms": self.generation_time_ms,
            "readability_score": self.readability_score,
            "seo_score": self.seo_score,
            "engagement_score": self.engagement_score,
            "quality_notes": self.quality_notes,
            "target_publish_date": self.target_publish_date.isoformat() if self.target_publish_date else None,
            "actual_publish_date": self.actual_publish_date.isoformat() if self.actual_publish_date else None,
            "publish_url": self.publish_url,
            "word_count": self.word_count,
            "reading_time_minutes": self.reading_time_minutes,
            "tags": self.tags,
            "categories": self.categories,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def to_summary_dict(self):
        """Return a summary version for list views"""
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "content_type": self.content_type.value,
            "status": self.status.value,
            "priority": self.priority.value,
            "primary_keyword": self.primary_keyword,
            "keyword_difficulty": self.keyword_difficulty,
            "search_volume": self.search_volume,
            "affiliate_offers_count": len(self.affiliate_offers) if self.affiliate_offers else 0,
            "word_count": self.word_count,
            "reading_time_minutes": self.reading_time_minutes,
            "seo_score": self.seo_score,
            "created_at": self.created_at.isoformat(),
            "target_publish_date": self.target_publish_date.isoformat() if self.target_publish_date else None
        }

    def is_draft(self) -> bool:
        """Check if content is in draft status"""
        return self.status == ContentStatus.DRAFT

    def is_published(self) -> bool:
        """Check if content is published"""
        return self.status == ContentStatus.PUBLISHED

    def is_high_priority(self) -> bool:
        """Check if content is high priority"""
        return self.priority in [ContentPriority.HIGH, ContentPriority.URGENT]

    def get_keyword_count(self) -> int:
        """Get total number of keywords"""
        count = 1  # Primary keyword
        if self.secondary_keywords:
            count += len(self.secondary_keywords)
        if self.enhanced_keywords:
            count += len(self.enhanced_keywords)
        return count

    def get_affiliate_offer_count(self) -> int:
        """Get number of affiliate offers"""
        return len(self.affiliate_offers) if self.affiliate_offers else 0

    def get_quality_score(self) -> float:
        """Calculate overall quality score"""
        scores = []
        if self.readability_score is not None:
            scores.append(self.readability_score)
        if self.seo_score is not None:
            scores.append(self.seo_score)
        if self.engagement_score is not None:
            scores.append(self.engagement_score)
        
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)

    def get_monetization_potential(self) -> str:
        """Assess monetization potential based on available data"""
        if not self.affiliate_offers and not self.expected_revenue:
            return "low"
        
        offer_count = self.get_affiliate_offer_count()
        if offer_count >= 3:
            return "high"
        elif offer_count >= 1:
            return "medium"
        else:
            return "low"

    def get_seo_potential(self) -> str:
        """Assess SEO potential based on keyword data"""
        if not self.keyword_difficulty and not self.search_volume:
            return "unknown"
        
        difficulty = self.keyword_difficulty or 50
        volume = self.search_volume or 0
        
        if difficulty <= 30 and volume >= 1000:
            return "high"
        elif difficulty <= 50 and volume >= 500:
            return "medium"
        else:
            return "low"

    def get_content_metrics(self) -> dict:
        """Get comprehensive content metrics"""
        return {
            "word_count": self.word_count or 0,
            "reading_time_minutes": self.reading_time_minutes or 0,
            "keyword_count": self.get_keyword_count(),
            "affiliate_offer_count": self.get_affiliate_offer_count(),
            "quality_score": self.get_quality_score(),
            "monetization_potential": self.get_monetization_potential(),
            "seo_potential": self.get_seo_potential(),
            "is_high_priority": self.is_high_priority(),
            "is_published": self.is_published()
        }
