"""
ContentIdeas model for TrendTap
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Enum, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from ..core.database import Base


class ContentType(PyEnum):
    """Content type enumeration"""
    ARTICLE = "article"
    GUIDE = "guide"
    REVIEW = "review"
    TUTORIAL = "tutorial"
    LISTICLE = "listicle"


class ContentAngle(PyEnum):
    """Content angle enumeration"""
    HOW_TO = "how-to"
    VS = "vs"
    LISTICLE = "listicle"
    PAIN_POINT = "pain-point"
    STORY = "story"


class ContentStatus(PyEnum):
    """Content status enumeration"""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class IdeaStatus(PyEnum):
    """Idea status enumeration (alias for ContentStatus)"""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class ContentIdeas(Base):
    """Content ideas model"""
    __tablename__ = "content_ideas"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    keyword_data_id = Column(Integer, ForeignKey("keyword_data.id"), nullable=True, index=True)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    content_type = Column(Enum(ContentType), nullable=False)
    angle = Column(Enum(ContentAngle), nullable=False)
    
    # Content scoring
    headline_score = Column(Float, nullable=True)  # CoSchedule headline score
    priority_score = Column(Float, nullable=True)  # Overall priority score
    
    # Content structure
    outline = Column(JSON, nullable=True)  # Content outline structure
    seo_recommendations = Column(JSON, nullable=True)  # SEO recommendations
    target_keywords = Column(JSON, nullable=True)  # Target keywords array
    
    # Affiliate integration
    affiliate_opportunities = Column(JSON, nullable=True)  # Affiliate opportunities
    monetization_potential = Column(Float, nullable=True)  # Monetization potential score
    
    # Content status
    status = Column(Enum(ContentStatus), default=ContentStatus.DRAFT, nullable=False)
    scheduled_date = Column(DateTime(timezone=True), nullable=True)
    published_date = Column(DateTime(timezone=True), nullable=True)
    
    # Content metadata
    word_count_target = Column(Integer, nullable=True)
    reading_time_estimate = Column(Integer, nullable=True)  # In minutes
    difficulty_level = Column(String(50), nullable=True)  # beginner, intermediate, advanced
    
    # Generation metadata
    generation_model = Column(String(50), nullable=True)
    generation_prompt = Column(Text, nullable=True)
    generation_parameters = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="content_ideas")
    keyword_data = relationship("KeywordData", back_populates="content_ideas")
    content_calendar = relationship("ContentCalendar", back_populates="content_idea")
    
    def __repr__(self):
        return f"<ContentIdeas(id={self.id}, title='{self.title}', type='{self.content_type.value}')>"
    
    def to_dict(self):
        """Convert content idea to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "keyword_data_id": self.keyword_data_id,
            "title": self.title,
            "description": self.description,
            "content_type": self.content_type.value,
            "angle": self.angle.value,
            "headline_score": self.headline_score,
            "priority_score": self.priority_score,
            "outline": self.outline,
            "seo_recommendations": self.seo_recommendations,
            "target_keywords": self.target_keywords,
            "affiliate_opportunities": self.affiliate_opportunities,
            "monetization_potential": self.monetization_potential,
            "status": self.status.value,
            "scheduled_date": self.scheduled_date.isoformat() if self.scheduled_date else None,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "word_count_target": self.word_count_target,
            "reading_time_estimate": self.reading_time_estimate,
            "difficulty_level": self.difficulty_level,
            "generation_model": self.generation_model,
            "generation_prompt": self.generation_prompt,
            "generation_parameters": self.generation_parameters,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def is_published(self) -> bool:
        """Check if content is published"""
        return self.status == ContentStatus.PUBLISHED
    
    def is_scheduled(self) -> bool:
        """Check if content is scheduled"""
        return self.status == ContentStatus.SCHEDULED
    
    def is_draft(self) -> bool:
        """Check if content is draft"""
        return self.status == ContentStatus.DRAFT
    
    def get_outline_sections(self) -> list:
        """Get outline sections"""
        if not self.outline or "sections" not in self.outline:
            return []
        
        return self.outline["sections"]
    
    def get_target_keywords_list(self) -> list:
        """Get target keywords as list"""
        if not self.target_keywords:
            return []
        
        return self.target_keywords
    
    def get_affiliate_opportunities_count(self) -> int:
        """Get count of affiliate opportunities"""
        if not self.affiliate_opportunities:
            return 0
        
        return len(self.affiliate_opportunities)
    
    def get_seo_recommendations(self) -> dict:
        """Get SEO recommendations"""
        if not self.seo_recommendations:
            return {}
        
        return self.seo_recommendations
    
    def get_meta_description(self) -> str:
        """Get meta description from SEO recommendations"""
        seo = self.get_seo_recommendations()
        return seo.get("meta_description", "")
    
    def get_word_count_target(self) -> int:
        """Get target word count"""
        if self.word_count_target:
            return self.word_count_target
        
        # Default word count based on content type
        default_counts = {
            ContentType.ARTICLE: 1500,
            ContentType.GUIDE: 3000,
            ContentType.REVIEW: 1000,
            ContentType.TUTORIAL: 2000,
            ContentType.LISTICLE: 1200
        }
        
        return default_counts.get(self.content_type, 1500)
    
    def get_reading_time_estimate(self) -> int:
        """Get estimated reading time in minutes"""
        if self.reading_time_estimate:
            return self.reading_time_estimate
        
        # Calculate based on word count (average 200 words per minute)
        word_count = self.get_word_count_target()
        return max(1, word_count // 200)
    
    def get_difficulty_level(self) -> str:
        """Get difficulty level"""
        if self.difficulty_level:
            return self.difficulty_level
        
        # Determine difficulty based on priority score
        if self.priority_score and self.priority_score >= 0.8:
            return "advanced"
        elif self.priority_score and self.priority_score >= 0.5:
            return "intermediate"
        else:
            return "beginner"
    
    def get_headline_score_category(self) -> str:
        """Get headline score category"""
        if not self.headline_score:
            return "unknown"
        
        if self.headline_score >= 80:
            return "excellent"
        elif self.headline_score >= 60:
            return "good"
        elif self.headline_score >= 40:
            return "fair"
        else:
            return "poor"
    
    def get_priority_score_category(self) -> str:
        """Get priority score category"""
        if not self.priority_score:
            return "unknown"
        
        if self.priority_score >= 0.8:
            return "high"
        elif self.priority_score >= 0.5:
            return "medium"
        else:
            return "low"
    
    def update_status(self, status: ContentStatus, scheduled_date: DateTime = None):
        """Update content status"""
        self.status = status
        
        if status == ContentStatus.SCHEDULED and scheduled_date:
            self.scheduled_date = scheduled_date
        elif status == ContentStatus.PUBLISHED:
            self.published_date = func.now()
    
    def add_affiliate_opportunity(self, opportunity: dict):
        """Add affiliate opportunity"""
        if not self.affiliate_opportunities:
            self.affiliate_opportunities = []
        
        self.affiliate_opportunities.append(opportunity)
    
    def calculate_monetization_potential(self) -> float:
        """Calculate monetization potential score"""
        if not self.affiliate_opportunities:
            return 0.0
        
        # Calculate based on affiliate opportunities
        total_epc = 0
        count = 0
        
        for opportunity in self.affiliate_opportunities:
            if "epc" in opportunity:
                total_epc += opportunity["epc"]
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_epc = total_epc / count
        
        # Normalize to 0-1 scale
        return min(avg_epc / 50.0, 1.0)  # Assuming max EPC of 50
    
    def get_content_summary(self) -> dict:
        """Get content summary"""
        return {
            "id": self.id,
            "title": self.title,
            "content_type": self.content_type.value,
            "angle": self.angle.value,
            "headline_score": self.headline_score,
            "priority_score": self.priority_score,
            "status": self.status.value,
            "word_count_target": self.get_word_count_target(),
            "reading_time_estimate": self.get_reading_time_estimate(),
            "difficulty_level": self.get_difficulty_level(),
            "affiliate_opportunities_count": self.get_affiliate_opportunities_count(),
            "monetization_potential": self.monetization_potential
        }
