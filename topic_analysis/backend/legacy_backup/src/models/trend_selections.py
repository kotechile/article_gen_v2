"""
TrendSelections model for TrendTap Enhanced Research Workflow
"""

from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, Enum, CheckConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from ..core.database import Base


class TrendCategory(PyEnum):
    """Trend category enumeration"""
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    LIFESTYLE = "lifestyle"
    HEALTH = "health"
    FINANCE = "finance"
    ENTERTAINMENT = "entertainment"


class CompetitionLevel(PyEnum):
    """Competition level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TrendSource(PyEnum):
    """Trend source enumeration"""
    LLM_ANALYSIS = "llm_analysis"
    CSV_UPLOAD = "csv_upload"


class TrendSelections(Base):
    """Trend selections model for enhanced research workflow"""
    __tablename__ = "trend_selections"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    trend_analysis_id = Column(String, ForeignKey("trend_analyses.id"), nullable=True, index=True)
    trend_name = Column(String(255), nullable=False)
    trend_description = Column(Text, nullable=True)
    trend_category = Column(Enum(TrendCategory), nullable=True)
    search_volume = Column(Integer, default=0, nullable=False)
    competition_level = Column(Enum(CompetitionLevel), nullable=True)
    source = Column(Enum(TrendSource), nullable=False)
    selected_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="trend_selections")
    trend_analysis = relationship("TrendAnalysis", back_populates="trend_selections")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('search_volume >= 0', name='check_search_volume_non_negative'),
    )
    
    def __repr__(self):
        return f"<TrendSelection(id={self.id}, name={self.trend_name}, source={self.source})>"
