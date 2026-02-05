"""
KeywordClusters model for TrendTap Enhanced Research Workflow
"""

from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, JSON, Enum, Numeric, CheckConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from ..core.database import Base


class SearchIntent(PyEnum):
    """Search intent enumeration"""
    INFORMATIONAL = "informational"
    COMMERCIAL = "commercial"
    TRANSACTIONAL = "transactional"


class KeywordClusters(Base):
    """Keyword clusters model for enhanced research workflow"""
    __tablename__ = "keyword_clusters"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    keyword_data_id = Column(String, ForeignKey("keyword_data.id"), nullable=True, index=True)
    cluster_name = Column(String(255), nullable=False)
    cluster_description = Column(Text, nullable=True)
    keywords = Column(JSON, nullable=False)
    cluster_size = Column(Integer, nullable=False)
    search_intent = Column(Enum(SearchIntent), nullable=True)
    content_theme = Column(String(255), nullable=True)
    priority_score = Column(Numeric(3, 2), default=0.0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="keyword_clusters")
    keyword_data = relationship("KeywordData", back_populates="keyword_clusters")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('cluster_size > 0', name='check_cluster_size_positive'),
        CheckConstraint('priority_score >= 0.0 AND priority_score <= 1.0', name='check_priority_score_range'),
    )
    
    def __repr__(self):
        return f"<KeywordCluster(id={self.id}, name={self.cluster_name}, size={self.cluster_size})>"
