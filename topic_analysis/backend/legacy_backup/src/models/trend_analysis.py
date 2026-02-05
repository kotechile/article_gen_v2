"""
TrendAnalysis model for storing trend analysis results
"""

import uuid
from sqlalchemy import Column, String, DateTime, JSON, ForeignKey, Enum, Boolean, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from ..core.database import Base

class TrendAnalysisStatus(PyEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class TrendAnalysisSource(PyEnum):
    GOOGLE_TRENDS = "google_trends"
    CSV_UPLOAD = "csv_upload"
    SEMRUSH = "semrush"
    AHREFS = "ahrefs"
    UBERSUGGEST = "ubersuggest"
    FALLBACK = "fallback"

class TrendAnalysis(Base):
    """
    TrendAnalysis model for storing trend analysis results and data
    """
    __tablename__ = "trend_analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    workflow_session_id = Column(UUID(as_uuid=True), ForeignKey("workflow_sessions.id", ondelete="CASCADE"), nullable=False)
    topic_decomposition_id = Column(UUID(as_uuid=True), ForeignKey("topic_decompositions.id", ondelete="SET NULL"), nullable=True)
    
    # Analysis metadata
    analysis_name = Column(String(255), nullable=False)
    description = Column(String(1000), nullable=True)
    keywords = Column(JSONB, nullable=False, default=[])  # List of keywords analyzed
    timeframe = Column(String(50), nullable=False, default="12m")  # 1h, 4h, 1d, 7d, 30d, 90d, 12m, 5y, all
    geo = Column(String(10), nullable=False, default="US")  # Geographic location
    category = Column(Integer, nullable=True)  # Google Trends category ID
    
    # Analysis results
    trend_data = Column(JSONB, nullable=False, default={})  # Raw trend data from APIs
    analysis_results = Column(JSONB, nullable=False, default={})  # Processed analysis results
    insights = Column(JSONB, nullable=False, default={})  # Generated insights and recommendations
    
    # Source and status
    source = Column(Enum(TrendAnalysisSource), nullable=False, default=TrendAnalysisSource.GOOGLE_TRENDS)
    status = Column(Enum(TrendAnalysisStatus), nullable=False, default=TrendAnalysisStatus.PENDING)
    error_message = Column(String(1000), nullable=True)
    
    # Processing metadata
    processing_time_ms = Column(Integer, nullable=True)  # Processing time in milliseconds
    api_calls_made = Column(Integer, nullable=False, default=0)
    cache_hit = Column(Boolean, nullable=False, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="trend_analyses")
    workflow_session = relationship("WorkflowSessions", back_populates="trend_analyses")
    topic_decomposition = relationship("TopicDecomposition", back_populates="trend_analyses")

    def __repr__(self):
        return f"<TrendAnalysis(id={self.id}, name='{self.analysis_name}', status='{self.status.value}')>"

    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "workflow_session_id": str(self.workflow_session_id),
            "topic_decomposition_id": str(self.topic_decomposition_id) if self.topic_decomposition_id else None,
            "analysis_name": self.analysis_name,
            "description": self.description,
            "keywords": self.keywords,
            "timeframe": self.timeframe,
            "geo": self.geo,
            "category": self.category,
            "trend_data": self.trend_data,
            "analysis_results": self.analysis_results,
            "insights": self.insights,
            "source": self.source.value,
            "status": self.status.value,
            "error_message": self.error_message,
            "processing_time_ms": self.processing_time_ms,
            "api_calls_made": self.api_calls_made,
            "cache_hit": self.cache_hit,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }

    def to_summary_dict(self):
        """Return a summary version for list views"""
        return {
            "id": str(self.id),
            "analysis_name": self.analysis_name,
            "description": self.description,
            "keywords_count": len(self.keywords) if self.keywords else 0,
            "timeframe": self.timeframe,
            "geo": self.geo,
            "source": self.source.value,
            "status": self.status.value,
            "processing_time_ms": self.processing_time_ms,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }

    def is_completed(self) -> bool:
        """Check if analysis is completed"""
        return self.status == TrendAnalysisStatus.COMPLETED

    def is_failed(self) -> bool:
        """Check if analysis failed"""
        return self.status == TrendAnalysisStatus.FAILED

    def is_pending(self) -> bool:
        """Check if analysis is pending"""
        return self.status == TrendAnalysisStatus.PENDING

    def is_in_progress(self) -> bool:
        """Check if analysis is in progress"""
        return self.status == TrendAnalysisStatus.IN_PROGRESS

    def get_trend_count(self) -> int:
        """Get number of trends analyzed"""
        if self.trend_data and "trends" in self.trend_data:
            return len(self.trend_data["trends"])
        return 0

    def get_insights_summary(self) -> Dict[str, Any]:
        """Get a summary of insights"""
        if not self.insights:
            return {}
        
        return {
            "total_insights": len(self.insights.get("insights", [])),
            "top_trending": self.insights.get("top_trending", []),
            "growth_opportunities": self.insights.get("growth_opportunities", []),
            "seasonal_patterns": self.insights.get("seasonal_patterns", []),
            "competition_level": self.insights.get("competition_level", "unknown")
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the analysis"""
        return {
            "processing_time_ms": self.processing_time_ms,
            "api_calls_made": self.api_calls_made,
            "cache_hit": self.cache_hit,
            "trends_analyzed": self.get_trend_count(),
            "keywords_analyzed": len(self.keywords) if self.keywords else 0,
            "success_rate": 1.0 if self.is_completed() else 0.0
        }