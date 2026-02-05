"""
ExternalToolResult model for storing results from external keyword research tools
"""

import uuid
from sqlalchemy import Column, String, DateTime, JSONB, ForeignKey, Integer, Float, Boolean, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..core.database import Base

class ExternalToolResult(Base):
    """
    ExternalToolResult model for storing results from external keyword research tools
    """
    __tablename__ = "external_tool_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    workflow_session_id = Column(UUID(as_uuid=True), ForeignKey("workflow_sessions.id", ondelete="CASCADE"), nullable=False)
    trend_analysis_id = Column(UUID(as_uuid=True), ForeignKey("trend_analyses.id", ondelete="SET NULL"), nullable=True)
    
    # Tool metadata
    tool_name = Column(String(50), nullable=False)  # semrush, ahrefs, ubersuggest, manual
    tool_version = Column(String(20), nullable=True)
    api_key_id = Column(UUID(as_uuid=True), ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True)
    
    # Query information
    query_type = Column(String(50), nullable=False)  # keyword_research, competitor_analysis, content_ideas, etc.
    query_parameters = Column(JSONB, nullable=False, default={})  # Original query parameters
    seed_keywords = Column(JSONB, nullable=False, default=[])  # Array of seed keywords used
    
    # Results data
    raw_results = Column(JSONB, nullable=False, default={})  # Raw API response
    processed_results = Column(JSONB, nullable=False, default={})  # Processed and cleaned data
    keywords_data = Column(JSONB, nullable=False, default=[])  # Array of keyword objects
    clusters_data = Column(JSONB, nullable=False, default=[])  # Array of keyword clusters
    
    # Statistics
    total_keywords = Column(Integer, nullable=False, default=0)
    total_clusters = Column(Integer, nullable=False, default=0)
    avg_search_volume = Column(Float, nullable=True)
    avg_keyword_difficulty = Column(Float, nullable=True)
    avg_cpc = Column(Float, nullable=True)
    total_search_volume = Column(Integer, nullable=True)
    
    # Quality metrics
    data_quality_score = Column(Float, nullable=True)  # 0-100 scale
    completeness_score = Column(Float, nullable=True)  # 0-100 scale
    relevance_score = Column(Float, nullable=True)  # 0-100 scale
    
    # Processing metadata
    processing_time_ms = Column(Integer, nullable=True)
    api_calls_made = Column(Integer, nullable=False, default=0)
    rate_limit_hit = Column(Boolean, nullable=False, default=False)
    error_count = Column(Integer, nullable=False, default=0)
    warning_count = Column(Integer, nullable=False, default=0)
    
    # Status and flags
    status = Column(String(50), nullable=False, default="pending")  # pending, processing, completed, failed
    is_processed = Column(Boolean, nullable=False, default=False)
    is_used_for_content = Column(Boolean, nullable=False, default=False)
    is_archived = Column(Boolean, nullable=False, default=False)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    error_details = Column(JSONB, nullable=False, default={})
    retry_count = Column(Integer, nullable=False, default=0)
    last_retry_at = Column(DateTime(timezone=True), nullable=True)
    
    # Notes and metadata
    processing_notes = Column(Text, nullable=True)
    user_notes = Column(Text, nullable=True)
    tags = Column(JSONB, nullable=False, default=[])  # Array of tags for organization
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    processed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="external_tool_results")
    workflow_session = relationship("WorkflowSessions", back_populates="external_tool_results")
    trend_analysis = relationship("TrendAnalysis", back_populates="external_tool_results")
    api_key = relationship("APIKey", back_populates="external_tool_results")

    def __repr__(self):
        return f"<ExternalToolResult(id={self.id}, tool='{self.tool_name}', status='{self.status}')>"

    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "workflow_session_id": str(self.workflow_session_id),
            "trend_analysis_id": str(self.trend_analysis_id) if self.trend_analysis_id else None,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "api_key_id": str(self.api_key_id) if self.api_key_id else None,
            "query_type": self.query_type,
            "query_parameters": self.query_parameters,
            "seed_keywords": self.seed_keywords,
            "raw_results": self.raw_results,
            "processed_results": self.processed_results,
            "keywords_data": self.keywords_data,
            "clusters_data": self.clusters_data,
            "total_keywords": self.total_keywords,
            "total_clusters": self.total_clusters,
            "avg_search_volume": self.avg_search_volume,
            "avg_keyword_difficulty": self.avg_keyword_difficulty,
            "avg_cpc": self.avg_cpc,
            "total_search_volume": self.total_search_volume,
            "data_quality_score": self.data_quality_score,
            "completeness_score": self.completeness_score,
            "relevance_score": self.relevance_score,
            "processing_time_ms": self.processing_time_ms,
            "api_calls_made": self.api_calls_made,
            "rate_limit_hit": self.rate_limit_hit,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "status": self.status,
            "is_processed": self.is_processed,
            "is_used_for_content": self.is_used_for_content,
            "is_archived": self.is_archived,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "retry_count": self.retry_count,
            "last_retry_at": self.last_retry_at.isoformat() if self.last_retry_at else None,
            "processing_notes": self.processing_notes,
            "user_notes": self.user_notes,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None
        }

    def to_summary_dict(self):
        """Return a summary version for list views"""
        return {
            "id": str(self.id),
            "tool_name": self.tool_name,
            "query_type": self.query_type,
            "total_keywords": self.total_keywords,
            "total_clusters": self.total_clusters,
            "avg_search_volume": self.avg_search_volume,
            "avg_keyword_difficulty": self.avg_keyword_difficulty,
            "data_quality_score": self.data_quality_score,
            "status": self.status,
            "is_processed": self.is_processed,
            "is_used_for_content": self.is_used_for_content,
            "processing_time_ms": self.processing_time_ms,
            "error_count": self.error_count,
            "created_at": self.created_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None
        }

    def is_completed(self) -> bool:
        """Check if processing is completed"""
        return self.status == "completed"

    def is_failed(self) -> bool:
        """Check if processing failed"""
        return self.status == "failed"

    def is_processing(self) -> bool:
        """Check if currently processing"""
        return self.status == "processing"

    def is_pending(self) -> bool:
        """Check if pending processing"""
        return self.status == "pending"

    def has_errors(self) -> bool:
        """Check if there are any errors"""
        return self.error_count > 0 or self.status == "failed"

    def has_warnings(self) -> bool:
        """Check if there are any warnings"""
        return self.warning_count > 0

    def get_success_rate(self) -> float:
        """Calculate success rate based on errors and warnings"""
        if self.api_calls_made == 0:
            return 0.0
        
        success_calls = self.api_calls_made - self.error_count
        return (success_calls / self.api_calls_made) * 100

    def get_quality_metrics(self) -> dict:
        """Get comprehensive quality metrics"""
        return {
            "data_quality_score": self.data_quality_score or 0,
            "completeness_score": self.completeness_score or 0,
            "relevance_score": self.relevance_score or 0,
            "overall_quality": self.get_overall_quality_score(),
            "success_rate": self.get_success_rate(),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "rate_limit_hit": self.rate_limit_hit
        }

    def get_overall_quality_score(self) -> float:
        """Calculate overall quality score"""
        scores = []
        if self.data_quality_score is not None:
            scores.append(self.data_quality_score)
        if self.completeness_score is not None:
            scores.append(self.completeness_score)
        if self.relevance_score is not None:
            scores.append(self.relevance_score)
        
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)

    def get_keyword_statistics(self) -> dict:
        """Get keyword-related statistics"""
        return {
            "total_keywords": self.total_keywords,
            "total_clusters": self.total_clusters,
            "avg_search_volume": self.avg_search_volume,
            "avg_keyword_difficulty": self.avg_keyword_difficulty,
            "avg_cpc": self.avg_cpc,
            "total_search_volume": self.total_search_volume,
            "keywords_per_cluster": self.total_keywords / self.total_clusters if self.total_clusters > 0 else 0
        }

    def get_processing_metrics(self) -> dict:
        """Get processing-related metrics"""
        return {
            "processing_time_ms": self.processing_time_ms,
            "api_calls_made": self.api_calls_made,
            "processing_speed": self.total_keywords / (self.processing_time_ms / 1000) if self.processing_time_ms and self.processing_time_ms > 0 else 0,
            "rate_limit_hit": self.rate_limit_hit,
            "retry_count": self.retry_count,
            "success_rate": self.get_success_rate()
        }

    def get_high_quality_keywords(self, min_volume: int = 1000, max_difficulty: float = 50.0) -> list:
        """Get high-quality keywords based on volume and difficulty"""
        if not self.keywords_data:
            return []
        
        high_quality = []
        for keyword in self.keywords_data:
            if isinstance(keyword, dict):
                volume = keyword.get("search_volume", 0)
                difficulty = keyword.get("difficulty", 100)
                if volume >= min_volume and difficulty <= max_difficulty:
                    high_quality.append(keyword)
        
        return high_quality

    def get_cluster_summary(self) -> dict:
        """Get summary of clusters"""
        if not self.clusters_data:
            return {
                "total_clusters": 0,
                "avg_cluster_size": 0,
                "largest_cluster_size": 0,
                "smallest_cluster_size": 0
            }
        
        cluster_sizes = [cluster.get("size", 0) for cluster in self.clusters_data if isinstance(cluster, dict)]
        
        return {
            "total_clusters": len(self.clusters_data),
            "avg_cluster_size": sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0,
            "largest_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "smallest_cluster_size": min(cluster_sizes) if cluster_sizes else 0
        }

    def can_retry(self) -> bool:
        """Check if the result can be retried"""
        return (
            self.status == "failed" and
            self.retry_count < 3 and
            not self.is_archived
        )

    def should_archive(self) -> bool:
        """Check if the result should be archived"""
        return (
            self.is_archived or
            (self.status == "failed" and self.retry_count >= 3) or
            (self.is_processed and self.is_used_for_content and self.created_at < func.now() - func.interval('30 days'))
        )

    def get_export_data(self) -> dict:
        """Get data formatted for export"""
        return {
            "tool_name": self.tool_name,
            "query_type": self.query_type,
            "seed_keywords": self.seed_keywords,
            "keywords": self.keywords_data,
            "clusters": self.clusters_data,
            "statistics": self.get_keyword_statistics(),
            "quality_metrics": self.get_quality_metrics(),
            "processing_metrics": self.get_processing_metrics(),
            "exported_at": func.now().isoformat()
        }
