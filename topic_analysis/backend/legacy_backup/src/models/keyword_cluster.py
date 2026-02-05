"""
KeywordCluster model for storing keyword clustering results
"""

import uuid
from sqlalchemy import Column, String, DateTime, JSONB, ForeignKey, Integer, Float, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..core.database import Base

class KeywordCluster(Base):
    """
    KeywordCluster model for storing keyword clustering results from external tools
    """
    __tablename__ = "keyword_clusters"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    workflow_session_id = Column(UUID(as_uuid=True), ForeignKey("workflow_sessions.id", ondelete="CASCADE"), nullable=False)
    trend_analysis_id = Column(UUID(as_uuid=True), ForeignKey("trend_analyses.id", ondelete="SET NULL"), nullable=True)
    
    # Cluster metadata
    cluster_name = Column(String(255), nullable=False)
    cluster_description = Column(String(1000), nullable=True)
    cluster_type = Column(String(50), nullable=False, default="semantic")  # semantic, topic, intent, difficulty
    
    # Keywords data
    keywords = Column(JSONB, nullable=False, default=[])  # Array of keyword objects
    primary_keyword = Column(String(255), nullable=False)
    secondary_keywords = Column(JSONB, nullable=False, default=[])  # Array of secondary keywords
    long_tail_keywords = Column(JSONB, nullable=False, default=[])  # Array of long-tail keywords
    
    # SEO metrics
    avg_search_volume = Column(Integer, nullable=True)
    avg_keyword_difficulty = Column(Float, nullable=True)
    avg_cpc = Column(Float, nullable=True)
    total_search_volume = Column(Integer, nullable=True)
    competition_level = Column(String(50), nullable=True)  # low, medium, high
    
    # Clustering metrics
    cluster_size = Column(Integer, nullable=False, default=0)
    cluster_density = Column(Float, nullable=True)  # How tightly clustered the keywords are
    semantic_similarity = Column(Float, nullable=True)  # Average semantic similarity
    intent_consistency = Column(Float, nullable=True)  # How consistent the search intent is
    
    # Content generation
    content_ideas = Column(JSONB, nullable=False, default=[])  # Array of content idea suggestions
    content_angles = Column(JSONB, nullable=False, default=[])  # Array of content angles
    target_audiences = Column(JSONB, nullable=False, default=[])  # Array of target audiences
    
    # External tool data
    source_tool = Column(String(50), nullable=True)  # semrush, ahrefs, ubersuggest, manual
    external_data = Column(JSONB, nullable=False, default={})  # Raw data from external tools
    processing_notes = Column(String(1000), nullable=True)
    
    # Quality metrics
    cluster_quality_score = Column(Float, nullable=True)  # 0-100 scale
    keyword_relevance_score = Column(Float, nullable=True)  # 0-100 scale
    content_potential_score = Column(Float, nullable=True)  # 0-100 scale
    
    # Status and flags
    is_active = Column(Boolean, nullable=False, default=True)
    is_processed = Column(Boolean, nullable=False, default=False)
    is_used_for_content = Column(Boolean, nullable=False, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="keyword_clusters")
    workflow_session = relationship("WorkflowSessions", back_populates="keyword_clusters")
    trend_analysis = relationship("TrendAnalysis", back_populates="keyword_clusters")

    def __repr__(self):
        return f"<KeywordCluster(id={self.id}, name='{self.cluster_name}', size={self.cluster_size})>"

    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "workflow_session_id": str(self.workflow_session_id),
            "trend_analysis_id": str(self.trend_analysis_id) if self.trend_analysis_id else None,
            "cluster_name": self.cluster_name,
            "cluster_description": self.cluster_description,
            "cluster_type": self.cluster_type,
            "keywords": self.keywords,
            "primary_keyword": self.primary_keyword,
            "secondary_keywords": self.secondary_keywords,
            "long_tail_keywords": self.long_tail_keywords,
            "avg_search_volume": self.avg_search_volume,
            "avg_keyword_difficulty": self.avg_keyword_difficulty,
            "avg_cpc": self.avg_cpc,
            "total_search_volume": self.total_search_volume,
            "competition_level": self.competition_level,
            "cluster_size": self.cluster_size,
            "cluster_density": self.cluster_density,
            "semantic_similarity": self.semantic_similarity,
            "intent_consistency": self.intent_consistency,
            "content_ideas": self.content_ideas,
            "content_angles": self.content_angles,
            "target_audiences": self.target_audiences,
            "source_tool": self.source_tool,
            "external_data": self.external_data,
            "processing_notes": self.processing_notes,
            "cluster_quality_score": self.cluster_quality_score,
            "keyword_relevance_score": self.keyword_relevance_score,
            "content_potential_score": self.content_potential_score,
            "is_active": self.is_active,
            "is_processed": self.is_processed,
            "is_used_for_content": self.is_used_for_content,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def to_summary_dict(self):
        """Return a summary version for list views"""
        return {
            "id": str(self.id),
            "cluster_name": self.cluster_name,
            "cluster_type": self.cluster_type,
            "cluster_size": self.cluster_size,
            "primary_keyword": self.primary_keyword,
            "avg_search_volume": self.avg_search_volume,
            "avg_keyword_difficulty": self.avg_keyword_difficulty,
            "competition_level": self.competition_level,
            "cluster_quality_score": self.cluster_quality_score,
            "is_active": self.is_active,
            "is_processed": self.is_processed,
            "is_used_for_content": self.is_used_for_content,
            "created_at": self.created_at.isoformat()
        }

    def get_keyword_count(self) -> int:
        """Get total number of keywords in cluster"""
        count = 0
        if self.keywords:
            count += len(self.keywords)
        if self.secondary_keywords:
            count += len(self.secondary_keywords)
        if self.long_tail_keywords:
            count += len(self.long_tail_keywords)
        return count

    def get_high_volume_keywords(self, threshold: int = 1000) -> list:
        """Get keywords with search volume above threshold"""
        high_volume = []
        if self.keywords:
            for keyword in self.keywords:
                if isinstance(keyword, dict) and keyword.get("search_volume", 0) >= threshold:
                    high_volume.append(keyword)
        return high_volume

    def get_low_difficulty_keywords(self, threshold: float = 30.0) -> list:
        """Get keywords with difficulty below threshold"""
        low_difficulty = []
        if self.keywords:
            for keyword in self.keywords:
                if isinstance(keyword, dict) and keyword.get("difficulty", 100) <= threshold:
                    low_difficulty.append(keyword)
        return low_difficulty

    def get_content_potential_keywords(self) -> list:
        """Get keywords with high content potential (low difficulty, high volume)"""
        potential_keywords = []
        if self.keywords:
            for keyword in self.keywords:
                if isinstance(keyword, dict):
                    volume = keyword.get("search_volume", 0)
                    difficulty = keyword.get("difficulty", 100)
                    if volume >= 500 and difficulty <= 50:
                        potential_keywords.append(keyword)
        return potential_keywords

    def calculate_cluster_metrics(self) -> dict:
        """Calculate cluster quality metrics"""
        if not self.keywords:
            return {
                "cluster_density": 0.0,
                "semantic_similarity": 0.0,
                "intent_consistency": 0.0,
                "cluster_quality_score": 0.0
            }
        
        # Calculate average metrics
        volumes = [k.get("search_volume", 0) for k in self.keywords if isinstance(k, dict)]
        difficulties = [k.get("difficulty", 0) for k in self.keywords if isinstance(k, dict)]
        cpcs = [k.get("cpc", 0) for k in self.keywords if isinstance(k, dict)]
        
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 0
        avg_cpc = sum(cpcs) / len(cpcs) if cpcs else 0
        
        # Calculate cluster density (simplified)
        cluster_density = min(1.0, len(self.keywords) / 10.0)  # Normalize to 0-1
        
        # Calculate semantic similarity (placeholder - would need actual NLP)
        semantic_similarity = 0.8  # Placeholder value
        
        # Calculate intent consistency (placeholder)
        intent_consistency = 0.7  # Placeholder value
        
        # Calculate overall quality score
        quality_factors = [
            cluster_density * 0.3,
            semantic_similarity * 0.3,
            intent_consistency * 0.2,
            (1.0 - min(avg_difficulty / 100.0, 1.0)) * 0.2  # Lower difficulty is better
        ]
        cluster_quality_score = sum(quality_factors) * 100
        
        return {
            "cluster_density": cluster_density,
            "semantic_similarity": semantic_similarity,
            "intent_consistency": intent_consistency,
            "cluster_quality_score": cluster_quality_score,
            "avg_search_volume": int(avg_volume),
            "avg_keyword_difficulty": round(avg_difficulty, 2),
            "avg_cpc": round(avg_cpc, 2)
        }

    def get_competition_level(self) -> str:
        """Determine competition level based on average difficulty"""
        if not self.avg_keyword_difficulty:
            return "unknown"
        
        if self.avg_keyword_difficulty <= 30:
            return "low"
        elif self.avg_keyword_difficulty <= 60:
            return "medium"
        else:
            return "high"

    def get_content_ideas_count(self) -> int:
        """Get number of content ideas generated"""
        return len(self.content_ideas) if self.content_ideas else 0

    def get_content_angles_count(self) -> int:
        """Get number of content angles identified"""
        return len(self.content_angles) if self.content_angles else 0

    def get_target_audiences_count(self) -> int:
        """Get number of target audiences identified"""
        return len(self.target_audiences) if self.target_audiences else 0

    def is_high_quality(self) -> bool:
        """Check if cluster is high quality"""
        return (self.cluster_quality_score or 0) >= 70

    def is_ready_for_content(self) -> bool:
        """Check if cluster is ready for content generation"""
        return (
            self.is_processed and
            self.is_active and
            self.cluster_size > 0 and
            (self.cluster_quality_score or 0) >= 50
        )

    def get_cluster_summary(self) -> dict:
        """Get comprehensive cluster summary"""
        return {
            "id": str(self.id),
            "name": self.cluster_name,
            "type": self.cluster_type,
            "size": self.cluster_size,
            "keyword_count": self.get_keyword_count(),
            "primary_keyword": self.primary_keyword,
            "avg_search_volume": self.avg_search_volume,
            "avg_difficulty": self.avg_keyword_difficulty,
            "competition_level": self.get_competition_level(),
            "quality_score": self.cluster_quality_score,
            "content_ideas_count": self.get_content_ideas_count(),
            "content_angles_count": self.get_content_angles_count(),
            "target_audiences_count": self.get_target_audiences_count(),
            "is_high_quality": self.is_high_quality(),
            "is_ready_for_content": self.is_ready_for_content(),
            "high_volume_keywords": len(self.get_high_volume_keywords()),
            "low_difficulty_keywords": len(self.get_low_difficulty_keywords()),
            "content_potential_keywords": len(self.get_content_potential_keywords())
        }
