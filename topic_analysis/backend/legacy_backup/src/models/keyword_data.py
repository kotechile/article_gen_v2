"""
KeywordData model for TrendTap
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Enum, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from ..core.database import Base


class KeywordSource(PyEnum):
    """Keyword source enumeration"""
    CSV_UPLOAD = "csv_upload"
    DATAFORSEO = "dataforseo"
    AHREFS = "ahrefs"
    SEMRUSH = "semrush"
    UBERSUGGEST = "ubersuggest"
    MANUAL = "manual"


class KeywordStatus(PyEnum):
    """Keyword processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class KeywordData(Base):
    """Keyword data model"""
    __tablename__ = "keyword_data"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    trend_analysis_id = Column(Integer, ForeignKey("trend_analyses.id"), nullable=True, index=True)
    status = Column(Enum(KeywordStatus), default=KeywordStatus.PENDING, nullable=False)
    source = Column(Enum(KeywordSource), nullable=False)
    
    # Keyword data
    keywords = Column(JSON, nullable=True)  # Array of keyword objects
    keyword_count = Column(Integer, default=0, nullable=False)
    
    # Processing metadata
    processing_duration = Column(Integer, nullable=True)  # Duration in seconds
    error_message = Column(Text, nullable=True)
    warnings = Column(JSON, nullable=True)  # Array of warning messages
    
    # Cost tracking (for DataForSEO)
    estimated_cost = Column(Float, nullable=True)
    actual_cost = Column(Float, nullable=True)
    
    # File information (for CSV uploads)
    original_filename = Column(String(255), nullable=True)
    file_size = Column(Integer, nullable=True)
    
    # External tool integration
    external_tool_source = Column(Enum(KeywordSource), nullable=True)
    external_tool_metrics = Column(JSON, default={}, nullable=True)
    cluster_id = Column(String, ForeignKey("keyword_clusters.id"), nullable=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="keyword_data")
    trend_analysis = relationship("TrendAnalysis", back_populates="keyword_data")
    content_ideas = relationship("ContentIdeas", back_populates="keyword_data")
    software_solutions = relationship("SoftwareSolutions", back_populates="keyword_data")
    keyword_clusters = relationship("KeywordClusters", back_populates="keyword_data")
    
    def __repr__(self):
        return f"<KeywordData(id={self.id}, keyword_count={self.keyword_count}, status='{self.status.value}')>"
    
    def to_dict(self):
        """Convert keyword data to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "trend_analysis_id": self.trend_analysis_id,
            "status": self.status.value,
            "source": self.source.value,
            "keywords": self.keywords,
            "keyword_count": self.keyword_count,
            "processing_duration": self.processing_duration,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "estimated_cost": self.estimated_cost,
            "actual_cost": self.actual_cost,
            "original_filename": self.original_filename,
            "file_size": self.file_size,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
    
    def is_completed(self) -> bool:
        """Check if keyword processing is completed"""
        return self.status == KeywordStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if keyword processing failed"""
        return self.status == KeywordStatus.FAILED
    
    def is_processing(self) -> bool:
        """Check if keyword processing is in progress"""
        return self.status == KeywordStatus.PROCESSING
    
    def get_top_keywords(self, limit: int = 10) -> list:
        """Get top keywords by priority score"""
        if not self.keywords:
            return []
        
        # Sort by priority score (highest first)
        sorted_keywords = sorted(
            self.keywords,
            key=lambda x: x.get("priority_score", 0),
            reverse=True
        )
        
        return sorted_keywords[:limit]
    
    def get_keywords_by_intent(self, intent: str) -> list:
        """Get keywords by search intent"""
        if not self.keywords:
            return []
        
        return [
            keyword for keyword in self.keywords
            if keyword.get("intent") == intent
        ]
    
    def get_keywords_by_difficulty(self, min_difficulty: int = 0, max_difficulty: int = 100) -> list:
        """Get keywords by difficulty range"""
        if not self.keywords:
            return []
        
        return [
            keyword for keyword in self.keywords
            if min_difficulty <= keyword.get("difficulty", 0) <= max_difficulty
        ]
    
    def get_high_priority_keywords(self, threshold: float = 0.7) -> list:
        """Get keywords with priority score above threshold"""
        if not self.keywords:
            return []
        
        return [
            keyword for keyword in self.keywords
            if keyword.get("priority_score", 0) >= threshold
        ]
    
    def get_keyword_by_term(self, term: str) -> dict:
        """Get specific keyword by term"""
        if not self.keywords:
            return {}
        
        for keyword in self.keywords:
            if keyword.get("keyword", "").lower() == term.lower():
                return keyword
        
        return {}
    
    def get_serp_analysis_summary(self) -> dict:
        """Get SERP analysis summary"""
        if not self.keywords:
            return {}
        
        total_keywords = len(self.keywords)
        avg_domain_authority = 0
        avg_page_authority = 0
        avg_content_length = 0
        avg_backlinks = 0
        avg_serp_weakness = 0
        
        count = 0
        for keyword in self.keywords:
            serp_analysis = keyword.get("serp_analysis", {})
            if serp_analysis:
                avg_domain_authority += serp_analysis.get("top_10_avg_domain_authority", 0)
                avg_page_authority += serp_analysis.get("top_10_avg_page_authority", 0)
                avg_content_length += serp_analysis.get("top_10_avg_content_length", 0)
                avg_backlinks += serp_analysis.get("top_10_avg_backlinks", 0)
                avg_serp_weakness += serp_analysis.get("serp_weakness_score", 0)
                count += 1
        
        if count > 0:
            avg_domain_authority /= count
            avg_page_authority /= count
            avg_content_length /= count
            avg_backlinks /= count
            avg_serp_weakness /= count
        
        return {
            "total_keywords": total_keywords,
            "avg_domain_authority": round(avg_domain_authority, 2),
            "avg_page_authority": round(avg_page_authority, 2),
            "avg_content_length": round(avg_content_length, 0),
            "avg_backlinks": round(avg_backlinks, 0),
            "avg_serp_weakness": round(avg_serp_weakness, 2)
        }
    
    def get_intent_distribution(self) -> dict:
        """Get distribution of search intents"""
        if not self.keywords:
            return {}
        
        intent_counts = {}
        for keyword in self.keywords:
            intent = keyword.get("intent", "unknown")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        return intent_counts
    
    def get_difficulty_distribution(self) -> dict:
        """Get distribution of keyword difficulties"""
        if not self.keywords:
            return {}
        
        difficulty_ranges = {
            "easy": 0,      # 0-30
            "medium": 0,    # 31-70
            "hard": 0       # 71-100
        }
        
        for keyword in self.keywords:
            difficulty = keyword.get("difficulty", 0)
            if difficulty <= 30:
                difficulty_ranges["easy"] += 1
            elif difficulty <= 70:
                difficulty_ranges["medium"] += 1
            else:
                difficulty_ranges["hard"] += 1
        
        return difficulty_ranges
    
    def add_warning(self, warning: str):
        """Add warning message"""
        if not self.warnings:
            self.warnings = []
        
        if warning not in self.warnings:
            self.warnings.append(warning)
    
    def mark_completed(self, keywords: list = None):
        """Mark keyword processing as completed"""
        self.status = KeywordStatus.COMPLETED
        self.completed_at = func.now()
        
        if keywords:
            self.keywords = keywords
            self.keyword_count = len(keywords)
    
    def mark_failed(self, error_message: str):
        """Mark keyword processing as failed"""
        self.status = KeywordStatus.FAILED
        self.error_message = error_message
        self.completed_at = func.now()
    
    def calculate_priority_scores(self):
        """Calculate priority scores for all keywords"""
        if not self.keywords:
            return
        
        for keyword in self.keywords:
            # Priority score = (affiliate EPC × search intent × SERP weakness) / keyword difficulty
            epc = keyword.get("epc", 0)
            search_volume = keyword.get("search_volume", 0)
            difficulty = keyword.get("difficulty", 50)
            serp_weakness = keyword.get("serp_analysis", {}).get("serp_weakness_score", 0.5)
            
            # Normalize search volume (0-1 scale)
            volume_score = min(search_volume / 10000, 1.0)
            
            # Calculate priority score
            if difficulty > 0:
                priority_score = (epc * volume_score * serp_weakness) / (difficulty / 100)
            else:
                priority_score = 0
            
            # Normalize to 0-1 range
            keyword["priority_score"] = min(max(priority_score, 0), 1)
