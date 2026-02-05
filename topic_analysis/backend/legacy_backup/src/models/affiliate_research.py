"""
AffiliateResearch model for TrendTap
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from ..core.database import Base


class ResearchStatus(PyEnum):
    """Research status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AffiliateResearch(Base):
    """Affiliate research model"""
    __tablename__ = "affiliate_researches"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    topic = Column(String(255), nullable=False)
    search_query = Column(String(500), nullable=True)
    status = Column(Enum(ResearchStatus), default=ResearchStatus.PENDING, nullable=False)
    
    # Research results
    results = Column(JSON, nullable=True)  # Array of affiliate program data
    selected_programs = Column(JSON, nullable=True)  # Array of selected programs
    total_programs_found = Column(Integer, default=0, nullable=False)
    total_networks_searched = Column(Integer, default=0, nullable=False)
    
    # Research metadata
    research_duration = Column(Integer, nullable=True)  # Duration in seconds
    error_message = Column(Text, nullable=True)
    warnings = Column(JSON, nullable=True)  # Array of warning messages
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="affiliate_researches")
    trend_analyses = relationship("TrendAnalysis", back_populates="affiliate_research")
    
    def __repr__(self):
        return f"<AffiliateResearch(id={self.id}, topic='{self.topic}', status='{self.status.value}')>"
    
    def to_dict(self):
        """Convert affiliate research to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "topic": self.topic,
            "search_query": self.search_query,
            "status": self.status.value,
            "results": self.results,
            "selected_programs": self.selected_programs,
            "total_programs_found": self.total_programs_found,
            "total_networks_searched": self.total_networks_searched,
            "research_duration": self.research_duration,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
    
    def is_completed(self) -> bool:
        """Check if research is completed"""
        return self.status == ResearchStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if research failed"""
        return self.status == ResearchStatus.FAILED
    
    def is_in_progress(self) -> bool:
        """Check if research is in progress"""
        return self.status == ResearchStatus.IN_PROGRESS
    
    def get_top_programs(self, limit: int = 10) -> list:
        """Get top affiliate programs by EPC"""
        if not self.results:
            return []
        
        # Sort by EPC (highest first)
        sorted_programs = sorted(
            self.results,
            key=lambda x: x.get("epc", 0),
            reverse=True
        )
        
        return sorted_programs[:limit]
    
    def get_programs_by_network(self, network: str) -> list:
        """Get programs from specific network"""
        if not self.results:
            return []
        
        return [program for program in self.results if program.get("network") == network]
    
    def get_selected_programs_count(self) -> int:
        """Get count of selected programs"""
        if not self.selected_programs:
            return 0
        
        return len([p for p in self.selected_programs if p.get("selected", False)])
    
    def get_average_epc(self) -> float:
        """Get average EPC of all programs"""
        if not self.results:
            return 0.0
        
        epcs = [program.get("epc", 0) for program in self.results if program.get("epc")]
        return sum(epcs) / len(epcs) if epcs else 0.0
    
    def get_network_coverage(self) -> dict:
        """Get coverage by network"""
        if not self.results:
            return {}
        
        network_counts = {}
        for program in self.results:
            network = program.get("network", "Unknown")
            network_counts[network] = network_counts.get(network, 0) + 1
        
        return network_counts
    
    def add_warning(self, warning: str):
        """Add warning message"""
        if not self.warnings:
            self.warnings = []
        
        if warning not in self.warnings:
            self.warnings.append(warning)
    
    def mark_completed(self, results: list = None):
        """Mark research as completed"""
        self.status = ResearchStatus.COMPLETED
        self.completed_at = func.now()
        
        if results:
            self.results = results
            self.total_programs_found = len(results)
    
    def mark_failed(self, error_message: str):
        """Mark research as failed"""
        self.status = ResearchStatus.FAILED
        self.error_message = error_message
        self.completed_at = func.now()
    
    def update_selected_programs(self, selected_programs: list):
        """Update selected programs"""
        self.selected_programs = selected_programs
