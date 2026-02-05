"""
WorkflowSessions model for TrendTap Enhanced Research Workflow
"""

from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, JSON, Enum, CheckConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from ..core.database import Base


class WorkflowStep(PyEnum):
    """Workflow step enumeration"""
    UPLOAD_CSV = "upload_csv"
    SELECT_TRENDS = "select_trends"
    GENERATE_KEYWORDS = "generate_keywords"
    EXPORT_KEYWORDS = "export_keywords"
    UPLOAD_EXTERNAL = "upload_external"
    ANALYZE_RESULTS = "analyze_results"


class WorkflowStatus(PyEnum):
    """Workflow status enumeration"""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class WorkflowSessions(Base):
    """Workflow sessions model for enhanced research workflow"""
    __tablename__ = "workflow_sessions"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    session_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    current_step = Column(Enum(WorkflowStep), default=WorkflowStep.UPLOAD_CSV, nullable=False)
    progress_percentage = Column(Integer, default=0, nullable=False)
    workflow_data = Column(JSON, default={}, nullable=False)
    completed_steps = Column(JSON, default=[], nullable=False)
    error_message = Column(Text, nullable=True)
    status = Column(Enum(WorkflowStatus), default=WorkflowStatus.ACTIVE, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="workflow_sessions")
    trend_analyses = relationship("TrendAnalysis", back_populates="workflow_session", cascade="all, delete-orphan")
    content_ideas = relationship("ContentIdea", back_populates="workflow_session", cascade="all, delete-orphan")
    keyword_clusters = relationship("KeywordCluster", back_populates="workflow_session", cascade="all, delete-orphan")
    external_tool_results = relationship("ExternalToolResult", back_populates="workflow_session", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('progress_percentage >= 0 AND progress_percentage <= 100', name='check_progress_percentage'),
    )
    
    def __repr__(self):
        return f"<WorkflowSession(id={self.id}, name={self.session_name}, status={self.status})>"
