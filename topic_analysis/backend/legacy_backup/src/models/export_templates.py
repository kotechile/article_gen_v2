"""
Export Templates model for TrendTap
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..core.database import Base


class ExportTemplate(Base):
    """Export template model"""
    __tablename__ = "export_templates"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    export_type = Column(String(50), nullable=False)  # affiliate_research, trend_analysis, etc.
    format = Column(String(10), nullable=False)  # csv, xlsx, json, etc.
    template_config = Column(JSON, nullable=False)
    is_public = Column(Boolean, default=False, nullable=False)
    usage_count = Column(Integer, default=0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="export_templates")
    
    def __repr__(self):
        return f"<ExportTemplate(id={self.id}, name='{self.name}', export_type='{self.export_type}')>"


