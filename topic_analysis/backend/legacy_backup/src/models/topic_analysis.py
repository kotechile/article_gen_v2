from sqlalchemy import Column, String, Text, Float, Integer, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from ..core.database import Base

class TopicAnalysis(Base):
    """Model for storing topic analysis results"""
    __tablename__ = "topic_analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic = Column(String(255), nullable=False, index=True)
    related_areas = Column(JSON, nullable=False)  # List of related areas
    affiliate_programs = Column(JSON, nullable=False)  # List of affiliate programs
    analysis_metadata = Column(JSON, nullable=True)  # LLM metadata, tokens used, etc.
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Usage tracking
    search_count = Column(Integer, default=0)
    last_searched = Column(DateTime, nullable=True)
    
    # Quality metrics
    relevance_score = Column(Float, default=0.0)
    user_rating = Column(Float, nullable=True)  # User feedback on quality
    
    def __repr__(self):
        return f"<TopicAnalysis(topic='{self.topic}', created_at='{self.created_at}')>"

class RelatedArea(Base):
    """Model for storing related areas (normalized)"""
    __tablename__ = "related_areas"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic_analysis_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    area_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    relevance_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

class TopicAffiliateProgram(Base):
    """Model for storing affiliate programs (normalized)"""
    __tablename__ = "topic_affiliate_programs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic_analysis_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    program_name = Column(String(255), nullable=False)
    commission_rate = Column(String(50), nullable=False)
    category = Column(String(100), nullable=False)
    difficulty_level = Column(String(20), nullable=False)
    description = Column(Text, nullable=True)
    estimated_traffic = Column(Integer, nullable=True)
    competition_level = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class TopicSearchLog(Base):
    """Model for logging topic searches for analytics"""
    __tablename__ = "topic_search_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic = Column(String(255), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    search_timestamp = Column(DateTime, default=datetime.utcnow)
    analysis_source = Column(String(50), nullable=False)  # 'database', 'llm', 'cache'
    response_time_ms = Column(Integer, nullable=True)
    success = Column(String(10), default='true')  # 'true', 'false'
    error_message = Column(Text, nullable=True)

