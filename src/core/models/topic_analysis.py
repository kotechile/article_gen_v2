"""
Topic analysis data models.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class KeywordMetrics(BaseModel):
    """Metrics for a keyword from DataForSEO."""
    volume: int = 0
    cpc: float = 0.0
    difficulty: int = 0
    competition_level: str = "ZERO" # ZERO, LOW, MEDIUM, HIGH

class SubtopicCreate(BaseModel):
    """Model for creating a subtopic."""
    name: str
    description: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    trend_direction: str = "stable" # up, down, stable
    metrics: Optional[KeywordMetrics] = None

class Subtopic(SubtopicCreate):
    """Model representing a subtopic in response."""
    id: Optional[str] = None
    topic_id: str
    status: str = "proposed" # proposed, approved, rejected
    created_at: datetime = Field(default_factory=datetime.utcnow)
