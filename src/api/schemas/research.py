"""
Research API schemas.

This module contains Pydantic schemas for research API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

from ...core.models.research import ResearchRequest, ResearchResponse, ResearchStatus


class ResearchRequestSchema(BaseModel):
    """Schema for research request validation."""
    
    brief: str = Field(..., min_length=10, max_length=3000)
    keywords: str = Field(..., min_length=1, max_length=500)
    provider: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    api_key: str = Field(..., min_length=1)
    depth: Optional[str] = Field("standard", pattern="^(standard|comprehensive|deep)$")
    tone: Optional[str] = Field("journalistic", pattern="^(academic|journalistic|casual|technical|persuasive)$")
    target_word_count: Optional[int] = Field(2000, ge=500, le=10000)
    claims_research_enabled: Optional[bool] = Field(True)
    rag_enabled: Optional[bool] = Field(True)
    rag_collection: Optional[str] = Field(None)
    rag_endpoint: Optional[str] = Field(None)
    rag_llm_provider: Optional[str] = Field(None)


class ResearchResponseSchema(BaseModel):
    """Schema for research response validation."""
    
    research_id: str
    status: str
    brief: str
    model: str
    depth: str
    tone: str
    target_word_count: int
    created_at: str
    estimated_completion: Optional[str] = None
