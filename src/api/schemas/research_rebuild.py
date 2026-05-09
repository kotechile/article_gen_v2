"""
Schemas for the research rebuild API surface.

These schemas are intentionally small and stable so the backend can start
exposing the new workflow incrementally behind feature flags.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


RejectionReasonTag = Literal[
    "too_broad",
    "off_brand",
    "duplicate",
    "technically_impossible",
    "weak_serp",
    "low_business_value",
    "low_confidence",
]


class GenerateResearchJobsRequestSchema(BaseModel):
    project_id: str = Field(..., min_length=1)
    primary_category_id: Optional[str] = None
    secondary_category_id: Optional[str] = None
    count: int = Field(30, ge=1, le=50)
    website_description: Optional[str] = None
    target_audience: Optional[str] = None
    focus_area: Optional[str] = Field(None, max_length=500)
    avoid_guidance: Optional[str] = Field(None, max_length=500)


class RejectResearchItemRequestSchema(BaseModel):
    rejection_reason_tags: List[RejectionReasonTag] = Field(default_factory=list)
    rejection_reason_free_text: Optional[str] = Field(None, max_length=1000)


class GenerateCandidatesRequestSchema(BaseModel):
    project_id: str = Field(..., min_length=1)
    user_job_id: str = Field(..., min_length=1)


class ValidateCandidatesRequestSchema(BaseModel):
    project_id: str = Field(..., min_length=1)
    candidate_ids: List[str] = Field(..., min_length=1, max_length=40)
    force_refresh: bool = False


class ResearchRebuildListResponseSchema(BaseModel):
    items: list[dict] = Field(default_factory=list)
    count: int = 0


class ValidationSummaryResponseSchema(BaseModel):
    candidate_id: str
    freshness_state: str
    eligibility_passed: bool
    achievability_score: Optional[float] = None
    route: Optional[str] = None
