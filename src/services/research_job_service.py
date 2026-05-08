"""
Job discovery service for the research rebuild.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from supabase_client import LLM_ROLE_RESEARCH_TOPIC_GENERATION

from .research_rebuild_base_service import ResearchRebuildBaseService
from .llm.llm_service import llm_service

logger = logging.getLogger(__name__)


class ResearchJobService(ResearchRebuildBaseService):
    """Persist and manage first-class user jobs."""

    table_name = "research_user_jobs"

    async def generate_jobs(
        self,
        *,
        context: Dict[str, Any],
        count: int = 30,
        negative_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate jobs from website/category context.

        Returns compact job records that can be persisted directly.
        """
        negative_notes = negative_context or {}
        prompt = f"""
You are generating high-signal user jobs for a content and software discovery workflow.

Return valid JSON with this shape:
{{
  "jobs": [
    {{
      "job_text": "string",
      "job_type_hint": "seo|editorial|software|hybrid",
      "generation_metadata": {{"why": "short reason"}}
    }}
  ]
}}

Rules:
- Generate exactly {max(1, min(count, 50))} jobs.
- Jobs must be specific user problems, not abstract topics.
- Favor concrete tasks, comparisons, decisions, workflows, calculators, templates, checklists, and recurring questions.
- Avoid duplicates and near-duplicates.
- Avoid jobs already rejected for being off-brand, too broad, or technically impossible.
- Keep each job_text short and practical.

Website context:
- Project name: {context.get("project_name") or ""}
- Website description: {context.get("website_description") or ""}
- Primary category: {context.get("primary_category_name") or ""}
- Primary category description: {context.get("primary_category_description") or ""}
- Secondary category: {context.get("secondary_category_name") or ""}
- Secondary category description: {context.get("secondary_category_description") or ""}
- Target audience: {context.get("target_audience") or ""}
- Trend titles: {", ".join(context.get("trend_titles") or [])}

Rejected patterns to avoid:
{negative_notes}
"""
        response = await llm_service.generate_json(
            prompt,
            task_role=LLM_ROLE_RESEARCH_TOPIC_GENERATION,
            max_tokens=2200,
        )
        jobs = response.get("jobs") if isinstance(response, dict) else []
        normalized: List[Dict[str, Any]] = []
        seen = set()
        for item in jobs or []:
            if not isinstance(item, dict):
                continue
            job_text = str(item.get("job_text") or "").strip()
            if not job_text:
                continue
            key = job_text.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                {
                    "job_text": job_text,
                    "job_type_hint": str(item.get("job_type_hint") or "hybrid").strip().lower(),
                    "job_source": "llm_generation",
                    "status": "draft",
                    "generation_metadata": item.get("generation_metadata") or {},
                }
            )
        logger.info("research rebuild generated jobs count=%s", len(normalized))
        return normalized[:count]

    async def save_jobs(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        primary_category_id: Optional[UUID],
        secondary_category_id: Optional[UUID],
        website_context_snapshot: Dict[str, Any],
        jobs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Persist generated jobs for later approval/rejection."""
        payloads: List[Dict[str, Any]] = []
        for job in jobs:
            payloads.append(
                {
                    "project_id": str(project_id),
                    "primary_category_id": str(primary_category_id) if primary_category_id else None,
                    "secondary_category_id": str(secondary_category_id) if secondary_category_id else None,
                    "job_text": str(job.get("job_text") or "").strip(),
                    "job_type_hint": job.get("job_type_hint"),
                    "job_source": job.get("job_source") or "llm_generation",
                    "status": job.get("status") or "draft",
                    "website_context_snapshot": website_context_snapshot,
                    "generation_metadata": job.get("generation_metadata") or {},
                }
            )
        return await self.bulk_create_records(user_id=user_id, data_list=payloads)

    async def create_job(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        job_text: str,
        primary_category_id: Optional[UUID] = None,
        secondary_category_id: Optional[UUID] = None,
        job_type_hint: Optional[str] = None,
        job_source: str = "manual",
        status: str = "draft",
        website_context_snapshot: Optional[Dict[str, Any]] = None,
        generation_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create a single user job."""
        return await self.create_record(
            user_id=user_id,
            data={
                "project_id": str(project_id),
                "primary_category_id": str(primary_category_id) if primary_category_id else None,
                "secondary_category_id": str(secondary_category_id) if secondary_category_id else None,
                "job_text": str(job_text or "").strip(),
                "job_type_hint": job_type_hint,
                "job_source": job_source,
                "status": status,
                "website_context_snapshot": website_context_snapshot or {},
                "generation_metadata": generation_metadata or {},
            },
        )

    async def list_jobs(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        primary_category_id: Optional[UUID] = None,
        secondary_category_id: Optional[UUID] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List jobs for a project/category scope."""
        filters: Dict[str, Any] = {"project_id": str(project_id)}
        if primary_category_id:
            filters["primary_category_id"] = str(primary_category_id)
        if secondary_category_id:
            filters["secondary_category_id"] = str(secondary_category_id)
        if status:
            filters["status"] = status
        return await self.list_records(
            user_id=user_id,
            filters=filters,
            order_by={"created_at": "desc"},
        )

    async def approve_job(self, *, job_id: UUID, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Move a job into the approved state."""
        return await self.update_record(record_id=job_id, user_id=user_id, data={"status": "approved"})

    async def reject_job(
        self,
        *,
        job_id: UUID,
        user_id: UUID,
        rejection_reason_tags: List[str],
        rejection_reason_free_text: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Reject a job and keep structured feedback for future generations."""
        return await self.update_record(
            record_id=job_id,
            user_id=user_id,
            data={
                "status": "rejected",
                "rejection_reason_tags": self.normalize_reason_tags(rejection_reason_tags),
                "rejection_reason_free_text": rejection_reason_free_text,
            },
        )

    async def build_negative_context(self, *, user_id: UUID, project_id: UUID) -> Dict[str, Any]:
        """
        Build a compact negative prompt context from rejected jobs.

        This keeps the future generation loop grounded in user feedback.
        """
        rejected_jobs = await self.list_records(
            user_id=user_id,
            filters={"project_id": str(project_id), "status": "rejected"},
            order_by={"updated_at": "desc"},
            limit=50,
        )
        return {
            "recent_rejected_jobs": [
                {
                    "job_text": item.get("job_text"),
                    "rejection_reason_tags": item.get("rejection_reason_tags") or [],
                    "rejection_reason_free_text": item.get("rejection_reason_free_text"),
                }
                for item in rejected_jobs
            ]
        }
