"""
Opportunity candidate service for the research rebuild.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from supabase_client import LLM_ROLE_RESEARCH_IDEA_GENERATION

from .research_rebuild_base_service import ResearchRebuildBaseService
from .llm.llm_service import llm_service

logger = logging.getLogger(__name__)


class ResearchCandidateService(ResearchRebuildBaseService):
    """Persist and manage opportunities derived from approved jobs."""

    table_name = "research_opportunity_candidates"

    async def derive_candidates_from_job(
        self,
        *,
        job: Dict[str, Any],
        website_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Derive article/software/editorial opportunities from an approved job.

        This uses the repo's existing LLM role configuration but returns a
        stricter candidate schema for the rebuild.
        """
        prompt = f"""
You are deriving concrete opportunity candidates from a user job.

Return valid JSON with this shape:
{{
  "candidates": [
    {{
      "candidate_type": "seo_article|software|editorial",
      "candidate_text": "string",
      "normalized_candidate_text": "lowercase normalized string",
      "candidate_metadata": {{
        "target_intent": "informational|commercial|transactional",
        "product_type": "optional string",
        "user_job_to_be_done": "optional string",
        "build_complexity": "low|medium|high",
        "distribution_angle": "optional string",
        "category_context": {{}}
      }},
      "source_keywords_json": ["seed 1", "seed 2", "seed 3"]
    }}
  ]
}}

Rules:
- Generate 2-5 candidates total.
- Include a software candidate only if the job implies repeated action, calculation, conversion, comparison, planning, or tracking.
- Include an SEO article candidate when there is clear search intent.
- Include an editorial candidate only when the job is strategically useful but not obviously search-first.
- candidate_text must be specific and publishable/buildable.
- source_keywords_json must contain 4-6 short literal search phrases.
- source_keywords_json phrases must look like real Google queries, not article titles.
- Each source keyword should usually be 2-5 words, occasionally 6 if absolutely necessary.
- Prefer practical head terms like "calculator", "cost", "vs", "tool", "checklist", "template", "comparison".
- Do not include poetic phrasing, colons, long sentence fragments, or branded headline language in source_keywords_json.
- If candidate_text is long or title-like, source_keywords_json must translate it into shorter, more searchable query variants.

Website context:
- Website description: {website_context.get("website_description") or ""}
- Primary category: {website_context.get("primary_category_name") or ""}
- Secondary category: {website_context.get("secondary_category_name") or ""}
- Target audience: {website_context.get("target_audience") or ""}

User job:
- job_text: {job.get("job_text") or ""}
- job_type_hint: {job.get("job_type_hint") or ""}
"""
        response = await llm_service.generate_json(
            prompt,
            task_role=LLM_ROLE_RESEARCH_IDEA_GENERATION,
            max_tokens=2200,
        )
        raw_candidates = response.get("candidates") if isinstance(response, dict) else []
        normalized: List[Dict[str, Any]] = []
        seen = set()
        for item in raw_candidates or []:
            if not isinstance(item, dict):
                continue
            candidate_type = str(item.get("candidate_type") or "").strip().lower()
            candidate_text = str(item.get("candidate_text") or "").strip()
            if candidate_type not in {"seo_article", "software", "editorial"} or not candidate_text:
                continue
            key = (candidate_type, candidate_text.lower())
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                {
                    "candidate_type": candidate_type,
                    "candidate_text": candidate_text,
                    "normalized_candidate_text": str(
                        item.get("normalized_candidate_text") or candidate_text.lower()
                    ).strip(),
                    "status": "draft",
                    "candidate_metadata": item.get("candidate_metadata") or {},
                    "source_keywords_json": item.get("source_keywords_json") or [],
                }
            )
        logger.info(
            "research rebuild derived candidates count=%s job=%r",
            len(normalized),
            job.get("job_text"),
        )
        return normalized

    async def save_candidates(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        user_job_id: UUID,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Persist new opportunity candidates for later validation."""
        payloads: List[Dict[str, Any]] = []
        for candidate in candidates:
            payloads.append(
                {
                    "project_id": str(project_id),
                    "user_job_id": str(user_job_id),
                    "candidate_type": candidate.get("candidate_type"),
                    "candidate_text": str(candidate.get("candidate_text") or "").strip(),
                    "normalized_candidate_text": candidate.get("normalized_candidate_text"),
                    "status": candidate.get("status") or "draft",
                    "candidate_metadata": candidate.get("candidate_metadata") or {},
                    "source_keywords_json": candidate.get("source_keywords_json") or [],
                }
            )
        return await self.bulk_create_records(user_id=user_id, data_list=payloads)

    async def create_candidate(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        user_job_id: UUID,
        candidate_type: str,
        candidate_text: str,
        normalized_candidate_text: Optional[str] = None,
        status: str = "draft",
        candidate_metadata: Optional[Dict[str, Any]] = None,
        source_keywords_json: Optional[List[Dict[str, Any]] | List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create a single opportunity candidate."""
        return await self.create_record(
            user_id=user_id,
            data={
                "project_id": str(project_id),
                "user_job_id": str(user_job_id),
                "candidate_type": candidate_type,
                "candidate_text": str(candidate_text or "").strip(),
                "normalized_candidate_text": normalized_candidate_text,
                "status": status,
                "candidate_metadata": candidate_metadata or {},
                "source_keywords_json": source_keywords_json or [],
            },
        )

    async def list_candidates(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        user_job_id: Optional[UUID] = None,
        candidate_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List candidates for a project/job scope."""
        filters: Dict[str, Any] = {"project_id": str(project_id)}
        if user_job_id:
            filters["user_job_id"] = str(user_job_id)
        if candidate_type:
            filters["candidate_type"] = candidate_type
        if status:
            filters["status"] = status
        return await self.list_records(
            user_id=user_id,
            filters=filters,
            order_by={"created_at": "desc"},
        )

    async def reject_candidate(
        self,
        *,
        candidate_id: UUID,
        user_id: UUID,
        rejection_reason_tags: List[str],
        rejection_reason_free_text: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Reject a candidate with structured feedback."""
        return await self.update_record(
            record_id=candidate_id,
            user_id=user_id,
            data={
                "status": "rejected",
                "rejection_reason_tags": self.normalize_reason_tags(rejection_reason_tags),
                "rejection_reason_free_text": rejection_reason_free_text,
            },
        )
