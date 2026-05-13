"""
Job discovery service for the research rebuild.
"""

from __future__ import annotations

from datetime import datetime
import logging
import re
from typing import Any, Dict, List, Optional
from uuid import UUID

from supabase_client import LLM_ROLE_RESEARCH_TOPIC_GENERATION

from .research_rebuild_base_service import ResearchRebuildBaseService
from .llm.llm_service import llm_service

logger = logging.getLogger(__name__)


class ResearchJobService(ResearchRebuildBaseService):
    """Persist and manage first-class user jobs."""

    table_name = "research_user_jobs"

    _LEADING_LABEL_PATTERN = re.compile(
        r"^(decision tree|decision matrix|comparison|workflow|prompt sequence|checklist|template|calculator)\s*:\s*",
        re.IGNORECASE,
    )

    @classmethod
    def _normalize_job_text(cls, raw_job_text: str) -> str:
        """Rewrite generated jobs into straightforward JTBD language."""
        text = re.sub(r"\s+", " ", str(raw_job_text or "").strip())
        if not text:
            return ""

        text = cls._LEADING_LABEL_PATTERN.sub("", text).strip()
        text = re.sub(r"^[\"'“”]+|[\"'“”]+$", "", text).strip()

        replacements = (
            (r"^using ai to\s+", "I need to use AI to "),
            (r"^how to\s+", "I need to "),
            (r"^selecting\s+", "I need to choose "),
            (r"^choosing\s+", "I need to choose "),
            (r"^analyz(?:e|ing)\s+", "I need to analyze "),
            (r"^extract and compare\s+", "I need to extract and compare "),
            (r"^compare\s+", "I need to compare "),
            (r"^choose\s+", "I need to choose "),
            (r"^find\s+", "I need to find "),
            (r"^check\s+", "I need to check "),
            (r"^track\s+", "I need to track "),
            (r"^calculate\s+", "I need to calculate "),
            (r"^estimate\s+", "I need to estimate "),
            (r"^plan\s+", "I need to plan "),
            (r"^use\s+", "I need to use "),
            (r"^review\s+", "I need to review "),
            (r"^decide\s+", "I need to decide "),
            (r"^pick\s+", "I need to pick "),
        )
        for pattern, replacement in replacements:
            if re.match(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE).strip()
                break

        if re.match(r"^(ai-powered|manual timers|perplexity|searchgpt)\b", text, re.IGNORECASE):
            text = f"I need to compare {text}"

        if not re.match(r"^(i need to|i want to)\b", text, re.IGNORECASE):
            text = f"I need to {text[:1].lower()}{text[1:]}" if len(text) > 1 else text

        text = text.rstrip(" .")
        return text

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
        current_date = datetime.now().strftime("%B %-d, %Y")
        current_year = datetime.now().year
        prompt = f"""
You are generating high-signal user jobs for a content and software discovery workflow.

Today's date is {current_date}. The current year is {current_year}.

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
- Jobs must be specific user problems, not abstract topics or content ideas.
- Treat the output as "jobs to be done" statements first. Do not pre-decide whether the solution is a comparison, calculator, workflow, template, article, or software tool.
- Write every job_text in straightforward everyday language that a normal reader can understand quickly.
- Job text must sound like a plain user need, not an internal strategy label, content format, or framework name.
- Prefer JTBD phrasing that starts with "I need to..." or occasionally "I want to...".
- Good examples: "I need to compare AI newsletter tools for quick daily industry updates", "I need to check which travel credit card gives me the best rewards for my spending", "I need to compare warranty terms before buying a dishwasher".
- Bad examples: "Decision tree: Choosing between...", "Prompt sequence to analyze...", "Workflow: Using AI to extract...", "Comparison: X vs. Y...".
- Avoid prefixes such as "Decision tree:", "Decision matrix:", "Comparison:", "Workflow:", and "Prompt sequence:".
- Avoid naming output formats in the job itself unless the user is explicitly seeking that format.
- The goal is to produce jobs that can later generate keyword candidates with measurable related-keyword support in DataForSEO.
- Prefer jobs with clear search language, clear intent, and concrete nouns a real person would search for.
- Avoid duplicates and near-duplicates.
- Avoid jobs already rejected for being off-brand, too broad, or technically impossible.
- Avoid overlap with previously generated jobs unless the focus area clearly creates a narrower or materially different angle.
- If the focus area overlaps with existing jobs, generate narrower sub-angles, different user intents, or more execution-specific variants instead of returning nothing.
- Treat "different audience + different task + different output" as distinct enough when the focus area is narrower than earlier broad jobs.
- Keep each job_text short and practical.
- Prefer one sentence, ideally under 16 words when possible.
- If a focus area is provided, make at least 80% of jobs clearly centered on that focus.
- If avoid guidance is provided, actively steer away from those patterns or themes.
- Prefer literal, searchable phrasing over essay-like or poetic titles.
- Use the current date and year as ground truth. Do not refer to 2025 or any earlier year unless the job is explicitly historical.
- If a year is needed in job_text, prefer {current_year}.
- Only return fewer than the requested count if the focus area is truly exhausted after attempting narrower, materially distinct variants.

Website context:
- Project name: {context.get("project_name") or ""}
- Website description: {context.get("website_description") or ""}
- Primary category: {context.get("primary_category_name") or ""}
- Primary category description: {context.get("primary_category_description") or ""}
- Secondary category: {context.get("secondary_category_name") or ""}
- Secondary category description: {context.get("secondary_category_description") or ""}
- Target audience: {context.get("target_audience") or ""}
- Focus area to prioritize: {context.get("focus_area") or ""}
- Avoid guidance from user: {context.get("avoid_guidance") or ""}
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
            job_text = self._normalize_job_text(str(item.get("job_text") or ""))
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
        include_archived: bool = False,
        batch_id: Optional[str] = None,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """List jobs for a project/category scope."""
        filters: Dict[str, Any] = {"project_id": str(project_id)}
        if primary_category_id:
            filters["primary_category_id"] = str(primary_category_id)
        if secondary_category_id:
            filters["secondary_category_id"] = str(secondary_category_id)
        if status:
            filters["status"] = status
        records = await self.list_records(
            user_id=user_id,
            filters=filters,
            order_by={"created_at": "desc"},
        )
        if not include_archived:
            records = [record for record in records if str(record.get("status") or "").strip().lower() != "archived"]
        if active_only and not status:
            records = [
                record
                for record in records
                if str(record.get("status") or "").strip().lower() in {"draft", "approved"}
            ]
        if batch_id:
            records = [
                record
                for record in records
                if str((record.get("generation_metadata") or {}).get("batch_id") or "") == batch_id
            ]
        return records

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

    async def build_negative_context(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        primary_category_id: Optional[UUID] = None,
        secondary_category_id: Optional[UUID] = None,
    ) -> Dict[str, Any]:
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
        existing_jobs = await self.list_records(
            user_id=user_id,
            filters={
                "project_id": str(project_id),
                "primary_category_id": str(primary_category_id) if primary_category_id else None,
                "secondary_category_id": str(secondary_category_id) if secondary_category_id else None,
            },
            order_by={"updated_at": "desc"},
            limit=75,
        )
        return {
            "recent_rejected_jobs": [
                {
                    "job_text": item.get("job_text"),
                    "rejection_reason_tags": item.get("rejection_reason_tags") or [],
                    "rejection_reason_free_text": item.get("rejection_reason_free_text"),
                }
                for item in rejected_jobs
            ],
            "recent_existing_jobs": [
                {
                    "job_text": item.get("job_text"),
                    "status": item.get("status"),
                    "job_type_hint": item.get("job_type_hint"),
                }
                for item in existing_jobs
                if str(item.get("status") or "").strip().lower() != "rejected"
            ],
        }

    async def archive_active_jobs_in_scope(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        primary_category_id: Optional[UUID] = None,
        secondary_category_id: Optional[UUID] = None,
    ) -> int:
        """Archive non-rejected jobs in the current scope to start a clean batch."""
        existing_jobs = await self.list_jobs(
            user_id=user_id,
            project_id=project_id,
            primary_category_id=primary_category_id,
            secondary_category_id=secondary_category_id,
            include_archived=False,
        )
        active_jobs = [
            job for job in existing_jobs
            if str(job.get("status") or "").strip().lower() in {"draft", "approved"}
        ]
        if not active_jobs:
            return 0

        updates = [
            {
                "id": str(job.get("id")),
                "status": "archived",
            }
            for job in active_jobs
            if job.get("id")
        ]
        await self.supabase_service.bulk_update(self.table_name, updates, user_id)
        return len(updates)
