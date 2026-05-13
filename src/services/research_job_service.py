"""
Job discovery service for the research rebuild.
"""

from __future__ import annotations

from datetime import datetime
import math
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
    _JTBD_PATTERN = re.compile(
        r"^(?:i need to\s+)?when\s+(?P<situation>.+?),\s*i want to\s+(?P<action>.+?),\s*so i can\s+(?P<outcome>.+?)[.]?$",
        re.IGNORECASE,
    )

    @classmethod
    def _simplify_action_text(cls, raw_action: str) -> str:
        """Rewrite action phrasing into short, searchable everyday language."""
        action = re.sub(r"\s+", " ", str(raw_action or "").strip())
        if not action:
            return ""
        substitutions = (
            (r"\bfeed the ai\b", "use AI with"),
            (r"\busing an ai chain\b", "with AI"),
            (r"\busing ai tools?\b", "with AI"),
            (r"\busing ai\b", "with AI"),
            (r"\bgenerate a first draft\b", "draft a first version"),
            (r"\bmy unique writing style\b", "my writing style"),
            (r"\bmultiple research papers\b", "research papers"),
            (r"\bmultiple industry reports\b", "industry reports"),
            (r"\bcross-reference\b", "compare"),
        )
        for pattern, replacement in substitutions:
            action = re.sub(pattern, replacement, action, flags=re.IGNORECASE)
        action = re.sub(r"\s+", " ", action).strip(" .")
        return action

    @classmethod
    def _extract_jtbd_parts(cls, raw_job_text: str) -> Optional[Dict[str, str]]:
        """Extract JTBD parts from a full JTBD sentence when present."""
        text = re.sub(r"\s+", " ", str(raw_job_text or "").strip())
        if not text:
            return None
        match = cls._JTBD_PATTERN.match(text)
        if not match:
            return None
        parts = {
            "situation": match.group("situation").strip(" ."),
            "action": match.group("action").strip(" ."),
            "outcome": match.group("outcome").strip(" ."),
        }
        if not all(parts.values()):
            return None
        return parts

    @classmethod
    def _build_jtbd_statement(cls, parts: Dict[str, str]) -> str:
        """Rebuild a clean JTBD statement from extracted parts."""
        return (
            f"When {parts['situation']}, I want to {parts['action']}, "
            f"so I can {parts['outcome']}."
        )

    @classmethod
    def _normalize_job_text(cls, raw_job_text: str) -> str:
        """Rewrite generated jobs into straightforward JTBD language."""
        text = re.sub(r"\s+", " ", str(raw_job_text or "").strip())
        if not text:
            return ""

        text = cls._LEADING_LABEL_PATTERN.sub("", text).strip()
        text = re.sub(r"^[\"'“”]+|[\"'“”]+$", "", text).strip()

        jtbd_parts = cls._extract_jtbd_parts(text)
        if jtbd_parts:
            action = cls._simplify_action_text(jtbd_parts["action"])
            outcome = jtbd_parts["outcome"]
            if re.match(r"^(choose|select|pick|find)\b", action, re.IGNORECASE):
                action = f"{action} that {outcome[:1].lower()}{outcome[1:]}" if len(outcome) > 1 else action
            text = f"I need to {action}"

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

    def _normalize_generated_jobs(self, jobs: Any, *, source: str = "llm_generation") -> List[Dict[str, Any]]:
        """Normalize raw LLM jobs into persisted job records."""
        normalized: List[Dict[str, Any]] = []
        seen = set()
        for item in jobs or []:
            if not isinstance(item, dict):
                continue
            raw_job_text = str(item.get("job_text") or "")
            jtbd_parts = self._extract_jtbd_parts(raw_job_text)
            job_text = self._normalize_job_text(raw_job_text)
            if not job_text:
                continue
            key = job_text.lower()
            if key in seen:
                continue
            seen.add(key)
            generation_metadata = dict(item.get("generation_metadata") or {})
            if jtbd_parts:
                generation_metadata["jtbd_statement"] = self._build_jtbd_statement(jtbd_parts)
                generation_metadata["situation"] = jtbd_parts["situation"]
                generation_metadata["outcome"] = jtbd_parts["outcome"]
            raw_search_seeds = generation_metadata.get("search_seeds") or item.get("search_seeds") or []
            search_seeds: List[str] = []
            for seed in raw_search_seeds if isinstance(raw_search_seeds, list) else []:
                cleaned_seed = re.sub(r"\s+", " ", str(seed or "").strip().lower())
                if not cleaned_seed:
                    continue
                if cleaned_seed not in search_seeds:
                    search_seeds.append(cleaned_seed)
            if search_seeds:
                generation_metadata["search_seeds"] = search_seeds[:3]
            intent_type = str(
                generation_metadata.get("intent_type") or item.get("intent_type") or ""
            ).strip().lower()
            if intent_type in {"informational", "navigational", "transactional"}:
                generation_metadata["intent_type"] = intent_type
            normalized.append(
                {
                    "job_text": job_text,
                    "job_type_hint": str(item.get("job_type_hint") or "hybrid").strip().lower(),
                    "job_source": source,
                    "status": "draft",
                    "generation_metadata": generation_metadata,
                }
            )
        return normalized

    @classmethod
    def _build_context_guardrails(cls, context: Dict[str, Any]) -> str:
        """Add prompt guardrails based on the category and website context."""
        combined = " ".join(
            str(
                context.get(key) or ""
            )
            for key in (
                "website_description",
                "primary_category_name",
                "primary_category_description",
                "secondary_category_name",
                "secondary_category_description",
                "focus_area",
            )
        ).lower()
        if any(
            phrase in combined
            for phrase in (
                "daily work",
                "daily life",
                "real daily life",
                "save time",
                "reduce admin",
                "automation",
                "personal workflow",
            )
        ):
            return """
- Prioritize real-life automation and everyday workflow needs over abstract knowledge-work tasks.
- Favor jobs tied to saving time, reducing admin, staying organized, turning information into action, and getting routine tasks done faster.
- Good directions include scheduling, note-taking, inbox cleanup, task follow-up, summarizing long content, comparing options before buying, planning, budgeting, travel, shopping, family coordination, and personal learning workflows.
- Avoid drifting into enterprise, analyst, consultant, or academic-language tasks unless the website context clearly makes that the main audience.
- If the focus area mentions prompt engineering, context management, or AI chains, translate that into the human outcome the person wants in daily life rather than making the prompt mechanic itself the job.
""".strip()
        return ""

    def _build_generate_jobs_prompt(
        self,
        *,
        context: Dict[str, Any],
        count: int,
        negative_notes: Dict[str, Any],
        current_date: str,
        current_year: int,
        relaxed_overlap: bool = False,
    ) -> str:
        overlap_rules = """
- Avoid overlap with previously generated jobs unless the focus area clearly creates a narrower or materially different angle.
- If the focus area overlaps with existing jobs, generate narrower sub-angles, different user intents, or more execution-specific variants instead of returning nothing.
- Treat "different audience + different task + different output" as distinct enough when the focus area is narrower than earlier broad jobs.
""".strip()
        if relaxed_overlap:
            overlap_rules = """
- Use previous jobs only to avoid near-exact duplicates.
- If older jobs were broader, you may generate narrower variants for this focus area.
- When in doubt, prefer a fresh focused JTBD variant over returning nothing.
""".strip()
        context_guardrails = self._build_context_guardrails(context)

        return f"""
Role: You are an expert Product Strategist and SEO Analyst specializing in the Jobs-to-be-Done (JTBD) framework.

Today's date is {current_date}. The current year is {current_year}.

Return valid JSON with this shape:
{{
  "jobs": [
    {{
      "job_text": "I need to [plain-language action].",
      "job_type_hint": "seo|editorial|software|hybrid",
      "generation_metadata": {{
        "why": "short reason",
        "jtbd_statement": "When [situation], I want to [action], so I can [expected outcome].",
        "intent_type": "informational|navigational|transactional",
        "search_seeds": ["seed one", "seed two", "seed three"],
        "category": "primary or secondary category label"
      }}
    }}
  ]
}}

Rules:
- Generate exactly {max(1, min(count, 50))} jobs.
- Jobs must be specific user problems, not abstract topics or content ideas.
- Treat the output as JTBD statements first. Do not pre-decide whether the solution is a comparison, calculator, workflow, template, article, or software tool.
- Write every job_text as a short, straightforward everyday-language sentence that starts with "I need to ...".
- Put the full JTBD formula in generation_metadata.jtbd_statement using this structure: "When [situation], I want to [action], so I can [expected outcome]."
- job_text should be the readable summary of the job, not the full three-clause JTBD sentence.
- job_text should usually be about 6-16 words and should read like a clear, searchable task.
- Prefer user-task wording over prompt-engineering wording. Say "I need to compare research papers with AI" instead of "I need to build an AI chain to cross-reference papers."
- Avoid phrasing that sounds like model instructions or tool internals, such as "feed the AI", "prompt chain", "multi-step prompting", or "generate a first draft in my voice".
- Focus on functional jobs that are easy to translate into SEO seed keywords.
- Good job_text examples: "I need to compare travel card reward programs", "I need to compare appliance warranty terms", "I need to choose an AI meal planner that fits my diet"
- Good job_text examples for this category: "I need to compare research papers with AI", "I need to draft reports in my writing style", "I need to turn long videos into quick action notes"
- Good JTBD examples: "When I am picking a travel card, I want to compare reward programs, so I can choose the best one for my spending.", "When I am buying an appliance, I want to compare warranty terms, so I can avoid expensive surprises later."
- Bad examples: "Decision tree: Choosing between...", "Prompt sequence to analyze...", "Workflow: Using AI to extract...", "Comparison: X vs. Y...".
- Bad job_text examples: "When I am drowning in back-to-back video calls, I want to compare AI meeting assistants...", "I need to when I am...", "Prompt sequence to analyze..."
- Avoid prefixes such as "Decision tree:", "Decision matrix:", "Comparison:", "Workflow:", and "Prompt sequence:".
- The goal is to produce jobs that can later generate keyword candidates with measurable related-keyword support in DataForSEO.
- For each job, include exactly 3 search seeds in generation_metadata.search_seeds.
- Search seeds must be raw 2-4 word phrases that a real person would type into Google.
- Search seeds must avoid headline language and should likely have search volume.
- Label each job with one intent_type: informational, navigational, or transactional.
- Prefer jobs with clear search language, clear intent, and concrete nouns a real person would search for.
- Avoid duplicates and near-duplicates.
- Avoid jobs already rejected for being off-brand, too broad, or technically impossible.
{context_guardrails}
{overlap_rules}
- Keep each job_text concise but complete enough to preserve the JTBD structure.
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
        prompt = self._build_generate_jobs_prompt(
            context=context,
            count=count,
            negative_notes=negative_notes,
            current_date=current_date,
            current_year=current_year,
            relaxed_overlap=False,
        )
        response = await llm_service.generate_json(
            prompt,
            task_role=LLM_ROLE_RESEARCH_TOPIC_GENERATION,
            max_tokens=2200,
        )
        jobs = response.get("jobs") if isinstance(response, dict) else []
        normalized = self._normalize_generated_jobs(jobs)
        minimum_viable_count = min(count, max(3, math.ceil(count * 0.6))) if count > 1 else 1
        if len(normalized) < minimum_viable_count and count > len(normalized):
            retry_negative_notes = dict(negative_notes)
            retry_existing_jobs = list(retry_negative_notes.get("recent_existing_jobs") or [])
            retry_existing_jobs.extend(
                {
                    "job_text": item.get("job_text"),
                    "status": item.get("status"),
                    "job_type_hint": item.get("job_type_hint"),
                }
                for item in normalized
            )
            retry_negative_notes["recent_existing_jobs"] = retry_existing_jobs
            retry_prompt = self._build_generate_jobs_prompt(
                context=context,
                count=count - len(normalized),
                negative_notes=retry_negative_notes,
                current_date=current_date,
                current_year=current_year,
                relaxed_overlap=True,
            )
            retry_response = await llm_service.generate_json(
                retry_prompt,
                task_role=LLM_ROLE_RESEARCH_TOPIC_GENERATION,
                max_tokens=2200,
            )
            retry_jobs = retry_response.get("jobs") if isinstance(retry_response, dict) else []
            retry_normalized = self._normalize_generated_jobs(retry_jobs, source="llm_generation_retry")
            existing_keys = {str(item.get("job_text") or "").strip().lower() for item in normalized}
            normalized.extend(
                item for item in retry_normalized
                if str(item.get("job_text") or "").strip().lower() not in existing_keys
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
        include_existing_jobs: bool = True,
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
        existing_jobs: List[Dict[str, Any]] = []
        if include_existing_jobs:
            existing_jobs = await self.list_records(
                user_id=user_id,
                filters={
                    "project_id": str(project_id),
                    "primary_category_id": str(primary_category_id) if primary_category_id else None,
                    "secondary_category_id": str(secondary_category_id) if secondary_category_id else None,
                    "status": "draft",
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
            ],
        }

    async def delete_active_jobs_in_scope(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        primary_category_id: Optional[UUID] = None,
        secondary_category_id: Optional[UUID] = None,
    ) -> int:
        """Delete draft and approved jobs in the current scope to start a clean batch."""
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
        deleted_count = 0
        for job in active_jobs:
            record_id = str(job.get("id") or "").strip()
            if not record_id:
                continue
            deleted = await self.delete_record(record_id=UUID(record_id), user_id=user_id)
            if deleted:
                deleted_count += 1
        return deleted_count
