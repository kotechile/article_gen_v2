"""
Outcome generation service for the research rebuild.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID

from supabase_client import LLM_ROLE_RESEARCH_IDEA_GENERATION

from .research_rebuild_base_service import ResearchRebuildBaseService
from .llm.llm_service import llm_service


class ResearchGenerationService(ResearchRebuildBaseService):
    """Generate persisted article/software/editorial outcomes from routed candidates."""

    table_name = "research_generated_outcomes"

    async def generate_outcome(
        self,
        *,
        candidate: Dict[str, Any],
        routing_decision: Dict[str, Any],
        keyword_pack: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate the final research outcome payload."""
        route = str(routing_decision.get("route") or "")
        outcome_type = "software" if "software" in route else ("editorial" if route == "editorial_only" else "article")
        prompt = f"""
Return valid JSON for a single {outcome_type} outcome.

Shape:
{{
  "title": "string",
  "description": "string",
  "target_intent": "informational|commercial|transactional",
  "product_type": "optional string",
  "user_job_to_be_done": "optional string",
  "build_complexity": "low|medium|high",
  "distribution_angle": "optional string",
  "category": "{'software_tool' if outcome_type == 'software' else 'seo_optimized'}",
  "subtopic": "string"
}}

Candidate:
- type: {candidate.get("candidate_type")}
- text: {candidate.get("candidate_text")}
- metadata: {candidate.get("candidate_metadata") or {}}

Keyword pack:
{keyword_pack or {}}
"""
        response = await llm_service.generate_json(
            prompt,
            task_role=LLM_ROLE_RESEARCH_IDEA_GENERATION,
            max_tokens=1200,
        )
        if not isinstance(response, dict):
            response = {}
        return {
            "outcome_type": outcome_type,
            "status": "generated",
            "outcome_metadata": response,
        }

    async def save_generated_outcome(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        candidate_id: UUID,
        payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Persist a generated outcome record."""
        data = dict(payload)
        data["project_id"] = str(project_id)
        data["candidate_id"] = str(candidate_id)
        return await self.create_record(user_id=user_id, data=data)

    async def list_generated_outcomes(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        candidate_id: Optional[UUID] = None,
        outcome_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[dict]:
        """List generated outcomes for a project/candidate scope."""
        filters: Dict[str, Any] = {"project_id": str(project_id)}
        if candidate_id:
            filters["candidate_id"] = str(candidate_id)
        if outcome_type:
            filters["outcome_type"] = outcome_type
        if status:
            filters["status"] = status
        return await self.list_records(
            user_id=user_id,
            filters=filters,
            order_by={"created_at": "desc"},
        )
