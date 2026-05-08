"""
Routing service for the research rebuild.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID

from .research_rebuild_base_service import ResearchRebuildBaseService


class ResearchRoutingService(ResearchRebuildBaseService):
    """Route validated opportunities into the next product workflow."""

    table_name = "research_routing_decisions"

    async def decide_route(
        self,
        *,
        candidate: Dict[str, Any],
        validation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply the future article/software/editorial routing rules."""
        candidate_type = str(candidate.get("candidate_type") or "").strip().lower()
        eligible = bool(validation_result.get("eligibility_passed"))
        achievability = float(validation_result.get("achievability_score") or 0.0)
        feasibility = float(validation_result.get("feasibility_score") or 0.0)
        software_pattern = float(validation_result.get("software_pattern_score") or 0.0)

        if candidate_type == "software":
            if not eligible and feasibility < 0.45:
                route = "software_backlog_low_feasibility"
            elif eligible and achievability >= 0.68 and software_pattern >= 0.60:
                route = "software_ready"
            elif achievability >= 0.58:
                route = "needs_more_keyword_validation"
            else:
                route = "rejected_low_achievability"
        elif candidate_type == "editorial":
            route = "editorial_only" if achievability >= 0.60 else "rejected_low_achievability"
        else:
            if eligible and achievability >= 0.65:
                route = "article_ready"
            elif achievability >= 0.55:
                route = "needs_more_keyword_validation"
            else:
                route = "rejected_low_achievability"

        return {
            "candidate_id": candidate.get("id"),
            "route": route,
            "route_reason_codes": validation_result.get("validation_reason_codes") or [],
            "route_metadata": {
                "candidate_type": candidate_type,
                "achievability_score": achievability,
                "eligibility_passed": eligible,
            },
        }

    async def save_routing_decision(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        candidate_id: UUID,
        validation_run_id: UUID,
        route: str,
        route_reason_codes: Optional[list[str]] = None,
        route_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Persist a route selection for a candidate."""
        return await self.create_record(
            user_id=user_id,
            data={
                "project_id": str(project_id),
                "candidate_id": str(candidate_id),
                "validation_run_id": str(validation_run_id),
                "route": route,
                "route_reason_codes": route_reason_codes or [],
                "route_metadata": route_metadata or {},
            },
        )

    async def list_routing_decisions(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        candidate_id: Optional[UUID] = None,
        route: Optional[str] = None,
    ) -> list[dict]:
        """List routing decisions for a project/candidate scope."""
        filters: Dict[str, Any] = {"project_id": str(project_id)}
        if candidate_id:
            filters["candidate_id"] = str(candidate_id)
        if route:
            filters["route"] = route
        return await self.list_records(
            user_id=user_id,
            filters=filters,
            order_by={"created_at": "desc"},
        )
