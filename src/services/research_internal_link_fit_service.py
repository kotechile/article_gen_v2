"""
Internal-link fit service for the research rebuild.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID

from .research_rebuild_base_service import ResearchRebuildBaseService
from .supabase_service import SupabaseService


class ResearchInternalLinkFitService(ResearchRebuildBaseService):
    """Attach parent/child/hub candidates before idea handoff."""

    table_name = "research_internal_link_candidates"

    async def compute_internal_link_fit(
        self,
        *,
        candidate: Dict[str, Any],
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute internal-link opportunities from imported WordPress content.

        V1 is expected to use the current shallow imported post inventory.
        """
        metadata = candidate.get("candidate_metadata") or {}
        category_context = metadata.get("category_context") if isinstance(metadata, dict) else {}
        project_id = category_context.get("project_id") or candidate.get("project_id")
        user_id = candidate.get("user_id")
        if not user_id:
            raise ValueError("candidate user_id is required")

        rows = await self.supabase_service.get_by_filters(
            "wordpress_imported_posts",
            filters={},
            user_id=UUID(str(user_id)),
            order_by={"created_at": "desc"},
            limit=100,
        )
        query_tokens = {
            tok for tok in str(candidate.get("candidate_text") or "").lower().split()
            if len(tok) > 3
        }
        results = []
        for row in rows:
            title = str(row.get("title") or "")
            title_tokens = {tok for tok in title.lower().split() if len(tok) > 3}
            overlap = len(query_tokens & title_tokens)
            if overlap <= 0:
                continue
            score = min(1.0, overlap / max(1, len(query_tokens)))
            results.append(
                {
                    "project_id": str(project_id) if project_id else None,
                    "validation_run_id": (validation_result or {}).get("id"),
                    "wordpress_imported_post_id": row.get("id"),
                    "link_role": "parent_candidate" if score >= 0.6 else "sibling_candidate",
                    "match_score": round(score, 4),
                    "match_reason_codes": ["title_token_overlap"],
                    "match_metadata": {
                        "matched_title": title,
                        "matched_link": row.get("link"),
                    },
                }
            )
        return {"items": results[:10]}

    async def save_internal_link_candidate(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        candidate_id: UUID,
        payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Persist a single parent/child/sibling/hub suggestion."""
        data = dict(payload)
        data["project_id"] = str(project_id)
        data["candidate_id"] = str(candidate_id)
        return await self.create_record(user_id=user_id, data=data)

    async def list_internal_link_candidates(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        candidate_id: Optional[UUID] = None,
        link_role: Optional[str] = None,
    ) -> list[dict]:
        """List internal-link candidates for a project/candidate scope."""
        filters: Dict[str, Any] = {"project_id": str(project_id)}
        if candidate_id:
            filters["candidate_id"] = str(candidate_id)
        if link_role:
            filters["link_role"] = link_role
        return await self.list_records(
            user_id=user_id,
            filters=filters,
            order_by={"created_at": "desc"},
        )
