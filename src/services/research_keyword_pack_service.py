"""
Keyword-pack service for the research rebuild.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID

from .research_rebuild_base_service import ResearchRebuildBaseService


class ResearchKeywordPackService(ResearchRebuildBaseService):
    """Persist the final keyword pack gate before Content Studio handoff."""

    table_name = "research_keyword_packs"

    async def build_keyword_pack(
        self,
        *,
        candidate: Dict[str, Any],
        validation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the primary + secondary keyword pack for a candidate."""
        validation_metadata = validation_result.get("validation_metadata") or {}
        primary_keyword = str(
            validation_metadata.get("primary_search_seed")
            or validation_metadata.get("query")
            or candidate.get("candidate_text")
            or ""
        ).strip()
        secondary_candidates = []
        for raw in validation_metadata.get("seed_keywords_used") or candidate.get("source_keywords_json") or []:
            cleaned = str(raw or "").strip()
            if cleaned and cleaned.lower() != primary_keyword.lower() and cleaned not in secondary_candidates:
                secondary_candidates.append(cleaned)

        metrics_rows = validation_metadata.get("metrics_rows") or []
        keyword_metrics_json = {}
        for row in metrics_rows:
            if not isinstance(row, dict):
                continue
            keyword = str(row.get("keyword") or "").strip()
            if not keyword:
                continue
            keyword_metrics_json[keyword] = {
                "search_volume": row.get("search_volume"),
                "cpc": row.get("cpc"),
                "keyword_difficulty": row.get("keyword_difficulty"),
            }

        measurable_primary = int((keyword_metrics_json.get(primary_keyword) or {}).get("search_volume") or 0) > 0
        measurable_count = sum(
            1 for value in secondary_candidates
            if int((keyword_metrics_json.get(value) or {}).get("search_volume") or 0) > 0
        )
        status = "ready" if primary_keyword and len(secondary_candidates) >= 3 and measurable_count >= 3 else "cluster_too_thin"
        return {
            "primary_keyword": primary_keyword,
            "secondary_keywords_json": secondary_candidates[:8],
            "keyword_metrics_json": keyword_metrics_json,
            "keyword_pack_status": status if measurable_primary else "cluster_too_thin",
            "keyword_pack_reason_codes": [] if status == "ready" and measurable_primary else ["cluster_too_thin"],
        }

    async def save_keyword_pack(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        candidate_id: UUID,
        validation_run_id: UUID,
        payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Persist keyword readiness for a candidate."""
        data = dict(payload)
        data["project_id"] = str(project_id)
        data["candidate_id"] = str(candidate_id)
        data["validation_run_id"] = str(validation_run_id)
        return await self.create_record(user_id=user_id, data=data)

    async def list_keyword_packs(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        candidate_id: Optional[UUID] = None,
        keyword_pack_status: Optional[str] = None,
    ) -> list[dict]:
        """List persisted keyword packs."""
        filters: Dict[str, Any] = {"project_id": str(project_id)}
        if candidate_id:
            filters["candidate_id"] = str(candidate_id)
        if keyword_pack_status:
            filters["keyword_pack_status"] = keyword_pack_status
        return await self.list_records(
            user_id=user_id,
            filters=filters,
            order_by={"created_at": "desc"},
        )
