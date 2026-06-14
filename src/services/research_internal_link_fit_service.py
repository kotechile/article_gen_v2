"""Internal-link fit service for the research rebuild."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional
from uuid import UUID

from .research_rebuild_base_service import ResearchRebuildBaseService
from .supabase_service import SupabaseService


class ResearchInternalLinkFitService(ResearchRebuildBaseService):
    """Attach parent/child/hub candidates before idea handoff."""

    table_name = "research_internal_link_candidates"
    _TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
    _STOPWORDS = {
        "about",
        "after",
        "against",
        "between",
        "build",
        "choosing",
        "compare",
        "decision",
        "determine",
        "evaluate",
        "guide",
        "into",
        "just",
        "make",
        "making",
        "over",
        "stack",
        "than",
        "that",
        "their",
        "these",
        "this",
        "those",
        "through",
        "using",
        "when",
        "which",
        "why",
        "with",
        "your",
    }

    @classmethod
    def _tokenize(cls, value: str) -> set[str]:
        return {
            token
            for token in cls._TOKEN_PATTERN.findall((value or "").lower())
            if len(token) > 3 and token not in cls._STOPWORDS
        }

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
            limit=1000,
        )
        keyword_seed = " ".join(
            [
                str(candidate.get("candidate_text") or ""),
                str(metadata.get("primary_keyword") or ""),
                str(metadata.get("candidate_title") or ""),
            ]
        )
        query_tokens = self._tokenize(keyword_seed)
        if not query_tokens:
            return {"items": []}

        results = []
        seen_titles: set[str] = set()
        for row in rows:
            title = str(row.get("title") or "")
            normalized_title = title.strip().lower()
            if not normalized_title or normalized_title in seen_titles:
                continue

            title_tokens = self._tokenize(title)
            overlap_tokens = query_tokens & title_tokens
            overlap = len(overlap_tokens)
            if overlap < 2:
                continue

            score = min(1.0, overlap / max(1, min(len(query_tokens), len(title_tokens))))
            if score < 0.18:
                continue

            matched_link = row.get("link") or ""
            if "://cms." in matched_link:
                matched_link = matched_link.replace("://cms.", "://")
            matched_link = re.sub(r'/(\d{4}/\d{2}/\d{2}/|\d{4}/\d{2}/)', '/', matched_link)

            seen_titles.add(normalized_title)
            results.append(
                {
                    "project_id": str(project_id) if project_id else None,
                    "validation_run_id": (validation_result or {}).get("id"),
                    "wordpress_imported_post_id": row.get("id"),
                    "link_role": "parent_candidate" if score >= 0.45 else "sibling_candidate",
                    "match_score": round(score, 4),
                    "match_reason_codes": ["title_token_overlap_filtered"],
                    "match_metadata": {
                        "matched_title": title,
                        "matched_link": matched_link,
                        "overlap_tokens": sorted(overlap_tokens),
                    },
                }
            )
        results.sort(key=lambda item: item.get("match_score") or 0, reverse=True)
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
