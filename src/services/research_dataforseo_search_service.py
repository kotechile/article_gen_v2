"""
Manual DataForSEO lookup persistence for the research rebuild.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from src.integrations.dataforseo import dataforseo_api

from .research_rebuild_base_service import ResearchRebuildBaseService
from .supabase_service import SupabaseService


class ResearchDataforseoSearchService(ResearchRebuildBaseService):
    """Persist and execute user-driven DataForSEO lookups."""

    table_name = "research_dataforseo_searches"

    def __init__(self, supabase_service: Optional[SupabaseService] = None):
        super().__init__(supabase_service=supabase_service)

    @staticmethod
    def _normalize_query_text(value: Any) -> str:
        return " ".join(str(value or "").strip().lower().split())

    @staticmethod
    def _sanitize_for_json(value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, default=str))
        except Exception:
            return {"error": "serialization_failed"}

    @staticmethod
    def _summarize_related_keywords(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        top_items = []
        for row in (items or [])[:25]:
            keyword = str(row.get("keyword") or "").strip()
            if not keyword:
                continue
            top_items.append(
                {
                    "keyword": keyword,
                    "search_volume": row.get("search_volume"),
                    "cpc": row.get("cpc"),
                    "keyword_difficulty": row.get("keyword_difficulty"),
                    "competition": row.get("competition"),
                }
            )
        return {
            "result_count": len(items or []),
            "top_items": top_items,
        }

    @staticmethod
    def _summarize_keyword_overview(
        metric_items: List[Dict[str, Any]],
        kd_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        kd_map = {
            str(row.get("keyword") or "").strip().lower(): row
            for row in (kd_items or [])
            if str(row.get("keyword") or "").strip()
        }
        merged_items = []
        for row in metric_items or []:
            keyword = str(row.get("keyword") or "").strip()
            if not keyword:
                continue
            kd_row = kd_map.get(keyword.lower(), {})
            merged_items.append(
                {
                    "keyword": keyword,
                    "search_volume": row.get("search_volume"),
                    "cpc": row.get("cpc"),
                    "competition": row.get("competition"),
                    "keyword_difficulty": row.get("keyword_difficulty")
                    if row.get("keyword_difficulty") is not None
                    else kd_row.get("keyword_difficulty"),
                }
            )
        return {
            "result_count": len(merged_items),
            "top_items": merged_items[:50],
        }

    @staticmethod
    def _summarize_serp(query_text: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        top_items = []
        for row in (items or [])[:10]:
            top_items.append(
                {
                    "title": row.get("title"),
                    "url": row.get("url"),
                    "snippet": row.get("snippet"),
                    "date": row.get("date"),
                }
            )
        return {
            "query_text": query_text,
            "result_count": len(items or []),
            "top_items": top_items,
        }

    async def run_search(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        search_type: str,
        query_text: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        user_job_id: Optional[UUID] = None,
        primary_category_id: Optional[str] = None,
        secondary_category_id: Optional[str] = None,
        language_code: str = "en",
        location_code: int = 2840,
        limit: int = 25,
    ) -> Dict[str, Any]:
        search_type_normalized = str(search_type or "").strip().lower()
        now_iso = datetime.now(timezone.utc).isoformat()

        if search_type_normalized == "related_keywords":
            seed = str(query_text or "").strip()
            if not seed:
                raise ValueError("query_text is required for related_keywords searches")
            raw_result = await dataforseo_api.get_related_keywords_labs_live(
                [seed],
                language_name="English",
                location_code=int(location_code),
                limit_per_seed=max(1, min(int(limit or 25), 100)),
                return_raw=True,
            )
            items = (raw_result or {}).get("items") or []
            response_payload = self._sanitize_for_json(raw_result)
            result_summary_json = self._summarize_related_keywords(items)
            endpoint = "dataforseo_labs/google/related_keywords/live"
            normalized_query = self._normalize_query_text(seed)
            request_payload = {
                "seed_keyword": seed,
                "language_code": language_code,
                "location_code": int(location_code),
                "limit": max(1, min(int(limit or 25), 100)),
            }
        elif search_type_normalized == "keyword_overview":
            cleaned_keywords = []
            for keyword in keywords or []:
                cleaned = str(keyword or "").strip()
                if cleaned and cleaned.lower() not in {item.lower() for item in cleaned_keywords}:
                    cleaned_keywords.append(cleaned)
            if not cleaned_keywords:
                raise ValueError("keywords is required for keyword_overview searches")
            metric_result = await dataforseo_api.get_bulk_metrics_standard(
                cleaned_keywords,
                language_code=language_code,
                location_code=int(location_code),
                return_raw=True,
            )
            kd_result = await dataforseo_api.get_keyword_difficulty(
                cleaned_keywords[:150],
                language_code=language_code,
                location_code=int(location_code),
                return_raw=True,
            )
            metric_items = (metric_result or {}).get("items") or []
            kd_items = (kd_result or {}).get("items") or []
            response_payload = self._sanitize_for_json(
                {
                    "metrics": metric_result,
                    "keyword_difficulty": kd_result,
                }
            )
            result_summary_json = self._summarize_keyword_overview(metric_items, kd_items)
            endpoint = "keywords_data/google_ads/search_volume + dataforseo_labs/google/bulk_keyword_difficulty/live"
            normalized_query = self._normalize_query_text(", ".join(cleaned_keywords[:10]))
            request_payload = {
                "keywords": cleaned_keywords,
                "language_code": language_code,
                "location_code": int(location_code),
            }
            query_text = ", ".join(cleaned_keywords)
        elif search_type_normalized == "serp":
            seed = str(query_text or "").strip()
            if not seed:
                raise ValueError("query_text is required for serp searches")
            items = await dataforseo_api.get_serp_standard(
                seed,
                language_code=language_code,
                location_code=int(location_code),
                depth=max(10, min(int(limit or 10), 20)),
            )
            response_payload = self._sanitize_for_json(
                {
                    "items": items,
                    "request": {
                        "keyword": seed,
                        "language_code": language_code,
                        "location_code": int(location_code),
                        "depth": max(10, min(int(limit or 10), 20)),
                    },
                }
            )
            result_summary_json = self._summarize_serp(seed, items)
            endpoint = "serp/google/organic/task_post"
            normalized_query = self._normalize_query_text(seed)
            request_payload = {
                "keyword": seed,
                "language_code": language_code,
                "location_code": int(location_code),
                "depth": max(10, min(int(limit or 10), 20)),
            }
        else:
            raise ValueError("search_type must be one of related_keywords, keyword_overview, serp")

        item = await self.create_record(
            user_id=user_id,
            data={
                "project_id": str(project_id),
                "user_job_id": str(user_job_id) if user_job_id else None,
                "primary_category_id": primary_category_id,
                "secondary_category_id": secondary_category_id,
                "search_type": search_type_normalized,
                "endpoint": endpoint,
                "query_text": str(query_text or "").strip(),
                "normalized_query_text": normalized_query,
                "request_payload": self._sanitize_for_json(request_payload),
                "response_payload": response_payload,
                "result_summary_json": self._sanitize_for_json(result_summary_json),
                "searched_at": now_iso,
                "updated_at": now_iso,
            },
        )
        if not item:
            raise ValueError("Failed to persist DataForSEO search")
        return item

    async def list_searches(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        user_job_id: Optional[UUID] = None,
        search_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {"project_id": str(project_id)}
        if user_job_id:
            filters["user_job_id"] = str(user_job_id)
        if search_type:
            filters["search_type"] = str(search_type).strip().lower()
        return await self.list_records(
            user_id=user_id,
            filters=filters,
            order_by={"searched_at": "desc"},
            limit=max(1, min(int(limit or 20), 100)),
        )
