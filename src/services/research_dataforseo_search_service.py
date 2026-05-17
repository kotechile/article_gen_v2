"""
Manual DataForSEO lookup persistence for the research rebuild.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from datetime import timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from src.integrations.dataforseo import dataforseo_api

from .research_rebuild_base_service import ResearchRebuildBaseService
from .supabase_service import SupabaseService


class ResearchDataforseoSearchService(ResearchRebuildBaseService):
    """Persist and execute user-driven DataForSEO lookups."""

    table_name = "research_dataforseo_searches"
    NON_BLOCKING_SEARCH_TYPES = {"categories_for_domain", "category_index"}

    def __init__(self, supabase_service: Optional[SupabaseService] = None):
        super().__init__(supabase_service=supabase_service)

    @staticmethod
    def _normalize_query_text(value: Any) -> str:
        return " ".join(str(value or "").strip().lower().split())

    @classmethod
    def _build_cache_key(
        cls,
        *,
        search_type: str,
        query_text: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        target: Optional[str] = None,
        language_code: str = "en",
        location_code: int = 2840,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        parts = [str(search_type or "").strip().lower(), language_code.strip().lower(), str(location_code)]
        if query_text:
            parts.append(cls._normalize_query_text(query_text))
        if keywords:
            cleaned = sorted({cls._normalize_query_text(item) for item in keywords if str(item or "").strip()})
            parts.extend(cleaned)
        if target:
            parts.append(cls._normalize_query_text(target))
        if extra:
            for key in sorted(extra.keys()):
                parts.append(f"{key}:{cls._normalize_query_text(extra[key])}")
        return "|".join(part for part in parts if part)

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

    @staticmethod
    def _summarize_google_trends(items: List[Dict[str, Any]], keywords: List[str]) -> Dict[str, Any]:
        top_items = []
        for row in items or []:
            for item in row.get("items", []) or []:
                keyword = item.get("keyword") or item.get("title") or item.get("term")
                if keyword:
                    top_items.append(item)
        return {
            "query_text": ", ".join(keywords[:5]),
            "result_count": len(top_items),
            "top_items": top_items[:20],
        }

    @staticmethod
    def _summarize_ranked_keywords(items: List[Dict[str, Any]], target: str) -> Dict[str, Any]:
        return {
            "query_text": target,
            "result_count": len(items or []),
            "top_items": (items or [])[:100],
        }

    @staticmethod
    def _summarize_relevant_pages(items: List[Dict[str, Any]], query_text: str) -> Dict[str, Any]:
        return {
            "query_text": query_text,
            "result_count": len(items or []),
            "top_items": (items or [])[:50],
        }

    @staticmethod
    def _summarize_categories_for_domain(items: List[Dict[str, Any]], target: str) -> Dict[str, Any]:
        return {
            "query_text": target,
            "result_count": len(items or []),
            "top_items": (items or [])[:20],
        }

    @staticmethod
    def _summarize_category_index(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "query_text": "dataforseo_labs/categories",
            "result_count": len(items or []),
            "top_items": (items or [])[:100],
        }

    async def find_cached_search(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        search_type: str,
        normalized_query_text: str,
        ttl_days: int,
    ) -> Optional[Dict[str, Any]]:
        rows = await self.list_records(
            user_id=user_id,
            filters={
                "project_id": str(project_id),
                "search_type": str(search_type or "").strip().lower(),
                "normalized_query_text": normalized_query_text,
            },
            order_by={"searched_at": "desc"},
            limit=1,
        )
        if not rows:
            return None
        row = rows[0]
        searched_at = row.get("searched_at")
        if not searched_at:
            return None
        try:
            searched_dt = datetime.fromisoformat(str(searched_at).replace("Z", "+00:00"))
            if searched_dt >= datetime.now(timezone.utc) - timedelta(days=max(1, ttl_days)):
                return row
        except Exception:
            return None
        return None

    async def run_search(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        search_type: str,
        query_text: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        target: Optional[str] = None,
        user_job_id: Optional[UUID] = None,
        primary_category_id: Optional[str] = None,
        secondary_category_id: Optional[str] = None,
        language_code: str = "en",
        location_code: int = 2840,
        limit: int = 25,
        force_refresh: bool = False,
        cache_ttl_days: int = 30,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        search_type_normalized = str(search_type or "").strip().lower()
        extra = extra if isinstance(extra, dict) else {}
        now_iso = datetime.now(timezone.utc).isoformat()
        normalized_query = self._build_cache_key(
            search_type=search_type_normalized,
            query_text=query_text,
            keywords=keywords,
            target=target,
            language_code=language_code,
            location_code=location_code,
            extra=extra,
        )

        if not force_refresh:
            cached = await self.find_cached_search(
                user_id=user_id,
                project_id=project_id,
                search_type=search_type_normalized,
                normalized_query_text=normalized_query,
                ttl_days=cache_ttl_days,
            )
            if cached:
                return cached

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
            request_payload = {
                "keyword": seed,
                "language_code": language_code,
                "location_code": int(location_code),
                "depth": max(10, min(int(limit or 10), 20)),
            }
        elif search_type_normalized == "google_trends":
            cleaned_keywords = [str(keyword or "").strip() for keyword in (keywords or []) if str(keyword or "").strip()]
            if not cleaned_keywords:
                raise ValueError("keywords is required for google_trends searches")
            trend_result = await dataforseo_api.get_google_trends_explore_live(
                cleaned_keywords[:5],
                language_code=language_code,
                location_code=int(location_code),
                return_raw=True,
            )
            items = (trend_result or {}).get("items") or []
            response_payload = self._sanitize_for_json((trend_result or {}).get("raw") or trend_result)
            result_summary_json = self._summarize_google_trends(items, cleaned_keywords)
            endpoint = "keywords_data/google_trends/explore/live"
            request_payload = {
                "keywords": cleaned_keywords[:5],
                "language_code": language_code,
                "location_code": int(location_code),
            }
            query_text = ", ".join(cleaned_keywords[:5])
        elif search_type_normalized == "serp_probe":
            seed = str(query_text or "").strip()
            if not seed:
                raise ValueError("query_text is required for serp_probe searches")
            serp_result = await dataforseo_api.get_google_organic_live_advanced(
                seed,
                language_code=language_code,
                location_code=int(location_code),
                depth=min(int(limit or 10), 10),
                return_raw=True,
            )
            items = (serp_result or {}).get("items") or []
            response_payload = self._sanitize_for_json((serp_result or {}).get("raw") or serp_result)
            result_summary_json = self._summarize_serp(seed, items)
            endpoint = "serp/google/organic/live/advanced"
            request_payload = {
                "keyword": seed,
                "language_code": language_code,
                "location_code": int(location_code),
                "depth": min(int(limit or 10), 10),
            }
        elif search_type_normalized == "ranked_keywords":
            page_target = str(target or query_text or "").strip()
            if not page_target:
                raise ValueError("target is required for ranked_keywords searches")
            ranked_result = await dataforseo_api.get_ranked_keywords_live(
                page_target,
                language_code=language_code,
                location_code=int(location_code),
                limit=max(1, min(int(limit or 100), 100)),
                offset=max(0, int(extra.get("offset") or 0)),
                ignore_synonyms=bool(extra.get("ignore_synonyms", True)),
                item_types=extra.get("item_types") if isinstance(extra.get("item_types"), list) else ["organic"],
                historical_serp_mode=str(extra.get("historical_serp_mode") or "live"),
                filters=extra.get("filters") if isinstance(extra.get("filters"), list) else None,
                order_by=extra.get("order_by") if isinstance(extra.get("order_by"), list) else None,
                return_raw=True,
            )
            items = (ranked_result or {}).get("items") or []
            response_payload = self._sanitize_for_json((ranked_result or {}).get("raw") or ranked_result)
            result_summary_json = self._summarize_ranked_keywords(items, page_target)
            endpoint = "dataforseo_labs/google/ranked_keywords/live"
            request_payload = {
                "target": page_target,
                "language_code": language_code,
                "location_code": int(location_code),
                "limit": max(1, min(int(limit or 100), 100)),
                "offset": max(0, int(extra.get("offset") or 0)),
                "ignore_synonyms": bool(extra.get("ignore_synonyms", True)),
                "historical_serp_mode": str(extra.get("historical_serp_mode") or "live"),
                "item_types": extra.get("item_types") if isinstance(extra.get("item_types"), list) else ["organic"],
                "filters": extra.get("filters") if isinstance(extra.get("filters"), list) else None,
                "order_by": extra.get("order_by") if isinstance(extra.get("order_by"), list) else None,
            }
            query_text = page_target
        elif search_type_normalized == "relevant_pages":
            page_target = str(target or query_text or "").strip()
            if not page_target:
                raise ValueError("target is required for relevant_pages searches")
            relevant_result = await dataforseo_api.get_relevant_pages_live(
                page_target,
                language_code=language_code,
                location_code=int(location_code),
                limit=max(1, min(int(limit or 20), 1000)),
                offset=max(0, int(extra.get("offset") or 0)),
                item_types=extra.get("item_types") if isinstance(extra.get("item_types"), list) else ["organic"],
                historical_serp_mode=str(extra.get("historical_serp_mode") or "live"),
                ignore_synonyms=bool(extra.get("ignore_synonyms", False)),
                filters=extra.get("filters") if isinstance(extra.get("filters"), list) else None,
                order_by=extra.get("order_by") if isinstance(extra.get("order_by"), list) else None,
                return_raw=True,
            )
            items = (relevant_result or {}).get("items") or []
            response_payload = self._sanitize_for_json((relevant_result or {}).get("raw") or relevant_result)
            result_summary_json = self._summarize_relevant_pages(items, page_target)
            endpoint = "dataforseo_labs/google/relevant_pages/live"
            request_payload = {
                "target": page_target,
                "language_code": language_code,
                "location_code": int(location_code),
                "limit": max(1, min(int(limit or 20), 1000)),
                "offset": max(0, int(extra.get("offset") or 0)),
                "historical_serp_mode": str(extra.get("historical_serp_mode") or "live"),
                "ignore_synonyms": bool(extra.get("ignore_synonyms", False)),
                "item_types": extra.get("item_types") if isinstance(extra.get("item_types"), list) else ["organic"],
                "filters": extra.get("filters") if isinstance(extra.get("filters"), list) else None,
                "order_by": extra.get("order_by") if isinstance(extra.get("order_by"), list) else None,
            }
            query_text = page_target
        elif search_type_normalized == "categories_for_domain":
            page_target = str(target or query_text or "").strip()
            if not page_target:
                raise ValueError("target is required for categories_for_domain searches")
            categories_result = await dataforseo_api.get_categories_for_domain_live(
                page_target,
                language_code=language_code,
                location_code=int(location_code),
                limit=max(1, min(int(limit or 20), 1000)),
                include_subcategories=bool(extra.get("include_subcategories", False)),
                item_types=extra.get("item_types") if isinstance(extra.get("item_types"), list) else ["organic"],
                filters=extra.get("filters") if isinstance(extra.get("filters"), list) else None,
                order_by=extra.get("order_by") if isinstance(extra.get("order_by"), list) else None,
                return_raw=True,
            )
            items = (categories_result or {}).get("items") or []
            response_payload = self._sanitize_for_json((categories_result or {}).get("raw") or categories_result)
            result_summary_json = self._summarize_categories_for_domain(items, page_target)
            endpoint = "dataforseo_labs/google/categories_for_domain/live"
            request_payload = {
                "target": page_target,
                "language_code": language_code,
                "location_code": int(location_code),
                "limit": max(1, min(int(limit or 20), 1000)),
                "include_subcategories": bool(extra.get("include_subcategories", False)),
                "item_types": extra.get("item_types") if isinstance(extra.get("item_types"), list) else ["organic"],
                "filters": extra.get("filters") if isinstance(extra.get("filters"), list) else None,
                "order_by": extra.get("order_by") if isinstance(extra.get("order_by"), list) else None,
            }
            query_text = page_target
        elif search_type_normalized == "category_index":
            category_index_result = await dataforseo_api.get_labs_categories(return_raw=True)
            items = (category_index_result or {}).get("items") or []
            response_payload = self._sanitize_for_json((category_index_result or {}).get("raw") or category_index_result)
            result_summary_json = self._summarize_category_index(items)
            endpoint = "dataforseo_labs/categories"
            request_payload = {}
            query_text = "dataforseo_labs/categories"
        else:
            raise ValueError("search_type must be one of related_keywords, keyword_overview, serp, google_trends, serp_probe, ranked_keywords, relevant_pages, categories_for_domain, category_index")

        payload = {
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
        }

        try:
            item = await self.create_record(
                user_id=user_id,
                data=payload,
            )
            if not item:
                raise ValueError("Failed to persist DataForSEO search")
            return item
        except Exception:
            if search_type_normalized not in self.NON_BLOCKING_SEARCH_TYPES:
                raise

            # Domain-fit helpers should still return usable results even when the
            # production DB has not yet been migrated to allow the new search types.
            return {
                "id": None,
                "user_id": str(user_id),
                **payload,
                "created_at": now_iso,
                "persistence_state": "ephemeral_unpersisted",
            }

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
