"""
Topic-level keyword research service for the article ideas revamp.

This service is intentionally independent from the legacy subtopics-first flow.
It builds a keyword research run from a single research topic, persists candidate
keywords and clusters, and exposes read helpers for the API layer.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.integrations.dataforseo import dataforseo_api
from src.services.llm.llm_service import llm_service
from supabase_client import LLM_ROLE_RESEARCH

logger = logging.getLogger(__name__)


class TopicKeywordResearchService:
    """Orchestrates topic-level keyword research runs and persistence."""

    QUERY_STOPWORDS = {
        "a", "an", "the", "to", "for", "of", "in", "on", "at", "with", "without",
        "from", "into", "by", "my", "your", "our", "their", "you", "is", "are", "be",
        "have", "has", "had", "too", "much", "more", "less", "first", "second", "third",
        "what", "how", "when", "why", "can", "should", "could", "would", "do", "does",
        "did", "and", "or", "vs", "versus", "if", "does", "will", "would", "increase",
        "improve", "improves", "using", "use", "best", "guide",
    }

    FILTER_STOP_TERMS = {
        "jobs", "job", "career", "salary", "pdf", "near me", "nearby", "hiring",
    }

    SEED_NOISE_PATTERNS = (
        "building on existing content",
        "existing content about",
        "homeowners and property buyers",
        "target audience",
        "site description",
    )

    SOFTWARE_SIGNAL_TERMS = {
        "calculator", "tool", "template", "checker", "planner", "estimator",
        "generator", "audit", "tracker", "scorecard", "comparison", "compare",
        "cost", "roi", "pricing",
    }

    DEFAULT_FILTERS = {
        "min_search_volume": 10,
        "max_keyword_difficulty": 60,
        "min_competition_index": 15,
        "max_candidates_to_enrich": 250,
        "max_clusters": 12,
    }

    DEFAULT_SCORE_CONFIG = {
        "kd_weight": 0.35,
        "competition_weight": 0.25,
        "volume_weight": 0.15,
        "cpc_weight": 0.10,
        "trend_weight": 0.10,
        "fit_weight": 0.05,
    }

    def __init__(self, supabase, supabase_admin):
        self.supabase = supabase
        self.supabase_admin = supabase_admin or supabase

    async def run_topic_research(
        self,
        topic_id: str,
        user_id: str,
        replace_existing: bool = False,
        filters: Optional[Dict[str, Any]] = None,
        score_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        filters = {**self.DEFAULT_FILTERS, **(filters or {})}
        score_config = {**self.DEFAULT_SCORE_CONFIG, **(score_config or {})}
        topic_context = self._load_topic_context(topic_id=topic_id, user_id=user_id)
        if not topic_context:
            raise ValueError("Research topic not found")

        if replace_existing:
            self.delete_topic_research(topic_id=topic_id, user_id=user_id)

        seed_package = await self._build_seed_keywords(topic_context)
        seeds = seed_package["seed_keywords"]
        run_row = self._create_run(
            topic_id=topic_id,
            user_id=user_id,
            seed_keywords=seeds,
            filters=filters,
            score_config=score_config,
            topic_context=topic_context,
            seed_package=seed_package,
        )
        run_id = run_row["id"]

        try:
            discovered_rows, raw_data = await self._discover_keyword_candidates(seeds=seeds)
            enriched_rows = await self._enrich_and_score_candidates(
                candidates=discovered_rows,
                topic_context=topic_context,
                filters=filters,
                score_config=score_config,
            )
            clusters = self._cluster_candidates(
                candidates=enriched_rows,
                topic_context=topic_context,
                filters=filters,
            )

            self._replace_candidates(run_id=run_id, topic_id=topic_id, user_id=user_id, rows=enriched_rows)
            self._replace_clusters(run_id=run_id, topic_id=topic_id, user_id=user_id, rows=clusters)

            summary = self._build_run_summary(
                topic_context=topic_context,
                seed_keywords=seeds,
                candidate_rows=enriched_rows,
                clusters=clusters,
                seed_package=seed_package,
            )
            self._update_run(
                run_id=run_id,
                topic_id=topic_id,
                user_id=user_id,
                status="completed",
                summary_json=summary,
                raw_data_json={**raw_data, "seed_generation": seed_package},
                error_message=None,
            )
            run = self.get_run(run_id=run_id, topic_id=topic_id, user_id=user_id)
            return {
                "run": run,
                "keywords": enriched_rows,
                "clusters": clusters,
                "summary": summary,
            }
        except Exception as err:
            logger.error(
                "Topic keyword research failed topic_id=%s user_id=%s err=%s",
                topic_id,
                user_id,
                err,
                exc_info=True,
            )
            self._update_run(
                run_id=run_id,
                topic_id=topic_id,
                user_id=user_id,
                status="failed",
                summary_json={
                    "pipeline_version": "topic_keyword_pipeline_v1",
                    "status": "failed",
                },
                raw_data_json={"seed_keywords": seeds, "seed_generation": seed_package},
                error_message=str(err),
            )
            raise

    def get_latest_run(self, topic_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        response = (
            self.supabase
            .table("topic_keyword_research_runs")
            .select("*")
            .eq("topic_id", topic_id)
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else None

    def get_run(self, run_id: str, topic_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        response = (
            self.supabase
            .table("topic_keyword_research_runs")
            .select("*")
            .eq("id", run_id)
            .eq("topic_id", topic_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else None

    def list_keywords(self, run_id: str, topic_id: str, user_id: str, include_filtered: bool = True) -> List[Dict[str, Any]]:
        query = (
            self.supabase
            .table("topic_keyword_candidates")
            .select("*")
            .eq("research_run_id", run_id)
            .eq("topic_id", topic_id)
            .eq("user_id", user_id)
            .order("opportunity_score", desc=True)
        )
        if not include_filtered:
            query = query.eq("is_filtered_out", False)
        response = query.execute()
        return response.data or []

    def list_clusters(self, run_id: str, topic_id: str, user_id: str) -> List[Dict[str, Any]]:
        response = (
            self.supabase
            .table("topic_keyword_clusters")
            .select("*")
            .eq("research_run_id", run_id)
            .eq("topic_id", topic_id)
            .eq("user_id", user_id)
            .order("opportunity_score", desc=True)
            .execute()
        )
        return response.data or []

    def delete_topic_research(self, topic_id: str, user_id: str) -> Dict[str, int]:
        latest_runs = (
            self.supabase
            .table("topic_keyword_research_runs")
            .select("id")
            .eq("topic_id", topic_id)
            .eq("user_id", user_id)
            .execute()
        )
        run_ids = [row.get("id") for row in (latest_runs.data or []) if row.get("id")]

        deleted_clusters = 0
        deleted_keywords = 0
        deleted_runs = 0

        if run_ids:
            try:
                clusters_res = (
                    self.supabase_admin
                    .table("topic_keyword_clusters")
                    .delete()
                    .eq("topic_id", topic_id)
                    .eq("user_id", user_id)
                    .execute()
                )
                deleted_clusters = len(clusters_res.data or [])
            except Exception:
                logger.warning("Failed deleting topic keyword clusters topic_id=%s", topic_id, exc_info=True)

            try:
                keywords_res = (
                    self.supabase_admin
                    .table("topic_keyword_candidates")
                    .delete()
                    .eq("topic_id", topic_id)
                    .eq("user_id", user_id)
                    .execute()
                )
                deleted_keywords = len(keywords_res.data or [])
            except Exception:
                logger.warning("Failed deleting topic keyword candidates topic_id=%s", topic_id, exc_info=True)

            try:
                runs_res = (
                    self.supabase_admin
                    .table("topic_keyword_research_runs")
                    .delete()
                    .eq("topic_id", topic_id)
                    .eq("user_id", user_id)
                    .execute()
                )
                deleted_runs = len(runs_res.data or [])
            except Exception:
                logger.warning("Failed deleting topic keyword runs topic_id=%s", topic_id, exc_info=True)

        return {
            "deleted_runs": deleted_runs,
            "deleted_keywords": deleted_keywords,
            "deleted_clusters": deleted_clusters,
        }

    def _load_topic_context(self, topic_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        topic_res = (
            self.supabase
            .table("research_topics")
            .select(
                "id, title, description, project_id, primary_category_id, secondary_category_id, "
                "intent_bucket, decision_focus, angle_question, value_layer_tags, target_audience"
            )
            .eq("id", topic_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        topic_rows = topic_res.data or []
        if not topic_rows:
            return None

        topic = topic_rows[0]
        project = {}
        category_map: Dict[str, Dict[str, Any]] = {}

        project_id = topic.get("project_id")
        if project_id:
            try:
                project_res = (
                    self.supabase
                    .table("projects")
                    .select("id, domain, app_name, site_description, websitedescription, targetaudiencedescription")
                    .eq("id", project_id)
                    .limit(1)
                    .execute()
                )
                project_rows = project_res.data or []
                project = project_rows[0] if project_rows else {}
            except Exception:
                logger.warning("Failed loading project context topic_id=%s", topic_id, exc_info=True)

        category_ids = [cid for cid in [topic.get("primary_category_id"), topic.get("secondary_category_id")] if cid]
        if category_ids:
            try:
                cat_res = (
                    self.supabase
                    .table("project_categories")
                    .select("id, name, description")
                    .in_("id", category_ids)
                    .execute()
                )
                category_map = {row["id"]: row for row in (cat_res.data or []) if row.get("id")}
            except Exception:
                logger.warning("Failed loading category context topic_id=%s", topic_id, exc_info=True)

        primary = category_map.get(topic.get("primary_category_id")) or {}
        secondary = category_map.get(topic.get("secondary_category_id")) or {}
        category_path = " / ".join([part for part in [primary.get("name"), secondary.get("name")] if part]) or None

        return {
            "topic": topic,
            "project": project,
            "primary_category": primary,
            "secondary_category": secondary,
            "category_path": category_path,
        }

    def _create_run(
        self,
        topic_id: str,
        user_id: str,
        seed_keywords: List[str],
        filters: Dict[str, Any],
        score_config: Dict[str, Any],
        topic_context: Dict[str, Any],
        seed_package: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "topic_id": topic_id,
            "user_id": user_id,
            "status": "running",
            "seed_keywords_json": seed_keywords,
            "filters_json": filters,
            "score_config_json": score_config,
            "summary_json": {
                "pipeline_version": "topic_keyword_pipeline_v1",
                "status": "running",
                "topic_title": (topic_context.get("topic") or {}).get("title"),
            },
            "raw_data_json": {
                "topic_context": self._sanitize_for_json(topic_context),
                "seed_keywords": seed_keywords,
                "seed_generation": self._sanitize_for_json(seed_package or {}),
            },
            "updated_at": datetime.utcnow().isoformat(),
        }
        res = self.supabase_admin.table("topic_keyword_research_runs").insert(payload).execute()
        rows = res.data or []
        if not rows:
            raise RuntimeError("Failed to create topic keyword research run")
        return rows[0]

    def _update_run(
        self,
        run_id: str,
        topic_id: str,
        user_id: str,
        status: str,
        summary_json: Dict[str, Any],
        raw_data_json: Dict[str, Any],
        error_message: Optional[str],
    ) -> None:
        payload = {
            "status": status,
            "summary_json": self._sanitize_for_json(summary_json),
            "raw_data_json": self._sanitize_for_json(raw_data_json),
            "error_message": error_message,
            "updated_at": datetime.utcnow().isoformat(),
        }
        (
            self.supabase_admin
            .table("topic_keyword_research_runs")
            .update(payload)
            .eq("id", run_id)
            .eq("topic_id", topic_id)
            .eq("user_id", user_id)
            .execute()
        )

    async def _discover_keyword_candidates(self, seeds: List[str]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        primary_seeds = seeds[:4]
        expansion_seeds = seeds[:8]
        direct_metric_seeds = seeds[:30]

        raw_data: Dict[str, Any] = {
            "seed_keywords": seeds,
            "pipeline_version": "topic_keyword_pipeline_v1",
            "sources": {},
        }

        related_live = await dataforseo_api.get_related_keywords_labs_live(
            primary_seeds,
            limit_per_seed=25,
            depth=1,
            return_raw=True,
        )
        related_standard = await dataforseo_api.get_related_keywords_standard(
            expansion_seeds,
            limit_per_seed=20,
            return_raw=True,
        )
        direct_metrics = await dataforseo_api.get_bulk_metrics_standard(
            direct_metric_seeds,
            return_raw=True,
        )

        raw_data["sources"]["related_keywords_live"] = self._sanitize_for_json((related_live or {}).get("raw"))
        raw_data["sources"]["keywords_for_keywords_standard"] = self._sanitize_for_json((related_standard or {}).get("raw"))
        raw_data["sources"]["seed_metrics"] = self._sanitize_for_json((direct_metrics or {}).get("raw"))

        candidate_rows: List[Dict[str, Any]] = []

        for row in (related_live or {}).get("items") or []:
            candidate_rows.append({
                **self._normalize_candidate_row(row),
                "source_endpoints": ["dataforseo_labs/google/related_keywords/live"],
            })

        for row in (related_standard or {}).get("items") or []:
            candidate_rows.append({
                **self._normalize_candidate_row(row),
                "source_endpoints": ["keywords_data/google_ads/keywords_for_keywords"],
            })

        for row in (direct_metrics or {}).get("items") or []:
            candidate_rows.append({
                **self._normalize_candidate_row(row),
                "source_endpoints": ["keywords_data/google_ads/search_volume"],
            })

        for seed in seeds:
            normalized = self._normalize_candidate_row({"keyword": seed})
            if normalized.get("keyword"):
                normalized["source_endpoints"] = ["seed_keywords"]
                candidate_rows.append(normalized)

        return candidate_rows, raw_data

    async def _enrich_and_score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        topic_context: Dict[str, Any],
        filters: Dict[str, Any],
        score_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for row in candidates or []:
            keyword = str(row.get("keyword") or "").strip()
            canonical = self._normalize_keyword_key(keyword)
            if not canonical:
                continue
            existing = merged.get(canonical)
            if not existing:
                merged[canonical] = {
                    "keyword": keyword,
                    "canonical_keyword": canonical,
                    "variant_keywords": [keyword],
                    "source_endpoints": list(row.get("source_endpoints") or []),
                    "search_volume": row.get("search_volume"),
                    "cpc": row.get("cpc"),
                    "competition": row.get("competition"),
                    "competition_index": self._coerce_competition_index(row),
                    "keyword_difficulty": row.get("keyword_difficulty"),
                }
                continue

            if keyword and keyword not in existing["variant_keywords"]:
                existing["variant_keywords"].append(keyword)
            existing["source_endpoints"] = self._merge_strings(existing.get("source_endpoints"), row.get("source_endpoints"))
            existing["search_volume"] = self._prefer_number(existing.get("search_volume"), row.get("search_volume"))
            existing["cpc"] = self._prefer_number(existing.get("cpc"), row.get("cpc"))
            existing["keyword_difficulty"] = self._prefer_number(existing.get("keyword_difficulty"), row.get("keyword_difficulty"))
            existing["competition"] = existing.get("competition") or row.get("competition")
            existing["competition_index"] = max(
                int(existing.get("competition_index") or 0),
                int(self._coerce_competition_index(row) or 0),
            ) or None

        candidate_keywords = list(merged.keys())[: int(filters.get("max_candidates_to_enrich") or 250)]
        bulk_metrics = await dataforseo_api.get_bulk_metrics_standard(candidate_keywords, return_raw=False)
        kd_metrics = await dataforseo_api.get_keyword_difficulty(candidate_keywords[:150], return_raw=False)

        bulk_map = {
            self._normalize_keyword_key(row.get("keyword")): row
            for row in (bulk_metrics or [])
            if self._normalize_keyword_key(row.get("keyword"))
        }
        kd_map = {
            self._normalize_keyword_key(row.get("keyword")): row
            for row in (kd_metrics or [])
            if self._normalize_keyword_key(row.get("keyword"))
        }

        scored_rows: List[Dict[str, Any]] = []
        for canonical, row in merged.items():
            metrics = bulk_map.get(canonical) or {}
            kd_row = kd_map.get(canonical) or {}

            row["search_volume"] = self._prefer_number(row.get("search_volume"), metrics.get("search_volume"))
            row["cpc"] = self._prefer_number(row.get("cpc"), metrics.get("cpc"))
            row["keyword_difficulty"] = self._prefer_number(row.get("keyword_difficulty"), kd_row.get("keyword_difficulty"))
            row["competition"] = row.get("competition") or metrics.get("competition") or kd_row.get("competition")
            row["competition_index"] = max(
                int(row.get("competition_index") or 0),
                int(self._competition_index_from_values(metrics.get("competition"), metrics.get("competition_level")) or 0),
                int(self._competition_index_from_values(kd_row.get("competition"), kd_row.get("competition_level")) or 0),
            ) or None
            row["intent_label"] = self._infer_intent_label(row.get("keyword") or canonical)
            row["topical_fit_score"] = self._topical_fit_score(canonical, topic_context)
            row["trend_json"] = {
                "validation_mode": "not_fetched_yet",
                "trend_score": 50,
            }

            is_filtered_out, filter_reason = self._filter_candidate_row(row=row, filters=filters)
            row["is_filtered_out"] = is_filtered_out
            row["filter_reason"] = filter_reason
            row["opportunity_score"] = self._compute_keyword_opportunity_score(
                row=row,
                score_config=score_config,
            )
            scored_rows.append(row)

        scored_rows.sort(key=lambda item: (item.get("opportunity_score") or 0), reverse=True)
        return scored_rows

    def _cluster_candidates(
        self,
        candidates: List[Dict[str, Any]],
        topic_context: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        usable = [row for row in (candidates or []) if not row.get("is_filtered_out")]
        usable.sort(key=lambda item: (item.get("opportunity_score") or 0), reverse=True)

        clusters: List[Dict[str, Any]] = []
        for row in usable:
            tokens = self._token_set(row.get("canonical_keyword") or row.get("keyword") or "")
            if not tokens:
                continue

            assigned_cluster = None
            best_overlap = 0.0
            for cluster in clusters:
                overlap = self._cluster_overlap(tokens, cluster["token_set"])
                if overlap >= 0.55 and overlap > best_overlap:
                    best_overlap = overlap
                    assigned_cluster = cluster

            if assigned_cluster is None:
                assigned_cluster = {
                    "token_set": set(tokens),
                    "rows": [],
                }
                clusters.append(assigned_cluster)
            else:
                assigned_cluster["token_set"].update(tokens)

            assigned_cluster["rows"].append(row)

        materialized_clusters: List[Dict[str, Any]] = []
        max_clusters = int(filters.get("max_clusters") or 12)
        for cluster in clusters[:max_clusters]:
            rows = sorted(cluster["rows"], key=lambda item: (item.get("opportunity_score") or 0), reverse=True)
            primary = rows[0]
            primary_keyword = primary.get("keyword") or primary.get("canonical_keyword")
            secondary_keywords = []
            for row in rows[1:6]:
                keyword = row.get("keyword") or row.get("canonical_keyword")
                if keyword and keyword not in secondary_keywords and keyword != primary_keyword:
                    secondary_keywords.append(keyword)

            cluster_name = self._build_cluster_name(primary_keyword or "", rows)
            intent_label = self._resolve_cluster_intent(rows)
            opportunity_score = round(sum(float(item.get("opportunity_score") or 0.0) for item in rows[:3]) / max(1, min(3, len(rows))), 2)
            software_opportunity_score = self._software_opportunity_score(rows)
            article_angle = self._build_article_angle(primary_keyword or "", intent_label)
            serp_validation_json = self._build_serp_validation_summary(primary_keyword or "", intent_label, rows)

            materialized_clusters.append({
                "cluster_name": cluster_name,
                "primary_keyword": primary_keyword,
                "secondary_keywords_json": secondary_keywords,
                "keyword_candidates_json": [
                    {
                        "keyword": row.get("keyword"),
                        "canonical_keyword": row.get("canonical_keyword"),
                        "opportunity_score": row.get("opportunity_score"),
                        "search_volume": row.get("search_volume"),
                        "keyword_difficulty": row.get("keyword_difficulty"),
                        "cpc": row.get("cpc"),
                    }
                    for row in rows[:8]
                ],
                "intent_label": intent_label,
                "serp_validation_json": serp_validation_json,
                "opportunity_score": opportunity_score,
                "software_opportunity_score": software_opportunity_score,
                "article_angle": article_angle,
                "rationale": self._build_cluster_rationale(primary_keyword or "", rows, topic_context),
            })

        materialized_clusters.sort(key=lambda item: (item.get("opportunity_score") or 0), reverse=True)
        return materialized_clusters

    def _replace_candidates(self, run_id: str, topic_id: str, user_id: str, rows: List[Dict[str, Any]]) -> None:
        try:
            (
                self.supabase_admin
                .table("topic_keyword_candidates")
                .delete()
                .eq("research_run_id", run_id)
                .eq("topic_id", topic_id)
                .eq("user_id", user_id)
                .execute()
            )
        except Exception:
            logger.warning("Failed clearing candidate rows run_id=%s", run_id, exc_info=True)

        payload = []
        now_iso = datetime.utcnow().isoformat()
        for row in rows:
            payload.append({
                "research_run_id": run_id,
                "topic_id": topic_id,
                "user_id": user_id,
                "keyword": row.get("keyword"),
                "canonical_keyword": row.get("canonical_keyword"),
                "variant_keywords_json": row.get("variant_keywords") or [],
                "source_endpoints_json": row.get("source_endpoints") or [],
                "search_volume": int(row.get("search_volume")) if row.get("search_volume") is not None else None,
                "cpc": round(float(row.get("cpc")), 2) if row.get("cpc") is not None else None,
                "competition": row.get("competition"),
                "competition_index": int(row.get("competition_index")) if row.get("competition_index") is not None else None,
                "keyword_difficulty": round(float(row.get("keyword_difficulty")), 2) if row.get("keyword_difficulty") is not None else None,
                "trend_json": row.get("trend_json") or {},
                "intent_label": row.get("intent_label"),
                "topical_fit_score": round(float(row.get("topical_fit_score")), 2) if row.get("topical_fit_score") is not None else None,
                "opportunity_score": round(float(row.get("opportunity_score")), 2) if row.get("opportunity_score") is not None else None,
                "is_filtered_out": bool(row.get("is_filtered_out")),
                "filter_reason": row.get("filter_reason"),
                "updated_at": now_iso,
            })

        for chunk_start in range(0, len(payload), 200):
            chunk = payload[chunk_start:chunk_start + 200]
            if not chunk:
                continue
            self.supabase_admin.table("topic_keyword_candidates").insert(chunk).execute()

    def _replace_clusters(self, run_id: str, topic_id: str, user_id: str, rows: List[Dict[str, Any]]) -> None:
        try:
            (
                self.supabase_admin
                .table("topic_keyword_clusters")
                .delete()
                .eq("research_run_id", run_id)
                .eq("topic_id", topic_id)
                .eq("user_id", user_id)
                .execute()
            )
        except Exception:
            logger.warning("Failed clearing cluster rows run_id=%s", run_id, exc_info=True)

        payload = []
        now_iso = datetime.utcnow().isoformat()
        for row in rows:
            payload.append({
                "research_run_id": run_id,
                "topic_id": topic_id,
                "user_id": user_id,
                "cluster_name": row.get("cluster_name"),
                "primary_keyword": row.get("primary_keyword"),
                "secondary_keywords_json": row.get("secondary_keywords_json") or [],
                "keyword_candidates_json": row.get("keyword_candidates_json") or [],
                "intent_label": row.get("intent_label"),
                "serp_validation_json": row.get("serp_validation_json") or {},
                "opportunity_score": round(float(row.get("opportunity_score")), 2) if row.get("opportunity_score") is not None else None,
                "software_opportunity_score": round(float(row.get("software_opportunity_score")), 2) if row.get("software_opportunity_score") is not None else None,
                "article_angle": row.get("article_angle"),
                "rationale": row.get("rationale"),
                "updated_at": now_iso,
            })

        for chunk_start in range(0, len(payload), 100):
            chunk = payload[chunk_start:chunk_start + 100]
            if not chunk:
                continue
            self.supabase_admin.table("topic_keyword_clusters").insert(chunk).execute()

    async def _build_seed_keywords(self, topic_context: Dict[str, Any]) -> Dict[str, Any]:
        deterministic_seeds = self._build_deterministic_seed_keywords(topic_context)
        llm_seeds = await self._generate_llm_seed_keywords(topic_context, deterministic_seeds)
        if not llm_seeds:
            raise ValueError(
                "Topic keyword research could not generate usable LLM seed keywords for this topic. "
                "Please revise the topic context or provide manual seeds."
            )
        return {
            "generation_mode": "llm_only_v1",
            "seed_keywords": llm_seeds,
            "deterministic_seeds": deterministic_seeds,
            "llm_seeds": llm_seeds,
            "seed_sources": {seed: "llm" for seed in llm_seeds},
        }

    def _build_deterministic_seed_keywords(self, topic_context: Dict[str, Any]) -> List[str]:
        topic = topic_context.get("topic") or {}
        primary_category = topic_context.get("primary_category") or {}
        secondary_category = topic_context.get("secondary_category") or {}
        project = topic_context.get("project") or {}

        title = str(topic.get("title") or "").strip()
        description = str(topic.get("description") or "").strip()
        primary_name = str(primary_category.get("name") or "").strip()
        secondary_name = str(secondary_category.get("name") or "").strip()
        audience = str(topic.get("target_audience") or project.get("targetaudiencedescription") or "").strip()
        decision_focus = str(topic.get("decision_focus") or "").strip()
        angle_question = str(topic.get("angle_question") or "").strip()

        meaningful_title_tokens = self._meaningful_tokens(title)
        title_core = " ".join(meaningful_title_tokens[:4]).strip()
        title_head = " ".join(meaningful_title_tokens[:3]).strip()
        title_tail = " ".join(meaningful_title_tokens[-3:]).strip()
        decision_tokens = self._meaningful_tokens(decision_focus)
        angle_tokens = self._meaningful_tokens(angle_question)
        context_tokens = set(meaningful_title_tokens + decision_tokens + angle_tokens)

        decision_phrases = self._extract_ngrams(decision_tokens, min_size=2, max_size=4)
        angle_phrases = self._extract_ngrams(angle_tokens, min_size=2, max_size=4)
        category_candidates = [
            name for name in [primary_name, secondary_name]
            if self._category_matches_topic(name, context_tokens)
        ]
        audience_seed = audience if self._audience_matches_topic(audience, context_tokens) else ""

        candidates = [
            title,
            title_core,
            title_head,
            title_tail,
            self._queryish_fragment(description, max_words=5),
            self._queryish_fragment(decision_focus, max_words=5),
            self._queryish_fragment(angle_question, max_words=5),
            *category_candidates,
            *[
                f"{category_name} {title_head}".strip()
                for category_name in category_candidates
                if title_head
            ],
            *[
                f"{title_head} {category_name}".strip()
                for category_name in category_candidates
                if title_head
            ],
            f"{title_head} {title_tail}".strip(),
            audience_seed,
            f"{title_head} {audience_seed}".strip() if title_head and audience_seed else "",
        ]

        title_ngrams = self._extract_ngrams(meaningful_title_tokens, min_size=2, max_size=4)
        candidates.extend(title_ngrams)
        candidates.extend(decision_phrases)
        candidates.extend(angle_phrases)

        if len(category_candidates) == 2:
            candidates.append(f"{primary_name} {secondary_name}".strip())

        seen = set()
        seeds: List[str] = []
        for raw in candidates:
            candidate = self._clean_seed_phrase(raw)
            if not candidate:
                continue
            if not self._looks_like_useful_seed(candidate):
                continue
            normalized = candidate.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            seeds.append(candidate)
        return seeds[:20]

    def _build_run_summary(
        self,
        topic_context: Dict[str, Any],
        seed_keywords: List[str],
        candidate_rows: List[Dict[str, Any]],
        clusters: List[Dict[str, Any]],
        seed_package: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        filtered_out = [row for row in candidate_rows if row.get("is_filtered_out")]
        active = [row for row in candidate_rows if not row.get("is_filtered_out")]
        top_keywords = [
            {
                "keyword": row.get("keyword"),
                "opportunity_score": row.get("opportunity_score"),
                "search_volume": row.get("search_volume"),
                "keyword_difficulty": row.get("keyword_difficulty"),
                "cpc": row.get("cpc"),
            }
            for row in active[:10]
        ]
        top_clusters = [
            {
                "cluster_name": row.get("cluster_name"),
                "primary_keyword": row.get("primary_keyword"),
                "opportunity_score": row.get("opportunity_score"),
                "software_opportunity_score": row.get("software_opportunity_score"),
            }
            for row in clusters[:8]
        ]
        return {
            "pipeline_version": "topic_keyword_pipeline_v1",
            "status": "completed",
            "topic_title": ((topic_context.get("topic") or {}).get("title") or ""),
            "category_path": topic_context.get("category_path"),
            "seed_count": len(seed_keywords),
            "seed_generation_mode": (seed_package or {}).get("generation_mode") or "llm_only_v1",
            "llm_seed_count": len((seed_package or {}).get("llm_seeds") or []),
            "deterministic_seed_count": len((seed_package or {}).get("deterministic_seeds") or []),
            "candidate_count": len(candidate_rows),
            "active_candidate_count": len(active),
            "filtered_candidate_count": len(filtered_out),
            "cluster_count": len(clusters),
            "top_keywords": top_keywords,
            "top_clusters": top_clusters,
            "validation_mode": "heuristic_v1",
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _normalize_candidate_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "keyword": str(row.get("keyword") or "").strip(),
            "search_volume": self._safe_int(row.get("search_volume")),
            "cpc": self._safe_float(row.get("cpc")),
            "competition": row.get("competition"),
            "competition_level": self._safe_int(row.get("competition_level")),
            "keyword_difficulty": self._safe_float(row.get("keyword_difficulty")),
        }

    def _filter_candidate_row(self, row: Dict[str, Any], filters: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        keyword = (row.get("keyword") or "").lower().strip()
        canonical = row.get("canonical_keyword") or ""
        if not canonical or len(self._token_set(canonical)) < 2:
            return True, "not_specific_enough"

        for stop_term in self.FILTER_STOP_TERMS:
            if stop_term in keyword:
                return True, f"blocked_term:{stop_term}"

        search_volume = int(row.get("search_volume") or 0)
        keyword_difficulty = float(row.get("keyword_difficulty") or 0.0)
        competition_index = int(row.get("competition_index") or 0)

        if search_volume <= 0 and (row.get("cpc") or 0) <= 0 and keyword_difficulty <= 0:
            return True, "no_measurable_demand"
        if search_volume > 0 and search_volume < int(filters.get("min_search_volume") or 0):
            return True, "below_min_search_volume"
        if keyword_difficulty > float(filters.get("max_keyword_difficulty") or 999):
            return True, "above_max_keyword_difficulty"
        if competition_index > 0 and competition_index < int(filters.get("min_competition_index") or 0):
            return True, "below_min_competition"
        return False, None

    def _compute_keyword_opportunity_score(self, row: Dict[str, Any], score_config: Dict[str, Any]) -> float:
        search_volume = max(0, int(row.get("search_volume") or 0))
        cpc = max(0.0, float(row.get("cpc") or 0.0))
        keyword_difficulty = min(100.0, max(0.0, float(row.get("keyword_difficulty") or 45.0)))
        competition_index = min(100.0, max(0.0, float(row.get("competition_index") or 0.0)))
        topical_fit_score = min(100.0, max(0.0, float(row.get("topical_fit_score") or 0.0)))
        trend_score = min(100.0, max(0.0, float(((row.get("trend_json") or {}).get("trend_score") or 50))))

        kd_score = 100.0 - keyword_difficulty
        volume_score = min(100.0, math.log10(search_volume + 1) / math.log10(10001) * 100.0) if search_volume > 0 else 0.0
        cpc_score = min(100.0, cpc * 12.0)

        score = (
            kd_score * float(score_config.get("kd_weight") or 0.35)
            + competition_index * float(score_config.get("competition_weight") or 0.25)
            + volume_score * float(score_config.get("volume_weight") or 0.15)
            + cpc_score * float(score_config.get("cpc_weight") or 0.10)
            + trend_score * float(score_config.get("trend_weight") or 0.10)
            + topical_fit_score * float(score_config.get("fit_weight") or 0.05)
        )
        return round(max(0.0, min(100.0, score)), 2)

    def _topical_fit_score(self, keyword: str, topic_context: Dict[str, Any]) -> float:
        topic = topic_context.get("topic") or {}
        topic_tokens = self._token_set(
            " ".join([
                str(topic.get("title") or ""),
                str(topic.get("description") or ""),
                str((topic_context.get("primary_category") or {}).get("name") or ""),
                str((topic_context.get("secondary_category") or {}).get("name") or ""),
            ])
        )
        keyword_tokens = self._token_set(keyword)
        if not topic_tokens or not keyword_tokens:
            return 0.0
        overlap = len(topic_tokens & keyword_tokens)
        if overlap <= 0:
            return 20.0
        return round(min(100.0, 30.0 + (overlap / max(1, len(keyword_tokens))) * 70.0), 2)

    def _infer_intent_label(self, keyword: str) -> str:
        normalized = str(keyword or "").lower()
        if any(term in normalized for term in ["calculator", "template", "tool", "checker", "generator", "planner"]):
            return "utility"
        if any(term in normalized for term in ["best", "compare", "comparison", "vs", "review", "cost", "pricing", "roi"]):
            return "commercial_investigation"
        return "informational"

    def _resolve_cluster_intent(self, rows: List[Dict[str, Any]]) -> str:
        counts: Dict[str, int] = defaultdict(int)
        for row in rows:
            counts[str(row.get("intent_label") or "informational")] += 1
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return ranked[0][0] if ranked else "informational"

    def _software_opportunity_score(self, rows: List[Dict[str, Any]]) -> float:
        best_score = max(float(row.get("opportunity_score") or 0.0) for row in rows)
        signal_hits = 0
        for row in rows[:5]:
            normalized = str(row.get("keyword") or "").lower()
            if any(term in normalized for term in self.SOFTWARE_SIGNAL_TERMS):
                signal_hits += 1
        return round(min(100.0, best_score * 0.65 + signal_hits * 12.5), 2)

    def _build_cluster_name(self, primary_keyword: str, rows: List[Dict[str, Any]]) -> str:
        keyword = primary_keyword.strip()
        if keyword:
            return keyword.title()
        fallback = (rows[0].get("canonical_keyword") or "Keyword Cluster").strip()
        return fallback.title()

    def _build_article_angle(self, primary_keyword: str, intent_label: str) -> str:
        keyword = primary_keyword.strip()
        if not keyword:
            return "Cluster-backed article opportunity"
        if intent_label == "commercial_investigation":
            return f"Best ways to evaluate {keyword} and what matters before committing"
        if intent_label == "utility":
            return f"Practical tools, workflows, and calculators built around {keyword}"
        return f"What {keyword} means, when it matters, and how to act on it"

    def _build_serp_validation_summary(self, primary_keyword: str, intent_label: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        keyword = str(primary_keyword or "").lower()
        blocked = any(stop_term in keyword for stop_term in self.FILTER_STOP_TERMS)
        article_friendly = not blocked and intent_label in {"informational", "commercial_investigation", "utility"}
        confidence = 80 if article_friendly else 35
        if intent_label == "utility":
            confidence = 70
        if "near me" in keyword or "jobs" in keyword:
            confidence = 20
        return {
            "status": "heuristic_only",
            "validation_mode": "heuristic_v1",
            "article_friendly": article_friendly,
            "article_intent_confidence": confidence,
            "observations": [
                f"Primary keyword: {primary_keyword}",
                f"Detected intent: {intent_label}",
                f"Cluster size: {len(rows)}",
            ],
        }

    def _build_cluster_rationale(self, primary_keyword: str, rows: List[Dict[str, Any]], topic_context: Dict[str, Any]) -> str:
        category_path = topic_context.get("category_path") or "the selected topic"
        best_score = round(float(rows[0].get("opportunity_score") or 0.0), 2)
        return (
            f"Cluster anchored on '{primary_keyword}' with {len(rows)} related keywords. "
            f"It stays aligned to {category_path} and currently surfaces a top opportunity score of {best_score}."
        )

    def _cluster_overlap(self, left_tokens: set[str], right_tokens: set[str]) -> float:
        if not left_tokens or not right_tokens:
            return 0.0
        shared = len(left_tokens & right_tokens)
        baseline = max(1, min(len(left_tokens), len(right_tokens)))
        return shared / baseline

    def _token_set(self, text: str) -> set[str]:
        return set(self._meaningful_tokens(text))

    def _meaningful_tokens(self, text: str) -> List[str]:
        normalized = self._normalize_keyword_key(text)
        if not normalized:
            return []
        return [token for token in normalized.split(" ") if token and token not in self.QUERY_STOPWORDS]

    def _extract_ngrams(self, tokens: List[str], min_size: int = 2, max_size: int = 4) -> List[str]:
        out: List[str] = []
        for size in range(min_size, max_size + 1):
            for index in range(0, max(0, len(tokens) - size + 1)):
                chunk = tokens[index:index + size]
                if len(chunk) < size:
                    continue
                out.append(" ".join(chunk))
        return out

    def _clean_seed_phrase(self, text: str) -> str:
        normalized = self._normalize_keyword_key(text)
        if not normalized:
            return ""
        tokens = normalized.split(" ")
        if len(tokens) < 2:
            return ""
        if len(tokens) > 6:
            tokens = tokens[:6]
        phrase = " ".join(tokens)
        if self._has_repeated_halves(phrase):
            return ""
        return phrase

    async def _generate_llm_seed_keywords(
        self,
        topic_context: Dict[str, Any],
        deterministic_seeds: List[str],
    ) -> List[str]:
        topic = topic_context.get("topic") or {}
        project = topic_context.get("project") or {}
        primary_category = topic_context.get("primary_category") or {}
        secondary_category = topic_context.get("secondary_category") or {}

        topic_title = str(topic.get("title") or "").strip()
        topic_description = str(topic.get("description") or "").strip()
        category_path = topic_context.get("category_path") or ""
        decision_focus = str(topic.get("decision_focus") or "").strip()
        angle_question = str(topic.get("angle_question") or "").strip()
        intent_bucket = str(topic.get("intent_bucket") or "").strip()
        audience = str(topic.get("target_audience") or project.get("targetaudiencedescription") or "").strip()
        domain = str(project.get("domain") or project.get("app_name") or "").strip()
        primary_name = str(primary_category.get("name") or "").strip()
        secondary_name = str(secondary_category.get("name") or "").strip()
        hint_text = ", ".join(deterministic_seeds[:10])

        prompt = f"""
Role: You are a veteran SEO researcher translating strategy language into real Google searches.

Topic Title: {topic_title}
Topic Description: {topic_description}
Category Path: {category_path}
Primary Category: {primary_name}
Secondary Category: {secondary_name}
Decision Focus: {decision_focus}
Angle Question: {angle_question}
Intent Bucket: {intent_bucket}
Target Audience: {audience}
Project / Domain Context: {domain}
Existing Seed Hints: {hint_text}

Goal:
- Automatically infer 4-6 search lanes a real person would explore around this topic.
- Translate abstract topic wording into practical search language.
- Produce seed phrases that a real person would type, not internal strategy labels.

Critical Priority Rule:
- Topic title, decision focus, and angle question are the source of truth.
- Category path, project/domain context, and audience are only supporting hints.
- If the category context conflicts with the topic itself, IGNORE the conflicting category context.

Lane Design Rules:
- Favor concrete lanes such as durability, maintenance, compatibility, lifecycle, replacement, pricing, resale, support, upgradeability, or risk when relevant.
- Do not drift into adjacent business categories unless the topic clearly asks for that.
- Each lane should represent a distinct user search path.

Seed Rules:
- Each seed must be 2-5 words.
- Prefer plain English over consultant-speak.
- Keep each seed query-like and natural.
- Include pricing, comparison, alternatives, tool, maintenance, support, upgrade, or failure-mode wording only when it genuinely fits the topic.
- Avoid headings, punctuation-heavy phrasing, and sentence fragments.
- Avoid generic filler like "ultimate guide", "best guide", "tips", or "overview".
- Avoid phrases that are too broad to be useful.

Output Contract:
Return valid JSON with this shape:
{{
  "lanes": [
    {{
      "name": "short lane name",
      "reason": "why this lane matters",
      "seeds": ["seed one", "seed two", "seed three"]
    }}
  ]
}}

Requirements:
- 4 to 6 lanes
- 3 to 5 seeds per lane
- No extra keys
"""
        try:
            response = await asyncio.wait_for(
                llm_service.generate_json(
                    prompt=prompt,
                    max_tokens=500,
                    temperature=0.2,
                    task_role=LLM_ROLE_RESEARCH,
                ),
                timeout=25.0,
            )
            lanes = response.get("lanes") if isinstance(response, dict) else []
            seeds: List[str] = []
            seen = set()
            for lane in lanes or []:
                lane_seeds = lane.get("seeds") if isinstance(lane, dict) else []
                for raw in lane_seeds or []:
                    normalized = self._normalize_llm_seed_phrase(raw)
                    if not normalized:
                        continue
                    key = normalized.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    seeds.append(normalized)
            logger.info(
                "LLM topic seed generation topic=%r generated=%s sample=%s",
                topic_title,
                len(seeds),
                seeds[:8],
            )
            return seeds[:24]
        except Exception as exc:
            logger.warning(
                "LLM topic seed generation failed topic=%r err=%s",
                topic_title,
                exc,
            )
            return []

    def _extract_seed_candidates(self, content: str) -> List[str]:
        if not content:
            return []
        candidates: List[str] = []
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        for line in lines:
            lowered = line.lower()
            if lowered.startswith(("step ", "task ", "output ", "role:", "constraints:", "quality rules:", "goal:")):
                continue
            line = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
            parts = [part.strip() for part in re.split(r",|;|\|", line) if part.strip()]
            if parts:
                candidates.extend(parts)
            else:
                candidates.append(line)
        return candidates

    def _normalize_llm_seed_phrase(self, text: str) -> str:
        normalized = self._normalize_keyword_key(text)
        if not normalized:
            return ""
        tokens = [token for token in normalized.split(" ") if token]
        if len(tokens) < 2:
            return ""
        if len(tokens) > 5:
            tokens = tokens[:5]
        phrase = " ".join(tokens)
        if not self._looks_human_search_like(phrase):
            return ""
        return phrase

    def _looks_human_search_like(self, phrase: str) -> bool:
        lowered = phrase.lower().strip()
        if not lowered:
            return False
        tokens = lowered.split()
        if len(tokens) < 2 or len(tokens) > 5:
            return False
        blocked_patterns = (
            "framework",
            "methodology",
            "enablement",
            "solutioning",
            "leverage",
            "synergy",
            "playbook",
            "north star",
        )
        if any(pattern in lowered for pattern in blocked_patterns):
            return False
        if sum(1 for token in tokens if len(token) <= 1) > 1:
            return False
        return True

    def _looks_like_useful_seed(self, phrase: str) -> bool:
        lowered = phrase.lower().strip()
        if not lowered:
            return False
        if any(pattern in lowered for pattern in self.SEED_NOISE_PATTERNS):
            return False
        if not self._looks_human_search_like(phrase):
            return False
        return True

    def _category_matches_topic(self, category_name: str, context_tokens: set[str]) -> bool:
        if not category_name or not context_tokens:
            return False
        category_tokens = set(self._meaningful_tokens(category_name))
        if not category_tokens:
            return False
        return bool(category_tokens & context_tokens)

    def _audience_matches_topic(self, audience: str, context_tokens: set[str]) -> bool:
        if not audience or not context_tokens:
            return False
        audience_tokens = set(self._meaningful_tokens(audience))
        if not audience_tokens:
            return False
        return len(audience_tokens & context_tokens) >= 2

    def _queryish_fragment(self, text: str, max_words: int = 5) -> str:
        normalized = self._normalize_keyword_key(text)
        if not normalized:
            return ""
        tokens = [token for token in normalized.split(" ") if token]
        if len(tokens) < 2:
            return ""
        return " ".join(tokens[:max_words])

    def _has_repeated_halves(self, phrase: str) -> bool:
        tokens = phrase.split()
        if len(tokens) < 4 or len(tokens) % 2 != 0:
            return False
        midpoint = len(tokens) // 2
        return tokens[:midpoint] == tokens[midpoint:]

    def _normalize_keyword_key(self, text: Any) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "").strip().lower())
        cleaned = cleaned.replace("&", " and ")
        cleaned = re.sub(r"[^\w\s-]", " ", cleaned)
        cleaned = re.sub(r"\b(202\d|203\d)\b", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _coerce_competition_index(self, row: Dict[str, Any]) -> Optional[int]:
        value = self._competition_index_from_values(
            row.get("competition"),
            row.get("competition_level"),
        )
        return int(value) if value is not None else None

    def _competition_index_from_values(self, competition: Any, competition_level: Any) -> Optional[int]:
        try:
            level = int(float(competition_level))
            if level > 0:
                return max(0, min(100, level))
        except Exception:
            pass
        comp = str(competition or "").upper().strip()
        if comp == "HIGH":
            return 85
        if comp == "MEDIUM":
            return 60
        if comp == "LOW":
            return 30
        return None

    def _merge_strings(self, left: Optional[List[str]], right: Optional[List[str]]) -> List[str]:
        seen = set()
        out = []
        for value in (left or []) + (right or []):
            item = str(value or "").strip()
            if not item or item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    def _prefer_number(self, left: Any, right: Any) -> Any:
        left_val = self._safe_float(left)
        right_val = self._safe_float(right)
        if left_val is None:
            return right_val
        if right_val is None:
            return left_val
        return max(left_val, right_val)

    def _safe_int(self, value: Any) -> Optional[int]:
        try:
            if value is None or str(value).strip() == "":
                return None
            return int(float(value))
        except Exception:
            return None

    def _safe_float(self, value: Any) -> Optional[float]:
        try:
            if value is None or str(value).strip() == "":
                return None
            return float(value)
        except Exception:
            return None

    def _sanitize_for_json(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._sanitize_for_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._sanitize_for_json(item) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize_for_json(item) for item in value]
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
