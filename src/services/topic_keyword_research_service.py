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
from urllib.parse import urlparse

from src.integrations.dataforseo import dataforseo_api
from src.services.llm.llm_service import llm_service
from supabase_client import LLM_ROLE_RESEARCH

logger = logging.getLogger(__name__)


class TopicKeywordResearchService:
    """Orchestrates topic-level keyword research runs and persistence."""

    EXCLUDED_AUTHORITY_DOMAINS = {
        "forbes.com",
        "www.forbes.com",
        "wikipedia.org",
        "www.wikipedia.org",
        "reddit.com",
        "www.reddit.com",
        "investopedia.com",
        "www.investopedia.com",
        "nerdwallet.com",
        "www.nerdwallet.com",
        "homedepot.com",
        "www.homedepot.com",
        "amazon.com",
        "www.amazon.com",
    }

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

    BUSINESS_DRIFT_TERMS = {
        "supply chain", "procurement", "construction", "enterprise", "b2b",
        "business", "industrial", "vendor", "warehouse", "logistics",
        "manufacturing", "asset management",
    }

    FILTER_STOP_PREFIXES = (
        "deciding between",
        "between ",
        "building on",
    )

    SEED_NOISE_PATTERNS = (
        "building on existing content",
        "existing content about",
        "homeowners and property buyers",
        "target audience",
        "site description",
    )

    GENERIC_TOPIC_TERMS = {
        "total", "cost", "ownership", "value", "analysis", "planning", "timing",
        "guide", "selection", "decision", "support", "comparison", "compare",
        "investing", "investment", "spending", "power", "capital", "allocation",
        "market", "consumer", "intelligence", "pricing", "roi", "reality",
        "period", "periods", "still", "make", "determining", "true", "point",
        "changing", "using", "driven", "future", "strategy", "strategic",
    }

    TOKEN_ALIAS_MAP = {
        "ev": {"ev", "electric", "vehicle", "vehicles", "car", "cars", "auto", "automotive"},
        "hybrid": {"hybrid", "phev", "plugin", "plug", "car", "cars", "vehicle", "vehicles"},
        "phev": {"phev", "plugin", "plug", "hybrid", "car", "cars", "vehicle", "vehicles"},
        "solar": {"solar", "panel", "panels", "roof", "rooftop"},
        "mortgage": {"mortgage", "home", "house", "housing"},
        "insurance": {"insurance", "coverage", "policy"},
    }

    SOFTWARE_SIGNAL_TERMS = {
        "calculator", "tool", "template", "checker", "planner", "estimator",
        "generator", "audit", "tracker", "scorecard", "comparison", "compare",
        "cost", "roi", "pricing",
    }

    DEFAULT_FILTERS = {
        "min_search_volume": 30,
        "max_keyword_difficulty": 55,
        "min_competition_index": 0,
        "max_candidates_to_enrich": 150,
        "max_clusters": 12,
        "target_device": "desktop",
        "target_language_code": "en",
        "target_location_code": 2840,
        "competitor_page_limit": 6,
        "ranked_keywords_per_page": 35,
        "expansion_seed_limit": 8,
        "expansion_keywords_per_seed": 20,
        "research_scope": "focused",
    }

    DEFAULT_SCORE_CONFIG = {
        "serp_weight": 0.30,
        "kd_weight": 0.25,
        "volume_weight": 0.20,
        "commercial_weight": 0.15,
        "fit_weight": 0.10,
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
        manual_seed_keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        filters = {**self.DEFAULT_FILTERS, **(filters or {})}
        score_config = {**self.DEFAULT_SCORE_CONFIG, **(score_config or {})}
        topic_context = self._load_topic_context(topic_id=topic_id, user_id=user_id)
        if not topic_context:
            raise ValueError("Research topic not found")

        if replace_existing:
            self.delete_topic_research(topic_id=topic_id, user_id=user_id)

        pending_seed_package = {
            "generation_mode": "pending",
            "seed_keywords": [],
            "deterministic_seeds": [],
            "llm_seeds": [],
            "seed_sources": {},
        }
        run_row = self._create_run(
            topic_id=topic_id,
            user_id=user_id,
            seed_keywords=[],
            filters=filters,
            score_config=score_config,
            topic_context=topic_context,
            seed_package=pending_seed_package,
        )
        run_id = run_row["id"]
        seed_package = pending_seed_package
        seeds: List[str] = []

        try:
            seed_package = await self._build_probe_queries(topic_context, manual_seed_keywords=manual_seed_keywords)
            seeds = seed_package["seed_keywords"]
            probe_results, probe_raw_data = await self._run_serp_probe_searches(
                probe_queries=seed_package.get("probe_queries") or [],
                filters=filters,
            )
            serp_gate = self._evaluate_serp_opportunity_gate(
                probe_queries=seed_package.get("probe_queries") or [],
                probe_results=probe_results,
                topic_context=topic_context,
            )
            if not serp_gate.get("passed"):
                summary = self._build_run_summary(
                    topic_context=topic_context,
                    seed_keywords=seeds,
                    candidate_rows=[],
                    clusters=[],
                    seed_package=seed_package,
                    serp_gate=serp_gate,
                    filters=filters,
                )
                self._update_run(
                    run_id=run_id,
                    topic_id=topic_id,
                    user_id=user_id,
                    status="completed",
                    summary_json=summary,
                    raw_data_json={
                        "serp_probes": probe_raw_data,
                        "serp_gate": serp_gate,
                        "seed_generation": seed_package,
                    },
                    error_message=None,
                )
                run = self.get_run(run_id=run_id, topic_id=topic_id, user_id=user_id)
                return {
                    "run": run,
                    "keywords": [],
                    "clusters": [],
                    "summary": summary,
                }

            competitor_pages = self._select_competitor_pages_from_probe_results(
                probe_results=probe_results,
                serp_gate=serp_gate,
                topic_context=topic_context,
                limit=int(filters.get("competitor_page_limit") or 6),
            )
            discovered_rows, raw_data = await self._discover_keyword_candidates_from_competitors(
                competitors=competitor_pages,
                probe_queries=seed_package.get("probe_queries") or [],
                serp_gate=serp_gate,
                topic_context=topic_context,
                filters=filters,
            )
            enriched_rows = await self._enrich_and_score_candidates(
                candidates=discovered_rows,
                topic_context=topic_context,
                filters=filters,
                score_config=score_config,
                topic_components=(seed_package or {}).get("topic_components") or [],
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
                serp_gate=serp_gate,
                filters=filters,
            )
            self._update_run(
                run_id=run_id,
                topic_id=topic_id,
                user_id=user_id,
                status="completed",
                summary_json=summary,
                raw_data_json={
                    **probe_raw_data,
                    **raw_data,
                    "serp_gate": serp_gate,
                    "competitor_pages": competitor_pages,
                    "seed_generation": seed_package,
                },
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
                "intent_bucket, decision_focus, angle_question, value_layer_tags, target_audience, related_terms"
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

    async def _build_probe_queries(
        self,
        topic_context: Dict[str, Any],
        manual_seed_keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        manual_hints = [
            self._clean_seed_phrase(seed)
            for seed in (manual_seed_keywords or [])
            if self._clean_seed_phrase(seed)
        ]
        topic = topic_context.get("topic") or {}
        project = topic_context.get("project") or {}
        primary_category = topic_context.get("primary_category") or {}
        secondary_category = topic_context.get("secondary_category") or {}

        topic_title = str(topic.get("title") or "").strip()
        topic_description = str(topic.get("description") or "").strip()
        category_path = topic_context.get("category_path") or ""
        audience = str(topic.get("target_audience") or project.get("targetaudiencedescription") or "").strip()
        decision_focus = str(topic.get("decision_focus") or "").strip()
        angle_question = str(topic.get("angle_question") or "").strip()
        hint_text = ", ".join(manual_hints[:8]) or "None"

        prompt = f"""
Role: You are an SEO strategist. Generate exactly 3 Google search probes for a topic.

Topic: {topic_title}
Description: {topic_description}
Category Path: {category_path}
Primary Category: {primary_category.get("name") or ""}
Secondary Category: {secondary_category.get("name") or ""}
Target Audience: {audience}
Decision Focus: {decision_focus}
Angle Question: {angle_question}
Manual Hints: {hint_text}

Rules:
- Output exactly 3 search probes.
- Probe 1 must be a broad practical query.
- Probe 2 must be an ROI, value, comparison, or investment-oriented query.
- Probe 3 must be a question or article-intent query.
- Keep each probe natural and between 3 and 10 words.
- These are SERP probes, not a large keyword list.
- Preserve the topic's concrete subject. Do not drift into adjacent industries.

Return only:
PRACTICAL:: query text
ROI:: query text
QUESTION:: query text
"""
        raw_output = ""
        probe_queries: List[str] = []
        try:
            response = await asyncio.wait_for(
                llm_service.generate_text(
                    prompt=prompt,
                    max_tokens=120,
                    temperature=0.2,
                    task_role=LLM_ROLE_RESEARCH,
                ),
                timeout=20.0,
            )
            raw_output = response.content or ""
            for marker in ("PRACTICAL::", "ROI::", "QUESTION::"):
                match = re.search(rf"{marker}\s*(.+)", raw_output, re.IGNORECASE)
                if match:
                    candidate = self._clean_probe_query(match.group(1))
                    if candidate:
                        probe_queries.append(candidate)
        except Exception:
            logger.warning("LLM probe generation failed topic=%r", topic_title, exc_info=True)

        if len(probe_queries) < 3:
            base = (
                self._clean_seed_phrase(topic_title)
                or self._build_probe_fallback_query(topic_title)
                or self._clean_seed_phrase(topic_description)
                or self._build_probe_fallback_query(topic_description)
                or self._build_probe_fallback_query(" ".join(manual_hints))
                or self._build_probe_fallback_query(
                    " ".join(
                        [
                            str(primary_category.get("name") or ""),
                            str(secondary_category.get("name") or ""),
                            topic_title,
                        ]
                    )
                )
            )
            practical = base
            roi = (
                self._clean_seed_phrase(f"{base} roi")
                or self._clean_seed_phrase(f"{base} value")
                or self._build_probe_fallback_query(f"{base} roi")
                or self._build_probe_fallback_query(f"{base} value")
            )
            question = (
                self._clean_seed_phrase(f"is {base} worth it")
                or self._clean_seed_phrase(f"should you use {base}")
                or self._build_probe_fallback_query(f"is {base} worth it")
                or self._build_probe_fallback_query(f"should you use {base}")
            )
            probe_queries = [item for item in [practical, roi, question] if item][:3]

        final_queries: List[str] = []
        seen = set()
        for candidate in probe_queries + manual_hints[:3]:
            cleaned = self._clean_probe_query(candidate)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            final_queries.append(cleaned)
            if len(final_queries) >= 3:
                break

        if len(final_queries) < 3:
            fallback_candidates = [
                self._build_probe_fallback_query(topic_title),
                self._build_probe_fallback_query(topic_description),
                self._build_probe_fallback_query(" ".join(manual_hints)),
                self._build_probe_fallback_query(
                    " ".join(
                        [
                            str(primary_category.get("name") or ""),
                            topic_title,
                        ]
                    )
                ),
                self._build_probe_fallback_query(
                    " ".join(
                        [
                            str(secondary_category.get("name") or ""),
                            topic_title,
                        ]
                    )
                ),
            ]
            base = fallback_candidates[0] or fallback_candidates[1] or fallback_candidates[3] or fallback_candidates[4]
            if base:
                fallback_candidates.extend([
                    base,
                    self._build_probe_fallback_query(f"{base} roi"),
                    self._build_probe_fallback_query(f"is {base} worth it"),
                ])

            for candidate in fallback_candidates:
                if not candidate:
                    continue
                key = candidate.lower()
                if key in seen:
                    continue
                seen.add(key)
                final_queries.append(candidate)
                if len(final_queries) >= 3:
                    break

        if len(final_queries) < 3:
            raise ValueError("Topic keyword research could not generate 3 SERP probes for this topic.")

        return {
            "generation_mode": "serp_probe_v2",
            "seed_keywords": final_queries,
            "probe_queries": final_queries,
            "deterministic_seeds": final_queries,
            "llm_seeds": final_queries,
            "topic_components": [],
            "seed_sources": {query: "serp_probe" for query in final_queries},
            "llm_raw_output": raw_output,
        }

    async def _run_serp_probe_searches(
        self,
        *,
        probe_queries: List[str],
        filters: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        probe_results: List[Dict[str, Any]] = []
        raw_payloads: List[Dict[str, Any]] = []
        language_code = str(filters.get("target_language_code") or "en")
        location_code = int(filters.get("target_location_code") or 2840)
        device = str(filters.get("target_device") or "desktop")

        for query in probe_queries[:3]:
            serp_result = await dataforseo_api.get_google_organic_standard_regular(
                query,
                language_code=language_code,
                location_code=location_code,
                device=device,
                depth=10,
                return_raw=True,
            )
            items = (serp_result or {}).get("items") or []
            if not items:
                serp_result = await dataforseo_api.get_google_organic_live_advanced(
                    query,
                    language_code=language_code,
                    location_code=location_code,
                    device=device,
                    depth=10,
                    return_raw=True,
                )
                items = (serp_result or {}).get("items") or []

            normalized_rows = []
            for row in items[:10]:
                url = str(row.get("url") or "").strip()
                domain = str(row.get("domain") or urlparse(url).netloc).lower()
                normalized_rows.append({
                    "query": query,
                    "rank": int(row.get("rank_group") or row.get("rank_absolute") or 0) or None,
                    "domain": domain,
                    "url": url,
                    "title": str(row.get("title") or "").strip(),
                    "snippet": str(row.get("snippet") or "").strip(),
                    "result_type": str(row.get("result_type") or "organic").strip() or "organic",
                })
            probe_results.append({"query": query, "rows": normalized_rows})
            raw_payloads.append({
                "query": query,
                "raw": self._sanitize_for_json((serp_result or {}).get("raw") or {}),
            })

        return probe_results, {"serp_probes": raw_payloads}

    def _evaluate_serp_opportunity_gate(
        self,
        *,
        probe_queries: List[str],
        probe_results: List[Dict[str, Any]],
        topic_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        domain_hits: Dict[str, int] = defaultdict(int)
        content_page_count = 0
        niche_page_count = 0
        weak_page_count = 0
        authority_like_count = 0
        ecommerce_or_service_count = 0
        probe_intents: List[str] = []
        repeated_urls: Dict[str, int] = defaultdict(int)

        for probe in probe_results:
            rows = probe.get("rows") or []
            probe_intents.append(self._classify_probe_intent(rows))
            seen_domains = set()
            seen_urls = set()
            for row in rows[:10]:
                url = str(row.get("url") or "").strip()
                domain = str(row.get("domain") or "").strip().lower()
                title = str(row.get("title") or "")
                snippet = str(row.get("snippet") or "")
                if url and url not in seen_urls:
                    repeated_urls[url] += 1
                    seen_urls.add(url)
                if domain and domain not in seen_domains:
                    domain_hits[domain] += 1
                    seen_domains.add(domain)
                if self._is_usable_content_page(url, title):
                    content_page_count += 1
                if self._is_niche_competitor(domain, url):
                    niche_page_count += 1
                if self._is_weak_content_page(title, snippet, url):
                    weak_page_count += 1
                if self._is_authority_domain(domain):
                    authority_like_count += 1
                if self._is_service_or_ecommerce_page(url, title):
                    ecommerce_or_service_count += 1

        repeated_domains = sorted(
            [{"domain": domain, "hits": hits} for domain, hits in domain_hits.items() if hits >= 2],
            key=lambda item: item["hits"],
            reverse=True,
        )
        repeated_url_rows = sorted(
            [{"url": url, "hits": hits} for url, hits in repeated_urls.items() if hits >= 2],
            key=lambda item: item["hits"],
            reverse=True,
        )
        dominant_intent = max(set(probe_intents), key=probe_intents.count) if probe_intents else "mixed"
        consistent_intent = probe_intents.count(dominant_intent) >= 2

        signals = {
            "stable_competitor_set": bool(repeated_domains),
            "article_friendly_results": content_page_count >= 2,
            "niche_sites_present": niche_page_count >= 2,
            "consistent_intent": consistent_intent,
            "weak_pages_present": weak_page_count >= 2,
            "not_authority_dominated": authority_like_count <= 12,
        }
        pass_count = sum(1 for passed in signals.values() if passed)

        killer_reasons: List[str] = []
        if authority_like_count >= 18:
            killer_reasons.append("authority_dominated_serp")
        if not consistent_intent:
            killer_reasons.append("intent_mismatch_across_probes")
        if ecommerce_or_service_count >= 18:
            killer_reasons.append("service_or_ecommerce_dominant_serp")
        if content_page_count == 0:
            killer_reasons.append("no_usable_content_pages_ranking")

        minimum_viable_opportunity = (
            consistent_intent
            and content_page_count >= 2
            and (
                niche_page_count >= 2
                or weak_page_count >= 2
            )
        )

        serp_weakness_score = round(
            max(
                0.0,
                min(
                    1.0,
                    (
                        min(1.0, niche_page_count / 6.0) * 0.25
                        + min(1.0, weak_page_count / 6.0) * 0.30
                        + (0.20 if content_page_count >= 2 else 0.0)
                        + (0.15 if repeated_domains else 0.0)
                        + (0.10 if authority_like_count <= 12 else 0.0)
                    ),
                ),
            ),
            4,
        )

        return {
            "passed": minimum_viable_opportunity and not killer_reasons,
            "signal_count": pass_count,
            "signals": signals,
            "killer_reasons": killer_reasons,
            "intent_classification": dominant_intent,
            "serp_weakness_score": serp_weakness_score,
            "repeated_domains": repeated_domains[:10],
            "repeated_urls": repeated_url_rows[:10],
            "probe_queries": probe_queries[:3],
        }

    def _select_competitor_pages_from_probe_results(
        self,
        *,
        probe_results: List[Dict[str, Any]],
        serp_gate: Dict[str, Any],
        topic_context: Dict[str, Any],
        limit: int,
    ) -> List[Dict[str, Any]]:
        candidate_map: Dict[str, Dict[str, Any]] = {}
        repeated_domain_lookup = {
            str(item.get("domain") or "").lower(): int(item.get("hits") or 0)
            for item in (serp_gate.get("repeated_domains") or [])
        }
        for probe in probe_results:
            for row in probe.get("rows") or []:
                url = str(row.get("url") or "").strip()
                domain = str(row.get("domain") or urlparse(url).netloc).lower()
                title = str(row.get("title") or "").strip()
                if not url or not domain:
                    continue
                if self._is_authority_domain(domain):
                    continue
                if not self._is_usable_content_page(url, title):
                    continue
                if self._is_service_or_ecommerce_page(url, title):
                    continue
                entry = candidate_map.setdefault(
                    url,
                    {
                        "url": url,
                        "domain": domain,
                        "title": title,
                        "best_rank": 99,
                        "query_hits": 0,
                        "domain_hits": repeated_domain_lookup.get(domain, 0),
                        "content_gap_score": 0,
                    },
                )
                entry["query_hits"] += 1
                entry["best_rank"] = min(int(row.get("rank") or 99), entry["best_rank"])
                if self._is_weak_content_page(title, str(row.get("snippet") or ""), url):
                    entry["content_gap_score"] += 1

        competitors = sorted(
            candidate_map.values(),
            key=lambda item: (
                int(item.get("query_hits") or 0),
                int(item.get("domain_hits") or 0),
                int(item.get("content_gap_score") or 0),
                -int(item.get("best_rank") or 99),
            ),
            reverse=True,
        )
        return competitors[: max(3, min(limit, 8))]

    async def _discover_keyword_candidates_from_competitors(
        self,
        *,
        competitors: List[Dict[str, Any]],
        probe_queries: List[str],
        serp_gate: Dict[str, Any],
        topic_context: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        language_code = str(filters.get("target_language_code") or "en")
        location_code = int(filters.get("target_location_code") or 2840)
        limit_per_page = int(filters.get("ranked_keywords_per_page") or 35)
        harvested_rows: List[Dict[str, Any]] = []
        raw_data: Dict[str, Any] = {
            "selected_competitors": competitors,
            "ranked_keyword_searches": [],
            "controlled_expansion": [],
        }

        for competitor in competitors:
            target_url = str(competitor.get("url") or "").strip()
            if not target_url:
                continue
            ranked_result = await dataforseo_api.get_ranked_keywords_live(
                target_url,
                language_code=language_code,
                location_code=location_code,
                limit=limit_per_page,
                return_raw=True,
            )
            raw_data["ranked_keyword_searches"].append({
                "target": target_url,
                "raw": self._sanitize_for_json((ranked_result or {}).get("raw") or {}),
            })
            for row in (ranked_result or {}).get("items") or []:
                candidate = self._normalize_candidate_row(row)
                candidate["source_endpoints"] = ["dataforseo_labs/google/ranked_keywords/live:url"]
                candidate["trend_json"] = {
                    "source_url": target_url,
                    "source_urls": [target_url],
                    "source_domain": competitor.get("domain"),
                    "source_domains": [competitor.get("domain")] if competitor.get("domain") else [],
                    "serp_weakness_score": serp_gate.get("serp_weakness_score"),
                    "competitor_support_score": min(
                        1.0,
                        0.45
                        + 0.20 * min(2, int(competitor.get("query_hits") or 1))
                        + 0.10 * min(2, int(competitor.get("content_gap_score") or 0)),
                    ),
                    "probe_queries": probe_queries[:3],
                }
                if self._passes_keyword_harvest_filters(candidate, topic_context=topic_context, filters=filters):
                    harvested_rows.append(candidate)

        expansion_seed_rows = self._select_expansion_seed_rows(
            harvested_rows,
            limit=int(filters.get("expansion_seed_limit") or 8),
        )
        for seed_row in expansion_seed_rows:
            seed_keyword = str(seed_row.get("keyword") or "").strip()
            if not seed_keyword:
                continue
            expansion_result = await dataforseo_api.get_related_keywords_standard(
                [seed_keyword],
                language_code=language_code,
                location_code=location_code,
                limit_per_seed=int(filters.get("expansion_keywords_per_seed") or 20),
                return_raw=True,
            )
            raw_data["controlled_expansion"].append({
                "seed_keyword": seed_keyword,
                "raw": self._sanitize_for_json((expansion_result or {}).get("raw") or {}),
            })
            for row in (expansion_result or {}).get("items") or []:
                candidate = self._normalize_candidate_row(row)
                candidate["source_endpoints"] = ["keywords_data/google_ads/keywords_for_keywords:controlled"]
                candidate["trend_json"] = {
                    "seed_keyword": seed_keyword,
                    "source_type": "controlled_expansion",
                    "serp_weakness_score": serp_gate.get("serp_weakness_score"),
                    "competitor_support_score": 0.55,
                    "probe_queries": probe_queries[:3],
                    "source_urls": list((seed_row.get("trend_json") or {}).get("source_urls") or []),
                    "source_domains": list((seed_row.get("trend_json") or {}).get("source_domains") or []),
                }
                if self._passes_keyword_harvest_filters(candidate, topic_context=topic_context, filters=filters):
                    harvested_rows.append(candidate)

        return harvested_rows, raw_data

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
        topic_components: Optional[List[Dict[str, Any]]] = None,
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
                    "trend_json": dict(row.get("trend_json") or {}),
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
            existing["trend_json"] = {
                **dict(existing.get("trend_json") or {}),
                **dict(row.get("trend_json") or {}),
                "source_urls": self._merge_strings(
                    list((existing.get("trend_json") or {}).get("source_urls") or []),
                    list((row.get("trend_json") or {}).get("source_urls") or []),
                ),
                "source_domains": self._merge_strings(
                    list((existing.get("trend_json") or {}).get("source_domains") or []),
                    list((row.get("trend_json") or {}).get("source_domains") or []),
                ),
            }

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
            row["topical_fit_score"] = self._topical_fit_score(
                canonical,
                topic_context,
                topic_components=topic_components,
                research_scope=str(filters.get("research_scope") or "focused"),
            )
            row["trend_json"] = {
                "validation_mode": "serp_probe_competitor_v2",
                **dict(row.get("trend_json") or {}),
            }

            is_filtered_out, filter_reason = self._filter_candidate_row(
                row=row,
                filters=filters,
                topic_context=topic_context,
                topic_components=topic_components,
            )
            row["is_filtered_out"] = is_filtered_out
            row["filter_reason"] = filter_reason
            row["opportunity_score"] = self._compute_keyword_opportunity_score(
                row=row,
                score_config=score_config,
            )
            scored_rows.append(row)

        scored_rows.sort(key=lambda item: (item.get("opportunity_score") or 0), reverse=True)
        return scored_rows

    def _passes_keyword_harvest_filters(
        self,
        row: Dict[str, Any],
        *,
        topic_context: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        keyword = str(row.get("keyword") or "").strip()
        canonical = self._normalize_keyword_key(keyword)
        if not canonical:
            return False
        research_scope = str((filters or {}).get("research_scope") or "focused").lower().strip()
        tokens = canonical.split()
        if len(tokens) < 2 or len(tokens) > 8:
            return False
        if int(row.get("rank_group") or 99) > 20:
            return False
        if (row.get("search_volume") or 0) and int(row.get("search_volume") or 0) < 30:
            return False
        if self._keyword_contains_brand(keyword, topic_context):
            return False
        anchor_overlap = self._anchor_overlap_count(canonical, self._topic_anchor_terms(topic_context))
        if research_scope == "expanded":
            if anchor_overlap <= 0 and self._category_neighborhood_overlap_count(canonical, topic_context) <= 0:
                return False
        elif anchor_overlap <= 0:
            return False
        if self._is_service_or_ecommerce_page(str(row.get("url") or ""), str(row.get("title") or "")):
            return False
        return True

    def _select_expansion_seed_rows(self, rows: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
        ordered = sorted(
            rows,
            key=lambda item: (
                float((item.get("trend_json") or {}).get("competitor_support_score") or 0.0),
                float(item.get("search_volume") or 0.0),
                -float(item.get("keyword_difficulty") or 100.0),
            ),
            reverse=True,
        )
        selected: List[Dict[str, Any]] = []
        seen = set()
        for row in ordered:
            keyword = self._normalize_keyword_key(row.get("keyword"))
            if not keyword or keyword in seen:
                continue
            seen.add(keyword)
            selected.append(row)
            if len(selected) >= limit:
                break
        return selected

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

    async def _build_seed_keywords(self, topic_context: Dict[str, Any], manual_seed_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        manual_seed_keywords = [
            self._clean_seed_phrase(seed)
            for seed in (manual_seed_keywords or [])
            if self._clean_seed_phrase(seed)
        ]
        if manual_seed_keywords:
            return {
                "generation_mode": "manual_override_v1",
                "seed_keywords": manual_seed_keywords[:20],
                "deterministic_seeds": self._build_deterministic_seed_keywords(topic_context),
                "llm_seeds": [],
                "topic_components": [],
                "component_query_seeds": [],
                "seed_sources": {seed: "manual_override" for seed in manual_seed_keywords[:20]},
                "llm_parse_strategy": None,
                "llm_raw_output": None,
                "llm_raw_seed_count": 0,
                "llm_accepted_seed_count": 0,
                "llm_rejected_candidates": [],
            }
        deterministic_seeds = self._build_deterministic_seed_keywords(topic_context)
        topic_components = await self._decompose_topic_into_search_components(topic_context, deterministic_seeds)
        component_query_seeds = self._component_query_seed_candidates(topic_components)
        llm_result = await self._generate_llm_seed_keywords(topic_context, deterministic_seeds, topic_components)
        llm_seeds = llm_result.get("accepted_seeds") or []
        final_seeds = self._diversify_seed_pool(
            topic_components=topic_components,
            llm_seeds=llm_seeds,
            component_query_seeds=component_query_seeds,
            deterministic_seeds=deterministic_seeds,
        )
        seed_package = {
            "generation_mode": "llm_subtopics_v1",
            "seed_keywords": final_seeds,
            "deterministic_seeds": deterministic_seeds,
            "llm_seeds": llm_seeds,
            "topic_components": topic_components,
            "component_query_seeds": component_query_seeds,
            "seed_sources": {
                **{seed: "component_query" for seed in component_query_seeds},
                **{seed: "llm" for seed in llm_seeds},
            },
            "llm_parse_strategy": llm_result.get("parse_strategy"),
            "llm_raw_output": llm_result.get("raw_output"),
            "llm_raw_seed_count": llm_result.get("raw_seed_count"),
            "llm_accepted_seed_count": llm_result.get("accepted_seed_count"),
            "llm_rejected_candidates": llm_result.get("rejected_candidates") or [],
        }
        if not final_seeds:
            raise ValueError(
                "Topic keyword research could not generate usable LLM seed keywords for this topic. "
                "Please revise the topic context or provide manual seeds."
            )
        return seed_package

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
        related_terms = [str(term).strip() for term in (topic.get("related_terms") or []) if str(term).strip()]
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
            *related_terms,
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
        serp_gate: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
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
            "pipeline_version": "topic_keyword_pipeline_v2",
            "status": "completed",
            "topic_title": ((topic_context.get("topic") or {}).get("title") or ""),
            "category_path": topic_context.get("category_path"),
            "seed_count": len(seed_keywords),
            "probe_count": len((seed_package or {}).get("probe_queries") or seed_keywords),
            "seed_generation_mode": (seed_package or {}).get("generation_mode") or "serp_probe_v2",
            "research_scope": str((filters or {}).get("research_scope") or "focused"),
            "llm_seed_count": len((seed_package or {}).get("llm_seeds") or []),
            "deterministic_seed_count": len((seed_package or {}).get("deterministic_seeds") or []),
            "llm_parse_strategy": (seed_package or {}).get("llm_parse_strategy"),
            "candidate_count": len(candidate_rows),
            "active_candidate_count": len(active),
            "filtered_candidate_count": len(filtered_out),
            "cluster_count": len(clusters),
            "serp_opportunity_passed": bool((serp_gate or {}).get("passed")),
            "serp_signal_count": int((serp_gate or {}).get("signal_count") or 0),
            "serp_intent_classification": (serp_gate or {}).get("intent_classification"),
            "serp_weakness_score": (serp_gate or {}).get("serp_weakness_score"),
            "top_repeated_domains": (serp_gate or {}).get("repeated_domains") or [],
            "top_repeated_urls": (serp_gate or {}).get("repeated_urls") or [],
            "gate_killer_reasons": (serp_gate or {}).get("killer_reasons") or [],
            "top_keywords": top_keywords,
            "top_clusters": top_clusters,
            "validation_mode": "serp_probe_competitor_v2",
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

    def _filter_candidate_row(
        self,
        row: Dict[str, Any],
        filters: Dict[str, Any],
        topic_context: Dict[str, Any],
        topic_components: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[bool, Optional[str]]:
        keyword = (row.get("keyword") or "").lower().strip()
        canonical = row.get("canonical_keyword") or ""
        research_scope = str(filters.get("research_scope") or "focused").lower().strip()
        if not canonical or len(self._token_set(canonical)) < 2:
            return True, "not_specific_enough"

        for stop_prefix in self.FILTER_STOP_PREFIXES:
            if keyword.startswith(stop_prefix):
                return True, f"blocked_prefix:{stop_prefix.strip()}"

        for stop_term in self.FILTER_STOP_TERMS:
            if stop_term in keyword:
                return True, f"blocked_term:{stop_term}"

        core_anchor_terms = self._core_topic_anchor_terms(topic_context)
        expanded_anchor_terms = self._topic_anchor_terms(topic_context, topic_components=topic_components)
        anchor_overlap = self._anchor_overlap_count(canonical, core_anchor_terms)
        neighborhood_overlap = self._category_neighborhood_overlap_count(canonical, topic_context)
        if core_anchor_terms and anchor_overlap <= 0:
            if research_scope != "expanded" or neighborhood_overlap <= 0:
                return True, "missing_topic_anchor"
        anchor_coverage = self._anchor_coverage_ratio(canonical, core_anchor_terms)
        keyword_token_count = max(1, len(self._token_set(canonical)))
        if core_anchor_terms and (
            anchor_coverage < 0.34
            or (anchor_overlap <= 1 and keyword_token_count >= 4)
        ):
            if research_scope != "expanded" or neighborhood_overlap <= 0:
                return True, "weak_topic_anchor"
        if (
            expanded_anchor_terms
            and self._anchor_overlap_count(canonical, expanded_anchor_terms) <= 0
        ):
            if research_scope != "expanded" or neighborhood_overlap <= 0:
                return True, "missing_expanded_topic_anchor"

        if (
            self._audience_mode(topic_context) == "consumer"
            and any(term in keyword for term in self.BUSINESS_DRIFT_TERMS)
        ):
            return True, "business_domain_drift"

        search_volume = int(row.get("search_volume") or 0)
        keyword_difficulty = float(row.get("keyword_difficulty") or 0.0)
        competition_index = int(row.get("competition_index") or 0)

        measurable_demand = (
            search_volume > 0
            or float(row.get("cpc") or 0) > 0
            or competition_index > 0
        )

        if not measurable_demand:
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
        serp_weakness_score = min(100.0, max(0.0, float(((row.get("trend_json") or {}).get("serp_weakness_score") or 0.35) * 100.0)))
        competitor_support_score = min(100.0, max(0.0, float(((row.get("trend_json") or {}).get("competitor_support_score") or 0.4) * 100.0)))

        kd_score = 100.0 - keyword_difficulty
        volume_score = min(100.0, math.log10(search_volume + 1) / math.log10(10001) * 100.0) if search_volume > 0 else 0.0
        commercial_score = min(100.0, min(100.0, cpc * 15.0) * 0.55 + competition_index * 0.45)
        serp_score = min(100.0, serp_weakness_score * 0.7 + competitor_support_score * 0.3)

        score = (
            serp_score * float(score_config.get("serp_weight") or 0.30)
            + kd_score * float(score_config.get("kd_weight") or 0.25)
            + volume_score * float(score_config.get("volume_weight") or 0.20)
            + commercial_score * float(score_config.get("commercial_weight") or 0.15)
            + topical_fit_score * float(score_config.get("fit_weight") or 0.10)
        )
        return round(max(0.0, min(100.0, score)), 2)

    def _topical_fit_score(
        self,
        keyword: str,
        topic_context: Dict[str, Any],
        topic_components: Optional[List[Dict[str, Any]]] = None,
        research_scope: str = "focused",
    ) -> float:
        topic = topic_context.get("topic") or {}
        topic_tokens = self._token_set(
            " ".join([
                str(topic.get("title") or ""),
                str(topic.get("description") or ""),
                " ".join([str(term).strip() for term in (topic.get("related_terms") or []) if str(term).strip()]),
                str((topic_context.get("primary_category") or {}).get("name") or ""),
                str((topic_context.get("secondary_category") or {}).get("name") or ""),
            ])
        )
        keyword_tokens = self._token_set(keyword)
        if not topic_tokens or not keyword_tokens:
            return 0.0
        overlap = len(topic_tokens & keyword_tokens)
        core_anchor_terms = self._core_topic_anchor_terms(topic_context)
        expanded_anchor_terms = self._topic_anchor_terms(topic_context, topic_components=topic_components)
        anchor_overlap = self._anchor_overlap_count(keyword, core_anchor_terms)
        anchor_coverage = self._anchor_coverage_ratio(keyword, core_anchor_terms)
        expanded_overlap = self._anchor_overlap_count(keyword, expanded_anchor_terms)
        expanded_coverage = self._anchor_coverage_ratio(keyword, expanded_anchor_terms)
        neighborhood_overlap = self._category_neighborhood_overlap_count(keyword, topic_context)
        if research_scope == "expanded" and neighborhood_overlap > 0 and anchor_overlap <= 0:
            anchor_overlap = 1
            anchor_coverage = max(anchor_coverage, 0.28)
        if core_anchor_terms and anchor_overlap <= 0:
            return 5.0
        if overlap <= 0:
            return 20.0
        anchor_bonus = 18.0 if anchor_overlap > 0 else 0.0
        coverage_bonus = min(18.0, anchor_coverage * 20.0)
        expanded_bonus = 8.0 if expanded_overlap > anchor_overlap else 0.0
        expanded_coverage_bonus = min(8.0, max(0.0, expanded_coverage - anchor_coverage) * 16.0)
        return round(
            min(
                100.0,
                24.0
                + (overlap / max(1, len(keyword_tokens))) * 58.0
                + anchor_bonus
                + coverage_bonus
                + expanded_bonus
                + expanded_coverage_bonus,
            ),
            2,
        )

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
        first_row = rows[0] if rows else {}
        serp_weakness = float(((first_row.get("trend_json") or {}).get("serp_weakness_score") or 0.0))
        competitor_support = float(((first_row.get("trend_json") or {}).get("competitor_support_score") or 0.0))
        confidence = 80 if article_friendly else 35
        if intent_label == "utility":
            confidence = 70
        if "near me" in keyword or "jobs" in keyword:
            confidence = 20
        return {
            "status": "competitor_proven",
            "validation_mode": "serp_probe_competitor_v2",
            "article_friendly": article_friendly,
            "article_intent_confidence": round(min(100.0, confidence + serp_weakness * 12.0 + competitor_support * 8.0), 1),
            "serp_weakness_score": round(serp_weakness, 4),
            "competitor_support_score": round(competitor_support, 4),
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

    def _classify_probe_intent(self, rows: List[Dict[str, Any]]) -> str:
        content_count = 0
        service_count = 0
        ecommerce_count = 0
        tool_count = 0
        for row in rows[:10]:
            url = str(row.get("url") or "")
            title = str(row.get("title") or "")
            if self._is_usable_content_page(url, title):
                content_count += 1
            if self._is_service_or_ecommerce_page(url, title):
                if any(token in f"{url} {title}".lower() for token in ["/product/", "/category/", "/shop/", "buy now", "price"]):
                    ecommerce_count += 1
                else:
                    service_count += 1
            if any(token in f"{url} {title}".lower() for token in ["calculator", "tool", "/tool/", "/tools/"]):
                tool_count += 1
        if tool_count >= 4:
            return "tool"
        if ecommerce_count >= 4:
            return "ecommerce"
        if service_count >= 4:
            return "service"
        if content_count >= 3:
            return "article"
        return "mixed"

    def _is_usable_content_page(self, url: str, title: str) -> bool:
        normalized = f"{url} {title}".lower()
        if not url:
            return False
        if self._is_service_or_ecommerce_page(url, title):
            return False
        if any(token in normalized for token in ["/blog/", "/guide", "/guides/", "/resources/", "/article", "/learn/", "/news/"]):
            return True
        if any(token in normalized for token in [" how ", " worth ", " roi ", " should ", " best ", " compare ", " vs "]):
            return True
        path = urlparse(url).path.strip("/")
        title_starts_with_number = bool(re.match(r"^\d+\b", str(title or "").strip().lower()))
        editorial_terms = any(
            token in normalized
            for token in [
                "simplify",
                "tips",
                "ways",
                "ideas",
                "habits",
                "routine",
                "routines",
                "technology tools",
                "productivity tools",
                "smart home",
                "life",
            ]
        )
        has_slug = bool(path and "-" in path)
        deepish_path = len([part for part in path.split("/") if part]) >= 1
        return title_starts_with_number or editorial_terms or (has_slug and deepish_path)

    def _is_service_or_ecommerce_page(self, url: str, title: str) -> bool:
        normalized = f"{url} {title}".lower()
        return any(
            token in normalized
            for token in [
                "/services/",
                "/service/",
                "/product/",
                "/products/",
                "/category/",
                "/collections/",
                "/shop/",
                "near me",
                "book now",
                "buy now",
                "request a quote",
            ]
        )

    def _is_authority_domain(self, domain: str) -> bool:
        normalized = str(domain or "").lower().strip()
        if not normalized:
            return True
        if normalized in self.EXCLUDED_AUTHORITY_DOMAINS:
            return True
        return any(
            token in normalized
            for token in [".gov", "wikipedia.", "reddit.", "forbes.", "amazon.", "investopedia.", "nerdwallet."]
        )

    def _is_niche_competitor(self, domain: str, url: str) -> bool:
        normalized = str(domain or urlparse(str(url or "")).netloc).lower().strip()
        if not normalized or self._is_authority_domain(normalized):
            return False
        return normalized.count(".") >= 1 and not any(
            token in normalized for token in ["youtube.", "facebook.", "instagram.", "linkedin."]
        )

    def _is_weak_content_page(self, title: str, snippet: str, url: str) -> bool:
        normalized_title = str(title or "").strip().lower()
        normalized_snippet = str(snippet or "").strip().lower()
        path = urlparse(str(url or "")).path.strip("/").lower()
        generic_title = len(normalized_title.split()) <= 5 or any(
            token in normalized_title for token in ["guide", "overview", "tips", "basics"]
        )
        shallow_path = path.count("/") == 0
        thin_snippet = len(normalized_snippet.split()) < 12
        return generic_title or shallow_path or thin_snippet

    def _keyword_contains_brand(self, keyword: str, topic_context: Dict[str, Any]) -> bool:
        normalized = self._normalize_keyword_key(keyword)
        project = topic_context.get("project") or {}
        brand_candidates = [
            str(project.get("domain") or "").split(".")[0],
            str(project.get("app_name") or ""),
        ]
        for brand in brand_candidates:
            brand_key = self._normalize_keyword_key(brand)
            if brand_key and brand_key in normalized:
                return True
        return False

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
        tokens = [token for token in normalized.split(" ") if token]
        if not tokens:
            return ""
        significant_tokens = [token for token in tokens if token not in self.QUERY_STOPWORDS]
        trimmed_tokens = [token for token in significant_tokens if token not in self.GENERIC_TOPIC_TERMS]
        if len(trimmed_tokens) >= 2:
            tokens = trimmed_tokens
        elif len(significant_tokens) >= 2:
            tokens = significant_tokens
        if len(tokens) < 2:
            return ""
        if len(tokens) > 6:
            tokens = tokens[:6]
        phrase = " ".join(tokens)
        if self._has_repeated_halves(phrase):
            return ""
        return phrase

    def _clean_probe_query(self, text: str) -> str:
        normalized = self._normalize_keyword_key(text)
        if not normalized:
            return ""
        tokens = [token for token in normalized.split(" ") if token]
        if not tokens:
            return ""
        significant_tokens = [token for token in tokens if token not in self.QUERY_STOPWORDS]
        trimmed_tokens = [token for token in significant_tokens if token not in self.GENERIC_TOPIC_TERMS]
        if len(trimmed_tokens) >= 2:
            tokens = trimmed_tokens
        elif len(significant_tokens) >= 2:
            tokens = significant_tokens
        if len(tokens) < 2:
            return ""
        if len(tokens) > 10:
            tokens = tokens[:10]
        phrase = " ".join(tokens)
        if self._has_repeated_halves(phrase):
            return ""
        return phrase

    def _build_probe_fallback_query(self, text: str) -> str:
        normalized = self._normalize_keyword_key(text)
        if not normalized:
            return ""
        tokens = [token for token in normalized.split(" ") if token]
        if not tokens:
            return ""

        preferred = [token for token in tokens if token not in self.QUERY_STOPWORDS]
        if len(preferred) < 2:
            preferred = tokens

        if len(preferred) < 2:
            return ""

        if len(preferred) > 8:
            preferred = preferred[:8]

        phrase = " ".join(preferred).strip()
        if self._has_repeated_halves(phrase):
            return ""
        return phrase

    async def _generate_llm_seed_keywords(
        self,
        topic_context: Dict[str, Any],
        deterministic_seeds: List[str],
        topic_components: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
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
        component_text = self._serialize_topic_components(topic_components or [])

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
Topic Components:
{component_text}

Goal:
- Automatically infer 4-6 search lanes a real person would explore around this topic.
- Translate abstract topic wording into practical search language.
- Produce seed phrases that a real person would type, not internal strategy labels.
- If the topic is abstract, first ground it in concrete subcomponents, need-buckets, or use-cases before proposing lanes.

Critical Priority Rule:
- Topic title, decision focus, and angle question are the source of truth.
- Category path, project/domain context, and audience are only supporting hints.
- If the category context conflicts with the topic itself, IGNORE the conflicting category context.
- Preserve the concrete subject of the topic, not just the abstract decision frame.
- If the topic is about a specific consumer object or comparison (for example EV vs hybrid ownership), seeds must keep that object/domain explicit instead of collapsing into a generic finance or business phrase.
- Honor the target audience. If the audience is consumer-facing, avoid drifting into enterprise, procurement, supply chain, or industrial search language unless the topic clearly asks for it.
- When topic components are provided, stay anchored to those components while keeping the website/category context in mind.

Lane Design Rules:
- Favor concrete user search lanes grounded in the topic.
- Example lanes may include durability, maintenance, compatibility, lifecycle, replacement, pricing, resale, support, upgradeability, or risk when relevant.
- Do not limit yourself to those examples if the topic clearly suggests stronger lanes such as engineering, thermal management, materials, firmware, batteries, modularity, or environmental control.
- Do not drift into adjacent business categories unless the topic clearly asks for that.
- Each lane should represent a distinct user search path.
- For abstract topics, cover multiple concrete components instead of repeating one abstract phrase.

Seed Rules:
- Each seed must be 2-5 words.
- Prefer plain English over consultant-speak.
- Keep each seed query-like and natural.
- Include pricing, comparison, alternatives, tool, maintenance, support, upgrade, or failure-mode wording only when it genuinely fits the topic.
- Avoid headings, punctuation-heavy phrasing, and sentence fragments.
- Avoid generic filler like "ultimate guide", "best guide", "tips", or "overview".
- Avoid phrases that are too broad to be useful.

Output Contract:
Return ONLY plain text using these exact delimiters:
LANE:: short lane name
WHY:: one short reason
SEED:: seed phrase one
SEED:: seed phrase two
SEED:: seed phrase three
ENDLANE

Requirements:
- 4 to 6 lanes
- 3 to 5 seeds per lane
- Every seed must appear on its own SEED:: line
- Do not return JSON
- Do not return bullets
- Do not return commentary
"""
        try:
            response = await asyncio.wait_for(
                llm_service.generate_text(
                    prompt=prompt,
                    max_tokens=500,
                    temperature=0.2,
                    task_role=LLM_ROLE_RESEARCH,
                ),
                timeout=25.0,
            )
            raw_content = response.content or ""
            parsed = self._extract_delimited_seed_lines(raw_content)
            parse_strategy = "delimited"
            if not parsed:
                parsed = self._extract_fallback_seed_lines(raw_content)
                parse_strategy = "fallback_lines"
            seeds: List[str] = []
            seen = set()
            rejected_candidates: List[str] = []
            for raw in parsed:
                normalized = self._normalize_llm_seed_phrase(raw)
                if not normalized:
                    cleaned = self._normalize_keyword_key(raw)
                    if cleaned:
                        rejected_candidates.append(cleaned)
                    continue
                key = normalized.lower()
                if key in seen:
                    continue
                seen.add(key)
                seeds.append(normalized)
            if len(seeds) < 6:
                salvage_candidates = self._salvage_seed_candidates_from_hints(
                    llm_candidates=parsed,
                    deterministic_hints=deterministic_seeds + self._component_hint_keywords(topic_components or []),
                    existing_seeds=seeds,
                )
                parse_strategy = f"{parse_strategy}+salvage" if salvage_candidates else parse_strategy
                for candidate in salvage_candidates:
                    key = candidate.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    seeds.append(candidate)
            logger.info(
                "LLM topic seed generation topic=%r generated=%s strategy=%s sample=%s",
                topic_title,
                len(seeds),
                parse_strategy,
                seeds[:8],
            )
            return {
                "accepted_seeds": seeds[:24],
                "raw_output": raw_content,
                "parse_strategy": parse_strategy,
                "raw_seed_count": len(parsed),
                "accepted_seed_count": len(seeds[:24]),
                "rejected_candidates": rejected_candidates[:30],
            }
        except Exception as exc:
            logger.warning(
                "LLM topic seed generation failed topic=%r err=%s",
                topic_title,
                exc,
            )
            return {
                "accepted_seeds": [],
                "raw_output": "",
                "parse_strategy": "exception",
                "raw_seed_count": 0,
                "accepted_seed_count": 0,
                "rejected_candidates": [],
            }

    async def _decompose_topic_into_search_components(
        self,
        topic_context: Dict[str, Any],
        deterministic_seeds: List[str],
    ) -> List[Dict[str, Any]]:
        topic = topic_context.get("topic") or {}
        project = topic_context.get("project") or {}
        primary_category = topic_context.get("primary_category") or {}
        secondary_category = topic_context.get("secondary_category") or {}

        topic_title = str(topic.get("title") or "").strip()
        topic_description = str(topic.get("description") or "").strip()
        category_path = topic_context.get("category_path") or ""
        decision_focus = str(topic.get("decision_focus") or "").strip()
        angle_question = str(topic.get("angle_question") or "").strip()
        audience = str(topic.get("target_audience") or project.get("targetaudiencedescription") or "").strip()
        domain = str(project.get("domain") or project.get("app_name") or "").strip()
        primary_name = str(primary_category.get("name") or "").strip()
        secondary_name = str(secondary_category.get("name") or "").strip()
        hint_text = ", ".join(deterministic_seeds[:10])

        prompt = f"""
Role: You are a content strategist and SEO researcher.

Topic Title: {topic_title}
Topic Description: {topic_description}
Category Path: {category_path}
Primary Category: {primary_name}
Secondary Category: {secondary_name}
Decision Focus: {decision_focus}
Angle Question: {angle_question}
Target Audience: {audience}
Project / Domain Context: {domain}
Existing Seed Hints: {hint_text}

Goal:
- Break this topic into 4 to 8 concrete search subtopics/components.
- Each subtopic should represent a distinct part of the topic a real person would search about.
- Stay tightly aligned to the website/category context and target audience.
- For abstract topics, convert the abstraction into concrete need-buckets, use-cases, or decision-buckets.

Rules:
- Subtopics must stay inside the actual topic, not drift into adjacent industries.
- Use plain-English subtopic names.
- Query spaces should be specific search queries or short search phrases that describe what people might search within that subtopic.
- Prioritize subtopics that can produce useful seed keywords.
- Avoid vague philosophical or poetic framings if the topic can be broken into practical user questions.

Output contract:
Return ONLY plain text with these delimiters:
COMPONENT:: short component name
WHY:: one short reason
QUERYSPACE:: short search-like hint
QUERYSPACE:: short search-like hint
ENDCOMPONENT

Requirements:
- 4 to 8 components
- 6 to 10 QUERYSPACE lines per component
"""
        try:
            response = await asyncio.wait_for(
                llm_service.generate_text(
                    prompt=prompt,
                    max_tokens=350,
                    temperature=0.2,
                    task_role=LLM_ROLE_RESEARCH,
                ),
                timeout=20.0,
            )
            raw_components = self._extract_topic_components(response.content or "")
            return self._filter_topic_components(raw_components, topic_context)
        except Exception as exc:
            logger.warning(
                "LLM topic component decomposition failed topic=%r err=%s",
                topic_title,
                exc,
            )
            return []

    def _extract_topic_components(self, content: str) -> List[Dict[str, Any]]:
        if not content:
            return []
        components: List[Dict[str, Any]] = []
        current: Dict[str, Any] = {}
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("COMPONENT::"):
                if current.get("name"):
                    components.append(current)
                current = {
                    "name": line.split("::", 1)[1].strip(),
                    "why": "",
                    "query_spaces": [],
                }
            elif upper.startswith("WHY::"):
                if current:
                    current["why"] = line.split("::", 1)[1].strip()
            elif upper.startswith("QUERYSPACE::"):
                if current:
                    value = self._clean_seed_phrase(line.split("::", 1)[1].strip())
                    if value:
                        current.setdefault("query_spaces", []).append(value)
            elif upper.startswith("ENDCOMPONENT"):
                if current.get("name"):
                    components.append(current)
                current = {}

        if current.get("name"):
            components.append(current)

        cleaned: List[Dict[str, Any]] = []
        seen = set()
        for component in components:
            name = str(component.get("name") or "").strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append({
                "name": name,
                "why": str(component.get("why") or "").strip(),
                "query_spaces": self._merge_strings(component.get("query_spaces"), []),
            })
            if len(cleaned) >= 5:
                break
        return cleaned

    def _serialize_topic_components(self, components: List[Dict[str, Any]]) -> str:
        if not components:
            return "None"
        lines: List[str] = []
        for component in components[:5]:
            name = str(component.get("name") or "").strip()
            why = str(component.get("why") or "").strip()
            query_spaces = ", ".join((component.get("query_spaces") or [])[:4])
            if not name:
                continue
            lines.append(f"- {name}: {why or 'Concrete search component'}")
            if query_spaces:
                lines.append(f"  Query spaces: {query_spaces}")
        return "\n".join(lines) if lines else "None"

    def _component_query_seed_candidates(self, components: List[Dict[str, Any]]) -> List[str]:
        candidates: List[str] = []
        seen = set()
        for component in components[:8]:
            component_name = self._clean_seed_phrase(component.get("name") or "")
            if component_name and component_name.lower() not in seen and self._looks_like_useful_seed(component_name):
                seen.add(component_name.lower())
                candidates.append(component_name)
            for query_space in (component.get("query_spaces") or [])[:10]:
                cleaned = self._clean_seed_phrase(query_space)
                if not cleaned or cleaned.lower() in seen:
                    continue
                if not self._looks_like_useful_seed(cleaned):
                    continue
                seen.add(cleaned.lower())
                candidates.append(cleaned)
        return candidates[:40]

    def _diversify_seed_pool(
        self,
        topic_components: List[Dict[str, Any]],
        llm_seeds: List[str],
        component_query_seeds: List[str],
        deterministic_seeds: List[str],
    ) -> List[str]:
        ordered: List[str] = []
        seen = set()

        # First take a couple of seeds per component to force diversity.
        for component in topic_components[:8]:
            per_component: List[str] = []
            component_name = self._clean_seed_phrase(component.get("name") or "")
            if component_name and self._looks_like_useful_seed(component_name):
                per_component.append(component_name)
            for query_space in (component.get("query_spaces") or [])[:10]:
                cleaned = self._clean_seed_phrase(query_space)
                if cleaned and self._looks_like_useful_seed(cleaned):
                    per_component.append(cleaned)
            added = 0
            for candidate in per_component:
                key = candidate.lower()
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(candidate)
                added += 1
                if added >= 2:
                    break

        for pool in (llm_seeds, component_query_seeds, deterministic_seeds):
            for candidate in pool:
                cleaned = self._clean_seed_phrase(candidate)
                if not cleaned:
                    continue
                key = cleaned.lower()
                if key in seen:
                    continue
                if not self._looks_like_useful_seed(cleaned):
                    continue
                seen.add(key)
                ordered.append(cleaned)

        return ordered[:24]

    def _component_hint_keywords(self, components: List[Dict[str, Any]]) -> List[str]:
        hints: List[str] = []
        for component in components[:5]:
            name = self._clean_seed_phrase(component.get("name") or "")
            if name:
                hints.append(name)
            for query_space in (component.get("query_spaces") or [])[:4]:
                cleaned = self._clean_seed_phrase(query_space)
                if cleaned:
                    hints.append(cleaned)
        return self._merge_strings(hints, [])

    def _filter_topic_components(
        self,
        components: List[Dict[str, Any]],
        topic_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not components:
            return []

        core_anchors = self._core_topic_anchor_terms(topic_context)
        if not core_anchors:
            return components[:5]

        filtered: List[Dict[str, Any]] = []
        seen_names = set()

        for component in components:
            name = str(component.get("name") or "").strip()
            if not name:
                continue

            clean_name = self._clean_seed_phrase(name) or self._normalize_keyword_key(name)
            if not clean_name:
                continue

            query_spaces = []
            for query_space in component.get("query_spaces") or []:
                cleaned = self._clean_seed_phrase(query_space) or self._normalize_keyword_key(query_space)
                if not cleaned:
                    continue
                if self._anchor_overlap_count(cleaned, core_anchors) <= 0:
                    continue
                query_spaces.append(cleaned)

            name_overlap = self._anchor_overlap_count(clean_name, core_anchors)
            query_overlap = any(self._anchor_overlap_count(query, core_anchors) > 0 for query in query_spaces)
            if name_overlap <= 0 and not query_overlap:
                continue

            key = clean_name.lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            filtered.append({
                "name": clean_name,
                "why": str(component.get("why") or "").strip(),
                "query_spaces": self._merge_strings(query_spaces, []),
            })
            if len(filtered) >= 5:
                break

        return filtered

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

    def _extract_delimited_seed_lines(self, content: str) -> List[str]:
        if not content:
            return []
        seeds: List[str] = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.upper().startswith("SEED::"):
                value = stripped.split("::", 1)[1].strip()
                if value:
                    seeds.append(value)
        return seeds

    def _extract_fallback_seed_lines(self, content: str) -> List[str]:
        if not content:
            return []
        candidates: List[str] = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            upper = stripped.upper()
            if upper.startswith(("LANE::", "WHY::", "ENDLANE")):
                continue
            if "::" in stripped:
                stripped = stripped.split("::", 1)[1].strip()
            candidates.extend(self._extract_seed_candidates(stripped))
        return candidates

    def _normalize_llm_seed_phrase(self, text: str) -> str:
        phrase = self._clean_seed_phrase(text)
        if not phrase:
            return ""
        if not self._looks_human_search_like(phrase):
            return ""
        return phrase

    def _looks_human_search_like(self, phrase: str) -> bool:
        lowered = phrase.lower().strip()
        if not lowered:
            return False
        tokens = lowered.split()
        if len(tokens) < 2 or len(tokens) > 6:
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

    def _salvage_seed_candidates_from_hints(
        self,
        llm_candidates: List[str],
        deterministic_hints: List[str],
        existing_seeds: List[str],
    ) -> List[str]:
        if len(existing_seeds) >= 6:
            return []
        hint_tokens = set()
        for candidate in llm_candidates[:24]:
            hint_tokens.update(self._meaningful_tokens(candidate))

        salvaged: List[str] = []
        existing_keys = {seed.lower() for seed in existing_seeds}
        for hint in deterministic_hints:
            cleaned = self._clean_seed_phrase(hint)
            if not cleaned or cleaned.lower() in existing_keys:
                continue
            if not self._looks_like_useful_seed(cleaned):
                continue
            if hint_tokens:
                overlap = len(set(self._meaningful_tokens(cleaned)) & hint_tokens)
                if overlap <= 0:
                    continue
            salvaged.append(cleaned)
            existing_keys.add(cleaned.lower())
            if len(existing_seeds) + len(salvaged) >= 8:
                break
        return salvaged

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

    def _category_neighborhood_overlap_count(self, keyword: str, topic_context: Dict[str, Any]) -> int:
        category_terms = self._token_set(
            " ".join(
                [
                    str((topic_context.get("primary_category") or {}).get("name") or ""),
                    str((topic_context.get("secondary_category") or {}).get("name") or ""),
                    str((topic_context.get("primary_category") or {}).get("description") or ""),
                    str((topic_context.get("secondary_category") or {}).get("description") or ""),
                ]
            )
        )
        if not category_terms:
            return 0
        return len(self._token_set(keyword) & category_terms)

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

    def _core_topic_anchor_terms(
        self,
        topic_context: Dict[str, Any],
    ) -> set[str]:
        topic = topic_context.get("topic") or {}
        raw_parts: List[str] = [
            str(topic.get("title") or ""),
            str(topic.get("decision_focus") or ""),
            str(topic.get("angle_question") or ""),
        ]
        raw_parts.extend([str(term).strip() for term in (topic.get("related_terms") or []) if str(term).strip()])
        raw_text = " ".join(raw_parts)
        base_tokens = [
            token for token in self._meaningful_tokens(raw_text)
            if token not in self.GENERIC_TOPIC_TERMS
        ]
        anchors: set[str] = set()
        for token in base_tokens:
            anchors.add(token)
            anchors.update(self.TOKEN_ALIAS_MAP.get(token, set()))
        return anchors

    def _topic_anchor_terms(
        self,
        topic_context: Dict[str, Any],
        topic_components: Optional[List[Dict[str, Any]]] = None,
    ) -> set[str]:
        anchors = set(self._core_topic_anchor_terms(topic_context))
        for component in topic_components or []:
            component_parts = [str(component.get("name") or "")]
            component_parts.extend([str(term).strip() for term in (component.get("query_spaces") or []) if str(term).strip()])
            for token in self._meaningful_tokens(" ".join(component_parts)):
                if token in self.GENERIC_TOPIC_TERMS:
                    continue
                anchors.add(token)
                anchors.update(self.TOKEN_ALIAS_MAP.get(token, set()))
        return anchors

    def _anchor_overlap_count(self, keyword: str, anchor_terms: set[str]) -> int:
        if not anchor_terms:
            return 0
        keyword_tokens = set(self._meaningful_tokens(keyword))
        return len(keyword_tokens & anchor_terms)

    def _anchor_coverage_ratio(self, keyword: str, anchor_terms: set[str]) -> float:
        if not anchor_terms:
            return 0.0
        keyword_tokens = set(self._meaningful_tokens(keyword))
        if not keyword_tokens:
            return 0.0
        return len(keyword_tokens & anchor_terms) / max(1, len(keyword_tokens))

    def _audience_mode(self, topic_context: Dict[str, Any]) -> str:
        topic = topic_context.get("topic") or {}
        project = topic_context.get("project") or {}
        combined = " ".join([
            str(topic.get("target_audience") or ""),
            str(project.get("targetaudiencedescription") or ""),
            str(project.get("site_description") or project.get("websitedescription") or ""),
        ]).lower()
        if any(term in combined for term in ["consumer", "homeowner", "buyer", "driver", "shopper", "family", "household"]):
            return "consumer"
        if any(term in combined for term in ["investor", "operator", "enterprise", "business", "b2b", "procurement", "finance team"]):
            return "business"
        return "general"

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
