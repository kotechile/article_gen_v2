"""
Content Ideas API endpoints.

Provides list, publish, and delete actions used by the frontend Idea Burst flow.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
import ast
from datetime import datetime
from uuid import uuid4
from flask import Blueprint, jsonify, request

from ...core.models.errors import ErrorResponse
from ...api.middleware.auth import require_api_key
from ...integrations.dataforseo import dataforseo_api
from ...services.affiliate_research_service import AffiliateResearchService

try:
    from supabase_client import get_supabase_client
except ImportError:
    import sys
    import os
    sys.path.append(os.getcwd())
    from supabase_client import get_supabase_client


logger = logging.getLogger(__name__)

content_ideas_bp = Blueprint("content_ideas", __name__, url_prefix="/api/content-ideas")
affiliate_research_service = AffiliateResearchService()

# Keep request latency bounded so reverse proxies don't return 504 on enrichment.
DATAFORSEO_BULK_TIMEOUT_SECONDS = 45
DATAFORSEO_KD_TIMEOUT_SECONDS = 20
AFFILIATE_ENRICH_TIMEOUT_SECONDS = 15
PER_IDEA_ENRICH_TIMEOUT_SECONDS = 70
MAX_KEYWORDS_FOR_METRICS = 15
MAX_RELATED_SEEDS = 3
MAX_RELATED_PER_SEED = 12

# Adaptive budget ladder (quality-first, cost-aware)
KEYWORD_QUALITY_MIN_NON_ZERO = 3
KEYWORD_QUALITY_MIN_BEST_VOLUME = 40
KEYWORD_BUDGET_LADDER = [
    {"name": "lite", "max_keywords_for_metrics": 12, "max_related_seeds": 0, "max_related_per_seed": 0},
    {"name": "balanced", "max_keywords_for_metrics": 20, "max_related_seeds": 2, "max_related_per_seed": 10},
    {"name": "deep", "max_keywords_for_metrics": 35, "max_related_seeds": 4, "max_related_per_seed": 18},
]

QUERY_STOPWORDS = {
    "a", "an", "the", "to", "for", "of", "in", "on", "at", "with", "without",
    "from", "into", "by", "my", "your", "our", "their", "you", "is", "are", "be", "have", "has", "had",
    "too", "much", "more", "less", "first", "second", "third", "best", "better",
    "what", "how", "when", "why", "can", "should", "could", "would", "do",
    "does", "did", "and", "or", "vs", "versus",
}

JARGON_STOPWORDS = {
    "framework", "paradigm", "architecture", "methodology", "optimization",
    "strategic", "strategy", "lens", "playbook",
}


def _get_admin_supabase_client(default_client):
    """Return a service-role Supabase client when available."""
    from supabase import create_client
    import os
    import httpx

    sb_url = os.environ.get('SUPABASE_URL')
    sb_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY') or os.environ.get('SUPABASE_SERVICE_KEY')

    if not (sb_url and sb_key):
        return default_client

    original_init = httpx.Client.__init__

    def new_init(self, *args, **kwargs):
        kwargs['verify'] = False
        original_init(self, *args, **kwargs)

    httpx.Client.__init__ = new_init

    try:
        return create_client(sb_url, sb_key)
    except Exception as admin_err:
        logger.error(f"Failed to initialize admin Supabase client: {admin_err}")
        return default_client


def _ensure_short_description(raw_description, title: str = "", keywords: list | None = None, subtopic: str = "") -> str:
    """Guarantee a non-empty description for ideas without truncating authored text."""
    description = str(raw_description or "").strip()
    keyword_list = [str(k).strip() for k in (keywords or []) if str(k).strip()]

    if not description:
        if keyword_list:
            description = f"Practical guide to {keyword_list[0]} with clear steps and decision-oriented takeaways."
        elif subtopic:
            description = f"Actionable breakdown of {subtopic} to help readers choose the right next move."
        elif title:
            description = f"Actionable breakdown of {title.lower()} with practical steps and clear outcomes."
        else:
            description = "Actionable, decision-focused article with practical steps readers can apply immediately."

    return description


def _coerce_json_field(value, default):
    """Normalize possibly-stringified JSON fields coming from legacy rows."""
    if value is None:
        return default
    if isinstance(value, type(default)):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return default
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, type(default)) else default
        except Exception:
            # Handle legacy python-literal strings and delimited lists.
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, type(default)):
                    return parsed
            except Exception:
                pass
            if isinstance(default, list):
                normalized = raw.replace("{", "").replace("}", "")
                return [part.strip() for part in re.split(r"[\n,]+", normalized) if part.strip()]
            return default
    return default


def _normalize_keyword_list(value) -> list[str]:
    """Normalize keyword fields that may arrive as arrays, JSON strings, or comma-separated text."""
    normalized: list[str] = []
    seen: set[str] = set()

    def _ingest(raw):
        if raw is None:
            return
        if isinstance(raw, (list, tuple, set)):
            for item in raw:
                _ingest(item)
            return
        if isinstance(raw, dict):
            keyword = raw.get("keyword") or raw.get("term")
            if keyword is not None:
                _ingest(keyword)
            return
        if isinstance(raw, str):
            parsed = _coerce_json_field(raw, [])
            if isinstance(parsed, list):
                cleaned_raw = raw.strip()
                if parsed != [cleaned_raw]:
                    _ingest(parsed)
                    return
            text = raw.strip()
        else:
            text = str(raw).strip()
        if not text:
            return
        for part in re.split(r"[\n,]+", text):
            candidate = str(part or "").strip()
            if not candidate:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(candidate)

    _ingest(value)
    return normalized


def _selected_keyword_metrics_to_map(payload) -> dict[str, dict]:
    """Convert selected_keyword_metrics_json payload into the shared metrics-map shape."""
    parsed = _coerce_json_field(payload, {})
    if not isinstance(parsed, dict):
        return {}

    metric_map: dict[str, dict] = {}

    def _add_row(row):
        if not isinstance(row, dict):
            return
        keyword = str(row.get("keyword") or "").strip()
        if not keyword:
            return
        metric_map[keyword] = {
            "search_volume": _coerce_numeric(row.get("search_volume"), int, None),
            "keyword_difficulty": _coerce_numeric(row.get("keyword_difficulty"), float, None),
            "cpc": _coerce_numeric(row.get("cpc"), float, None),
        }

    _add_row(parsed.get("primary"))
    for row in parsed.get("secondary") or []:
        _add_row(row)
    return metric_map


def _keyword_metrics_map_has_values(metrics_map) -> bool:
    parsed = _coerce_json_field(metrics_map, {})
    if not isinstance(parsed, dict):
        return False
    for row in parsed.values():
        if not isinstance(row, dict):
            continue
        if row.get("search_volume") is not None:
            return True
        if row.get("keyword_difficulty") is not None:
            return True
        if row.get("cpc") is not None:
            return True
    return False


def _build_titles_keyword_payload_from_idea(
    idea: dict,
    *,
    now_iso: str,
    selection_reason: str,
    strategy_version: str,
    selection_source: str,
) -> dict:
    """Build a Keyword Intelligence-compatible Titles payload from a content idea."""
    primary_candidates = _normalize_keyword_list(
        idea.get("primary_keywords") or idea.get("primary_keyword") or idea.get("keywords")
    )
    secondary_candidates = _normalize_keyword_list(
        idea.get("secondary_keywords") or idea.get("secondary_keywords_json")
    )
    candidate_terms = _normalize_keyword_list(idea.get("keyword_candidates_json"))

    if not primary_candidates and candidate_terms:
        primary_candidates = candidate_terms[:1]
    if not secondary_candidates and len(primary_candidates) > 1:
        secondary_candidates = primary_candidates[1:]
        primary_candidates = primary_candidates[:1]

    exact_keyword_metrics = _coerce_json_field(idea.get("keyword_metrics"), {})
    if not isinstance(exact_keyword_metrics, dict):
        exact_keyword_metrics = {}

    seo_offer_enrichment = {}
    idea_metadata = _coerce_json_field(idea.get("idea_metadata"), {})
    if isinstance(idea_metadata, dict):
        seo_offer_enrichment = _coerce_json_field(idea_metadata.get("seo_offer_enrichment"), {})
        if not isinstance(seo_offer_enrichment, dict):
            seo_offer_enrichment = {}

    metadata_keyword_metrics = _coerce_json_field(seo_offer_enrichment.get("keyword_metrics"), {})
    if not isinstance(metadata_keyword_metrics, dict):
        metadata_keyword_metrics = {}
    selected_keyword_metrics = _selected_keyword_metrics_to_map(idea.get("selected_keyword_metrics_json"))

    combined_metrics = dict(metadata_keyword_metrics)
    combined_metrics.update(selected_keyword_metrics)
    combined_metrics.update(exact_keyword_metrics)

    initial_primary_keyword = primary_candidates[0] if primary_candidates else ""
    normalized_metrics = {
        _normalize_keyword_term(keyword): value
        for keyword, value in combined_metrics.items()
        if isinstance(value, dict)
    }

    def _metric_row(keyword: str) -> dict:
        return normalized_metrics.get(_normalize_keyword_term(keyword), {}) if keyword else {}

    def _metric_strength(keyword: str) -> tuple[float, float, float]:
        row = _metric_row(keyword)
        volume = float(row.get("search_volume") or 0)
        cpc = float(row.get("cpc") or 0)
        kd = float(row.get("keyword_difficulty") or 0)
        return (volume, cpc, kd)

    primary_keyword = initial_primary_keyword
    alternative_keywords = _normalize_keyword_list([*secondary_candidates, *candidate_terms])
    primary_has_signal = any(value is not None for value in _metric_row(primary_keyword).values()) if primary_keyword else False
    if alternative_keywords and not primary_has_signal:
        best_keyword = max(alternative_keywords, key=_metric_strength, default="")
        if best_keyword and _metric_strength(best_keyword) > (0.0, 0.0, 0.0):
            primary_keyword = best_keyword

    secondary_keywords = []
    for keyword in [initial_primary_keyword, *secondary_candidates, *candidate_terms]:
        cleaned = str(keyword or "").strip()
        if not cleaned or cleaned.lower() == primary_keyword.lower():
            continue
        if cleaned not in secondary_keywords:
            secondary_keywords.append(cleaned)

    all_keywords = [primary_keyword] if primary_keyword else []
    for keyword in secondary_keywords:
        if keyword not in all_keywords:
            all_keywords.append(keyword)

    has_exact_metrics = _keyword_metrics_map_has_values(combined_metrics)
    selected_keyword_metrics_json = _build_keyword_metrics_payload(
        primary_keyword=primary_keyword,
        secondary_keywords=secondary_keywords,
        metrics_map=combined_metrics,
        source="dataforseo_exact" if has_exact_metrics else "llm_fallback",
        target_intent=idea.get("target_intent") or "informational",
    )
    primary_metric = (selected_keyword_metrics_json.get("primary") or {}) if primary_keyword else {}

    return {
        "Keywords": ", ".join(all_keywords),
        "keyword_candidates_json": all_keywords,
        "keyword_research_status": "ready" if primary_keyword else "fallback",
        "keyword_research_source": "dataforseo_exact" if has_exact_metrics else "llm_fallback",
        "keyword_research_confidence": 0.85 if has_exact_metrics else 0.35,
        "keyword_research_generated_at": now_iso,
        "primary_keywords": [primary_keyword] if primary_keyword else [],
        "primary_keyword": primary_keyword,
        "secondary_keywords": secondary_keywords,
        "secondary_keywords_json": secondary_keywords,
        "selected_keyword_search_volume": int(primary_metric.get("search_volume") or idea.get("total_search_volume") or 0),
        "selected_keyword_difficulty": float(primary_metric.get("keyword_difficulty") or idea.get("average_difficulty") or 0.0),
        "selected_keyword_intent": idea.get("target_intent") or "informational",
        "selected_keyword_metrics_json": selected_keyword_metrics_json,
        "keyword_selection_reason": selection_reason,
        "keyword_strategy_version": strategy_version,
        "keyword_selection_source": selection_source,
        "search_phrase": idea.get("search_phrase") or primary_keyword,
    }


def _sanitize_for_json(value):
    """Recursively sanitize values so PostgREST JSON encoding never fails."""
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _summarize_dataforseo_raw(raw_payload: dict | None) -> dict:
    """Extract high-signal diagnostics from raw DataForSEO response."""
    if not isinstance(raw_payload, dict):
        return {"raw_type": type(raw_payload).__name__}
    tasks = raw_payload.get("tasks")
    task_summaries = []
    if isinstance(tasks, list):
        for task in tasks[:5]:
            if not isinstance(task, dict):
                continue
            result = task.get("result")
            items_count = None
            if isinstance(result, list) and result and isinstance(result[0], dict):
                items_count = result[0].get("items_count")
            task_summaries.append({
                "task_status_code": task.get("status_code"),
                "task_status_message": task.get("status_message"),
                "result_count": task.get("result_count"),
                "items_count": items_count,
                "seed_keyword": ((task.get("data") or {}).get("keyword") if isinstance(task.get("data"), dict) else None),
            })
    return {
        "status_code": raw_payload.get("status_code"),
        "status_message": raw_payload.get("status_message"),
        "tasks_count": raw_payload.get("tasks_count"),
        "tasks_error": raw_payload.get("tasks_error"),
        "cost": raw_payload.get("cost"),
        "task_summaries": task_summaries,
    }


def _coerce_numeric(value, cast, default):
    try:
        if value is None:
            return default
        return cast(value)
    except Exception:
        return default


def _extract_existing_keyword_metrics_for_enrichment(idea: dict) -> tuple[dict, dict, list[dict]]:
    """
    Reuse already-persisted exact keyword metrics when available so enrichment
    does not re-call DataForSEO for ideas that already have real numbers.
    """
    if not isinstance(idea, dict):
        return {}, {}, []

    idea_metadata = _coerce_json_field(idea.get("idea_metadata"), {})
    seo_offer = (idea_metadata.get("seo_offer_enrichment") or {}) if isinstance(idea_metadata, dict) else {}

    keyword_metrics = _coerce_json_field(idea.get("keyword_metrics"), {})
    if not isinstance(keyword_metrics, dict) or not keyword_metrics:
        keyword_metrics = _coerce_json_field(
            seo_offer.get("keyword_metrics") or seo_offer.get("keyword_metrics_map"),
            {},
        )
    if not isinstance(keyword_metrics, dict):
        keyword_metrics = {}

    normalized_metrics: dict[str, dict] = {}
    for raw_keyword, raw_metric in keyword_metrics.items():
        keyword = _normalize_keyword_term(str(raw_keyword or ""))
        if not keyword or not isinstance(raw_metric, dict):
            continue
        normalized_metrics[keyword] = {
            "search_volume": _coerce_numeric(raw_metric.get("search_volume"), int, None),
            "keyword_difficulty": _coerce_numeric(raw_metric.get("keyword_difficulty"), float, None),
            "cpc": _coerce_numeric(raw_metric.get("cpc"), float, None),
        }

    non_zero_rows = [
        row for row in normalized_metrics.values()
        if int(row.get("search_volume") or 0) > 0
        or float(row.get("keyword_difficulty") or 0.0) > 0
        or float(row.get("cpc") or 0.0) > 0
    ]
    if not non_zero_rows:
        return {}, {}, []

    raw_dataforseo_output = _coerce_json_field(
        idea.get("raw_dataforseo_output") or seo_offer.get("raw_dataforseo_output"),
        {},
    )
    ranked_candidates = _coerce_json_field(seo_offer.get("keyword_ranked_candidates"), [])
    if not isinstance(ranked_candidates, list):
        ranked_candidates = []
    return normalized_metrics, raw_dataforseo_output if isinstance(raw_dataforseo_output, dict) else {}, ranked_candidates


def _hydrate_legacy_idea_seo_fields(row_copy: dict) -> dict:
    """
    Backfill SEO fields for legacy rows that were persisted with minimal columns.
    Prefers exact keyword-level data from `keyword_metrics`, then metadata payloads.
    """
    if not isinstance(row_copy, dict):
        return row_copy

    idea_metadata = _coerce_json_field(row_copy.get("idea_metadata"), {})
    seo_offer = (idea_metadata.get("seo_offer_enrichment") or {}) if isinstance(idea_metadata, dict) else {}

    # Normalize keyword metrics map from multiple possible payload locations.
    keyword_metrics = _coerce_json_field(row_copy.get("keyword_metrics"), {})
    if not isinstance(keyword_metrics, dict) or not keyword_metrics:
        keyword_metrics = _coerce_json_field(
            seo_offer.get("keyword_metrics") or seo_offer.get("keyword_metrics_map"),
            {},
        )
    if not isinstance(keyword_metrics, dict):
        keyword_metrics = {}
    row_copy["keyword_metrics"] = keyword_metrics
    row_copy["affiliate_offer_count"] = _coerce_numeric(
        row_copy.get("affiliate_offer_count"),
        int,
        _coerce_numeric(seo_offer.get("affiliate_offer_count"), int, 0),
    )
    affiliate_offers_preview = _coerce_json_field(
        row_copy.get("affiliate_offers_preview") or seo_offer.get("affiliate_offers_preview"),
        [],
    )
    row_copy["affiliate_offers_preview"] = affiliate_offers_preview if isinstance(affiliate_offers_preview, list) else []
    affiliate_search_status = str(
        row_copy.get("affiliate_search_status")
        or seo_offer.get("affiliate_search_status")
        or ("success" if row_copy["affiliate_offer_count"] is not None else "")
    ).strip()
    if affiliate_search_status:
        row_copy["affiliate_search_status"] = affiliate_search_status
    affiliate_search_error = str(
        row_copy.get("affiliate_search_error")
        or seo_offer.get("affiliate_search_error")
        or ""
    ).strip()
    if affiliate_search_error:
        row_copy["affiliate_search_error"] = affiliate_search_error

    # Reconstruct keyword arrays when missing.
    keywords = _coerce_json_field(row_copy.get("keywords"), [])
    primary_keywords = _coerce_json_field(row_copy.get("primary_keywords"), [])
    secondary_keywords = _coerce_json_field(row_copy.get("secondary_keywords"), [])

    if not keywords:
        metadata_keywords = _coerce_json_field(
            seo_offer.get("keywords_used") or idea_metadata.get("keywords_used"),
            [],
        )
        if metadata_keywords:
            keywords = metadata_keywords

    if not keywords and keyword_metrics:
        keywords = [str(key).strip() for key in keyword_metrics.keys() if str(key).strip()]

    if not primary_keywords and keywords:
        primary_keywords = list(keywords)
    if not secondary_keywords and len(primary_keywords) > 1:
        secondary_keywords = list(primary_keywords[1:])

    row_copy["keywords"] = keywords
    row_copy["primary_keywords"] = primary_keywords
    row_copy["secondary_keywords"] = secondary_keywords

    # Backfill aggregate metrics when not populated on the row.
    total_search_volume = _coerce_numeric(row_copy.get("total_search_volume"), int, None)
    average_difficulty = _coerce_numeric(row_copy.get("average_difficulty"), float, None)
    average_cpc = _coerce_numeric(row_copy.get("average_cpc"), float, None)

    if not total_search_volume:
        total_search_volume = _coerce_numeric(seo_offer.get("total_search_volume"), int, None)
    if not average_difficulty:
        average_difficulty = _coerce_numeric(seo_offer.get("average_difficulty"), float, None)
    if not average_cpc:
        average_cpc = _coerce_numeric(seo_offer.get("average_cpc"), float, None)

    if (
        (not total_search_volume or not average_difficulty or not average_cpc)
        and keyword_metrics
    ):
        volumes = []
        difficulties = []
        cpcs = []
        for value in keyword_metrics.values():
            if not isinstance(value, dict):
                continue
            vol = _coerce_numeric(value.get("search_volume"), int, 0)
            kd = _coerce_numeric(value.get("keyword_difficulty"), float, 0.0)
            cpc = _coerce_numeric(value.get("cpc"), float, 0.0)
            if vol > 0:
                volumes.append(vol)
            if kd > 0:
                difficulties.append(kd)
            if cpc > 0:
                cpcs.append(cpc)

        if not total_search_volume and volumes:
            total_search_volume = int(sum(volumes))
        if not average_difficulty and difficulties:
            average_difficulty = round(sum(difficulties) / len(difficulties), 1)
        if not average_cpc and cpcs:
            average_cpc = round(sum(cpcs) / len(cpcs), 2)

    row_copy["total_search_volume"] = int(total_search_volume) if total_search_volume else None
    row_copy["average_difficulty"] = round(float(average_difficulty), 1) if average_difficulty else None
    row_copy["average_cpc"] = round(float(average_cpc), 2) if average_cpc else None
    return row_copy


def _resolve_user_id_from_request(supabase, data=None):
    auth_header = request.headers.get("Authorization")
    user_id = None

    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split("Bearer ")[1]
        try:
            user_response = supabase.auth.get_user(token)
            if user_response and user_response.user:
                user_id = user_response.user.id
        except Exception as auth_error:
            logger.warning(f"Failed to validate token or get user: {auth_error}")

    if not user_id and data and data.get("user_id"):
        user_id = data["user_id"]

    return user_id


def _extract_keywords_for_enrichment(idea: dict) -> list[str]:
    """Collect a normalized keyword list from idea payload fields."""
    def _query_like_terms(raw: str) -> list[str]:
        cleaned = re.sub(r"[^a-zA-Z0-9\s-]", " ", str(raw or "").lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return []

        out = []
        for part in re.split(r"\b(?:or|vs|versus|and)\b", cleaned):
            part = re.sub(r"\s+", " ", part).strip()
            if not part:
                continue
            tokens = [t for t in part.split(" ") if t and t not in QUERY_STOPWORDS and t not in JARGON_STOPWORDS]
            if len(tokens) < 2:
                if tokens and len(tokens[0]) >= 4:
                    out.append(tokens[0])
                continue
            if len(tokens) > 3:
                tokens = tokens[:3]
            phrase = " ".join(tokens)
            out.append(phrase)
            if len(tokens) >= 3:
                out.append(" ".join(tokens[:2]))
                out.append(" ".join(tokens[-2:]))
        return out

    candidates = []
    for field in ("primary_keywords", "keywords", "secondary_keywords"):
        value = idea.get(field)
        if isinstance(value, list):
            for item in value:
                candidates.extend(_query_like_terms(str(item)))
        elif isinstance(value, str) and value.strip():
            # Handle comma-separated fallback shapes.
            for part in value.split(","):
                candidates.extend(_query_like_terms(part))

    search_phrase = str(idea.get("search_phrase") or "").strip()
    if search_phrase:
        candidates.extend(_query_like_terms(search_phrase))

    # Pass-1 seed keywords are stored in metadata and should feed DataForSEO related expansion.
    idea_metadata = idea.get("idea_metadata") or {}
    if isinstance(idea_metadata, dict):
        seed_pack = idea_metadata.get("keyword_seed_pack") or {}
        if isinstance(seed_pack, dict):
            for item in (seed_pack.get("input_keywords") or []):
                candidates.extend(_query_like_terms(str(item)))
        seed_inputs = idea_metadata.get("input_keywords")
        if isinstance(seed_inputs, list):
            for item in seed_inputs:
                candidates.extend(_query_like_terms(str(item)))

    if not candidates:
        # Fallback: derive a few keyword-like tokens from title.
        title = str(idea.get("title") or "").strip()
        candidates.extend(_query_like_terms(title))

    seen = set()
    normalized = []
    for kw in candidates:
        key = _normalize_keyword_term(kw)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized[:20]


def _normalize_keyword_term(term: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(term or "").strip().lower())
    cleaned = re.sub(r"[^\w\s-]", " ", cleaned)
    cleaned = re.sub(r"\b(202\d|203\d)\b", " ", cleaned)
    cleaned = re.sub(r"\b(vs|versus|and|or)\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    tokens = [tok for tok in cleaned.split(" ") if tok and tok not in QUERY_STOPWORDS and tok not in JARGON_STOPWORDS]
    if len(tokens) < 2:
        return tokens[0] if tokens and len(tokens[0]) >= 4 else ""
    if len(tokens) > 3:
        tokens = tokens[:3]
    return " ".join(tokens)


def _shorten_keyword_term(term: str) -> str:
    stop_phrases = {
        "how to", "best", "guide", "complete guide", "ultimate guide", "tips", "strategy", "strategies",
        "for beginners", "step by step", "in 2026", "in 2025",
    }
    normalized = _normalize_keyword_term(term)
    if not normalized:
        return ""
    for phrase in stop_phrases:
        normalized = normalized.replace(phrase, " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    tokens = [t for t in normalized.split(" ") if t]
    if len(tokens) <= 2:
        return normalized
    # Keep compact 2-3 token terms likely to have measurable demand.
    return " ".join(tokens[:3])


def _build_keyword_candidates(seed_keywords: list[str], title: str = "") -> list[str]:
    candidates: list[str] = []
    for kw in seed_keywords or []:
        base = _normalize_keyword_term(kw)
        if not base:
            continue
        candidates.append(base)
        short = _shorten_keyword_term(base)
        if short and short != base:
            candidates.append(short)
        tokens = short.split(" ") if short else base.split(" ")
        if len(tokens) >= 2:
            # Last 2 tokens often gives a broader, measurable phrase.
            tail = " ".join(tokens[-2:])
            if tail:
                candidates.append(tail)
            # First 2 tokens often preserves core entity.
            head = " ".join(tokens[:2])
            if head:
                candidates.append(head)

    if title:
        title_tokens = re.findall(r"[a-z0-9]{3,}", title.lower())
        if len(title_tokens) >= 2:
            candidates.append(" ".join(title_tokens[:2]))
            candidates.append(" ".join(title_tokens[-2:]))

    # de-dupe preserving order
    seen = set()
    out = []
    for c in candidates:
        c2 = _normalize_keyword_term(c)
        if not c2 or c2 in seen:
            continue
        seen.add(c2)
        out.append(c2)
    return out[:40]


def _simplify_seed_keyword(raw_term: str) -> str:
    """
    Convert arbitrary idea keyword text into a short, human-searchable seed phrase.
    Target: 1-3 tokens, plain language.
    """
    normalized = _normalize_keyword_term(raw_term)
    if not normalized:
        return ""

    tokens = [t for t in normalized.split(" ") if t]
    if not tokens:
        return ""

    # Remove filler/function words that often create broken fragments.
    stop_tokens = {
        "the", "a", "an", "for", "to", "of", "in", "on", "with", "by",
        "and", "or", "is", "are", "be", "you", "your", "my", "we", "our",
        "how", "what", "when", "why", "which",
    }
    cleaned = [t for t in tokens if t not in stop_tokens]
    if not cleaned:
        cleaned = tokens

    # Keep seed compact for DataForSEO related_keywords/live.
    cleaned = cleaned[:3]
    if not cleaned:
        return ""
    return " ".join(cleaned)


def _looks_human_seed(seed: str) -> bool:
    """Basic quality gate to avoid malformed seeds like single letters."""
    if not seed:
        return False
    tokens = [t for t in seed.split(" ") if t]
    if not tokens:
        return False
    if len(tokens) > 3:
        return False
    # Require at least one meaningful alpha token (3+ chars) or numeric signal.
    has_meaningful = any((len(t) >= 3 and re.search(r"[a-z]", t)) for t in tokens)
    has_numeric = any(re.search(r"\d", t) for t in tokens)
    return has_meaningful or has_numeric


def _score_seed_candidate(seed: str) -> int:
    """Heuristic scoring to prefer simple, query-like seed phrases."""
    tokens = [t for t in str(seed or "").split(" ") if t]
    if not tokens:
        return -999

    score = 0
    token_count = len(tokens)
    if token_count == 2:
        score += 6
    elif token_count == 1:
        score += 4
    elif token_count == 3:
        score += 3

    common_query_terms = {
        "mortgage", "stocks", "invest", "investment", "home", "equity",
        "market", "crash", "retirement", "debt", "payoff", "portfolio",
        "tax", "roi", "budget", "net", "worth", "cash", "rate",
    }
    score += sum(2 for t in tokens if t in common_query_terms)

    jargon_penalties = {
        "framework", "methodology", "architecture", "lens", "paradigm",
        "deployment", "synthesis", "liquidity", "audit", "ratio",
    }
    score -= sum(2 for t in tokens if t in jargon_penalties)

    if any(len(t) == 1 and t.isalpha() for t in tokens):
        score -= 3
    if any(t in QUERY_STOPWORDS for t in tokens):
        score -= 1
    return score


def _context_tokens_for_seed_ranking(text: str) -> list[str]:
    normalized = _normalize_keyword_term(text)
    if not normalized:
        return []
    return [t for t in normalized.split(" ") if t]


def _context_relevance_score(seed: str, title: str = "", description: str = "") -> int:
    """
    Score how tightly a seed keyword matches first-pass idea context.
    High relevance requires overlap with title and/or description terms.
    """
    seed_tokens = [t for t in _normalize_keyword_term(seed).split(" ") if t]
    if not seed_tokens:
        return -999

    title_norm = _normalize_keyword_term(title)
    desc_norm = _normalize_keyword_term(description)
    title_tokens = _context_tokens_for_seed_ranking(title)
    desc_tokens = _context_tokens_for_seed_ranking(description)
    context_tokens = set(title_tokens + desc_tokens)

    seed_text = " ".join(seed_tokens)
    score = 0

    # Strong signal: exact seed phrase appears in title/description.
    if title_norm and seed_text and seed_text in title_norm:
        score += 14
    if desc_norm and seed_text and seed_text in desc_norm:
        score += 8

    # Prefer candidates whose terms are present in idea context.
    overlap = sum(1 for token in seed_tokens if token in context_tokens)
    if overlap <= 0:
        score -= 10
    else:
        score += overlap * 5
        if overlap == len(seed_tokens):
            score += 4

    # Small bonus for alignment with title lead terms (usually main topic intent).
    title_lead = set(title_tokens[:6])
    if title_lead:
        lead_overlap = sum(1 for token in seed_tokens if token in title_lead)
        score += lead_overlap * 2

    return score


def _select_primary_seed_keyword(
    keywords: list[str],
    title: str = "",
    description: str = "",
    search_phrase: str = "",
) -> str:
    """
    Choose one simple seed keyword per idea for DataForSEO.
    Priority: explicit search phrase -> extracted keywords -> title-derived candidates.
    """
    ordered_inputs: list[str] = []
    if search_phrase:
        ordered_inputs.append(search_phrase)
    ordered_inputs.extend(keywords or [])
    ordered_inputs.extend(_build_keyword_candidates(keywords or [], title=title))

    seen = set()
    ranked_candidates: list[tuple[int, str]] = []
    for raw in ordered_inputs:
        seed = _simplify_seed_keyword(raw)
        if not seed or seed in seen:
            continue
        seen.add(seed)
        if _looks_human_seed(seed):
            base_score = _score_seed_candidate(seed)
            relevance_score = _context_relevance_score(
                seed,
                title=title,
                description=description,
            )
            ranked_candidates.append((base_score + relevance_score, seed))

    if ranked_candidates:
        ranked_candidates.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
        return ranked_candidates[0][1]

    # Last resort fallback: short title fragment.
    title_tokens = [t for t in _normalize_keyword_term(title).split(" ") if t][:3]
    fallback = " ".join(title_tokens)
    return fallback if _looks_human_seed(fallback) else ""


async def _fetch_metrics_map_for_keywords(
    keywords: list[str],
    max_keywords_for_metrics: int = MAX_KEYWORDS_FOR_METRICS,
    diagnostics: dict | None = None,
    raw_capture: dict | None = None,
) -> dict:
    """Fetch keyword metrics using DataForSEO Labs related_keywords/live and return normalized metrics map."""
    if not keywords:
        return {}

    scoped_keywords = keywords[:max_keywords_for_metrics]
    metrics_map: dict = {}
    if diagnostics is not None:
        diagnostics["scoped_keyword_count"] = len(scoped_keywords)
        diagnostics["scoped_keywords_sample"] = scoped_keywords[:10]
    try:
        related_response = await asyncio.wait_for(
            dataforseo_api.get_related_keywords_labs_live(
                scoped_keywords,
                limit_per_seed=20,
                return_raw=True,
            ),
            timeout=DATAFORSEO_BULK_TIMEOUT_SECONDS,
        )
        if isinstance(related_response, dict):
            related_rows = related_response.get("items") or []
            related_raw = related_response.get("raw")
        else:
            related_rows = related_response or []
            related_raw = None
        if raw_capture is not None and related_raw is not None:
            raw_capture["related_keywords_live"] = related_raw
        logger.info(
            "DataForSEO related_keywords raw summary: %s",
            _summarize_dataforseo_raw(related_raw),
        )
        if diagnostics is not None:
            diagnostics["related_rows_returned"] = len(related_rows or [])
            diagnostics["dataforseo_raw_summary"] = _summarize_dataforseo_raw(related_raw)
        for item in (related_rows or []):
            keyword = str(item.get("keyword") or "").strip().lower()
            if not keyword:
                continue
            raw_search_volume = item.get("search_volume")
            raw_cpc = item.get("cpc")
            raw_kd = item.get("keyword_difficulty")
            search_volume = int(raw_search_volume) if raw_search_volume is not None and str(raw_search_volume).strip() != "" else None
            cpc = float(raw_cpc) if raw_cpc is not None and str(raw_cpc).strip() != "" else None
            keyword_difficulty = float(raw_kd) if raw_kd is not None and str(raw_kd).strip() != "" else None
            metrics_map[keyword] = {
                "search_volume": search_volume,
                "cpc": cpc,
                "keyword_difficulty": keyword_difficulty,
            }
    except Exception:
        logger.warning("DataForSEO labs related_keywords request failed for candidate batch", exc_info=True)
        if diagnostics is not None:
            diagnostics["related_error"] = "related_keywords_request_failed"

    if diagnostics is not None:
        non_null_count = 0
        for row in metrics_map.values():
            if (
                row.get("search_volume") is not None
                or row.get("keyword_difficulty") is not None
                or row.get("cpc") is not None
            ):
                non_null_count += 1
        diagnostics["metrics_keywords_non_null"] = non_null_count

    return metrics_map


def _rank_keywords_by_opportunity(candidates: list[str], metrics_map: dict) -> list[dict]:
    ranked = []
    for kw in candidates:
        row = metrics_map.get(kw.lower(), {}) or {}
        vol = int(row.get("search_volume") or 0)
        kd = float(row.get("keyword_difficulty") or 0.0)
        cpc = float(row.get("cpc") or 0.0)
        # Opportunity: favor measurable volume + manageable difficulty.
        vol_score = min(vol / 50.0, 100.0)
        kd_score = max(0.0, 100.0 - kd)
        cpc_score = min(cpc * 10.0, 30.0)
        opp = (vol_score * 0.6) + (kd_score * 0.3) + (cpc_score * 0.1)
        ranked.append({
            "keyword": kw,
            "search_volume": vol,
            "keyword_difficulty": kd,
            "cpc": cpc,
            "opportunity": round(opp, 2),
        })
    ranked.sort(
        # Selection rule v2: prefer measurable keywords with lowest KD, then highest volume.
        # Unknown KD (0) is treated as low confidence and ranked after known KD.
        key=lambda x: (
            x["search_volume"] <= 0,
            (x["keyword_difficulty"] <= 0),
            x["keyword_difficulty"] if x["keyword_difficulty"] > 0 else 999.0,
            -x["search_volume"],
            -x["cpc"],
            -x["opportunity"],
        ),
    )
    return ranked


def _keyword_quality_summary(ranked_candidates: list[dict]) -> dict:
    non_zero = [row for row in ranked_candidates if int(row.get("search_volume") or 0) > 0]
    best = ranked_candidates[0] if ranked_candidates else {}
    return {
        "non_zero_count": len(non_zero),
        "best_volume": int(best.get("search_volume") or 0),
        "best_opportunity": float(best.get("opportunity") or 0.0),
        "best_keyword": str(best.get("keyword") or ""),
        "best_keyword_difficulty": float(best.get("keyword_difficulty") or 0.0),
    }


def _normalize_title_keyword_term(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s-]", " ", str(value or "").lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_keyword_metrics_from_dataforseo_raw(raw_payload: dict | None) -> dict:
    """
    Build a keyword->metrics map from raw DataForSEO related_keywords/live response.
    Pulls metrics from:
    - result[].seed_keyword_data
    - result[].items[].keyword_data
    """
    if not isinstance(raw_payload, dict):
        return {}

    out: dict[str, dict] = {}

    def _ingest_keyword_data(keyword_data: dict | None) -> None:
        if not isinstance(keyword_data, dict):
            return
        keyword = str(keyword_data.get("keyword") or "").strip().lower()
        if not keyword:
            return

        keyword_info = keyword_data.get("keyword_info") if isinstance(keyword_data.get("keyword_info"), dict) else {}
        keyword_props = keyword_data.get("keyword_properties") if isinstance(keyword_data.get("keyword_properties"), dict) else {}

        def _to_int(v):
            try:
                if v is None or str(v).strip() == "":
                    return None
                return int(float(v))
            except Exception:
                return None

        def _to_float(v):
            try:
                if v is None or str(v).strip() == "":
                    return None
                return float(v)
            except Exception:
                return None

        search_volume = _to_int(keyword_info.get("search_volume"))
        cpc = _to_float(keyword_info.get("cpc"))
        keyword_difficulty = _to_float(keyword_props.get("keyword_difficulty"))

        existing = out.get(keyword) or {}
        # Prefer rows that contain at least one exact metric.
        existing_score = int(existing.get("search_volume") is not None) + int(existing.get("keyword_difficulty") is not None) + int(existing.get("cpc") is not None)
        incoming_score = int(search_volume is not None) + int(keyword_difficulty is not None) + int(cpc is not None)
        if incoming_score >= existing_score:
            out[keyword] = {
                "search_volume": search_volume,
                "keyword_difficulty": keyword_difficulty,
                "cpc": cpc,
            }

    tasks = raw_payload.get("tasks")
    if not isinstance(tasks, list):
        return out

    for task in tasks:
        if not isinstance(task, dict):
            continue
        for result in (task.get("result") or []):
            if not isinstance(result, dict):
                continue
            seed_keyword_data = result.get("seed_keyword_data")
            if isinstance(seed_keyword_data, dict):
                _ingest_keyword_data(seed_keyword_data)
            elif isinstance(seed_keyword_data, list):
                for entry in seed_keyword_data:
                    if isinstance(entry, dict):
                        _ingest_keyword_data(entry)

            for item in (result.get("items") or []):
                if not isinstance(item, dict):
                    continue
                keyword_data = item.get("keyword_data")
                if isinstance(keyword_data, dict):
                    _ingest_keyword_data(keyword_data)

    return out


def _quality_gate_passed(summary: dict) -> bool:
    return (
        int(summary.get("non_zero_count") or 0) >= KEYWORD_QUALITY_MIN_NON_ZERO
        and int(summary.get("best_volume") or 0) >= KEYWORD_QUALITY_MIN_BEST_VOLUME
    )


def _build_keyword_metrics_payload(
    primary_keyword: str,
    secondary_keywords: list[str],
    metrics_map: dict,
    source: str,
    target_intent: str | None = None,
) -> dict:
    """Create a structured keyword metrics payload from exact per-keyword enrichment data."""
    normalized_map = {}
    for key, value in (metrics_map or {}).items():
        if not key:
            continue
        normalized_map[str(key).strip().lower()] = value or {}

    source_normalized = str(source or "").strip().lower()
    is_fallback_source = source_normalized in {"llm_fallback", "unknown", "aggregate_estimate"}
    default_metric_source = "dataforseo_exact" if not is_fallback_source else "llm_fallback"

    def _row(keyword: str) -> dict:
        raw = normalized_map.get(str(keyword or "").strip().lower(), {})
        raw_search_volume = raw.get("search_volume")
        raw_keyword_difficulty = raw.get("keyword_difficulty")
        raw_cpc = raw.get("cpc")
        search_volume = int(raw_search_volume) if raw_search_volume is not None and str(raw_search_volume).strip() != "" else None
        keyword_difficulty = float(raw_keyword_difficulty) if raw_keyword_difficulty is not None and str(raw_keyword_difficulty).strip() != "" else None
        cpc = float(raw_cpc) if raw_cpc is not None and str(raw_cpc).strip() != "" else None
        has_exact_metrics = search_volume is not None or keyword_difficulty is not None or cpc is not None
        is_estimated = is_fallback_source or not has_exact_metrics
        return {
            "keyword": str(keyword or "").strip(),
            "search_volume": search_volume,
            "keyword_difficulty": keyword_difficulty,
            "cpc": cpc,
            "metric_source": default_metric_source if is_estimated else "dataforseo_exact",
            "is_estimated": is_estimated,
        }

    primary = _row(primary_keyword) if primary_keyword else {
        "keyword": "",
        "search_volume": None,
        "keyword_difficulty": None,
        "cpc": None,
        "metric_source": default_metric_source,
        "is_estimated": True,
    }
    secondary = [_row(keyword) for keyword in (secondary_keywords or []) if str(keyword).strip()]
    return {
        "primary": {
            **primary,
            "intent": str(target_intent or "").strip().lower() or "informational",
        },
        "secondary": secondary,
        "candidate_count": len(([primary_keyword] if primary_keyword else []) + [k for k in (secondary_keywords or []) if str(k).strip()]),
        "source": source_normalized or "dataforseo_exact",
        "generated_at": datetime.utcnow().isoformat(),
    }


def _sync_titles_keyword_fields_from_idea(supabase, idea: dict, user_id: str, now_iso: str) -> int:
    """Update Titles keyword fields from a content_ideas row; returns number of rows updated."""
    update_payload = _build_titles_keyword_payload_from_idea(
        idea,
        now_iso=now_iso,
        selection_reason="Refreshed from content_ideas keyword enrichment.",
        strategy_version="phase1_v4",
        selection_source="rebuild_keyword_dossier_sync",
    )
    updated_rows = 0
    for where_key, where_value in (("source_idea_id", idea.get("id")), ("id", idea.get("titles_record_id"))):
        if not where_value:
            continue
        try:
            response = (
                supabase.table("Titles")
                .update(update_payload)
                .eq("user_id", user_id)
                .eq(where_key, where_value)
                .execute()
            )
            updated_rows += len(response.data or [])
        except Exception as update_error:
            # Backward-compatible fallback for deployments missing some columns.
            err = str(update_error)
            missing_cols = re.findall(r"Could not find the '([^']+)' column", err)
            fallback_payload = dict(update_payload)
            for col in missing_cols:
                fallback_payload.pop(col, None)
            if not fallback_payload:
                continue
            response = (
                supabase.table("Titles")
                .update(fallback_payload)
                .eq("user_id", user_id)
                .eq(where_key, where_value)
                .execute()
            )
            updated_rows += len(response.data or [])
    return updated_rows


async def _compute_idea_enrichment(idea: dict) -> dict:
    """
    Compute SEO/offer enrichment for one idea.

    Returns aggregate metrics and a compact offers preview.
    """
    idea_id = idea.get("id")
    start_ts = time.perf_counter()
    keywords = _extract_keywords_for_enrichment(idea)
    candidates = _build_keyword_candidates(keywords, title=str(idea.get("title") or ""))
    logger.info(
        "Enrichment start for idea_id=%s title=%s keyword_count=%s candidate_count=%s",
        idea_id,
        (idea.get("title") or "")[:120],
        len(keywords),
        len(candidates),
    )
    if not keywords:
        logger.warning("Enrichment aborted for idea_id=%s: no keywords extracted", idea_id)
        return {
            "keywords_used": [],
            "total_search_volume": 0,
            "average_cpc": 0.0,
            "average_difficulty": 0.0,
            "affiliate_offer_count": 0,
            "affiliate_offers": [],
            "raw_dataforseo_output": {
                "idea_id": idea_id,
                "captured_at": datetime.utcnow().isoformat(),
                "tiers": [],
                "error": "no_keywords_extracted",
            },
            "status": "failed",
            "reason": "No usable keywords found on idea",
        }

    total_search_volume = 0
    average_cpc = 0.0
    average_difficulty = 0.0
    cpc_count = 0
    kd_count = 0

    primary_seed_keyword = _select_primary_seed_keyword(
        keywords=keywords,
        title=str(idea.get("title") or ""),
        description=str(idea.get("description") or ""),
        search_phrase=str(idea.get("search_phrase") or ""),
    )
    working_candidates = [primary_seed_keyword] if primary_seed_keyword else list(candidates[:1])
    existing_metrics_map, existing_raw_dataforseo_output, existing_ranked_candidates = _extract_existing_keyword_metrics_for_enrichment(idea)

    fallback_seed_candidates: list[str] = []
    for raw_candidate in ([str(idea.get("search_phrase") or "")] + keywords + candidates):
        seed_candidate = _simplify_seed_keyword(raw_candidate)
        if not seed_candidate:
            continue
        if seed_candidate in working_candidates or seed_candidate in fallback_seed_candidates:
            continue
        if _looks_human_seed(seed_candidate):
            fallback_seed_candidates.append(seed_candidate)
    fallback_seed_candidates = fallback_seed_candidates[:5]
    metrics_map: dict = {}
    ranked_candidates: list[dict] = []
    tier_diagnostics: list[dict] = []
    dataforseo_calls = 0
    raw_dataforseo_output: dict = {}

    if existing_metrics_map:
        metrics_map = dict(existing_metrics_map)
        ranked_pool = list(dict.fromkeys(
            [primary_seed_keyword] +
            [str(k).strip().lower() for k in metrics_map.keys() if str(k).strip()] +
            [str(k).strip().lower() for k in (keywords or []) if str(k).strip()]
        ))
        ranked_candidates = _rank_keywords_by_opportunity(ranked_pool, metrics_map)
        if existing_ranked_candidates:
            ranked_candidates = ranked_candidates or existing_ranked_candidates
        raw_dataforseo_output = existing_raw_dataforseo_output or {
            "idea_id": idea_id,
            "captured_at": datetime.utcnow().isoformat(),
            "tiers": [],
            "source": "reused_saved_metrics",
        }
        tier_diagnostics.append({
            "tier": "reused_saved_metrics",
            "primary_seed_keyword": primary_seed_keyword,
            "calls": 0,
            "quality": _keyword_quality_summary(ranked_candidates),
        })
        logger.info(
            "Reused saved keyword metrics for idea_id=%s metric_keywords=%s",
            idea_id,
            len(metrics_map),
        )
    else:
        # Single DataForSEO call path for content ideas:
        # dataforseo_labs/google/related_keywords/live
        tier_diag = {
            "tier": "labs_related_keywords_live",
            "max_keywords_for_metrics": min(MAX_KEYWORDS_FOR_METRICS, len(working_candidates)),
            "primary_seed_keyword": primary_seed_keyword,
        }
        metrics_map = await _fetch_metrics_map_for_keywords(
            working_candidates,
            max_keywords_for_metrics=MAX_KEYWORDS_FOR_METRICS,
            diagnostics=tier_diag,
            raw_capture=tier_diag.setdefault("raw_dataforseo", {}),
        )
        non_zero_metric_rows = sum(
            1
            for row in (metrics_map or {}).values()
            if int(row.get("search_volume") or 0) > 0
            or float(row.get("keyword_difficulty") or 0.0) > 0
            or float(row.get("cpc") or 0.0) > 0
        )

        if non_zero_metric_rows == 0:
            logger.warning(
                "DataForSEO returned zero measurable metrics for primary seed idea_id=%s seed=%r; retrying fallback seeds=%s",
                idea_id,
                primary_seed_keyword,
                fallback_seed_candidates,
            )
            for fallback_seed in fallback_seed_candidates:
                retry_diag = {
                    "tier": "labs_related_keywords_live_retry",
                    "primary_seed_keyword": fallback_seed,
                    "max_keywords_for_metrics": 1,
                }
                retry_metrics = await _fetch_metrics_map_for_keywords(
                    [fallback_seed],
                    max_keywords_for_metrics=1,
                    diagnostics=retry_diag,
                    raw_capture=retry_diag.setdefault("raw_dataforseo", {}),
                )
                retry_non_zero = sum(
                    1
                    for row in (retry_metrics or {}).values()
                    if int(row.get("search_volume") or 0) > 0
                    or float(row.get("keyword_difficulty") or 0.0) > 0
                    or float(row.get("cpc") or 0.0) > 0
                )
                retry_diag["quality"] = {"non_zero_metric_rows": retry_non_zero}
                retry_diag["calls"] = 1
                tier_diagnostics.append(retry_diag)
                dataforseo_calls += 1
                if retry_non_zero > 0:
                    metrics_map = retry_metrics
                    working_candidates = [fallback_seed]
                    primary_seed_keyword = fallback_seed
                    tier_diag = retry_diag
                    logger.info(
                        "DataForSEO fallback seed succeeded idea_id=%s seed=%r non_zero_metric_rows=%s",
                        idea_id,
                        fallback_seed,
                        retry_non_zero,
                    )
                    break
        ranked_pool = list(dict.fromkeys(
            [primary_seed_keyword] +
            [str(k).strip().lower() for k in metrics_map.keys() if str(k).strip()] +
            [str(k).strip().lower() for k in (keywords or []) if str(k).strip()]
        ))
        ranked_candidates = _rank_keywords_by_opportunity(ranked_pool, metrics_map)
        summary = _keyword_quality_summary(ranked_candidates)
        tier_diag["quality"] = summary
        tier_diag["calls"] = max(1, int(tier_diag.get("calls") or 1))
        if not tier_diagnostics or tier_diagnostics[-1] is not tier_diag:
            tier_diagnostics.append(tier_diag)
        if dataforseo_calls <= 0:
            dataforseo_calls = 1

        # Persist exact raw response from DataForSEO when present.
        raw_dataforseo_output = (
            (tier_diag.get("raw_dataforseo") or {}).get("related_keywords_live")
            or {
                "idea_id": idea_id,
                "captured_at": datetime.utcnow().isoformat(),
                "tiers": [],
            }
        )
        raw_metrics_map = _extract_keyword_metrics_from_dataforseo_raw(raw_dataforseo_output)
        if raw_metrics_map:
            merged = dict(metrics_map or {})
            for kw, raw_row in raw_metrics_map.items():
                existing = merged.get(kw) or {}
                search_volume = existing.get("search_volume") if existing.get("search_volume") is not None else raw_row.get("search_volume")
                keyword_difficulty = existing.get("keyword_difficulty") if existing.get("keyword_difficulty") is not None else raw_row.get("keyword_difficulty")
                cpc = existing.get("cpc") if existing.get("cpc") is not None else raw_row.get("cpc")
                merged[kw] = {
                    "search_volume": search_volume,
                    "keyword_difficulty": keyword_difficulty,
                    "cpc": cpc,
                }
            metrics_map = merged
            logger.info(
                "Merged raw DataForSEO metrics into map idea_id=%s raw_metric_keywords=%s merged_metric_keywords=%s",
                idea_id,
                len(raw_metrics_map),
                len(metrics_map),
            )
        logger.info(
            "Enrichment DataForSEO raw selected idea_id=%s seed=%r raw_summary=%s",
            idea_id,
            primary_seed_keyword,
            _summarize_dataforseo_raw(raw_dataforseo_output),
        )

    ranked_with_metrics = [
        row for row in ranked_candidates
        if int(row.get("search_volume") or 0) > 0
        or float(row.get("keyword_difficulty") or 0.0) > 0
        or float(row.get("cpc") or 0.0) > 0
    ]
    selected_keywords = [row["keyword"] for row in (ranked_with_metrics[:12] or ranked_candidates[:5])]
    all_keywords_found = [row["keyword"] for row in ranked_candidates[:20] if str(row.get("keyword") or "").strip()]
    selected_metric_rows = [
        metrics_map.get(str(keyword).strip().lower(), {}) or {}
        for keyword in selected_keywords
    ]
    has_exact_keyword_metrics = any(
        (int(row.get("search_volume") or 0) > 0)
        or (float(row.get("keyword_difficulty") or 0.0) > 0)
        or (float(row.get("cpc") or 0.0) > 0)
        for row in selected_metric_rows
    )

    for keyword in selected_keywords:
        row = metrics_map.get(keyword.lower(), {})
        search_volume = int(row.get("search_volume") or 0)
        cpc = float(row.get("cpc") or 0.0)
        difficulty = float(row.get("keyword_difficulty") or 0.0)

        total_search_volume += search_volume
        if cpc > 0:
            average_cpc += cpc
            cpc_count += 1
        if difficulty > 0:
            average_difficulty += difficulty
            kd_count += 1

    average_cpc = round((average_cpc / cpc_count) if cpc_count else 0.0, 2)
    average_difficulty = round((average_difficulty / kd_count) if kd_count else 0.0, 1)

    search_term = str(idea.get("title") or keywords[0]).strip()
    affiliate_offer_count = 0
    affiliate_offers_preview = []
    affiliate_search_status = "not_run"
    affiliate_search_error = None
    try:
        affiliate_start = time.perf_counter()
        affiliate_result = await asyncio.wait_for(
            affiliate_research_service.search_affiliate_programs(
                search_term=search_term,
                niche=str(idea.get("content_type") or "").strip() or None,
                user_id=idea.get("user_id"),
            ),
            timeout=AFFILIATE_ENRICH_TIMEOUT_SECONDS
        )
        logger.info(
            "Affiliate enrichment completed for idea_id=%s in %.2fs",
            idea_id,
            time.perf_counter() - affiliate_start,
        )
        programs = affiliate_result.get("programs") or []
        affiliate_offer_count = len(programs)
        affiliate_search_status = "success"
        affiliate_offers_preview = [
            {
                "name": program.get("name"),
                "network": program.get("network"),
                "commission_rate": program.get("commission_rate"),
            }
            for program in programs[:5]
        ]
    except Exception as exc:
        affiliate_search_status = "failed"
        affiliate_search_error = str(exc)[:300]
        logger.warning("Affiliate search failed for idea_id=%s", idea_id, exc_info=True)

    total_elapsed = time.perf_counter() - start_ts
    logger.info(
        "Enrichment complete for idea_id=%s in %.2fs volume=%s cpc=%s kd=%s offers=%s calls=%s",
        idea_id,
        total_elapsed,
        int(total_search_volume),
        average_cpc,
        average_difficulty,
        affiliate_offer_count,
        dataforseo_calls,
    )

    return {
        "keywords_used": selected_keywords or keywords,
        "all_keywords_found": all_keywords_found or selected_keywords or keywords,
        "total_search_volume": int(total_search_volume),
        "average_cpc": average_cpc,
        "average_difficulty": average_difficulty,
        "has_exact_keyword_metrics": has_exact_keyword_metrics,
        "keyword_metrics_map": metrics_map,
        "keyword_ranked_candidates": ranked_candidates[:10],
        "keyword_quality_summary": _keyword_quality_summary(ranked_candidates),
        "keyword_budget_ladder_used": [],
        "dataforseo_diagnostics": {
            "tier_diagnostics": tier_diagnostics,
            "initial_keyword_count": len(keywords),
            "initial_candidate_count": len(candidates),
            "selected_keywords": selected_keywords,
            "all_keywords_found_count": len(all_keywords_found),
            "has_exact_keyword_metrics": has_exact_keyword_metrics,
        },
        "raw_dataforseo_output": raw_dataforseo_output,
        "dataforseo_call_count_estimate": dataforseo_calls,
        "affiliate_offer_count": affiliate_offer_count,
        "affiliate_offers": affiliate_offers_preview,
        "affiliate_search_status": affiliate_search_status,
        "affiliate_search_error": affiliate_search_error,
        "status": "enriched",
        "reason": None,
    }


def _persist_raw_trace_for_idea(
    supabase,
    idea: dict,
    user_id: str,
    now_iso: str,
    raw_output: dict | None,
    reason: str | None = None,
) -> None:
    """
    Best-effort debug persistence for raw DataForSEO output, including failed enrichments.
    Keeps output in both raw_dataforseo_output column and idea_metadata for backward compatibility.
    """
    safe_raw = _sanitize_for_json(raw_output or {})
    idea_metadata = dict(idea.get("idea_metadata") or {})
    seo_offer = dict((idea_metadata.get("seo_offer_enrichment") or {}))
    seo_offer["raw_dataforseo_output"] = safe_raw
    if reason:
        seo_offer["raw_trace_reason"] = reason
    seo_offer["raw_trace_updated_at"] = now_iso
    idea_metadata["seo_offer_enrichment"] = seo_offer

    for payload in (
        {
            "raw_dataforseo_output": safe_raw,
            "idea_metadata": _sanitize_for_json(idea_metadata),
            "updated_at": now_iso,
        },
        {
            "raw_dataforseo_output": safe_raw,
            "updated_at": now_iso,
        },
        {
            "idea_metadata": _sanitize_for_json(idea_metadata),
            "updated_at": now_iso,
        },
        {"updated_at": now_iso},
    ):
        try:
            supabase.table("content_ideas").update(_sanitize_for_json(payload)).eq("id", idea.get("id")).eq("user_id", user_id).execute()
            return
        except Exception as e:
            logger.warning(
                "Raw trace persistence failed for idea_id=%s payload_keys=%s err=%s",
                idea.get("id"),
                sorted((payload or {}).keys()),
                str(e)[:700],
            )
            continue


def _verify_enrichment_persistence(row: dict, expected_keywords: list[str], expected_raw: dict) -> bool:
    if not isinstance(row, dict):
        return False

    stored_raw = _coerce_json_field(row.get("raw_dataforseo_output"), {})
    raw_expected_non_empty = isinstance(expected_raw, dict) and bool(expected_raw)
    raw_persisted = (not raw_expected_non_empty) or (isinstance(stored_raw, dict) and bool(stored_raw))

    stored_keywords = _coerce_json_field(
        row.get("primary_keywords") or row.get("keywords"),
        [],
    )
    stored_keywords_norm = {
        str(keyword).strip().lower()
        for keyword in (stored_keywords or [])
        if str(keyword).strip()
    }
    expected_keywords_norm = [
        str(keyword).strip().lower()
        for keyword in (expected_keywords or [])
        if str(keyword).strip()
    ]
    keywords_persisted = not expected_keywords_norm or any(
        keyword in stored_keywords_norm for keyword in expected_keywords_norm
    )

    return raw_persisted and keywords_persisted


def _apply_enrichment_update_with_fallback(
    supabase,
    *,
    idea_id: str,
    user_id: str,
    payloads: list[dict],
    expected_keywords: list[str],
    expected_raw: dict,
) -> bool:
    for payload in payloads:
        candidate_payload = _sanitize_for_json(dict(payload or {}))
        if not candidate_payload:
            continue
        update_succeeded = False
        while candidate_payload and not update_succeeded:
            try:
                supabase.table("content_ideas").update(candidate_payload).eq("id", idea_id).eq("user_id", user_id).execute()
                update_succeeded = True
            except Exception as update_error:
                err_text = str(update_error or "")

                missing_cols = re.findall(r"Could not find the '([^']+)' column", err_text)
                if missing_cols:
                    removed_cols = []
                    for col in missing_cols:
                        if col in candidate_payload:
                            candidate_payload.pop(col, None)
                            removed_cols.append(col)
                    logger.warning(
                        "Enrichment update retry after missing columns idea_id=%s removed=%s remaining_keys=%s err=%s",
                        idea_id,
                        removed_cols,
                        sorted(candidate_payload.keys()),
                        err_text[:600],
                    )
                    continue

                risky_fields = ("status", "title", "updated_at")
                narrowed_payload = {
                    k: v for k, v in candidate_payload.items()
                    if k not in risky_fields
                }
                if narrowed_payload and narrowed_payload != candidate_payload:
                    logger.warning(
                        "Enrichment update retry with narrowed payload idea_id=%s dropped=%s keys=%s err=%s",
                        idea_id,
                        [k for k in risky_fields if k in candidate_payload],
                        sorted(narrowed_payload.keys()),
                        err_text[:600],
                    )
                    candidate_payload = narrowed_payload
                    continue

                logger.warning(
                    "Enrichment update failed idea_id=%s payload_keys=%s err=%s",
                    idea_id,
                    sorted(candidate_payload.keys()),
                    err_text[:700],
                )
                candidate_payload = {}
                break

        if not update_succeeded:
            continue

        try:
            verify_resp = (
                supabase
                .table("content_ideas")
                .select("id,keywords,primary_keywords,raw_dataforseo_output")
                .eq("id", idea_id)
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            verify_row = (verify_resp.data or [None])[0]
        except Exception:
            verify_row = None

        if _verify_enrichment_persistence(verify_row or {}, expected_keywords, expected_raw):
            return True

    # Final minimal fallback: persist only raw output + keyword list.
    final_raw = _sanitize_for_json(expected_raw or {})
    final_keywords = [str(keyword).strip() for keyword in (expected_keywords or []) if str(keyword).strip()]
    minimal_payload = {
        "raw_dataforseo_output": final_raw,
        "keywords": final_keywords,
    }
    try:
        supabase.table("content_ideas").update(minimal_payload).eq("id", idea_id).eq("user_id", user_id).execute()
        verify_resp = (
            supabase
            .table("content_ideas")
                .select("id,keywords,raw_dataforseo_output")
            .eq("id", idea_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        verify_row = (verify_resp.data or [None])[0]
        if _verify_enrichment_persistence(verify_row or {}, final_keywords, final_raw):
            return True
    except Exception as e:
        logger.error(
            "Final minimal enrichment persistence failed for idea_id=%s keywords=%s raw_summary=%s err=%s",
            idea_id,
            final_keywords,
            _summarize_dataforseo_raw(final_raw),
            str(e)[:700],
        )

    logger.error(
        "Enrichment persistence exhausted all fallback payloads for idea_id=%s keywords=%s raw_summary=%s",
        idea_id,
        final_keywords,
        _summarize_dataforseo_raw(final_raw),
    )
    return False


def _select_preferred_project_category_row(
    category_rows: list[dict],
    primary_category_id=None,
    secondary_category_id=None,
) -> dict:
    if not category_rows:
        return {}

    by_id = {
        str(row.get("id")): row
        for row in category_rows
        if row.get("id")
    }
    for category_id in [primary_category_id, secondary_category_id]:
        if category_id and str(category_id) in by_id:
            return by_id[str(category_id)]

    def _sort_key(row: dict):
        sort_order = row.get("sort_order")
        try:
            normalized_sort = int(sort_order)
        except Exception:
            normalized_sort = 999999
        return (normalized_sort, str(row.get("created_at") or ""))

    wp_rows = [row for row in category_rows if row.get("wordpress_category_id") is not None]
    wp_root_rows = [row for row in wp_rows if row.get("parent_category_id") in (None, "")]
    if wp_root_rows:
        return sorted(wp_root_rows, key=_sort_key)[0]
    if wp_rows:
        return sorted(wp_rows, key=_sort_key)[0]

    root_rows = [row for row in category_rows if row.get("parent_category_id") in (None, "")]
    if root_rows:
        return sorted(root_rows, key=_sort_key)[0]
    return sorted(category_rows, key=_sort_key)[0]


def _detach_titles_source_idea_link(supabase, user_id: str, idea_id: str) -> int:
    """Detach Titles->content_ideas association so Content Library rows are preserved."""
    if not str(idea_id or "").strip():
        return 0
    try:
        response = (
            supabase
            .table("Titles")
            .update({
                "source_idea_id": None,
                "updated_at": datetime.utcnow().isoformat(),
            })
            .eq("user_id", user_id)
            .eq("source_idea_id", idea_id)
            .execute()
        )
        return len(response.data or [])
    except Exception as e:
        logger.warning(
            "Failed to detach Titles.source_idea_id for idea_id=%s user_id=%s err=%s",
            idea_id,
            user_id,
            e,
        )
        return 0


def _resolve_publish_context_from_idea(supabase_admin, idea: dict, user_id: str) -> dict:
    """
    Ensure publish has content_ideas category/domain fields; backfill from project context when absent.
    """
    resolved = {
        "wordpress_category_id": idea.get("wordpress_category_id"),
        "wordpress_parent_category_id": idea.get("wordpress_parent_category_id"),
        "category": idea.get("category"),
        "domain": idea.get("domain"),
    }
    category_context = _resolve_category_context_from_idea(supabase_admin, idea, user_id)

    if all(resolved.values()):
        return resolved

    topic_id = idea.get("topic_id")
    project_id = category_context.get("project_id")
    primary_category_id = category_context.get("primary_category_id")
    secondary_category_id = category_context.get("secondary_category_id")

    if not project_id:
        return resolved

    if not resolved.get("domain"):
        try:
            project_response = (
                supabase_admin
                .table("projects")
                .select("id, domain, app_name")
                .eq("id", project_id)
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            project_row = (project_response.data or [None])[0] or {}
            resolved["domain"] = project_row.get("domain") or project_row.get("app_name")
        except Exception:
            logger.warning("Could not resolve project domain for project_id=%s", project_id, exc_info=True)

    needs_category_backfill = (
        resolved.get("wordpress_category_id") is None
        or resolved.get("wordpress_parent_category_id") is None
        or not str(resolved.get("category") or "").strip()
    )
    if not needs_category_backfill:
        return resolved

    try:
        categories_response = (
            supabase_admin
            .table("project_categories")
            .select(
                "id, project_id, parent_category_id, sort_order, created_at, "
                "name, description, wordpress_category_id, wordpress_parent_category_id"
            )
            .eq("project_id", project_id)
            .eq("user_id", user_id)
            .execute()
        )
        selected_row = _select_preferred_project_category_row(
            categories_response.data or [],
            primary_category_id=primary_category_id,
            secondary_category_id=secondary_category_id,
        )
        if selected_row:
            if resolved.get("wordpress_category_id") is None:
                resolved["wordpress_category_id"] = selected_row.get("wordpress_category_id")
            if resolved.get("wordpress_parent_category_id") is None:
                resolved["wordpress_parent_category_id"] = selected_row.get("wordpress_parent_category_id")
            if not str(resolved.get("category") or "").strip():
                resolved["category"] = selected_row.get("description") or selected_row.get("name")
    except Exception:
        logger.warning("Could not resolve project category context for project_id=%s", project_id, exc_info=True)

    return resolved


def _resolve_category_context_from_idea(supabase_admin, idea: dict, user_id: str) -> dict:
    """
    Recover project/category lineage for legacy and rebuild-origin ideas.

    Supports both:
    - legacy topic-backed ideas (`topic_id -> research_topics`)
    - rebuild ideas where `topic_id` may actually be a candidate id
    """
    idea_metadata = _coerce_json_field(idea.get("idea_metadata"), {})
    existing = dict((idea_metadata.get("category_context") or {}) if isinstance(idea_metadata, dict) else {})
    context = {
        "project_id": existing.get("project_id"),
        "primary_category_id": existing.get("primary_category_id"),
        "secondary_category_id": existing.get("secondary_category_id"),
        "project_name": existing.get("project_name"),
        "primary_category_name": existing.get("primary_category_name"),
        "secondary_category_name": existing.get("secondary_category_name"),
        "category_path": existing.get("category_path"),
    }

    topic_id = idea.get("topic_id")
    candidate_row = None
    if topic_id:
        try:
            topic_response = (
                supabase_admin
                .table("research_topics")
                .select("project_id, primary_category_id, secondary_category_id")
                .eq("id", topic_id)
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            topic_row = (topic_response.data or [None])[0] or {}
            context["project_id"] = topic_row.get("project_id") or context.get("project_id")
            context["primary_category_id"] = topic_row.get("primary_category_id") or context.get("primary_category_id")
            context["secondary_category_id"] = topic_row.get("secondary_category_id") or context.get("secondary_category_id")
        except Exception:
            logger.warning("Could not resolve research topic category context for topic_id=%s", topic_id, exc_info=True)

        if not context.get("project_id"):
            try:
                candidate_response = (
                    supabase_admin
                    .table("research_opportunity_candidates")
                    .select("project_id, user_job_id, candidate_metadata")
                    .eq("id", topic_id)
                    .eq("user_id", user_id)
                    .limit(1)
                    .execute()
                )
                candidate_row = (candidate_response.data or [None])[0] or None
            except Exception:
                logger.warning("Could not resolve rebuild candidate context for topic_id=%s", topic_id, exc_info=True)

    if candidate_row:
        candidate_metadata = candidate_row.get("candidate_metadata") or {}
        rebuild_context = (candidate_metadata.get("category_context") or {}) if isinstance(candidate_metadata, dict) else {}
        context["project_id"] = candidate_row.get("project_id") or context.get("project_id")
        context["primary_category_name"] = rebuild_context.get("primary") or rebuild_context.get("primary_category_name") or context.get("primary_category_name")
        context["secondary_category_name"] = rebuild_context.get("secondary") or rebuild_context.get("secondary_category_name") or context.get("secondary_category_name")

        user_job_id = candidate_row.get("user_job_id")
        if user_job_id:
            try:
                job_response = (
                    supabase_admin
                    .table("research_user_jobs")
                    .select("project_id, primary_category_id, secondary_category_id")
                    .eq("id", user_job_id)
                    .eq("user_id", user_id)
                    .limit(1)
                    .execute()
                )
                job_row = (job_response.data or [None])[0] or {}
                context["project_id"] = job_row.get("project_id") or context.get("project_id")
                context["primary_category_id"] = job_row.get("primary_category_id") or context.get("primary_category_id")
                context["secondary_category_id"] = job_row.get("secondary_category_id") or context.get("secondary_category_id")
            except Exception:
                logger.warning("Could not resolve research user job context for user_job_id=%s", user_job_id, exc_info=True)

    if context.get("project_id") and not context.get("project_name"):
        try:
            project_response = (
                supabase_admin
                .table("projects")
                .select("id, domain, app_name")
                .eq("id", context["project_id"])
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            project_row = (project_response.data or [None])[0] or {}
            context["project_name"] = project_row.get("domain") or project_row.get("app_name") or context.get("project_name")
        except Exception:
            logger.warning("Could not resolve project for category context project_id=%s", context.get("project_id"), exc_info=True)

    category_ids = [cid for cid in [context.get("primary_category_id"), context.get("secondary_category_id")] if cid]
    if category_ids:
        try:
            categories_response = (
                supabase_admin
                .table("project_categories")
                .select("id, name")
                .in_("id", category_ids)
                .eq("user_id", user_id)
                .execute()
            )
            category_map = {str(row.get("id")): row.get("name") for row in (categories_response.data or [])}
            context["primary_category_name"] = category_map.get(str(context.get("primary_category_id"))) or context.get("primary_category_name")
            context["secondary_category_name"] = category_map.get(str(context.get("secondary_category_id"))) or context.get("secondary_category_name")
        except Exception:
            logger.warning("Could not resolve project categories for context ids=%s", category_ids, exc_info=True)

    category_path = [context.get("primary_category_name"), context.get("secondary_category_name")]
    context["category_path"] = " / ".join([part for part in category_path if str(part or "").strip()]) or context.get("category_path")
    return context


def _publish_article_content_idea_to_titles(
    supabase,
    supabase_admin,
    *,
    idea: dict,
    user_id: str,
    now: str,
) -> tuple[bool, str | None]:
    """
    Promote one non-software content idea into Titles.

    Returns (published, titles_record_id). If a Titles row already exists for
    this source idea, it is reused instead of creating a duplicate.
    """
    existing_rows = []
    try:
        existing_response = (
            supabase
            .table("Titles")
            .select("id")
            .eq("user_id", user_id)
            .eq("source_idea_id", idea.get("id"))
            .limit(1)
            .execute()
        )
        existing_rows = existing_response.data or []
    except Exception:
        logger.warning("Could not check existing Titles row for idea_id=%s", idea.get("id"), exc_info=True)

    if existing_rows:
        return True, existing_rows[0].get("id")

    category_context = _resolve_category_context_from_idea(supabase_admin, idea, user_id)
    publish_context = _resolve_publish_context_from_idea(
        supabase_admin,
        idea,
        user_id,
    )
    idea_metadata = _coerce_json_field(idea.get("idea_metadata"), {})
    if not isinstance(idea_metadata, dict):
        idea_metadata = {}
    if category_context:
        idea_metadata["category_context"] = category_context
    keyword_payload = _build_titles_keyword_payload_from_idea(
        idea,
        now_iso=now,
        selection_reason="Initialized from research idea publish payload.",
        strategy_version="phase1_v4",
        selection_source="research_rebuild_publish",
    )
    title_payload = {
        "id": str(uuid4()),
        "user_id": user_id,
        "Title": idea.get("title") or "Untitled Article",
        "userDescription": idea.get("description") or "",
        "status": "New",
        "published": False,
        "dateCreatedOn": now,
        "wordpress_category_id": publish_context.get("wordpress_category_id"),
        "wordpress_parent_category_id": publish_context.get("wordpress_parent_category_id"),
        "category": publish_context.get("category"),
        "domain": publish_context.get("domain"),
        "source_idea_id": idea.get("id"),
        "topic_id": idea.get("topic_id"),
        "idea_metadata": idea_metadata,
        "raw_dataforseo_output": idea.get("raw_dataforseo_output"),
        "keyword_clusters_json": [],
    }
    title_payload.update(keyword_payload)
    try:
        response = supabase.table("Titles").insert(title_payload).execute()
        inserted = (response.data or [{}])[0]
        return True, inserted.get("id") or title_payload["id"]
    except Exception as insert_error:
        err = str(insert_error)
        missing_cols = re.findall(r"Could not find the '([^']+)' column", err)
        if missing_cols:
            fallback_payload = dict(title_payload)
            for col in missing_cols:
                fallback_payload.pop(col, None)
            try:
                response = supabase.table("Titles").insert(fallback_payload).execute()
                inserted = (response.data or [{}])[0]
                logger.warning(
                    "Inserted Titles row for idea_id=%s after dropping missing columns: %s",
                    idea.get("id"),
                    ", ".join(missing_cols),
                )
                return True, inserted.get("id") or fallback_payload["id"]
            except Exception:
                logger.warning("Could not insert Titles row for idea_id=%s", idea.get("id"), exc_info=True)
                return False, None
        logger.warning("Could not insert Titles row for idea_id=%s", idea.get("id"), exc_info=True)
        return False, None


def _mark_content_idea_published(
    supabase,
    *,
    idea_id: str,
    user_id: str,
    now: str,
    titles_record_id: str | None = None,
) -> int:
    """Best-effort status update after an idea is promoted."""
    base_payload = {
        "status": "published",
        "published": True,
        "published_to_titles": True,
        "published_at": now,
        "updated_at": now,
    }
    if titles_record_id:
        base_payload["titles_record_id"] = titles_record_id

    try:
        supabase.table("content_ideas").update(base_payload).eq("id", idea_id).eq("user_id", user_id).execute()
        return 1
    except Exception:
        pass

    fallback_payload = {
        "status": "published",
        "updated_at": now,
    }
    if titles_record_id:
        fallback_payload["titles_record_id"] = titles_record_id
    try:
        supabase.table("content_ideas").update(fallback_payload).eq("id", idea_id).eq("user_id", user_id).execute()
        return 1
    except Exception:
        pass

    final_payload = {"updated_at": now}
    if titles_record_id:
        final_payload["titles_record_id"] = titles_record_id
    try:
        supabase.table("content_ideas").update(final_payload).eq("id", idea_id).eq("user_id", user_id).execute()
        return 1
    except Exception:
        result = (
            supabase
            .table("content_ideas")
            .update({"description": ""})
            .eq("id", idea_id)
            .eq("user_id", user_id)
            .execute()
        )
        return 1 if result.data else 0


def _build_released_software_payload(idea: dict, user_id: str, released_at: str) -> dict:
    """Project a software content idea into the durable released_software_ideas table."""
    idea_metadata = _coerce_json_field(idea.get("idea_metadata"), {})
    keyword_metrics = _coerce_json_field(idea.get("keyword_metrics"), {})
    primary_keywords = _coerce_json_field(
        idea.get("primary_keywords") or idea.get("keywords"),
        [],
    )
    secondary_keywords = _coerce_json_field(idea.get("secondary_keywords"), [])
    key_inputs = _coerce_json_field(idea_metadata.get("key_inputs") or idea.get("key_inputs"), [])
    ranking_breakdown = _coerce_json_field(idea.get("ranking_breakdown"), {})

    return {
        "user_id": user_id,
        "source_idea_id": idea.get("id"),
        "topic_id": idea.get("topic_id"),
        "title": idea.get("title") or "Untitled Software Idea",
        "description": _ensure_short_description(
            raw_description=idea.get("description"),
            title=idea.get("title") or "",
            keywords=primary_keywords,
            subtopic=idea.get("subtopic") or "",
        ),
        "status": "saved",
        "released_at": released_at,
        "published": True,
        "content_type": "software",
        "subtopic": idea.get("subtopic"),
        "category": idea.get("category"),
        "domain": idea.get("domain"),
        "keywords": primary_keywords,
        "primary_keywords": primary_keywords,
        "secondary_keywords": secondary_keywords,
        "search_phrase": idea.get("search_phrase"),
        "total_search_volume": idea.get("total_search_volume"),
        "average_difficulty": idea.get("average_difficulty"),
        "average_cpc": idea.get("average_cpc"),
        "affiliate_offer_count": idea.get("affiliate_offer_count"),
        "topic_rating": idea.get("topic_rating") or 0,
        "viability_score": idea.get("viability_score"),
        "trend_score": idea.get("trend_score"),
        "monetization_score": idea.get("monetization_score"),
        "seo_ease_score": idea.get("seo_ease_score"),
        "opportunity_score": idea.get("opportunity_score"),
        "product_type": idea.get("product_type") or idea_metadata.get("product_type"),
        "user_job_to_be_done": idea.get("user_job_to_be_done") or idea_metadata.get("user_job_to_be_done"),
        "key_inputs": key_inputs,
        "output_result": idea.get("output_result") or idea_metadata.get("output_result"),
        "build_complexity": idea.get("build_complexity") or idea_metadata.get("build_complexity"),
        "distribution_angle": idea.get("distribution_angle") or idea_metadata.get("distribution_angle"),
        "target_intent": idea.get("target_intent") or idea_metadata.get("target_intent"),
        "content_outline": _coerce_json_field(idea.get("content_outline"), []),
        "ranking_breakdown": ranking_breakdown,
        "keyword_metrics": keyword_metrics,
        "idea_metadata": idea_metadata,
        "raw_dataforseo_output": idea.get("raw_dataforseo_output"),
        "raw_supabase_output": idea.get("raw_supabase_output"),
        "updated_at": released_at,
    }


def _upsert_released_software_idea(supabase, idea: dict, user_id: str, released_at: str) -> tuple[bool, str | None]:
    """
    Persist a software idea into released_software_ideas without a topic FK so it
    survives topic deletion. Returns (persisted, released_row_id).
    """
    payload = _build_released_software_payload(idea=idea, user_id=user_id, released_at=released_at)
    source_idea_id = str(idea.get("id") or "").strip()

    try:
        existing = []
        if source_idea_id:
            existing = (
                supabase
                .table("released_software_ideas")
                .select("id")
                .eq("user_id", user_id)
                .eq("source_idea_id", source_idea_id)
                .limit(1)
                .execute()
                .data
                or []
            )

        if existing:
            released_id = existing[0].get("id")
            update_payload = dict(payload)
            update_payload.pop("user_id", None)
            update_payload.pop("source_idea_id", None)
            update_payload.pop("released_at", None)
            response = (
                supabase
                .table("released_software_ideas")
                .update(update_payload)
                .eq("id", released_id)
                .eq("user_id", user_id)
                .execute()
            )
            row = (response.data or [{}])[0]
            return True, row.get("id") or released_id

        insert_payload = dict(payload)
        insert_payload["id"] = str(uuid4())
        response = supabase.table("released_software_ideas").insert(insert_payload).execute()
        row = (response.data or [{}])[0]
        return True, row.get("id") or insert_payload["id"]
    except Exception:
        logger.warning(
            "Could not persist released software idea source_idea_id=%s user_id=%s",
            source_idea_id or None,
            user_id,
            exc_info=True,
        )
        return False, None


@content_ideas_bp.route("/list", methods=["POST"])
@require_api_key
def list_content_ideas():
    try:
        if not request.is_json:
            return jsonify(ErrorResponse(
                error="invalid_content_type",
                message="Content-Type must be application/json",
                error_code="INVALID_CONTENT_TYPE",
                status=400,
            ).dict()), 400

        data = request.get_json() or {}
        supabase = get_supabase_client()
        supabase_admin = _get_admin_supabase_client(supabase)
        request_user_id = _resolve_user_id_from_request(supabase, data)
        if not request_user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401,
            ).dict()), 401

        user_id = data.get("user_id") or request_user_id
        if user_id != request_user_id:
            return jsonify(ErrorResponse(
                error="forbidden",
                message="You do not have access to this user_id",
                error_code="FORBIDDEN",
                status=403,
            ).dict()), 403

        topic_id = data.get("topic_id")
        content_type = data.get("content_type")

        query = (
            supabase
            .table("content_ideas")
            .select("*")
            .eq("user_id", user_id)
        )
        if topic_id:
            query = query.eq("topic_id", topic_id)
        if content_type:
            query = query.eq("content_type", content_type)

        response = query.order("created_at", desc=True).execute()
        rows = response.data or []

        # Ensure every idea has a short description (including legacy rows).
        normalized_rows = []
        for row in rows:
            row_copy = dict(row)
            row_copy["idea_metadata"] = _coerce_json_field(row_copy.get("idea_metadata"), {})
            row_copy["keyword_metrics"] = _coerce_json_field(row_copy.get("keyword_metrics"), {})
            row_copy["primary_keywords"] = _coerce_json_field(row_copy.get("primary_keywords"), [])
            row_copy["secondary_keywords"] = _coerce_json_field(row_copy.get("secondary_keywords"), [])
            row_copy["keywords"] = _coerce_json_field(row_copy.get("keywords"), [])
            row_copy = _hydrate_legacy_idea_seo_fields(row_copy)
            normalized_description = _ensure_short_description(
                raw_description=row_copy.get("description"),
                title=row_copy.get("title") or "",
                keywords=(row_copy.get("primary_keywords") or row_copy.get("keywords") or []),
                subtopic=row_copy.get("subtopic") or "",
            )
            row_copy["description"] = normalized_description
            normalized_rows.append(row_copy)

            # Best-effort backfill for legacy blanks.
            if not str(row.get("description") or "").strip():
                try:
                    (
                        supabase
                        .table("content_ideas")
                        .update({"description": normalized_description})
                        .eq("id", row_copy.get("id"))
                        .eq("user_id", user_id)
                        .execute()
                    )
                except Exception:
                    logger.warning(
                        "Could not backfill description for content_idea id=%s",
                        row_copy.get("id"),
                        exc_info=True,
                    )

        return jsonify(normalized_rows), 200

    except Exception as e:
        logger.error(f"Error listing content ideas: {e}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500,
        ).dict()), 500


@content_ideas_bp.route("/publish", methods=["POST"])
@require_api_key
def publish_content_ideas():
    try:
        if not request.is_json:
            return jsonify(ErrorResponse(
                error="invalid_content_type",
                message="Content-Type must be application/json",
                error_code="INVALID_CONTENT_TYPE",
                status=400,
            ).dict()), 400

        data = request.get_json() or {}
        idea_ids = data.get("idea_ids") or []
        if not idea_ids:
            return jsonify(ErrorResponse(
                error="validation_error",
                message="idea_ids is required",
                error_code="VALIDATION_ERROR",
                status=400,
            ).dict()), 400

        supabase = get_supabase_client()
        supabase_admin = _get_admin_supabase_client(supabase)
        request_user_id = _resolve_user_id_from_request(supabase, data)
        if not request_user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401,
            ).dict()), 401

        user_id = data.get("user_id") or request_user_id
        if user_id != request_user_id:
            return jsonify(ErrorResponse(
                error="forbidden",
                message="You do not have access to this user_id",
                error_code="FORBIDDEN",
                status=403,
            ).dict()), 403

        now = datetime.utcnow().isoformat()
        updated_count = 0
        published_to_titles_count = 0
        published_to_software_count = 0
        for idea_id in idea_ids:
            # 1) Fetch idea row first (works across old/new schemas).
            idea_resp = (
                supabase
                .table("content_ideas")
                .select("*")
                .eq("id", idea_id)
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            if not idea_resp.data:
                continue
            idea = idea_resp.data[0]
            titles_record_id = None
            released_software_id = None

            # 2) Persist released ideas into their durable destination table.
            if (idea.get("content_type") or "").lower() == "software":
                software_persisted, released_software_id = _upsert_released_software_idea(
                    supabase=supabase,
                    idea=idea,
                    user_id=user_id,
                    released_at=now,
                )
                if software_persisted:
                    published_to_software_count += 1
                    try:
                        supabase.table("content_ideas").update({
                            "titles_record_id": released_software_id,
                            "updated_at": now,
                        }).eq("id", idea_id).eq("user_id", user_id).execute()
                    except Exception:
                        logger.warning(
                            "Could not stamp content_ideas.titles_record_id for released software idea_id=%s",
                            idea_id,
                            exc_info=True,
                        )
            else:
                title_published, titles_record_id = _publish_article_content_idea_to_titles(
                    supabase,
                    supabase_admin,
                    idea=idea,
                    user_id=user_id,
                    now=now,
                )
                if title_published:
                    published_to_titles_count += 1

            # 3) Best-effort status update on content_ideas with progressive fallbacks.
            updated_count += _mark_content_idea_published(
                supabase,
                idea_id=idea_id,
                user_id=user_id,
                now=now,
                titles_record_id=titles_record_id if (idea.get("content_type") or "").lower() != "software" else released_software_id,
            )

        success = (updated_count > 0) or (published_to_titles_count > 0) or (published_to_software_count > 0)
        status_code = 200 if success else 400
        return jsonify({
            "success": success,
            "published_count": updated_count,
            "published_to_titles_count": published_to_titles_count,
            "published_to_software_count": published_to_software_count,
            "requested_count": len(idea_ids),
            "message": None if success else "No ideas were published. Verify idea IDs and schema.",
        }), status_code

    except Exception as e:
        logger.error(f"Error publishing content ideas: {e}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500,
        ).dict()), 500


@content_ideas_bp.route("/enrich", methods=["POST"])
@require_api_key
def enrich_content_ideas():
    """
    Enrich selected content ideas with SEO metrics and affiliate offer signals.
    """
    try:
        if not request.is_json:
            return jsonify(ErrorResponse(
                error="invalid_content_type",
                message="Content-Type must be application/json",
                error_code="INVALID_CONTENT_TYPE",
                status=400,
            ).dict()), 400

        data = request.get_json() or {}
        idea_ids = data.get("idea_ids") or []
        if not idea_ids:
            return jsonify(ErrorResponse(
                error="validation_error",
                message="idea_ids is required",
                error_code="VALIDATION_ERROR",
                status=400,
            ).dict()), 400

        supabase = get_supabase_client()
        supabase_admin = _get_admin_supabase_client(supabase)
        request_user_id = _resolve_user_id_from_request(supabase, data)
        if not request_user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401,
            ).dict()), 401

        user_id = data.get("user_id") or request_user_id
        if user_id != request_user_id:
            return jsonify(ErrorResponse(
                error="forbidden",
                message="You do not have access to this user_id",
                error_code="FORBIDDEN",
                status=403,
            ).dict()), 403

        req_start = time.perf_counter()
        now = datetime.utcnow().isoformat()
        results = []
        enriched_count = 0
        logger.info(
            "Enrich request received user_id=%s idea_count=%s",
            user_id,
            len(idea_ids),
        )

        for idea_id in idea_ids:
            idea_start = time.perf_counter()
            fetch_resp = (
                supabase_admin
                .table("content_ideas")
                .select("*")
                .eq("id", idea_id)
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            if not fetch_resp.data:
                logger.warning("Enrich idea fetch failed idea_id=%s user_id=%s", idea_id, user_id)
                results.append({
                    "idea_id": idea_id,
                    "status": "failed",
                    "reason": "Idea not found for user",
                })
                continue

            idea = fetch_resp.data[0]
            try:
                enrichment = asyncio.run(
                    asyncio.wait_for(
                        _compute_idea_enrichment(idea),
                        timeout=PER_IDEA_ENRICH_TIMEOUT_SECONDS
                    )
                )
            except TimeoutError:
                logger.warning("Per-idea enrichment timed out for idea_id=%s", idea_id)
                results.append({
                    "idea_id": idea_id,
                    "status": "failed",
                    "reason": "Enrichment timed out for this idea",
                })
                continue

            if enrichment.get("status") != "enriched":
                logger.warning(
                    "Enrichment failed for idea_id=%s reason=%s",
                    idea_id,
                    enrichment.get("reason"),
                )
                _persist_raw_trace_for_idea(
                    supabase=supabase_admin,
                    idea=idea,
                    user_id=user_id,
                    now_iso=now,
                    raw_output=enrichment.get("raw_dataforseo_output") or {},
                    reason=enrichment.get("reason") or "enrichment_failed",
                )
                results.append({
                    "idea_id": idea_id,
                    "status": "failed",
                    "reason": enrichment.get("reason") or "Enrichment failed",
                })
                continue

            keywords_used = [
                str(keyword).strip()
                for keyword in (enrichment.get("keywords_used") or [])
                if str(keyword).strip()
            ]
            all_keywords_found = [
                str(keyword).strip()
                for keyword in (enrichment.get("all_keywords_found") or [])
                if str(keyword).strip()
            ]
            if not keywords_used:
                keywords_used = all_keywords_found
            original_title = str(idea.get("title") or "").strip()
            keyword_projection_payload = {
                "keywords": keywords_used,
            }
            update_payload = {
                "affiliate_offer_count": enrichment["affiliate_offer_count"],
                "affiliate_offers_preview": _sanitize_for_json(enrichment.get("affiliate_offers") or []),
                "affiliate_search_status": enrichment.get("affiliate_search_status"),
                "affiliate_search_error": enrichment.get("affiliate_search_error"),
                "raw_dataforseo_output": _sanitize_for_json(enrichment.get("raw_dataforseo_output") or {}),
                "updated_at": now,
            }
            if enrichment.get("has_exact_keyword_metrics"):
                update_payload["total_search_volume"] = enrichment["total_search_volume"]
                update_payload["average_cpc"] = enrichment["average_cpc"]
                update_payload["average_difficulty"] = enrichment["average_difficulty"]
            enrichment_metadata = {
                **(idea.get("idea_metadata") or {}),
                "seo_offer_enrichment": {
                    "keywords_used": enrichment["keywords_used"],
                    "keyword_metrics": enrichment.get("keyword_metrics_map") or {},
                    "keyword_ranked_candidates": enrichment.get("keyword_ranked_candidates") or [],
                    "keyword_quality_summary": enrichment.get("keyword_quality_summary") or {},
                    "keyword_budget_ladder_used": enrichment.get("keyword_budget_ladder_used") or [],
                    "dataforseo_diagnostics": enrichment.get("dataforseo_diagnostics") or {},
                    "raw_dataforseo_output": enrichment.get("raw_dataforseo_output") or {},
                    "dataforseo_call_count_estimate": enrichment.get("dataforseo_call_count_estimate") or 0,
                    "affiliate_offer_count": enrichment["affiliate_offer_count"],
                    "affiliate_offers_preview": enrichment["affiliate_offers"],
                    "affiliate_search_status": enrichment.get("affiliate_search_status"),
                    "affiliate_search_error": enrichment.get("affiliate_search_error"),
                    "enriched_at": now,
                },
                "keyword_pass_2": {
                    "selection_rule_version": "kd_asc_volume_desc_v1",
                    "keyword_ranked_candidates": enrichment.get("keyword_ranked_candidates") or [],
                    "keyword_quality_summary": enrichment.get("keyword_quality_summary") or {},
                    "original_title": original_title,
                },
            }

            # Try richest payload first; gracefully degrade for older schemas.
            # Start with safe/minimal persistence so raw output + keywords are not blocked by optional fields.
            safe_minimal_payload = {
                "raw_dataforseo_output": _sanitize_for_json(enrichment.get("raw_dataforseo_output") or {}),
                "keywords": _sanitize_for_json(keyword_projection_payload.get("keywords") or []),
                "updated_at": now,
            }
            payload_attempts = [
                {
                    **update_payload,
                    **keyword_projection_payload,
                    "keyword_metrics": _sanitize_for_json(enrichment.get("keyword_metrics_map") or {}),
                    "idea_metadata": _sanitize_for_json(enrichment_metadata),
                },
                {
                    **update_payload,
                    **keyword_projection_payload,
                    "idea_metadata": _sanitize_for_json(enrichment_metadata),
                },
                {
                    **update_payload,
                    **keyword_projection_payload,
                    "keyword_metrics": _sanitize_for_json(enrichment.get("keyword_metrics_map") or {}),
                },
                {
                    **update_payload,
                    **keyword_projection_payload,
                },
                {
                    **update_payload,
                    "keyword_metrics": _sanitize_for_json(enrichment.get("keyword_metrics_map") or {}),
                    "idea_metadata": _sanitize_for_json(enrichment_metadata),
                },
                {
                    **update_payload,
                    "idea_metadata": _sanitize_for_json(enrichment_metadata),
                },
                {
                    **update_payload,
                    "keyword_metrics": _sanitize_for_json(enrichment.get("keyword_metrics_map") or {}),
                },
                {
                    **safe_minimal_payload,
                    "keyword_metrics": _sanitize_for_json(enrichment.get("keyword_metrics_map") or {}),
                },
                update_payload,
                safe_minimal_payload,
            ]

            updated = _apply_enrichment_update_with_fallback(
                supabase_admin,
                idea_id=idea_id,
                user_id=user_id,
                payloads=payload_attempts,
                expected_keywords=keywords_used,
                expected_raw=enrichment.get("raw_dataforseo_output") or {},
            )

            if updated:
                enriched_count += 1
                logger.info(
                    "Enrichment persisted for idea_id=%s in %.2fs",
                    idea_id,
                    time.perf_counter() - idea_start,
                )
                results.append({
                    "idea_id": idea_id,
                    "status": "enriched",
                    "metrics": {
                        "total_search_volume": enrichment["total_search_volume"],
                        "average_cpc": enrichment["average_cpc"],
                        "average_difficulty": enrichment["average_difficulty"],
                        "affiliate_offer_count": enrichment["affiliate_offer_count"],
                    },
                    "keywords_used": enrichment["keywords_used"],
                    "keyword_metrics_map": enrichment.get("keyword_metrics_map") or {},
                    "affiliate_offers_preview": enrichment["affiliate_offers"],
                    "affiliate_search_status": enrichment.get("affiliate_search_status"),
                    "affiliate_search_error": enrichment.get("affiliate_search_error"),
                    "offers_preview": enrichment["affiliate_offers"],
                })
            else:
                logger.error(
                    "Enrichment persistence failed for idea_id=%s after %.2fs",
                    idea_id,
                    time.perf_counter() - idea_start,
                )
                results.append({
                    "idea_id": idea_id,
                    "status": "failed",
                    "reason": "Could not persist enrichment values",
                })

        logger.info(
            "Enrich request finished user_id=%s requested=%s enriched=%s elapsed=%.2fs",
            user_id,
            len(idea_ids),
            enriched_count,
            time.perf_counter() - req_start,
        )
        return jsonify({
            "success": enriched_count > 0,
            "requested_count": len(idea_ids),
            "enriched_count": enriched_count,
            "results": results,
        }), 200 if enriched_count > 0 else 400

    except Exception as e:
        logger.error("Error enriching content ideas: %s", e, exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500,
        ).dict()), 500


@content_ideas_bp.route("/refresh-keywords", methods=["POST"])
@require_api_key
def refresh_keywords_for_library():
    """
    Refresh keyword metrics for content_ideas and sync linked Titles rows.
    Accepts idea_ids directly and/or title_ids that are mapped via Titles.source_idea_id.
    """
    try:
        if not request.is_json:
            return jsonify(ErrorResponse(
                error="invalid_content_type",
                message="Content-Type must be application/json",
                error_code="INVALID_CONTENT_TYPE",
                status=400,
            ).dict()), 400

        data = request.get_json() or {}
        supabase = get_supabase_client()
        supabase_admin = _get_admin_supabase_client(supabase)
        request_user_id = _resolve_user_id_from_request(supabase, data)
        if not request_user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401,
            ).dict()), 401

        user_id = data.get("user_id") or request_user_id
        if user_id != request_user_id:
            return jsonify(ErrorResponse(
                error="forbidden",
                message="You do not have access to this user_id",
                error_code="FORBIDDEN",
                status=403,
            ).dict()), 403

        input_idea_ids = [str(x).strip() for x in (data.get("idea_ids") or []) if str(x).strip()]
        title_ids = [str(x).strip() for x in (data.get("title_ids") or []) if str(x).strip()]
        if not input_idea_ids and not title_ids:
            return jsonify(ErrorResponse(
                error="validation_error",
                message="Provide idea_ids and/or title_ids",
                error_code="VALIDATION_ERROR",
                status=400,
            ).dict()), 400

        idea_ids = set(input_idea_ids)
        if title_ids:
            title_rows = (
                supabase.table("Titles")
                .select("id,source_idea_id")
                .eq("user_id", user_id)
                .in_("id", title_ids)
                .execute()
                .data
                or []
            )
            for row in title_rows:
                source_idea_id = row.get("source_idea_id")
                if source_idea_id:
                    idea_ids.add(str(source_idea_id))

        if not idea_ids:
            return jsonify({
                "success": False,
                "requested_count": 0,
                "enriched_count": 0,
                "titles_synced_count": 0,
                "results": [],
                "message": "No linked content ideas found for selected titles."
            }), 400

        now = datetime.utcnow().isoformat()
        enriched_count = 0
        titles_synced_count = 0
        results = []

        for idea_id in sorted(idea_ids):
            fetch_resp = (
                supabase_admin.table("content_ideas")
                .select("*")
                .eq("id", idea_id)
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            if not fetch_resp.data:
                results.append({"idea_id": idea_id, "status": "failed", "reason": "Idea not found for user"})
                continue

            idea = fetch_resp.data[0]
            try:
                enrichment = asyncio.run(
                    asyncio.wait_for(
                        _compute_idea_enrichment(idea),
                        timeout=PER_IDEA_ENRICH_TIMEOUT_SECONDS,
                    )
                )
            except TimeoutError:
                results.append({"idea_id": idea_id, "status": "failed", "reason": "Enrichment timed out for this idea"})
                continue
            except Exception:
                logger.warning("Keyword refresh enrichment failed for idea_id=%s", idea_id, exc_info=True)
                results.append({"idea_id": idea_id, "status": "failed", "reason": "Enrichment request failed"})
                continue
            if enrichment.get("status") != "enriched":
                _persist_raw_trace_for_idea(
                    supabase=supabase_admin,
                    idea=idea,
                    user_id=user_id,
                    now_iso=now,
                    raw_output=enrichment.get("raw_dataforseo_output") or {},
                    reason=enrichment.get("reason") or "enrichment_failed",
                )
                results.append({"idea_id": idea_id, "status": "failed", "reason": enrichment.get("reason") or "Enrichment failed"})
                continue

            keywords_used = [
                str(keyword).strip()
                for keyword in (enrichment.get("keywords_used") or [])
                if str(keyword).strip()
            ]
            keyword_projection_payload = {
                "keywords": keywords_used,
            }
            update_payload = {
                "affiliate_offer_count": enrichment["affiliate_offer_count"],
                "affiliate_offers_preview": _sanitize_for_json(enrichment.get("affiliate_offers") or []),
                "affiliate_search_status": enrichment.get("affiliate_search_status"),
                "affiliate_search_error": enrichment.get("affiliate_search_error"),
                "raw_dataforseo_output": _sanitize_for_json(enrichment.get("raw_dataforseo_output") or {}),
                "updated_at": now,
            }
            if enrichment.get("has_exact_keyword_metrics"):
                update_payload["total_search_volume"] = enrichment["total_search_volume"]
                update_payload["average_cpc"] = enrichment["average_cpc"]
                update_payload["average_difficulty"] = enrichment["average_difficulty"]
            payload_attempts = [
                {
                    **update_payload,
                    **keyword_projection_payload,
                    "keyword_metrics": _sanitize_for_json(enrichment.get("keyword_metrics_map") or {}),
                    "idea_metadata": {
                        **(idea.get("idea_metadata") or {}),
                        "seo_offer_enrichment": {
                            "keywords_used": enrichment["keywords_used"],
                            "keyword_metrics": _sanitize_for_json(enrichment.get("keyword_metrics_map") or {}),
                            "keyword_ranked_candidates": enrichment.get("keyword_ranked_candidates") or [],
                            "keyword_quality_summary": enrichment.get("keyword_quality_summary") or {},
                            "keyword_budget_ladder_used": enrichment.get("keyword_budget_ladder_used") or [],
                            "dataforseo_diagnostics": enrichment.get("dataforseo_diagnostics") or {},
                            "raw_dataforseo_output": _sanitize_for_json(enrichment.get("raw_dataforseo_output") or {}),
                            "dataforseo_call_count_estimate": enrichment.get("dataforseo_call_count_estimate") or 0,
                            "affiliate_offer_count": enrichment["affiliate_offer_count"],
                            "affiliate_offers_preview": enrichment["affiliate_offers"],
                            "affiliate_search_status": enrichment.get("affiliate_search_status"),
                            "affiliate_search_error": enrichment.get("affiliate_search_error"),
                            "enriched_at": now,
                        },
                    },
                },
                {
                    "raw_dataforseo_output": _sanitize_for_json(enrichment.get("raw_dataforseo_output") or {}),
                    "keywords": _sanitize_for_json(keyword_projection_payload.get("keywords") or []),
                    "updated_at": now,
                },
                {
                    **update_payload,
                    **keyword_projection_payload,
                    "keyword_metrics": _sanitize_for_json(enrichment.get("keyword_metrics_map") or {}),
                },
                {
                    **update_payload,
                    **keyword_projection_payload,
                },
                {
                    **update_payload,
                    "keyword_metrics": _sanitize_for_json(enrichment.get("keyword_metrics_map") or {}),
                    "idea_metadata": {
                        **(idea.get("idea_metadata") or {}),
                        "seo_offer_enrichment": {
                            "keywords_used": enrichment["keywords_used"],
                            "keyword_metrics": _sanitize_for_json(enrichment.get("keyword_metrics_map") or {}),
                            "keyword_ranked_candidates": enrichment.get("keyword_ranked_candidates") or [],
                            "keyword_quality_summary": enrichment.get("keyword_quality_summary") or {},
                            "keyword_budget_ladder_used": enrichment.get("keyword_budget_ladder_used") or [],
                            "dataforseo_diagnostics": enrichment.get("dataforseo_diagnostics") or {},
                            "raw_dataforseo_output": _sanitize_for_json(enrichment.get("raw_dataforseo_output") or {}),
                            "dataforseo_call_count_estimate": enrichment.get("dataforseo_call_count_estimate") or 0,
                            "affiliate_offer_count": enrichment["affiliate_offer_count"],
                            "affiliate_offers_preview": enrichment["affiliate_offers"],
                            "affiliate_search_status": enrichment.get("affiliate_search_status"),
                            "affiliate_search_error": enrichment.get("affiliate_search_error"),
                            "enriched_at": now,
                        },
                    },
                },
                update_payload,
            ]
            updated = _apply_enrichment_update_with_fallback(
                supabase_admin,
                idea_id=idea_id,
                user_id=user_id,
                payloads=payload_attempts,
                expected_keywords=keywords_used,
                expected_raw=enrichment.get("raw_dataforseo_output") or {},
            )
            if not updated:
                results.append({
                    "idea_id": idea_id,
                    "status": "failed",
                    "reason": "Could not persist enrichment values",
                })
                continue

            # Re-fetch updated row and sync Titles projection.
            refreshed_idea_resp = (
                supabase_admin.table("content_ideas")
                .select("*")
                .eq("id", idea_id)
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            refreshed_idea = (refreshed_idea_resp.data or [idea])[0]
            synced = _sync_titles_keyword_fields_from_idea(
                supabase=supabase_admin,
                idea=refreshed_idea,
                user_id=user_id,
                now_iso=now,
            )
            titles_synced_count += synced
            enriched_count += 1
            results.append({
                "idea_id": idea_id,
                "status": "enriched",
                "titles_synced": synced,
                "metrics": {
                    "total_search_volume": enrichment["total_search_volume"],
                    "average_cpc": enrichment.get("average_cpc") or 0,
                    "average_difficulty": enrichment["average_difficulty"],
                    "affiliate_offer_count": enrichment.get("affiliate_offer_count") or 0,
                },
                "keywords_used": enrichment.get("keywords_used") or [],
                "selected_primary_keyword": enrichment.get("selected_primary_keyword"),
                "keyword_metrics_map": enrichment.get("keyword_metrics_map") or {},
                "raw_dataforseo_output": enrichment.get("raw_dataforseo_output"),
                "affiliate_offers_preview": enrichment.get("affiliate_offers") or [],
                "affiliate_search_status": enrichment.get("affiliate_search_status"),
                "affiliate_search_error": enrichment.get("affiliate_search_error"),
            })

        return jsonify({
            "success": enriched_count > 0,
            "requested_count": len(idea_ids),
            "enriched_count": enriched_count,
            "titles_synced_count": titles_synced_count,
            "results": results,
        }), 200 if enriched_count > 0 else 400
    except Exception as e:
        logger.error("Error refreshing keyword metrics for content library: %s", e, exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500,
        ).dict()), 500


@content_ideas_bp.route("/<idea_id>", methods=["DELETE"])
@require_api_key
def delete_content_idea(idea_id):
    try:
        supabase = get_supabase_client()
        request_user_id = _resolve_user_id_from_request(supabase, request.args)
        if not request_user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401,
            ).dict()), 401

        user_id = request.args.get("user_id") or request_user_id
        if user_id != request_user_id:
            return jsonify(ErrorResponse(
                error="forbidden",
                message="You do not have access to this user_id",
                error_code="FORBIDDEN",
                status=403,
            ).dict()), 403

        # Preserve Content Library rows even if DB foreign keys are configured
        # to cascade from content_ideas.
        detached_titles = _detach_titles_source_idea_link(
            supabase=supabase,
            user_id=user_id,
            idea_id=idea_id,
        )

        response = (
            supabase
            .table("content_ideas")
            .delete()
            .eq("id", idea_id)
            .eq("user_id", user_id)
            .execute()
        )

        if not response.data:
            return jsonify(ErrorResponse(
                error="not_found",
                message="Content idea not found",
                error_code="NOT_FOUND",
                status=404,
            ).dict()), 404

        return jsonify({
            "success": True,
            "id": idea_id,
            "detached_content_library_records": detached_titles,
        }), 200

    except Exception as e:
        logger.error(f"Error deleting content idea {idea_id}: {e}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500,
        ).dict()), 500


@content_ideas_bp.route("/keyword-lab/related", methods=["POST"])
@require_api_key
def keyword_lab_related():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        data = request.get_json() or {}
        seed = _normalize_keyword_term(data.get("seed_keyword") or "")
        limit = int(data.get("limit") or 12)
        min_search_volume = int(data.get("min_search_volume") or 0)
        raw_max_kd = data.get("max_keyword_difficulty")
        max_keyword_difficulty = float(raw_max_kd) if raw_max_kd is not None and str(raw_max_kd).strip() != "" else None
        exclude_keywords = data.get("exclude_keywords") or []
        if isinstance(exclude_keywords, str):
            exclude_keywords = [part.strip() for part in re.split(r"[\n,]+", exclude_keywords) if part.strip()]
        if not seed:
            return jsonify({"error": "seed_keyword is required"}), 400

        supabase = get_supabase_client()
        request_user_id = _resolve_user_id_from_request(supabase, data)
        if not request_user_id:
            return jsonify({"error": "Authorization bearer token is required"}), 401

        related_rows_response = asyncio.run(
            asyncio.wait_for(
                dataforseo_api.get_keyword_suggestions_labs_live(
                    [seed],
                    limit_per_seed=max(10, min(limit, 100)),
                    return_raw=True,
                    filters=[],
                ),
                timeout=DATAFORSEO_BULK_TIMEOUT_SECONDS,
            )
        )
        if isinstance(related_rows_response, dict):
            related_rows = related_rows_response.get("items") or []
        else:
            related_rows = related_rows_response or []

        metrics_map = {}
        for row in related_rows or []:
            kw_key = str(row.get("keyword") or "").strip().lower()
            if kw_key:
                metrics_map[kw_key] = {
                    "search_volume": row.get("search_volume"),
                    "cpc": row.get("cpc"),
                    "keyword_difficulty": row.get("keyword_difficulty"),
                }

        related_keywords = []
        seen = {seed}
        exclude_set = {_normalize_keyword_term(k) for k in exclude_keywords if _normalize_keyword_term(k)}
        for row in related_rows or []:
            kw = _normalize_keyword_term(row.get("keyword") or "")
            if not kw or kw in seen or kw in exclude_set:
                continue
            seen.add(kw)
            related_keywords.append(kw)
        # Keep seed included as requested, then append new related terms.
        candidate_keywords = [seed] + related_keywords
        ranked = _rank_keywords_by_opportunity(candidate_keywords, metrics_map)
        if min_search_volume > 0:
            ranked = [
                row for row in ranked
                if int(row.get("search_volume") or 0) >= min_search_volume
            ]
        if max_keyword_difficulty is not None:
            ranked = [
                row for row in ranked
                if row.get("keyword_difficulty") is not None and float(row.get("keyword_difficulty") or 0.0) <= max_keyword_difficulty
            ]
        return jsonify({
            "success": True,
            "seed_keyword": seed,
            "keywords": ranked,
            "quality": _keyword_quality_summary(ranked),
        }), 200
    except Exception as e:
        logger.error("keyword_lab_related failed: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500
