"""
Research Topics API endpoints.

This module provides endpoints for managing research topics.
"""

import logging
import json
import re
import time
from datetime import datetime
from uuid import uuid4
from flask import Blueprint, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from ...core.models.errors import ErrorResponse, ValidationErrorResponse
from ...api.middleware.auth import require_api_key

from ...core.models.topic_analysis import Subtopic

# Import supabase client
# Prefer the strict service-role singleton used by backend endpoints.
try:
    from src.core.supabase_singleton import get_supabase_client
except ImportError:
    try:
        from ...core.supabase_singleton import get_supabase_client
    except ImportError:
        # Legacy fallback for alternate runtime contexts.
        try:
            from supabase_client import get_supabase_client
        except ImportError:
            import sys
            import os
            sys.path.append(os.getcwd())
            from supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

# Create blueprint - Note: URL prefix is handled in app.py or here
# Frontend requests /api/research-topics/, check app.py registration
research_topics_bp = Blueprint('research_topics', __name__, url_prefix='/api/research-topics')

# Create rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1000 per hour", "60 per minute"],
    storage_uri="memory://"
)


def _resolve_user_id_from_request(supabase, data=None):
    """Resolve the authenticated Supabase user id from the bearer token or request payload."""
    auth_header = request.headers.get('Authorization')
    user_id = None

    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split('Bearer ')[1]
        try:
            user_response = supabase.auth.get_user(token)
            if user_response and user_response.user:
                user_id = user_response.user.id
        except Exception as auth_error:
            logger.warning(
                "Failed to validate bearer token path=%s method=%s token_len=%s error=%s",
                request.path,
                request.method,
                len(token or ""),
                auth_error,
            )
    elif auth_header:
        logger.warning(
            "Authorization header present but not Bearer path=%s method=%s header_prefix=%r",
            request.path,
            request.method,
            auth_header[:20],
        )

    if not user_id and data and data.get('user_id'):
        user_id = data['user_id']
        logger.info(
            "Resolved request user from payload path=%s method=%s user_id=%s",
            request.path,
            request.method,
            user_id,
        )

    return user_id


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


def _unlink_titles_for_idea_ids(supabase, user_id: str, idea_ids: list[str]) -> int:
    """
    Preserve Content Library records when content ideas are deleted by detaching
    the `Titles.source_idea_id` link first.
    """
    clean_ids = [str(idea_id).strip() for idea_id in (idea_ids or []) if str(idea_id).strip()]
    if not clean_ids:
        return 0

    detached = 0
    chunk_size = 100
    for index in range(0, len(clean_ids), chunk_size):
        chunk = clean_ids[index:index + chunk_size]
        try:
            res = (
                supabase
                .table("Titles")
                .update({
                    "source_idea_id": None,
                    "updated_at": datetime.utcnow().isoformat(),
                })
                .eq("user_id", user_id)
                .in_("source_idea_id", chunk)
                .execute()
            )
            detached += len(res.data or [])
        except Exception as detach_err:
            logger.warning(
                "Failed to detach Titles.source_idea_id for user_id=%s chunk_size=%s err=%s",
                user_id,
                len(chunk),
                detach_err,
            )
    return detached


def _normalize_keyword_metric_term(term) -> str:
    cleaned = re.sub(r"\s+", " ", str(term or "").strip().lower())
    cleaned = cleaned.replace("&", " and ")
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _coerce_keyword_metric_entries(raw_keywords) -> list[dict]:
    """
    Normalize mixed keyword payloads into a list of keyword objects.
    Accepts strings, dicts, and JSON-stringified arrays/objects.
    """
    if raw_keywords is None:
        return []

    if isinstance(raw_keywords, str):
        raw = raw_keywords.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            return _coerce_keyword_metric_entries(parsed)
        except Exception:
            return [{"keyword": part.strip()} for part in re.split(r"[\n,]+", raw) if part.strip()]

    if not isinstance(raw_keywords, list):
        return []

    entries: list[dict] = []
    for item in raw_keywords:
        if isinstance(item, str):
            keyword = item.strip()
            if keyword:
                entries.append({"keyword": keyword})
            continue
        if not isinstance(item, dict):
            continue

        keyword = str(item.get("keyword") or item.get("term") or item.get("seed_keyword") or "").strip()
        if not keyword:
            continue

        raw_search_volume = item.get("search_volume") if "search_volume" in item else item.get("volume")
        raw_cpc = item.get("cpc")
        raw_kd = item.get("keyword_difficulty")
        if raw_kd is None:
            raw_kd = item.get("difficulty")
        if raw_kd is None:
            raw_kd = item.get("seo_difficulty")
        try:
            search_volume = int(float(raw_search_volume)) if raw_search_volume is not None and str(raw_search_volume).strip() != "" else None
        except Exception:
            search_volume = None
        try:
            cpc = float(raw_cpc) if raw_cpc is not None and str(raw_cpc).strip() != "" else None
        except Exception:
            cpc = None
        try:
            keyword_difficulty = float(raw_kd) if raw_kd is not None and str(raw_kd).strip() != "" else None
        except Exception:
            keyword_difficulty = None

        entries.append({
            "keyword": keyword,
            "search_volume": max(0, search_volume) if search_volume is not None else None,
            "cpc": max(0.0, cpc) if cpc is not None else None,
            "keyword_difficulty": max(0.0, keyword_difficulty) if keyword_difficulty is not None else None,
        })
    return entries


def _build_keyword_metrics_map(raw_keywords) -> dict:
    metrics_map: dict = {}
    for item in _coerce_keyword_metric_entries(raw_keywords):
        keyword = str(item.get("keyword") or "").strip()
        normalized = _normalize_keyword_metric_term(keyword)
        if not normalized:
            continue
        metrics_map[normalized] = {
            "keyword": keyword,
            "search_volume": int(item.get("search_volume")) if item.get("search_volume") is not None else None,
            "cpc": round(float(item.get("cpc")), 2) if item.get("cpc") is not None else None,
            "keyword_difficulty": round(float(item.get("keyword_difficulty")), 1) if item.get("keyword_difficulty") is not None else None,
        }
    return metrics_map


def _build_idea_keyword_metrics_payload(idea: dict, source_metrics_map: dict) -> tuple[dict, dict]:
    """
    Match idea keywords against available keyword metrics and compute exact aggregates.
    Returns:
    - keyword_metrics map keyed by display keyword for UI compatibility
    - aggregate metrics summary
    """
    keyword_candidates = []
    for field in ("primary_keywords", "keywords", "secondary_keywords"):
        value = idea.get(field)
        if isinstance(value, list):
            keyword_candidates.extend([str(item).strip() for item in value if str(item).strip()])
        elif isinstance(value, str) and value.strip():
            keyword_candidates.extend([part.strip() for part in re.split(r"[\n,]+", value) if part.strip()])
    idea_metadata = idea.get("idea_metadata") or {}
    if isinstance(idea_metadata, dict):
        seed_keywords = idea_metadata.get("input_keywords")
        if isinstance(seed_keywords, list):
            keyword_candidates.extend([str(item).strip() for item in seed_keywords if str(item).strip()])
        seed_pack_keywords = ((idea_metadata.get("keyword_seed_pack") or {}).get("input_keywords")
                             if isinstance(idea_metadata.get("keyword_seed_pack"), dict) else [])
        if isinstance(seed_pack_keywords, list):
            keyword_candidates.extend([str(item).strip() for item in seed_pack_keywords if str(item).strip()])

    keyword_metrics: dict = {}
    volumes = []
    difficulties = []
    cpcs = []

    seen = set()
    for keyword in keyword_candidates:
        normalized = _normalize_keyword_metric_term(keyword)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)

        row = source_metrics_map.get(normalized)
        if not row:
            for candidate_key, candidate_row in source_metrics_map.items():
                if candidate_key == normalized or candidate_key in normalized or normalized in candidate_key:
                    row = candidate_row
                    break
        if not row:
            continue

        raw_search_volume = row.get("search_volume")
        raw_cpc = row.get("cpc")
        raw_keyword_difficulty = row.get("keyword_difficulty")
        search_volume = int(raw_search_volume) if raw_search_volume is not None and str(raw_search_volume).strip() != "" else None
        cpc = float(raw_cpc) if raw_cpc is not None and str(raw_cpc).strip() != "" else None
        keyword_difficulty = float(raw_keyword_difficulty) if raw_keyword_difficulty is not None and str(raw_keyword_difficulty).strip() != "" else None
        keyword_metrics[keyword] = {
            "search_volume": search_volume,
            "cpc": round(cpc, 2) if cpc is not None else None,
            "keyword_difficulty": round(keyword_difficulty, 1) if keyword_difficulty is not None else None,
        }
        if search_volume is not None and search_volume > 0:
            volumes.append(search_volume)
        if cpc is not None and cpc > 0:
            cpcs.append(cpc)
        if keyword_difficulty is not None and keyword_difficulty > 0:
            difficulties.append(keyword_difficulty)

    aggregates = {
        "total_search_volume": int(sum(volumes)) if volumes else None,
        "average_cpc": round((sum(cpcs) / len(cpcs)) if cpcs else 0.0, 2) if cpcs else None,
        "average_difficulty": round((sum(difficulties) / len(difficulties)) if difficulties else 0.0, 1) if difficulties else None,
        "keywords_used": list(keyword_metrics.keys()),
    }
    return keyword_metrics, aggregates


def _normalize_search_phrase_text(raw_phrase: str) -> str:
    phrase = re.sub(r"[^a-zA-Z0-9\s\-]", " ", str(raw_phrase or "")).lower()
    phrase = re.sub(r"\s+", " ", phrase).strip(" -")
    if not phrase:
        return ""
    tokens = [token for token in phrase.split(" ") if token]
    if not tokens:
        return ""
    if len(tokens) == 1 and len(tokens[0]) < 3:
        return ""
    if len(tokens) > 3:
        tokens = tokens[:3]
    return " ".join(tokens)


def _normalize_idea_title_text(raw_title: str) -> str:
    """Normalize LLM titles to plain language and reduce consultant-speak drift."""
    if not raw_title:
        return raw_title

    title = re.sub(r"\s+", " ", raw_title.strip())
    jargon_map = {
        r"\bframework\b": "guide",
        r"\bplaybook\b": "plan",
        r"\bmethodology\b": "method",
        r"\boptimization\b": "improvements",
        r"\bscenario\b": "plan",
        r"\bsolvency\b": "financial health",
        r"\barbitrage\b": "price gap",
        r"\bamortization\b": "paydown",
        r"\bvaluation\b": "value",
    }
    for pattern, replacement in jargon_map.items():
        title = re.sub(pattern, replacement, title, flags=re.IGNORECASE)

    title = re.sub(r"\s*[-–—]{2,}\s*", " - ", title)
    title = re.sub(r"\s{2,}", " ", title).strip(" -")
    return title


def _build_keyword_metrics_fallback_payload(raw_keywords) -> tuple[dict, dict]:
    """
    Build a fallback metrics payload from a cluster's stored keyword candidates.
    This keeps idea metrics available even when the generated phrasing does not
    exactly match the cluster keywords used during research.
    """
    keyword_metrics: dict = {}
    volumes = []
    difficulties = []
    cpcs = []

    for item in _coerce_keyword_metric_entries(raw_keywords):
        keyword = str(item.get("keyword") or "").strip()
        if not keyword or keyword in keyword_metrics:
            continue

        search_volume = item.get("search_volume")
        cpc = item.get("cpc")
        keyword_difficulty = item.get("keyword_difficulty")
        keyword_metrics[keyword] = {
            "search_volume": search_volume,
            "cpc": round(float(cpc), 2) if cpc is not None else None,
            "keyword_difficulty": round(float(keyword_difficulty), 1) if keyword_difficulty is not None else None,
        }
        if search_volume is not None and search_volume > 0:
            volumes.append(int(search_volume))
        if cpc is not None and cpc > 0:
            cpcs.append(float(cpc))
        if keyword_difficulty is not None and keyword_difficulty > 0:
            difficulties.append(float(keyword_difficulty))

    aggregates = {
        "total_search_volume": int(sum(volumes)) if volumes else None,
        "average_cpc": round((sum(cpcs) / len(cpcs)) if cpcs else 0.0, 2) if cpcs else None,
        "average_difficulty": round((sum(difficulties) / len(difficulties)) if difficulties else 0.0, 1) if difficulties else None,
        "keywords_used": list(keyword_metrics.keys()),
    }
    return keyword_metrics, aggregates


def _apply_keyword_metrics_to_idea(
    idea: dict,
    source_metrics_map: dict,
    *,
    fallback_keywords: list[str] | None = None,
    fallback_metric_entries=None,
    exact_source: str,
    fallback_source: str | None = None,
    estimate_only_source: str = "llm_estimate_only",
) -> dict:
    """
    Attach the strongest available keyword metrics to an idea.
    Preference order:
    1. Exact/fuzzy matches from the provided source metrics map
    2. Fallback aggregate metrics from supplied metric entries
    3. LLM-estimated keywords only, with no attached numeric metrics
    """
    idea_copy = dict(idea or {})
    idea_metadata = idea_copy.get("idea_metadata") or {}
    if not isinstance(idea_metadata, dict):
        idea_metadata = {}

    fallback_keywords = [str(keyword).strip() for keyword in (fallback_keywords or []) if str(keyword).strip()]
    fallback_keyword_metrics = {}
    fallback_aggregates = {"keywords_used": [], "total_search_volume": None, "average_cpc": None, "average_difficulty": None}
    if fallback_metric_entries:
        fallback_keyword_metrics, fallback_aggregates = _build_keyword_metrics_fallback_payload(fallback_metric_entries)

    keyword_metrics, aggregates = _build_idea_keyword_metrics_payload(
        idea_copy,
        source_metrics_map,
    )
    if keyword_metrics:
        idea_copy["keyword_metrics"] = keyword_metrics
        idea_copy["keywords"] = aggregates["keywords_used"] or idea_copy.get("keywords") or []
        idea_copy["primary_keywords"] = aggregates["keywords_used"] or idea_copy.get("primary_keywords") or []
        if not idea_copy.get("secondary_keywords"):
            idea_copy["secondary_keywords"] = (aggregates["keywords_used"] or [])[1:]
        idea_copy["total_search_volume"] = aggregates["total_search_volume"]
        idea_copy["average_cpc"] = aggregates["average_cpc"]
        idea_copy["average_difficulty"] = aggregates["average_difficulty"]
        idea_metadata["seo_offer_enrichment"] = {
            **(idea_metadata.get("seo_offer_enrichment") or {}),
            "keywords_used": aggregates["keywords_used"],
            "keyword_metrics": keyword_metrics,
            "source": exact_source,
            "enriched_at": datetime.utcnow().isoformat(),
        }
    elif fallback_keyword_metrics:
        idea_copy["keyword_metrics"] = fallback_keyword_metrics
        if not idea_copy.get("keywords"):
            idea_copy["keywords"] = fallback_aggregates["keywords_used"] or fallback_keywords
        if not idea_copy.get("primary_keywords"):
            idea_copy["primary_keywords"] = fallback_aggregates["keywords_used"] or fallback_keywords
        if not idea_copy.get("secondary_keywords"):
            fallback_secondary = (fallback_aggregates["keywords_used"] or fallback_keywords)[1:]
            idea_copy["secondary_keywords"] = fallback_secondary
        idea_copy["total_search_volume"] = fallback_aggregates["total_search_volume"]
        idea_copy["average_cpc"] = fallback_aggregates["average_cpc"]
        idea_copy["average_difficulty"] = fallback_aggregates["average_difficulty"]
        idea_metadata["seo_offer_enrichment"] = {
            **(idea_metadata.get("seo_offer_enrichment") or {}),
            "keywords_used": fallback_aggregates["keywords_used"] or fallback_keywords,
            "keyword_metrics": fallback_keyword_metrics,
            "source": fallback_source or exact_source,
            "enriched_at": datetime.utcnow().isoformat(),
        }
    elif fallback_keywords:
        idea_metadata["seo_offer_enrichment"] = {
            **(idea_metadata.get("seo_offer_enrichment") or {}),
            "keywords_used": idea_copy.get("primary_keywords") or idea_copy.get("keywords") or fallback_keywords,
            "source": estimate_only_source,
        }

    idea_copy["idea_metadata"] = idea_metadata
    return idea_copy


def _coerce_optional_int(value):
    try:
        return int(value) if value is not None and str(value).strip() != "" else None
    except Exception:
        return None


def _coerce_optional_float(value):
    try:
        return float(value) if value is not None and str(value).strip() != "" else None
    except Exception:
        return None


def _build_content_idea_persist_row(
    *,
    idea: dict,
    topic_id: str,
    user_id: str,
    default_subtopic_name: str,
    idea_wp_context: dict,
    category_path: str | None,
    category_context_project_id=None,
    category_context_primary_category_id=None,
    category_context_secondary_category_id=None,
    raw_dataforseo_output=None,
) -> dict:
    keywords = idea.get("primary_keywords") or idea.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(k).strip() for k in keywords if str(k).strip()]

    secondary_keywords = idea.get("secondary_keywords") or []
    if not isinstance(secondary_keywords, list):
        secondary_keywords = []
    secondary_keywords = [str(k).strip() for k in secondary_keywords if str(k).strip()]
    if not secondary_keywords and len(keywords) > 1:
        secondary_keywords = keywords[1:]

    content_type = (idea.get("content_type") or "blog").strip().lower()
    category = "software_tool" if content_type == "software" else "seo_optimized"
    mapped_category_description = str(idea_wp_context.get("category_description") or "").strip()
    if mapped_category_description:
        category = mapped_category_description

    idea_metadata = idea.get("idea_metadata") or {}
    if not isinstance(idea_metadata, dict):
        idea_metadata = {}
    idea_metadata["category_context"] = {
        "project_id": category_context_project_id,
        "primary_category_id": category_context_primary_category_id,
        "secondary_category_id": category_context_secondary_category_id,
        "primary_category_name": idea_wp_context.get("primary_category_name"),
        "secondary_category_name": idea_wp_context.get("secondary_category_name"),
        "primary_category_description": idea_wp_context.get("primary_category_description"),
        "secondary_category_description": idea_wp_context.get("secondary_category_description"),
        "category_path": idea_wp_context.get("category_path") or category_path,
    }

    total_search_volume = _coerce_optional_int(idea.get("total_search_volume"))
    average_difficulty = _coerce_optional_float(idea.get("average_difficulty"))
    average_cpc = _coerce_optional_float(idea.get("average_cpc"))

    return {
        "id": idea.get("id"),
        "title": idea.get("title") or "Untitled Idea",
        "description": idea.get("description") or "",
        "content_type": content_type,
        "category": category,
        "subtopic": str(idea.get("subtopic") or default_subtopic_name or "Idea").strip(),
        "topic_id": topic_id,
        "user_id": user_id,
        "keywords": keywords,
        "primary_keywords": keywords,
        "secondary_keywords": secondary_keywords,
        "wordpress_category_id": idea_wp_context.get("wordpress_category_id"),
        "wordpress_parent_category_id": idea_wp_context.get("wordpress_parent_category_id"),
        "domain": idea_wp_context.get("domain"),
        "search_phrase": idea.get("search_phrase") or "",
        "total_search_volume": total_search_volume,
        "average_difficulty": average_difficulty,
        "average_cpc": average_cpc,
        "viability_score": int(idea.get("viability_score") or 0),
        "traffic_potential_score": int(idea.get("traffic_potential_score") or 0),
        "seo_optimization_score": int(idea.get("seo_optimization_score") or 0),
        "target_intent": idea.get("target_intent") or "",
        "article_format": idea.get("article_format") or "",
        "user_decision_helped": idea.get("user_decision_helped") or "",
        "internal_link_hook": idea.get("internal_link_hook") or "",
        "monetization_hook": idea.get("monetization_hook") or "",
        "product_type": idea.get("product_type") or "",
        "user_job_to_be_done": idea.get("user_job_to_be_done") or "",
        "key_inputs": idea.get("key_inputs") or [],
        "output_result": idea.get("output_result") or "",
        "build_complexity": idea.get("build_complexity") or "",
        "distribution_angle": idea.get("distribution_angle") or "",
        "keyword_metrics": idea.get("keyword_metrics") or {},
        "raw_dataforseo_output": raw_dataforseo_output if raw_dataforseo_output is not None else (idea.get("raw_dataforseo_output") or {}),
        "idea_metadata": idea_metadata,
        "status": idea.get("status") or "draft",
        "created_at": idea.get("created_at") or datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }


def _insert_content_idea_with_schema_fallback(supabase_admin, row: dict, *, log_label: str) -> bool:
    payload = dict(row)
    last_error = None
    max_attempts = max(12, len(payload) + 4)
    for _ in range(max_attempts):
        try:
            supabase_admin.table("content_ideas").insert(payload).execute()
            return True
        except Exception as insert_error:
            last_error = insert_error
            err = str(insert_error)
            missing_cols = re.findall(r"Could not find the '([^']+)' column", err)
            if not missing_cols:
                logger.warning(
                    "%s insert failed without recoverable schema hint payload_keys=%s err=%s",
                    log_label,
                    sorted(payload.keys()),
                    err,
                    exc_info=True,
                )
                return False
            removed_any = False
            for col in missing_cols:
                if col in payload:
                    payload.pop(col, None)
                    removed_any = True
            if not payload:
                return False
            if not removed_any:
                break
    if last_error:
        logger.error(
            "%s insert exhausted schema fallback payload_keys=%s err=%s",
            log_label,
                sorted(payload.keys()),
            last_error,
        )
    return False


def _rank_idea_groups(
    *,
    blog_ideas: list[dict],
    software_ideas: list[dict],
    target_intent,
    tool_potential_score,
    serp_intent_match,
) -> tuple[list[dict], list[dict]]:
    ranked_blog_ideas = _rank_ideas(
        ideas=blog_ideas,
        content_type="blog",
        context_target_intent=target_intent,
        context_tool_potential_score=tool_potential_score,
        context_serp_intent_match=serp_intent_match,
    )
    ranked_software_ideas = _rank_ideas(
        ideas=software_ideas,
        content_type="software",
        context_target_intent=target_intent,
        context_tool_potential_score=tool_potential_score,
        context_serp_intent_match=serp_intent_match,
    )
    return ranked_blog_ideas, ranked_software_ideas


def _build_idea_generation_success_payload(
    *,
    blog_ideas: list[dict],
    software_ideas: list[dict],
    persisted_count: int,
    persisted_idea_ids: list[str],
    extra_fields: dict | None = None,
) -> dict:
    all_ideas = (blog_ideas or []) + (software_ideas or [])
    persistence_warning = None
    if all_ideas and persisted_count != len(all_ideas):
        persistence_warning = (
            f"Generated {len(all_ideas)} ideas but saved {persisted_count}. "
            "Some ideas may not persist across reloads."
        )

    payload = {
        "success": True,
        "blog_ideas": [idea.to_dict() if hasattr(idea, 'to_dict') else idea for idea in (blog_ideas or [])],
        "software_ideas": [idea.to_dict() if hasattr(idea, 'to_dict') else idea for idea in (software_ideas or [])],
        "generated_count": len(all_ideas),
        "persisted_count": persisted_count,
        "persisted_idea_ids": persisted_idea_ids,
        "persistence_warning": persistence_warning,
    }
    if extra_fields:
        payload.update(extra_fields)
    return payload


def _create_cluster_generated_idea(
    *,
    current_idea: dict,
    content_type: str,
    topic_id: str,
    user_id: str,
    run_id: str,
    matched_cluster: dict | None,
) -> dict:
    subtopic_name = (matched_cluster or {}).get("cluster_name") or "Keyword Cluster"
    current_idea = dict(current_idea or {})
    current_idea.setdefault("keywords", [str((matched_cluster or {}).get("primary_keyword") or "").strip()])
    current_idea.setdefault("input_keywords", [str((matched_cluster or {}).get("primary_keyword") or "").strip()])

    idea = create_idea_dict(
        current_idea,
        content_type,
        topic_id,
        user_id,
        subtopic_name,
        primary_user_outcome=(matched_cluster or {}).get("article_angle"),
    )
    if matched_cluster:
        idea_metadata = idea.get("idea_metadata") or {}
        raw_keyword_candidates = matched_cluster.get("keyword_candidates_json") or []
        compact_keyword_candidates = []
        qualified_keyword_candidates = []
        for row in raw_keyword_candidates[:30]:
            if not isinstance(row, dict):
                continue
            keyword = str(row.get("keyword") or "").strip()
            if not keyword:
                continue
            compact_row = {
                "keyword": keyword,
                "search_volume": _coerce_optional_int(row.get("search_volume")),
                "keyword_difficulty": _coerce_optional_float(row.get("keyword_difficulty")),
                "cpc": _coerce_optional_float(row.get("cpc")),
                "intent_label": row.get("intent_label"),
                "competition_level": row.get("competition_level"),
                "opportunity_score": _coerce_optional_float(row.get("opportunity_score")),
            }
            compact_keyword_candidates.append(compact_row)

            search_volume = compact_row.get("search_volume") or 0
            keyword_difficulty = compact_row.get("keyword_difficulty")
            if search_volume > 100 and keyword_difficulty is not None and keyword_difficulty < 35:
                qualified_keyword_candidates.append(compact_row)
        idea_metadata["topic_keyword_research"] = {
            "research_run_id": run_id,
            "keyword_cluster_id": matched_cluster.get("id"),
            "cluster_name": matched_cluster.get("cluster_name"),
            "primary_keyword": matched_cluster.get("primary_keyword"),
            "secondary_keywords": matched_cluster.get("secondary_keywords_json") or [],
            "keyword_candidates": compact_keyword_candidates,
            "qualified_keywords": qualified_keyword_candidates[:12],
            "generation_origin": "topic_keyword_pipeline_v1",
        }
        idea["idea_metadata"] = idea_metadata
    return idea


def _parse_cluster_idea_response_text(
    *,
    text: str,
    content_type: str,
    topic_id: str,
    user_id: str,
    run_id: str,
    selected_clusters: list[dict],
) -> list[dict]:
    ideas = []
    current_idea = {}
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^(BLOG_IDEA|SOFTWARE_IDEA):', line, re.IGNORECASE):
            if current_idea and 'title' in current_idea:
                cluster_id = current_idea.get("cluster_id")
                matched_cluster = next((item for item in selected_clusters if str(item.get("id")) == str(cluster_id)), None)
                ideas.append(_create_cluster_generated_idea(
                    current_idea=current_idea,
                    content_type=content_type,
                    topic_id=topic_id,
                    user_id=user_id,
                    run_id=run_id,
                    matched_cluster=matched_cluster,
                ))
            current_idea = {'id': str(uuid4())}
        elif line.upper().startswith('CLUSTER_ID:'):
            current_idea['cluster_id'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('TITLE:'):
            current_idea['title'] = _normalize_idea_title_text(line.split(':', 1)[1].strip())
        elif line.upper().startswith('DESCRIPTION:'):
            current_idea['description'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('SEARCH_PHRASE:'):
            current_idea['search_phrase'] = _normalize_search_phrase_text(line.split(':', 1)[1].strip())
        elif line.upper().startswith('INPUT_KEYWORDS:'):
            kw_text = line.split(':', 1)[1].strip()
            current_idea['input_keywords'] = [k.strip() for k in kw_text.split(',') if k.strip()]
        elif line.upper().startswith('MONETIZATION:'):
            current_idea['monetization_hook'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('INTENT:'):
            current_idea['target_intent'] = line.split(':', 1)[1].strip().lower()
        elif line.upper().startswith('FORMAT:'):
            current_idea['article_format'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('USER_DECISION_HELPED:'):
            current_idea['user_decision_helped'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('INTERNAL_LINK_HOOK:'):
            current_idea['internal_link_hook'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('PRODUCT_TYPE:'):
            current_idea['product_type'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('USER_JOB:'):
            current_idea['user_job_to_be_done'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('KEY_INPUTS:'):
            current_idea['key_inputs'] = [item.strip() for item in line.split(':', 1)[1].strip().split(',') if item.strip()]
        elif line.upper().startswith('OUTPUT_RESULT:'):
            current_idea['output_result'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('BUILD_COMPLEXITY:'):
            current_idea['build_complexity'] = line.split(':', 1)[1].strip().lower()
        elif line.upper().startswith('DISTRIBUTION_ANGLE:'):
            current_idea['distribution_angle'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('VIABILITY:'):
            try:
                via_text = line.split(':', 1)[1].strip()
                via_match = re.search(r'(\d+)', via_text)
                current_idea['viability_score'] = int(via_match.group(1)) if via_match else 50
            except Exception:
                current_idea['viability_score'] = 50
        elif re.match(r'^END_IDEA', line, re.IGNORECASE):
            if current_idea and 'title' in current_idea:
                cluster_id = current_idea.get("cluster_id")
                matched_cluster = next((item for item in selected_clusters if str(item.get("id")) == str(cluster_id)), None)
                ideas.append(_create_cluster_generated_idea(
                    current_idea=current_idea,
                    content_type=content_type,
                    topic_id=topic_id,
                    user_id=user_id,
                    run_id=run_id,
                    matched_cluster=matched_cluster,
                ))
                current_idea = {}

    if current_idea and 'title' in current_idea:
        cluster_id = current_idea.get("cluster_id")
        matched_cluster = next((item for item in selected_clusters if str(item.get("id")) == str(cluster_id)), None)
        ideas.append(_create_cluster_generated_idea(
            current_idea=current_idea,
            content_type=content_type,
            topic_id=topic_id,
            user_id=user_id,
            run_id=run_id,
            matched_cluster=matched_cluster,
        ))

    return ideas


def _clip_prompt_text(value: str, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[:limit - 3].rstrip()}..."


def _serialize_cluster_prompt_context(selected_clusters: list[dict]) -> tuple[list[dict], str]:
    prompt_clusters = []
    for cluster in selected_clusters or []:
        prompt_clusters.append({
            "id": cluster.get("id"),
            "cluster_name": cluster.get("cluster_name"),
            "primary_keyword": cluster.get("primary_keyword"),
            "secondary_keywords": (cluster.get("secondary_keywords_json") or [])[:5],
            "intent_label": cluster.get("intent_label"),
            "article_angle": cluster.get("article_angle"),
            "opportunity_score": cluster.get("opportunity_score"),
            "software_opportunity_score": cluster.get("software_opportunity_score"),
            "rationale": cluster.get("rationale"),
        })

    clusters_text = "\n".join(
        [
            (
                f"CLUSTER_ID: {cluster.get('id')}\n"
                f"CLUSTER_NAME: {_clip_prompt_text(cluster.get('cluster_name'), 140)}\n"
                f"PRIMARY_KEYWORD: {_clip_prompt_text(cluster.get('primary_keyword'), 120)}\n"
                f"SECONDARY_KEYWORDS: {', '.join([str(k).strip() for k in (cluster.get('secondary_keywords') or []) if str(k).strip()]) or 'N/A'}\n"
                f"INTENT: {_clip_prompt_text(cluster.get('intent_label'), 80)}\n"
                f"ARTICLE_ANGLE: {_clip_prompt_text(cluster.get('article_angle'), 180)}\n"
                f"OPPORTUNITY_SCORE: {cluster.get('opportunity_score') or 0}\n"
                f"SOFTWARE_OPPORTUNITY_SCORE: {cluster.get('software_opportunity_score') or 0}\n"
                f"RATIONALE: {_clip_prompt_text(cluster.get('rationale'), 220)}"
            )
            for cluster in prompt_clusters
        ]
    )
    return prompt_clusters, clusters_text


def _build_cluster_generation_prompts(
    *,
    topic_context: dict,
    category_path: str,
    effective_value_layer_tags: list,
    effective_intent_bucket: str,
    effective_decision_focus: str,
    effective_angle_question: str,
    effective_tool_potential_score: int,
    selected_clusters: list[dict],
    clusters_text: str,
) -> tuple[str, str]:
    blog_prompt = f"""
You are a veteran SEO content strategist. Generate article ideas from keyword clusters in plain, human language that sounds like real Google searches.
Also act as a Search Intent Specialist: reverse complex concepts into short query terms users actually type.

Current Year: 2026
Topic: {_clip_prompt_text(topic_context.get('title') or 'N/A', 140)}
Topic Description: {_clip_prompt_text(topic_context.get('description') or 'N/A', 220)}
Category Path: {_clip_prompt_text(category_path or 'N/A', 140)}
Intent Bucket: {_clip_prompt_text(effective_intent_bucket, 80)}
Decision Focus: {_clip_prompt_text(effective_decision_focus, 220)}
Angle Question: {_clip_prompt_text(effective_angle_question, 220)}
Value Tags: {', '.join([str(tag).strip() for tag in effective_value_layer_tags[:6] if str(tag).strip()]) or 'N/A'}

Selected Keyword Clusters:
{clusters_text}

Generate exactly {len(selected_clusters)} BLOG article ideas, using one idea per cluster.

Hard constraints:
1. Each idea MUST map clearly to one cluster and include the exact CLUSTER_ID in the output.
2. Use the cluster's primary keyword and supporting keywords as the basis for the article.
3. Every idea must target a meaningfully different user decision or question.
4. Avoid consultant/corporate jargon in titles and search phrases.
5. Each idea MUST include SEARCH_PHRASE (1-3 words, lowercase).
6. TITLE must include SEARCH_PHRASE verbatim.
7. INPUT_KEYWORDS must be 3-5 simple query-like phrases, aligned to that cluster only.
8. DESCRIPTION is required and cannot be empty.
9. Prefer article angles that are realistically rankable and monetizable.

For each idea, provide:
- Cluster ID
- Title
- Description
- Search Phrase
- Input Keywords
- Intent
- Format
- User Decision Helped
- Internal Link Hook
- Monetization Hook
- Viability

Output format (use exactly this format):
BLOG_IDEA: [number]
CLUSTER_ID: [cluster uuid]
TITLE: [title]
DESCRIPTION: [description]
SEARCH_PHRASE: [1-3 word query]
INPUT_KEYWORDS: [keyword1, keyword2, keyword3, keyword4]
INTENT: [informational/commercial/transactional]
FORMAT: [comparison/checklist/framework/case-study/how-to/calculator-guide]
USER_DECISION_HELPED: [decision]
INTERNAL_LINK_HOOK: [internal link strategy]
MONETIZATION: [monetization approach]
VIABILITY: [overall viability score 1-100]
END_IDEA
"""

    software_prompt = f"""
You are a product strategist generating software tools users can discover through search. Use plain language and practical naming.

Current Year: 2026
Topic: {_clip_prompt_text(topic_context.get('title') or 'N/A', 140)}
Category Path: {_clip_prompt_text(category_path or 'N/A', 140)}
Tool Potential Score: {effective_tool_potential_score}/100

Selected Keyword Clusters:
{clusters_text}

Generate up to {len(selected_clusters)} SOFTWARE ideas. Only generate a software idea for a cluster if it has real tool, calculator, planner, comparison, or workflow-helper potential.

Hard constraints:
1. Each idea MUST include the exact CLUSTER_ID in the output.
2. These are products/features to build, not articles.
3. Avoid duplicating the same tool concept across clusters.
4. Tool names must be plain-English and practical.
5. SEARCH_PHRASE must be 1-3 words and realistic search language.
6. INPUT_KEYWORDS must be 3-5 simple phrases tied to the cluster.

Output format (use exactly this format):
SOFTWARE_IDEA: [number]
CLUSTER_ID: [cluster uuid]
TITLE: [tool name]
DESCRIPTION: [what the tool does and user interaction]
SEARCH_PHRASE: [1-3 word query]
INPUT_KEYWORDS: [keyword1, keyword2, keyword3, keyword4]
PRODUCT_TYPE: [calculator/planner/evaluator/comparison-tool/dashboard/workflow-helper]
USER_JOB: [job to be done]
KEY_INPUTS: [input1, input2, input3]
OUTPUT_RESULT: [result]
MONETIZATION: [how to monetize the tool]
BUILD_COMPLEXITY: [low/medium/high]
DISTRIBUTION_ANGLE: [distribution strategy]
VIABILITY: [overall viability score 1-100]
END_IDEA
"""
    return blog_prompt, software_prompt


def _select_preferred_project_category_row(
    category_rows: list[dict],
    primary_category_id=None,
    secondary_category_id=None,
) -> dict:
    """Pick the best project_categories row for idea-level WP/category context."""
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


def _resolve_idea_wordpress_category_context(
    supabase_admin,
    *,
    project_id,
    user_id: str,
    primary_category_id=None,
    secondary_category_id=None,
) -> dict:
    """
    Resolve WordPress category metadata and domain for new content_ideas rows.
    """
    context = {
        "wordpress_category_id": None,
        "wordpress_parent_category_id": None,
        "category_description": None,
        "category_path": None,
        "primary_category_id": primary_category_id,
        "secondary_category_id": secondary_category_id,
        "primary_category_name": None,
        "secondary_category_name": None,
        "primary_category_description": None,
        "secondary_category_description": None,
        "domain": None,
    }
    if not project_id:
        return context

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
        context["domain"] = project_row.get("domain") or project_row.get("app_name")
    except Exception:
        logger.warning("Could not resolve project domain for project_id=%s", project_id, exc_info=True)

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
        category_rows = categories_response.data or []
        selected_row = _select_preferred_project_category_row(
            category_rows,
            primary_category_id=primary_category_id,
            secondary_category_id=secondary_category_id,
        )
        category_by_id = {
            str(row.get("id")): row
            for row in category_rows
            if row.get("id")
        }
        primary_row = category_by_id.get(str(primary_category_id)) if primary_category_id else None
        secondary_row = category_by_id.get(str(secondary_category_id)) if secondary_category_id else None

        context["primary_category_name"] = (primary_row or {}).get("name")
        context["secondary_category_name"] = (secondary_row or {}).get("name")
        context["primary_category_description"] = (primary_row or {}).get("description")
        context["secondary_category_description"] = (secondary_row or {}).get("description")
        context["category_path"] = " / ".join(
            [
                part
                for part in [
                    context.get("primary_category_name"),
                    context.get("secondary_category_name"),
                ]
                if str(part or "").strip()
            ]
        ) or None
        if selected_row:
            context["wordpress_category_id"] = selected_row.get("wordpress_category_id")
            context["wordpress_parent_category_id"] = selected_row.get("wordpress_parent_category_id")
            context["category_description"] = selected_row.get("description")
    except Exception:
        logger.warning("Could not resolve project category metadata for project_id=%s", project_id, exc_info=True)

    return context


def _enrich_research_topics(supabase, topics):
    """Attach project and category display names to research topics."""
    if not topics:
        return topics

    project_ids = sorted({topic.get('project_id') for topic in topics if topic.get('project_id')})
    category_ids = sorted({
        category_id
        for topic in topics
        for category_id in [topic.get('primary_category_id'), topic.get('secondary_category_id')]
        if category_id
    })

    projects_by_id = {}
    categories_by_id = {}

    if project_ids:
        project_response = supabase.table('projects').select('id, domain, app_name').in_('id', project_ids).execute()
        projects_by_id = {
            project['id']: project.get('domain') or project.get('app_name')
            for project in (project_response.data or [])
        }

    if category_ids:
        try:
            category_response = supabase.table('project_categories').select('id, name, description').in_('id', category_ids).execute()
        except Exception:
            category_response = supabase.table('project_categories').select('id, name').in_('id', category_ids).execute()
        categories_by_id = {
            category['id']: category.get('name')
            for category in (category_response.data or [])
        }

    for topic in topics:
        topic['project_name'] = projects_by_id.get(topic.get('project_id'))
        topic['primary_category_name'] = categories_by_id.get(topic.get('primary_category_id'))
        topic['secondary_category_name'] = categories_by_id.get(topic.get('secondary_category_id'))

    return topics


def _attach_topic_progress_counts(supabase, user_id, topics):
    """
    Attach lightweight progress counts used by the Research Topics list UI.

    Adds (best-effort) fields:
    - subtopics_count: number of subtopics for the topic
    - topic_keyword_candidate_count: number of ranked topic-level keywords
    - topic_keyword_cluster_count: number of topic-level intent clusters
    - topic_keyword_research_status: latest topic-level research run status
    - content_ideas_count: number of content ideas generated for the topic
    - in_library_count: number of ideas that were published/sent to library

    This is intentionally computed in-process to avoid N+1 requests from the frontend.
    For current paging sizes (e.g. 12 items), fetching minimal columns is fast enough.
    """
    if not topics:
        return topics

    topic_ids = [topic.get("id") for topic in topics if topic.get("id")]
    if not topic_ids:
        return topics

    # Default counts to 0 so the frontend can render "Empty" reliably.
    subtopics_by_topic = {tid: 0 for tid in topic_ids}
    researched_subtopics_by_topic = {tid: 0 for tid in topic_ids}
    has_underlying_data_by_topic = {tid: False for tid in topic_ids}
    topic_keyword_candidate_counts = {tid: 0 for tid in topic_ids}
    topic_keyword_cluster_counts = {tid: 0 for tid in topic_ids}
    topic_keyword_research_status_by_topic = {tid: None for tid in topic_ids}
    ideas_by_topic = {tid: 0 for tid in topic_ids}
    in_library_by_topic = {tid: 0 for tid in topic_ids}

    # Subtopics: primary key path is research_topic_id; fallback to project_id for legacy schemas.
    try:
        sub_resp = (
            supabase
            .table("subtopics")
            .select("research_topic_id,project_id,search_volume,seo_difficulty,cpc,affiliate_offer_count,monetization_data,trend_analysis")
            .eq("user_id", user_id)
            .in_("research_topic_id", topic_ids)
            .execute()
        )
        for row in (sub_resp.data or []):
            tid = row.get("research_topic_id")
            if tid in subtopics_by_topic:
                subtopics_by_topic[tid] += 1
                monetization_data = row.get("monetization_data") or {}
                trend_analysis = row.get("trend_analysis") or {}
                has_signal = bool(
                    (row.get("search_volume") or 0) > 0
                    or (row.get("seo_difficulty") or 0) > 0
                    or (row.get("cpc") or 0) > 0
                    or (row.get("affiliate_offer_count") or 0) > 0
                    or (monetization_data.get("offers") or [])
                )
                if has_signal:
                    has_underlying_data_by_topic[tid] = True
                if bool(trend_analysis.get("manual_researched")):
                    researched_subtopics_by_topic[tid] += 1
    except Exception:
        logger.debug("Could not compute subtopics_count via research_topic_id", exc_info=True)

    try:
        # Only apply fallback for topics that had zero via primary path.
        missing = [tid for tid, count in subtopics_by_topic.items() if count == 0]
        if missing:
            sub_fallback_resp = (
                supabase
                .table("subtopics")
                .select("research_topic_id,project_id,search_volume,seo_difficulty,cpc,affiliate_offer_count,monetization_data,trend_analysis")
                .eq("user_id", user_id)
                .in_("project_id", missing)
                .execute()
            )
            for row in (sub_fallback_resp.data or []):
                tid = row.get("project_id")
                if tid in subtopics_by_topic:
                    subtopics_by_topic[tid] += 1
                    monetization_data = row.get("monetization_data") or {}
                    trend_analysis = row.get("trend_analysis") or {}
                    has_signal = bool(
                        (row.get("search_volume") or 0) > 0
                        or (row.get("seo_difficulty") or 0) > 0
                        or (row.get("cpc") or 0) > 0
                        or (row.get("affiliate_offer_count") or 0) > 0
                        or (monetization_data.get("offers") or [])
                    )
                    if has_signal:
                        has_underlying_data_by_topic[tid] = True
                    if bool(trend_analysis.get("manual_researched")):
                        researched_subtopics_by_topic[tid] += 1
    except Exception:
        logger.debug("Could not compute subtopics_count via legacy project_id fallback", exc_info=True)

    # Topic-level keyword research: new pipeline counts and latest run status.
    try:
        runs_resp = (
            supabase
            .table("topic_keyword_research_runs")
            .select("id,topic_id,status,updated_at,created_at")
            .eq("user_id", user_id)
            .in_("topic_id", topic_ids)
            .execute()
        )
        latest_run_ids = {}
        latest_run_sort_keys = {}
        for row in (runs_resp.data or []):
            tid = row.get("topic_id")
            if tid not in topic_keyword_research_status_by_topic:
                continue
            sort_key = row.get("updated_at") or row.get("created_at") or ""
            if tid not in latest_run_sort_keys or str(sort_key) > str(latest_run_sort_keys[tid]):
                latest_run_sort_keys[tid] = sort_key
                latest_run_ids[tid] = row.get("id")
                topic_keyword_research_status_by_topic[tid] = row.get("status")

        candidate_run_ids = [run_id for run_id in latest_run_ids.values() if run_id]
        if candidate_run_ids:
            candidates_resp = (
                supabase
                .table("topic_keyword_candidates")
                .select("topic_id,research_run_id")
                .eq("user_id", user_id)
                .in_("research_run_id", candidate_run_ids)
                .execute()
            )
            for row in (candidates_resp.data or []):
                tid = row.get("topic_id")
                if tid in topic_keyword_candidate_counts:
                    topic_keyword_candidate_counts[tid] += 1
                    has_underlying_data_by_topic[tid] = True

            clusters_resp = (
                supabase
                .table("topic_keyword_clusters")
                .select("topic_id,research_run_id")
                .eq("user_id", user_id)
                .in_("research_run_id", candidate_run_ids)
                .execute()
            )
            for row in (clusters_resp.data or []):
                tid = row.get("topic_id")
                if tid in topic_keyword_cluster_counts:
                    topic_keyword_cluster_counts[tid] += 1
                    has_underlying_data_by_topic[tid] = True
    except Exception:
        logger.debug("Could not compute topic keyword research counts", exc_info=True)

    # Content ideas: created by idea-burst and linked by topic_id.
    try:
        ideas_resp = (
            supabase
            .table("content_ideas")
            .select("topic_id,published,published_to_titles,status")
            .eq("user_id", user_id)
            .in_("topic_id", topic_ids)
            .execute()
        )
        for row in (ideas_resp.data or []):
            tid = row.get("topic_id")
            if tid not in ideas_by_topic:
                continue
            ideas_by_topic[tid] += 1

            status = (row.get("status") or "").strip().lower()
            is_in_library = bool(
                row.get("published")
                or row.get("published_to_titles")
                or status == "published"
            )
            if is_in_library:
                in_library_by_topic[tid] += 1
    except Exception:
        logger.debug("Could not compute content_ideas_count/in_library_count", exc_info=True)

    for topic in topics:
        tid = topic.get("id")
        if not tid:
            continue
        topic["subtopics_count"] = subtopics_by_topic.get(tid, 0)
        researched_count = researched_subtopics_by_topic.get(tid, 0)
        total_subtopics = subtopics_by_topic.get(tid, 0)
        topic["researched_subtopics_count"] = researched_count
        topic["has_underlying_data"] = has_underlying_data_by_topic.get(tid, False)
        topic["all_subtopics_researched"] = bool(total_subtopics > 0 and researched_count == total_subtopics)
        topic["topic_keyword_candidate_count"] = topic_keyword_candidate_counts.get(tid, 0)
        topic["topic_keyword_cluster_count"] = topic_keyword_cluster_counts.get(tid, 0)
        topic["topic_keyword_research_status"] = topic_keyword_research_status_by_topic.get(tid)
        topic["content_ideas_count"] = ideas_by_topic.get(tid, 0)
        topic["in_library_count"] = in_library_by_topic.get(tid, 0)

    return topics


ANGLE_METADATA_FIELDS = [
    "intent_bucket",
    "decision_focus",
    "angle_question",
    "value_layer_tags",
    "target_audience",
    "evidence_sources",
    "related_terms",
]

TOPIC_MODE_FIELDS = [
    "topic_mode",
    "keyword_viability_score",
    "keyword_viability_label",
    "topic_generation_reasoning",
    "topic_generation_metadata",
]

VALID_TOPIC_MODES = {"keyword_first", "editorial_first", "hybrid"}
VALID_KEYWORD_VIABILITY_LABELS = {"high", "medium", "low"}


def _extract_angle_metadata(payload):
    """Extract optional angle metadata fields when explicitly provided."""
    metadata = {}
    for key in ANGLE_METADATA_FIELDS:
        if key not in payload:
            continue
        value = payload.get(key)
        if value is None:
            continue
        metadata[key] = value
    return metadata


def _coerce_topic_mode(value):
    normalized = _safe_string(value)
    if normalized and normalized.lower() in VALID_TOPIC_MODES:
        return normalized.lower()
    return "hybrid"


def _coerce_keyword_viability_score(value):
    if value is None or value == "":
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(100.0, score))


def _coerce_keyword_viability_label(value, score=None):
    normalized = _safe_string(value)
    if normalized and normalized.lower() in VALID_KEYWORD_VIABILITY_LABELS:
        return normalized.lower()
    numeric_score = _coerce_keyword_viability_score(score)
    if numeric_score is None:
        return "medium"
    if numeric_score >= 70:
        return "high"
    if numeric_score >= 40:
        return "medium"
    return "low"


def _extract_topic_mode_metadata(payload):
    metadata = {}
    metadata["topic_mode"] = _coerce_topic_mode(payload.get("topic_mode"))
    if "keyword_viability_score" in payload:
        metadata["keyword_viability_score"] = _coerce_keyword_viability_score(payload.get("keyword_viability_score"))
    else:
        metadata["keyword_viability_score"] = None
    metadata["keyword_viability_label"] = _coerce_keyword_viability_label(
        payload.get("keyword_viability_label"),
        metadata.get("keyword_viability_score"),
    )
    metadata["topic_generation_reasoning"] = _safe_string(payload.get("topic_generation_reasoning"))
    topic_generation_metadata = payload.get("topic_generation_metadata")
    metadata["topic_generation_metadata"] = topic_generation_metadata if isinstance(topic_generation_metadata, dict) else {}
    return metadata


def _safe_string(value):
    """Normalize optional values into compact strings."""
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = " ".join(value.split()).strip()
        return cleaned or None
    return str(value)


def _coerce_topic_rating(value):
    """Normalize topic rating into an integer between 0 and 5."""
    if value is None:
        return 0
    try:
        rating = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(5, rating))


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def _derive_intent_bucket(title: str, category_path: str) -> str:
    """Infer a default intent bucket from topic/category phrasing."""
    text = f"{title} {category_path}".lower()
    if any(term in text for term in ["vs", "compare", "comparison", "best", "top", "alternative"]):
        return "commercial_evaluation"
    if any(term in text for term in ["calculator", "tool", "template", "checklist", "framework"]):
        return "solution_enablement"
    if any(term in text for term in ["cost", "roi", "value", "profit", "returns", "pricing"]):
        return "decision_financial"
    return "informational_decision"


def _derive_value_layer_tags(title: str, category_path: str) -> list[str]:
    """Infer high-level value tags used later for decomposition and idea scoring."""
    text = f"{title} {category_path}".lower()
    tags: list[str] = []
    if any(term in text for term in ["roi", "return", "profit", "resale", "yield"]):
        tags.append("roi-focused")
    if any(term in text for term in ["cost", "price", "expense", "budget", "hidden cost"]):
        tags.append("cost-vs-value")
    if any(term in text for term in ["timing", "when to", "cycle", "market timing"]):
        tags.append("timing-decision")
    if any(term in text for term in ["location", "city", "state", "geo", "geographic"]):
        tags.append("location-decision")
    if any(term in text for term in ["audit", "scorecard", "framework", "evaluation"]):
        tags.append("hidden-cost-audit")
    if any(term in text for term in ["tool", "calculator", "dashboard", "app", "automation"]):
        tags.append("tool-builder")
    if not tags:
        tags.append("decision-support")
    return tags[:4]


def _derive_angle_metadata(
    title: str,
    description: str,
    primary_category_name: str,
    secondary_category_name: str,
    project_description: str,
) -> dict:
    """Build fallback angle metadata when the client did not provide structured fields."""
    title_clean = _safe_string(title) or "Untitled topic"
    description_clean = _safe_string(description)
    category_parts = [p for p in [primary_category_name, secondary_category_name] if p]
    category_path = " / ".join(category_parts)

    intent_bucket = _derive_intent_bucket(title_clean, category_path)
    decision_focus = (
        description_clean
        or f"Help users evaluate options and make a better decision about {title_clean}."
    )
    angle_question = f"How should someone evaluate {title_clean} and decide the best next action?"
    target_audience = None
    if project_description:
        lowered = project_description.lower()
        if any(term in lowered for term in ["investor", "investing", "portfolio", "capital"]):
            target_audience = "investors and operators"
        elif any(term in lowered for term in ["homeowner", "home", "property owner"]):
            target_audience = "homeowners and property buyers"

    related_terms = []
    for token in re.split(r"[^a-zA-Z0-9]+", title_clean.lower()):
        if token and len(token) > 3 and token not in related_terms:
            related_terms.append(token)
    if secondary_category_name:
        related_terms.append(secondary_category_name.lower())

    metadata = {
        "intent_bucket": intent_bucket,
        "decision_focus": decision_focus,
        "angle_question": angle_question,
        "value_layer_tags": _derive_value_layer_tags(title_clean, category_path),
        "target_audience": target_audience,
        "related_terms": related_terms[:8],
    }

    if category_path:
        metadata["evidence_sources"] = [f"category:{category_path}"]

    return {key: value for key, value in metadata.items() if value not in [None, "", []]}


def _hydrate_angle_metadata_for_payloads(
    supabase,
    payloads: list[dict],
):
    """Fill missing angle metadata fields using project/category context."""
    if not payloads:
        return payloads

    project_ids = sorted({item.get("project_id") for item in payloads if item.get("project_id")})
    category_ids = sorted({
        category_id
        for item in payloads
        for category_id in [item.get("primary_category_id"), item.get("secondary_category_id")]
        if category_id
    })

    projects_by_id = {}
    categories_by_id = {}

    if project_ids:
        project_response = (
            supabase
            .table("projects")
            .select("id, site_description, websitedescription, targetaudiencedescription")
            .in_("id", project_ids)
            .execute()
        )
        projects_by_id = {row["id"]: row for row in (project_response.data or [])}

    if category_ids:
        category_response = (
            supabase
            .table("project_categories")
            .select("id, name")
            .in_("id", category_ids)
            .execute()
        )
        categories_by_id = {
            row["id"]: (row.get("name") or "")
            for row in (category_response.data or [])
        }

    for item in payloads:
        existing = _extract_angle_metadata(item)
        missing = [field for field in ANGLE_METADATA_FIELDS if field not in existing]
        if not missing:
            continue

        project = projects_by_id.get(item.get("project_id")) or {}
        generated = _derive_angle_metadata(
            title=item.get("title") or "",
            description=item.get("description") or "",
            primary_category_name=categories_by_id.get(item.get("primary_category_id")) or "",
            secondary_category_name=categories_by_id.get(item.get("secondary_category_id")) or "",
            project_description=_safe_string(
                project.get("site_description")
                or project.get("websitedescription")
                or project.get("targetaudiencedescription")
            ) or "",
        )
        for field in missing:
            if field in generated and generated[field] is not None:
                item[field] = generated[field]

    return payloads


def _extract_trend_titles(last_trend_report):
    """Extract recent trend theme titles from the saved trend report."""
    if not isinstance(last_trend_report, dict):
        return []

    report_content = last_trend_report.get('report_content') or {}
    topics = report_content.get('topics') or []
    titles = []
    for item in topics:
        if not isinstance(item, dict):
            continue
        title = _safe_string(item.get('title'))
        if title:
            titles.append(title)
    return titles[:8]


def _build_decision_focus(topic, primary_category_name, secondary_category_name):
    """Create a compact statement describing what choice this topic should help users make."""
    title = _safe_string(topic.get('title')) or "this topic"
    topic_description = _safe_string(topic.get('description'))
    category_path = " / ".join([p for p in [primary_category_name, secondary_category_name] if p])

    if topic_description:
        return topic_description
    if category_path:
        return f"Use {title} to help the user make a better decision within {category_path}."
    return f"Use {title} to help the user evaluate options, compare tradeoffs, and choose an action."


def _build_category_strategy_hint(
    primary_category_name,
    secondary_category_name,
    primary_category_description=None,
    secondary_category_description=None,
):
    """Build a compact sentence that anchors decomposition to category strategy."""
    parts = []
    if _safe_string(primary_category_name):
        parts.append(f"Primary category: {_safe_string(primary_category_name)}.")
    if _safe_string(primary_category_description):
        parts.append(f"Primary category context: {_safe_string(primary_category_description)}")
    if _safe_string(secondary_category_name):
        parts.append(f"Sub-category: {_safe_string(secondary_category_name)}.")
    if _safe_string(secondary_category_description):
        parts.append(f"Sub-category context: {_safe_string(secondary_category_description)}")
    return " ".join(parts).strip()


def _build_decomposition_context(
    topic,
    project,
    primary_category_name=None,
    secondary_category_name=None,
    primary_category_description=None,
    secondary_category_description=None,
):
    """Build a richer topic packet for downstream decomposition prompts."""
    project_description = _safe_string(
        (project or {}).get('site_description')
        or (project or {}).get('websitedescription')
        or (project or {}).get('targetaudiencedescription')
    )
    topic_description = _safe_string(topic.get('description'))
    category_path = " / ".join([name for name in [primary_category_name, secondary_category_name] if name]) or None
    trend_titles = _extract_trend_titles((project or {}).get('last_trend_report'))

    signal_terms = []
    report_content = ((project or {}).get('last_trend_report') or {}).get('report_content') or {}
    for item in (report_content.get('topics') or [])[:6]:
        if isinstance(item, dict):
            for term in item.get('related_terms') or []:
                term_clean = _safe_string(term)
                if term_clean and term_clean not in signal_terms:
                    signal_terms.append(term_clean)

    constraints = [
        "Generate subtopics that can become meaningful article clusters, not just adjacent keywords.",
        "Prefer concrete decision angles, comparison paths, scorecards, frameworks, audits, calculators, or scenario-based topics.",
        "Stay tightly aligned to the website niche and selected category lens.",
        "Avoid generic interpretations of the topic if the site context points to a narrower intent.",
        "Honor primary/sub-category strategy context when choosing examples, terminology, and audience framing.",
    ]

    return {
        "project_name": _safe_string((project or {}).get('domain') or (project or {}).get('app_name')),
        "project_description": project_description,
        "topic_description": topic_description,
        "category_path": category_path,
        "primary_category_name": _safe_string(primary_category_name),
        "secondary_category_name": _safe_string(secondary_category_name),
        "primary_category_description": _safe_string(primary_category_description),
        "secondary_category_description": _safe_string(secondary_category_description),
        "category_strategy_hint": _build_category_strategy_hint(
            primary_category_name=primary_category_name,
            secondary_category_name=secondary_category_name,
            primary_category_description=primary_category_description,
            secondary_category_description=secondary_category_description,
        ),
        "intent_bucket": _safe_string(topic.get('intent_bucket')),
        "decision_focus": _build_decision_focus(topic, primary_category_name, secondary_category_name),
        "angle_question": _safe_string(topic.get('angle_question')),
        "value_layer_tags": topic.get('value_layer_tags') or [],
        "target_audience": _safe_string(topic.get('target_audience')),
        "evidence_sources": topic.get('evidence_sources') or [],
        "trend_titles": trend_titles,
        "signal_terms": (topic.get('related_terms') or []) + signal_terms,
        "decomposition_constraints": constraints,
    }

@research_topics_bp.route('/', methods=['GET'])
@require_api_key
def list_research_topics():
    """List research topics with pagination and filtering."""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify(ErrorResponse(
                error="database_error",
                message="Database connection failed",
                error_code="DB_CONNECTION_FAILED",
                status=500
            ).dict()), 500

        # Get query parameters
        page = request.args.get('page', 1, type=int)
        size = request.args.get('size', 10, type=int)
        status = request.args.get('status')
        order_by = request.args.get('order_by', 'created_at')
        order_direction = request.args.get('order_direction', 'desc')
        project_id = request.args.get('project_id')
        primary_category_id = request.args.get('primary_category_id')
        secondary_category_id = request.args.get('secondary_category_id')
        user_id = _resolve_user_id_from_request(supabase)

        if not user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401

        # Calculate range
        start = (page - 1) * size
        end = start + size - 1

        # Build query
        query = supabase.table('research_topics').select('*', count='exact')
        query = query.eq('user_id', user_id)

        if status:
            query = query.eq('status', status)
        if project_id:
            query = query.eq('project_id', project_id)
        if primary_category_id:
            query = query.eq('primary_category_id', primary_category_id)
        if secondary_category_id:
            query = query.eq('secondary_category_id', secondary_category_id)
        
        # Apply sorting
        query = query.order(order_by, desc=(order_direction == 'desc'))
        
        # Apply pagination
        query = query.range(start, end)

        # Execute
        response = query.execute()
        items = _enrich_research_topics(supabase, response.data or [])
        items = _attach_topic_progress_counts(supabase, user_id, items)

        return jsonify({
            "items": items,
            "total": response.count,
            "page": page,
            "size": size,
            "has_next": (start + len(items)) < response.count,
            "has_prev": page > 1
        }), 200

    except Exception as e:
        logger.error(f"Error listing research topics: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=f"An error occurred while listing research topics: {str(e)}",
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500

@research_topics_bp.route('/', methods=['POST'])
@require_api_key
def create_research_topic():
    """Create a new research topic."""
    try:
        if not request.is_json:
            return jsonify(ErrorResponse(
                error="invalid_content_type",
                message="Content-Type must be application/json",
                error_code="INVALID_CONTENT_TYPE",
                status=400
            ).dict()), 400

        data = request.get_json()
        
        # Basic validation
        if not data.get('title'):
            return jsonify(ErrorResponse(
                error="validation_error",
                message="Title is required",
                error_code="VALIDATION_ERROR",
                status=400
            ).dict()), 400

        supabase = get_supabase_client()
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401

        user_id = _resolve_user_id_from_request(supabase, data)

        if not user_id:
             return jsonify(ErrorResponse(
                error="authentication_required", 
                message="Could not resolve a valid user ID. Please ensure you are logged in.",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
             ).dict()), 401

        # Validate required fields
        if not data.get('project_id'):
            return jsonify(ErrorResponse(
                error="validation_error",
                message="project_id is required",
                error_code="VALIDATION_ERROR",
                status=400
            ).dict()), 400

        insert_data = {
            "title": data.get('title'),
            "description": data.get('description', ''),
            "status": data.get('status', 'active'),
            "is_archived": bool(data.get('is_archived', False)),
            "topic_rating": _coerce_topic_rating(data.get('topic_rating')),
            "updated_at": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "project_id": data.get('project_id'),
            "primary_category_id": data.get('primary_category_id'),
            "secondary_category_id": data.get('secondary_category_id'),
            "topic_source": data.get('topic_source'),
            "source_topic_id": data.get('source_topic_id'),
        }
        insert_data.update(_extract_angle_metadata(data))
        insert_data.update(_extract_topic_mode_metadata(data))
        hydrated = _hydrate_angle_metadata_for_payloads(supabase, [insert_data])
        insert_data = hydrated[0] if hydrated else insert_data

        supabase_admin = _get_admin_supabase_client(supabase)
        existing_response = (
            supabase_admin
            .table('research_topics')
            .select('*')
            .eq('user_id', user_id)
            .eq('title', insert_data["title"])
            .limit(1)
            .execute()
        )
        existing_rows = existing_response.data or []
        if existing_rows:
            enriched_existing = _enrich_research_topics(supabase, existing_rows)
            return jsonify(enriched_existing[0]), 200

        response = supabase_admin.table('research_topics').insert(insert_data).execute()
            
        if not response or not response.data:
            raise Exception("Failed to insert record")

        enriched = _enrich_research_topics(supabase, response.data)
        return jsonify(enriched[0]), 201

    except Exception as e:
        message = str(e)
        if 'idx_research_topics_user_title' in message or 'duplicate key value violates unique constraint' in message:
            try:
                supabase = get_supabase_client()
                user_id = _resolve_user_id_from_request(supabase, request.get_json(silent=True) or {})
                title = (request.get_json(silent=True) or {}).get('title')
                if user_id and title:
                    supabase_admin = _get_admin_supabase_client(supabase)
                    existing_response = (
                        supabase_admin
                        .table('research_topics')
                        .select('*')
                        .eq('user_id', user_id)
                        .eq('title', title)
                        .limit(1)
                        .execute()
                    )
                    existing_rows = existing_response.data or []
                    if existing_rows:
                        enriched_existing = _enrich_research_topics(supabase, existing_rows)
                        return jsonify(enriched_existing[0]), 200
            except Exception:
                logger.warning("Failed duplicate-topic recovery for title=%r", (request.get_json(silent=True) or {}).get('title'), exc_info=True)
        logger.error(f"Error creating research topic: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=f"An error occurred while creating research topic: {str(e)}",
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500

@research_topics_bp.route('/<topic_id>', methods=['GET'])
@require_api_key
def get_research_topic(topic_id):
    """Get a research topic by ID."""
    try:
        supabase = get_supabase_client()
        user_id = _resolve_user_id_from_request(supabase)
        if not user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401

        response = (
            supabase
            .table('research_topics')
            .select('*')
            .eq('id', topic_id)
            .eq('user_id', user_id)
            .single()
            .execute()
        )
        
        if not response.data: # Should throw error from single() usually if not found, but safe check
             return jsonify(ErrorResponse(
                error="not_found",
                message="Research topic not found",
                error_code="NOT_FOUND",
                status=404
            ).dict()), 404
        enriched = _enrich_research_topics(supabase, [response.data])
        return jsonify(enriched[0]), 200

    except Exception as e:
        # Check if it is a "no rows found" error often returned as exception by supabase-py
        if "JSON object must be str, bytes or bytearray" in str(e) or "Row not found" in str(e): 
             return jsonify(ErrorResponse(
                error="not_found",
                message="Research topic not found",
                error_code="NOT_FOUND",
                status=404
            ).dict()), 404

        logger.error(f"Error getting research topic: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message="An error occurred",
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500

@research_topics_bp.route('/<topic_id>', methods=['PUT'])
@require_api_key
def update_research_topic(topic_id):
    """Update a research topic."""
    try:
        data = request.get_json()
        supabase = get_supabase_client()
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401

        user_id = _resolve_user_id_from_request(supabase, data)
        if not user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401
        
        update_data = {
            k: v for k, v in data.items() if k in [
                'title',
                'description',
                'status',
                'is_archived',
                'topic_rating',
                'project_id',
                'primary_category_id',
                'secondary_category_id',
                'topic_source',
                'source_topic_id',
                *ANGLE_METADATA_FIELDS,
                *TOPIC_MODE_FIELDS,
            ]
        }
        update_data['updated_at'] = datetime.utcnow().isoformat()
        if 'topic_rating' in update_data:
            update_data['topic_rating'] = _coerce_topic_rating(update_data.get('topic_rating'))
        if any(field in update_data for field in TOPIC_MODE_FIELDS):
            update_data.update(_extract_topic_mode_metadata(update_data))
        
        response = (
            supabase
            .table('research_topics')
            .update(update_data)
            .eq('id', topic_id)
            .eq('user_id', user_id)
            .execute()
        )
        
        if not response.data:
             return jsonify(ErrorResponse(
                error="not_found",
                message="Research topic not found",
                error_code="NOT_FOUND",
                status=404
            ).dict()), 404

        enriched = _enrich_research_topics(supabase, response.data)
        return jsonify(enriched[0]), 200

    except Exception as e:
        logger.error(f"Error updating research topic: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message="An error occurred",
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@research_topics_bp.route('/bulk-create', methods=['POST'])
@require_api_key
def bulk_create_research_topics():
    """Create multiple research topics in a single request."""
    try:
        if not request.is_json:
            return jsonify(ErrorResponse(
                error="invalid_content_type",
                message="Content-Type must be application/json",
                error_code="INVALID_CONTENT_TYPE",
                status=400
            ).dict()), 400

        data = request.get_json() or {}
        items = data.get('items') or []
        if not isinstance(items, list) or not items:
            return jsonify(ErrorResponse(
                error="validation_error",
                message="A non-empty 'items' array is required",
                error_code="VALIDATION_ERROR",
                status=400
            ).dict()), 400

        supabase = get_supabase_client()
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401

        user_id = _resolve_user_id_from_request(supabase, data)

        if not user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401

        now = datetime.utcnow().isoformat()
        insert_payload = []
        for item in items:
            title = (item.get('title') or '').strip()
            if not title:
                return jsonify(ErrorResponse(
                    error="validation_error",
                    message="Each topic must include a title",
                    error_code="VALIDATION_ERROR",
                    status=400
                ).dict()), 400

            # Validate required fields
            if not item.get('project_id'):
                return jsonify(ErrorResponse(
                    error="validation_error",
                    message="Each topic must include a project_id",
                    error_code="VALIDATION_ERROR",
                    status=400
                ).dict()), 400

            item_payload = {
                "title": title,
                "description": item.get('description', ''),
                "status": item.get('status', 'active'),
                "is_archived": bool(item.get('is_archived', False)),
                "topic_rating": _coerce_topic_rating(item.get('topic_rating')),
                "updated_at": now,
                "user_id": user_id,
                "project_id": item.get('project_id'),
                "primary_category_id": item.get('primary_category_id'),
                "secondary_category_id": item.get('secondary_category_id'),
                "topic_source": item.get('topic_source'),
                "source_topic_id": item.get('source_topic_id'),
            }
            item_payload.update(_extract_angle_metadata(item))
            item_payload.update(_extract_topic_mode_metadata(item))
            insert_payload.append(item_payload)

        insert_payload = _hydrate_angle_metadata_for_payloads(supabase, insert_payload)

        supabase_admin = _get_admin_supabase_client(supabase)
        response = supabase_admin.table('research_topics').insert(insert_payload).execute()

        if not response or not response.data:
            raise Exception("Failed to insert records")

        enriched = _enrich_research_topics(supabase, response.data)
        return jsonify(enriched), 201

    except Exception as e:
        logger.error(f"Error bulk creating research topics: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message="An error occurred",
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500

@research_topics_bp.route('/<topic_id>', methods=['DELETE'])
@require_api_key
def delete_research_topic(topic_id):
    """Delete a research topic and best-effort cascade related records."""
    try:
        supabase = get_supabase_client()
        user_id = _resolve_user_id_from_request(supabase)
        if not user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401

        # Verify topic ownership first.
        topic_owner_res = (
            supabase
            .table('research_topics')
            .select('id,user_id')
            .eq('id', topic_id)
            .limit(1)
            .execute()
        )
        topic_owner_rows = topic_owner_res.data or []
        if not topic_owner_rows or topic_owner_rows[0].get('user_id') != user_id:
            return jsonify(ErrorResponse(
                error="forbidden",
                message="You do not have access to this research topic",
                error_code="FORBIDDEN",
                status=403
            ).dict()), 403

        # 1) Delete content ideas linked to this topic.
        deleted_content_ideas = 0
        detached_titles = 0
        try:
            idea_rows = (
                supabase
                .table('content_ideas')
                .select('id')
                .eq('topic_id', topic_id)
                .eq('user_id', user_id)
                .execute()
                .data
                or []
            )
            detached_titles = _unlink_titles_for_idea_ids(
                supabase=supabase,
                user_id=user_id,
                idea_ids=[row.get("id") for row in idea_rows if row.get("id")],
            )
            ideas_deleted = (
                supabase
                .table('content_ideas')
                .delete()
                .eq('topic_id', topic_id)
                .eq('user_id', user_id)
                .execute()
            )
            deleted_content_ideas = len(ideas_deleted.data or [])
        except Exception as ideas_err:
            logger.warning("Topic delete cascade: failed to delete content ideas topic_id=%s user_id=%s err=%s", topic_id, user_id, ideas_err)

        # 2) Resolve subtopics for evidence cleanup.
        subtopic_rows = []
        try:
            subtopic_res = (
                supabase
                .table('subtopics')
                .select('id')
                .eq('research_topic_id', topic_id)
                .eq('user_id', user_id)
                .execute()
            )
            subtopic_rows = subtopic_res.data or []
        except Exception:
            try:
                subtopic_res = (
                    supabase
                    .table('subtopics')
                    .select('id')
                    .eq('project_id', topic_id)
                    .eq('user_id', user_id)
                    .execute()
                )
                subtopic_rows = subtopic_res.data or []
            except Exception as subtopic_fetch_err:
                logger.warning("Topic delete cascade: failed to load subtopics topic_id=%s user_id=%s err=%s", topic_id, user_id, subtopic_fetch_err)

        subtopic_ids = [row.get('id') for row in subtopic_rows if row.get('id')]

        # 3) Delete evidence rows (optional tables).
        if subtopic_ids:
            try:
                supabase.table('subtopic_keyword_candidates').delete().eq('user_id', user_id).in_('subtopic_id', subtopic_ids).execute()
            except Exception as e:
                logger.debug("Topic delete cascade: keyword evidence cleanup skipped topic_id=%s err=%s", topic_id, e)
            try:
                supabase.table('subtopic_affiliate_evidence').delete().eq('user_id', user_id).in_('subtopic_id', subtopic_ids).execute()
            except Exception as e:
                logger.debug("Topic delete cascade: affiliate evidence cleanup skipped topic_id=%s err=%s", topic_id, e)

        # 4) Delete subtopics (both new + legacy linkage).
        deleted_subtopics = 0
        try:
            subtopics_deleted = (
                supabase
                .table('subtopics')
                .delete()
                .eq('user_id', user_id)
                .or_(f"research_topic_id.eq.{topic_id},project_id.eq.{topic_id}")
                .execute()
            )
            deleted_subtopics = len(subtopics_deleted.data or [])
        except Exception as subtopic_delete_err:
            logger.warning("Topic delete cascade: failed to delete subtopics topic_id=%s user_id=%s err=%s", topic_id, user_id, subtopic_delete_err)

        # 5) Delete topic row.
        response = (
            supabase
            .table('research_topics')
            .delete()
            .eq('id', topic_id)
            .eq('user_id', user_id)
            .execute()
        )

        deleted_topics = len(response.data or [])
        if deleted_topics == 0:
            return jsonify(ErrorResponse(
                error="not_found",
                message="Research topic not found",
                error_code="NOT_FOUND",
                status=404
            ).dict()), 404

        logger.info(
            "Topic delete cascade completed topic_id=%s user_id=%s deleted_topics=%s deleted_subtopics=%s deleted_content_ideas=%s detached_titles=%s",
            topic_id,
            user_id,
            deleted_topics,
            deleted_subtopics,
            deleted_content_ideas,
            detached_titles,
        )
        return jsonify({
            "message": "Topic deleted successfully",
            "deleted_topics": deleted_topics,
            "deleted_subtopics": deleted_subtopics,
            "deleted_content_ideas": deleted_content_ideas,
            "detached_content_library_records": detached_titles,
        }), 200

    except Exception as e:
        logger.error(f"Error deleting research topic: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message="An error occurred",
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@research_topics_bp.route('/<topic_id>/subtopics/<subtopic_id>', methods=['DELETE'])
@require_api_key
def delete_subtopic(topic_id, subtopic_id):
    """Delete a subtopic and all related content ideas for that topic/subtopic."""
    try:
        supabase = get_supabase_client()
        user_id = _resolve_user_id_from_request(supabase)
        if not user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401

        # Verify topic ownership.
        topic_owner_res = (
            supabase
            .table('research_topics')
            .select('id,user_id')
            .eq('id', topic_id)
            .limit(1)
            .execute()
        )
        topic_owner_rows = topic_owner_res.data or []
        if not topic_owner_rows or topic_owner_rows[0].get('user_id') != user_id:
            return jsonify(ErrorResponse(
                error="forbidden",
                message="You do not have access to this research topic",
                error_code="FORBIDDEN",
                status=403
            ).dict()), 403

        # Load subtopic name for content_ideas cleanup (ideas store subtopic name).
        subtopic_res = (
            supabase
            .table('subtopics')
            .select('id,name,research_topic_id,project_id')
            .eq('id', subtopic_id)
            .eq('user_id', user_id)
            .or_(f"research_topic_id.eq.{topic_id},project_id.eq.{topic_id}")
            .limit(1)
            .execute()
        )
        subtopic_rows = subtopic_res.data or []
        if not subtopic_rows:
            return jsonify(ErrorResponse(
                error="not_found",
                message="Subtopic not found",
                error_code="NOT_FOUND",
                status=404
            ).dict()), 404

        subtopic_name = (subtopic_rows[0].get('name') or '').strip()
        deleted_content_ideas = 0
        detached_titles = 0

        # Delete content ideas linked by topic + subtopic name.
        if subtopic_name:
            try:
                idea_rows = (
                    supabase
                    .table('content_ideas')
                    .select('id')
                    .eq('topic_id', topic_id)
                    .eq('subtopic', subtopic_name)
                    .eq('user_id', user_id)
                    .execute()
                    .data
                    or []
                )
                detached_titles = _unlink_titles_for_idea_ids(
                    supabase=supabase,
                    user_id=user_id,
                    idea_ids=[row.get("id") for row in idea_rows if row.get("id")],
                )
                ideas_deleted = (
                    supabase
                    .table('content_ideas')
                    .delete()
                    .eq('topic_id', topic_id)
                    .eq('subtopic', subtopic_name)
                    .eq('user_id', user_id)
                    .execute()
                )
                deleted_content_ideas += len(ideas_deleted.data or [])
            except Exception as ideas_err:
                logger.warning(
                    "Subtopic delete cascade: failed to delete content ideas topic_id=%s subtopic_id=%s subtopic_name=%r user_id=%s err=%s",
                    topic_id,
                    subtopic_id,
                    subtopic_name,
                    user_id,
                    ideas_err,
                )

        # Optional evidence tables cleanup.
        try:
            supabase.table('subtopic_keyword_candidates').delete().eq('user_id', user_id).eq('subtopic_id', subtopic_id).execute()
        except Exception as e:
            logger.debug("Subtopic delete cascade: keyword evidence cleanup skipped subtopic_id=%s err=%s", subtopic_id, e)
        try:
            supabase.table('subtopic_affiliate_evidence').delete().eq('user_id', user_id).eq('subtopic_id', subtopic_id).execute()
        except Exception as e:
            logger.debug("Subtopic delete cascade: affiliate evidence cleanup skipped subtopic_id=%s err=%s", subtopic_id, e)

        subtopic_delete_res = (
            supabase
            .table('subtopics')
            .delete()
            .eq('id', subtopic_id)
            .eq('user_id', user_id)
            .or_(f"research_topic_id.eq.{topic_id},project_id.eq.{topic_id}")
            .execute()
        )
        deleted_subtopics = len(subtopic_delete_res.data or [])
        if deleted_subtopics == 0:
            return jsonify(ErrorResponse(
                error="not_found",
                message="Subtopic not found",
                error_code="NOT_FOUND",
                status=404
            ).dict()), 404

        logger.info(
            "Subtopic delete cascade completed topic_id=%s subtopic_id=%s subtopic_name=%r user_id=%s deleted_subtopics=%s deleted_content_ideas=%s detached_titles=%s",
            topic_id,
            subtopic_id,
            subtopic_name,
            user_id,
            deleted_subtopics,
            deleted_content_ideas,
            detached_titles,
        )
        return jsonify({
            "message": "Subtopic deleted successfully",
            "deleted_subtopics": deleted_subtopics,
            "deleted_content_ideas": deleted_content_ideas,
            "detached_content_library_records": detached_titles,
        }), 200

    except Exception as e:
        logger.error("Error deleting subtopic topic_id=%s subtopic_id=%s err=%s", topic_id, subtopic_id, e, exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message="Failed to delete subtopic",
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500

@research_topics_bp.route('/stats/overview', methods=['GET'])
@require_api_key
def get_overview_stats():
    """Get overview statistics for research topics."""
    try:
        supabase = get_supabase_client()
        
        # We need counts for active, completed, archived.
        # This is a bit expensive with multiple queries, but simplest for now.
        # Or select status and count.
        
        # Supabase API doesn't support "group by" cleanly in simple client always.
        # We can just get total counts.
        
        total_res = supabase.table('research_topics').select('id', count='exact', head=True).execute()
        active_res = supabase.table('research_topics').select('id', count='exact', head=True).eq('status', 'active').execute()
        completed_res = supabase.table('research_topics').select('id', count='exact', head=True).eq('status', 'completed').execute()
        archived_res = supabase.table('research_topics').select('id', count='exact', head=True).eq('status', 'archived').execute()
        
        stats = {
            "total_topics": total_res.count or 0,
            "active_topics": active_res.count or 0,
            "completed_topics": completed_res.count or 0,
            "archived_topics": archived_res.count or 0,
            "total_subtopics": 0, # Implement if table exists
            "total_analyses": 0,
            "total_content_ideas": 0
        }
        
        return jsonify(stats), 200

    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message="An error occurred",
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500

@research_topics_bp.route('/<topic_id>/subtopics', methods=['GET'])
@require_api_key
def get_subtopics(topic_id):
    """Get subtopics for a research topic."""
    try:
        supabase = get_supabase_client()
        user_id = _resolve_user_id_from_request(supabase)
        if not user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401

        # Try primary lookup by research_topic_id (new schema).
        # If that fails or returns no rows, fallback to project_id (legacy/mixed schema).
        items = []
        primary_error = None

        try:
            response = (
                supabase
                .table('subtopics')
                .select('*')
                .eq('research_topic_id', topic_id)
                .eq('user_id', user_id)
                .execute()
            )
            items = response.data or []
            logger.info(
                "Subtopics GET used research_topic_id path topic_id=%s user_id=%s count=%s",
                topic_id, user_id, len(items)
            )
        except Exception as e:
            primary_error = e
            logger.warning(
                "Subtopics GET research_topic_id path failed topic_id=%s user_id=%s error=%s",
                topic_id, user_id, e
            )

        if not items:
            try:
                fallback_response = (
                    supabase
                    .table('subtopics')
                    .select('*')
                    .eq('project_id', topic_id)
                    .eq('user_id', user_id)
                    .execute()
                )
                items = fallback_response.data or []
                logger.info(
                    "Subtopics GET used project_id fallback path topic_id=%s user_id=%s count=%s primary_error=%s",
                    topic_id, user_id, len(items), primary_error
                )
            except Exception as fallback_error:
                logger.error(
                    "Subtopics GET failed for both paths topic_id=%s user_id=%s primary_error=%s fallback_error=%s",
                    topic_id, user_id, primary_error, fallback_error
                )
                raise

        return jsonify({
            "items": items,
            "total": len(items)
        }), 200

    except Exception as e:
        logger.error(f"Error getting subtopics: {str(e)}", exc_info=True)
        # Return empty list instead of 500 to avoid breaking UI if table missing
        return jsonify({"items": [], "total": 0}), 200


@research_topics_bp.route('/<topic_id>/subtopics/<subtopic_id>', methods=['PUT'])
@require_api_key
def update_subtopic(topic_id, subtopic_id):
    """Update a subtopic for a research topic."""
    try:
        if not request.is_json:
            return jsonify(ErrorResponse(
                error="invalid_content_type",
                message="Content-Type must be application/json",
                error_code="INVALID_CONTENT_TYPE",
                status=400
            ).dict()), 400

        data = request.get_json() or {}
        supabase = get_supabase_client()
        user_id = _resolve_user_id_from_request(supabase)
        if not user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401

        topic_owner = (
            supabase
            .table('research_topics')
            .select('id,user_id')
            .eq('id', topic_id)
            .single()
            .execute()
        )
        if not topic_owner.data or topic_owner.data.get('user_id') != user_id:
            return jsonify(ErrorResponse(
                error="forbidden",
                message="You do not have access to this research topic",
                error_code="FORBIDDEN",
                status=403
            ).dict()), 403

        update_data = {}
        allowed_fields = {
            "name",
            "is_archived",
            "topic_rating",
            "trend_direction",
            "trend_score",
            "seo_difficulty",
            "search_volume",
            "cpc",
            "affiliate_offer_count",
            "keywords",
            "trend_analysis",
            "monetization_data",
            "rationale",
            "target_audience",
            "intent_bucket",
            "decision_focus",
            "angle_question",
            "value_layer_tags",
            "cluster_type",
            "primary_user_outcome",
            "serp_intent_match",
            "tool_potential_score",
        }
        for key in allowed_fields:
            if key in data:
                update_data[key] = data.get(key)

        if "is_archived" in update_data:
            update_data["is_archived"] = _coerce_bool(update_data.get("is_archived"))
        if "topic_rating" in update_data:
            update_data["topic_rating"] = _coerce_topic_rating(update_data.get("topic_rating"))

        update_data["updated_at"] = datetime.utcnow().isoformat()

        updated = (
            supabase
            .table('subtopics')
            .update(update_data)
            .eq('id', subtopic_id)
            .eq('user_id', user_id)
            .or_(f"research_topic_id.eq.{topic_id},project_id.eq.{topic_id}")
            .execute()
        )

        rows = updated.data or []
        if not rows:
            return jsonify(ErrorResponse(
                error="not_found",
                message="Subtopic not found",
                error_code="NOT_FOUND",
                status=404
            ).dict()), 404

        return jsonify(rows[0]), 200

    except Exception as e:
        logger.error(f"Error updating subtopic {subtopic_id}: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message="Failed to update subtopic",
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500

# Additional imports for Enhanced Logic
from src.services.enhanced_topic_decomposition_service import EnhancedTopicDecompositionService
from src.services.subtopics_service import SubtopicsService
from src.services.topic_keyword_research_service import TopicKeywordResearchService
from src.core.models.enhanced_subtopic import EnhancedSubtopic

# Instantiate services
enhanced_decomposition_service = EnhancedTopicDecompositionService()
subtopics_service = SubtopicsService()


def _get_topic_keyword_research_service(supabase):
    return TopicKeywordResearchService(
        supabase=supabase,
        supabase_admin=_get_admin_supabase_client(supabase),
    )

@research_topics_bp.route('/<topic_id>/subtopics/generate', methods=['POST'])
@require_api_key
def generate_subtopics(topic_id):
    """
    Generate subtopics for a research topic using the Enhanced Decomposition Pipeline.

    Flask is a synchronous framework — async def routes are not natively supported.
    We bridge to the async service pipeline using asyncio.run() so the event loop
    is created fresh per request (safe for WSGI/Gunicorn workers).
    """
    import asyncio

    request_id = str(uuid4())
    request_started = time.perf_counter()
    try:
        supabase = get_supabase_client()
        request_user_id = _resolve_user_id_from_request(supabase)
        if not request_user_id:
            logger.warning(
                "Subtopic generation missing authenticated user request_id=%s topic_id=%s path=%s has_auth_header=%s",
                request_id,
                topic_id,
                request.path,
                bool(request.headers.get('Authorization')),
            )
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401

        # 1. Fetch topic metadata
        topic_fetch_started = time.perf_counter()
        topic_res = (
            supabase
            .table('research_topics')
            .select('id, title, description, user_id, project_id, primary_category_id, secondary_category_id, intent_bucket, decision_focus, angle_question, value_layer_tags, target_audience, evidence_sources, related_terms')
            .eq('id', topic_id)
            .single()
            .execute()
        )

        if not topic_res.data:
            return jsonify(ErrorResponse(
                error="not_found",
                message="Research topic not found",
                error_code="NOT_FOUND",
                status=404
            ).dict()), 404

        topic = topic_res.data
        topic_title = topic['title']
        user_id     = topic['user_id']
        logger.info(
            "Subtopic generation started request_id=%s topic_id=%s request_user_id=%s owner_user_id=%s title=%r topic_fetch_ms=%.1f",
            request_id,
            topic_id,
            request_user_id,
            user_id,
            topic_title,
            (time.perf_counter() - topic_fetch_started) * 1000,
        )
        if user_id != request_user_id:
            return jsonify(ErrorResponse(
                error="forbidden",
                message="You do not have access to this research topic",
                error_code="FORBIDDEN",
                status=403
            ).dict()), 403

        project = {}
        project_id = topic.get('project_id')
        if project_id:
            try:
                project_res = (
                    supabase
                    .table('projects')
                    .select('id, domain, app_name, site_description, websitedescription, targetaudiencedescription, last_trend_report')
                    .eq('id', project_id)
                    .limit(1)
                    .execute()
                )
                if project_res.data:
                    project = project_res.data[0] or {}
            except Exception as project_err:
                logger.warning(f"Failed to load project context for decomposition: {project_err}")

        category_names = {}
        category_ids = [
            cid for cid in [topic.get('primary_category_id'), topic.get('secondary_category_id')] if cid
        ]
        if category_ids:
            try:
                try:
                    category_res = (
                        supabase
                        .table('project_categories')
                        .select('id, name, description')
                        .in_('id', category_ids)
                        .execute()
                    )
                except Exception:
                    category_res = (
                        supabase
                        .table('project_categories')
                        .select('id, name')
                        .in_('id', category_ids)
                        .execute()
                    )

                category_names = {
                    item['id']: {
                        "name": item.get('name'),
                        "description": item.get('description'),
                    }
                    for item in (category_res.data or [])
                }
            except Exception as category_err:
                logger.warning(f"Failed to load category names for decomposition: {category_err}")

        primary_category = category_names.get(topic.get('primary_category_id')) or {}
        secondary_category = category_names.get(topic.get('secondary_category_id')) or {}
        decomposition_context = _build_decomposition_context(
            topic,
            project,
            primary_category_name=primary_category.get("name"),
            secondary_category_name=secondary_category.get("name"),
            primary_category_description=primary_category.get("description"),
            secondary_category_description=secondary_category.get("description"),
        )

        # 2. Run the async decomposition pipeline synchronously
        async def _run():
            decomposition_started = time.perf_counter()
            result = await enhanced_decomposition_service.decompose_topic_enhanced(
                query=topic_title,
                user_id=user_id,
                max_subtopics=12,
                decomposition_context=decomposition_context,
            )
            logger.info(
                "Enhanced decomposition finished request_id=%s success=%s subtopic_count=%s message=%r methods=%s decomposition_ms=%.1f",
                request_id,
                result.get("success"),
                len(result.get("subtopics") or []),
                result.get("message"),
                result.get("enhancement_methods"),
                (time.perf_counter() - decomposition_started) * 1000,
            )

            if not result.get("success"):
                warning_message = result.get("message", "Decomposition failed")
                logger.warning(
                    "Decomposition unsuccessful request_id=%s topic=%s. Returning empty set. Message=%s",
                    request_id,
                    topic_id,
                    warning_message
                )
                return [], {
                    "success": False,
                    "message": warning_message,
                    "processing_time": result.get("processing_time"),
                    "enhancement_methods": result.get("enhancement_methods", []),
                    "debug": result.get("debug"),
                }

            enhanced_subtopics_data = result.get("subtopics", [])
            saved_subtopics = []
            failed_subtopics = []
            persistence_started = time.perf_counter()
            logger.info(
                "Persisting generated subtopics request_id=%s generated_count=%s titles=%s",
                request_id,
                len(enhanced_subtopics_data),
                [item.get("title") for item in enhanced_subtopics_data[:12]]
            )

            for sub_data in enhanced_subtopics_data:
                # Debug logging to trace metrics
                logger.info(f"DEBUG Subtopic '{sub_data.get('title')}': vol={sub_data.get('search_volume')}, cpc={sub_data.get('cpc')}, kd={sub_data.get('keyword_difficulty')}")

                trend_data = {
                    "trend_score":    80,
                    "seo_difficulty": sub_data.get("keyword_difficulty", 50),
                    "search_volume":  sub_data.get("search_volume", 0),
                    "cpc":            sub_data.get("cpc", 0.0),
                    "keywords":       sub_data.get("seed_keywords", []),
                    "rationale":      sub_data.get("rationale"),
                    "target_audience": sub_data.get("target_audience"),
                    "trend_analysis": sub_data.get("trend_analysis"),
                    "monetization":   sub_data.get("monetization_data"),
                    "intent_bucket": sub_data.get("intent_bucket"),
                    "decision_focus": sub_data.get("decision_focus"),
                    "angle_question": sub_data.get("angle_question"),
                    "value_layer_tags": sub_data.get("value_layer_tags", []),
                    "cluster_type": sub_data.get("cluster_type"),
                    "primary_user_outcome": sub_data.get("primary_user_outcome"),
                    "serp_intent_match": sub_data.get("serp_intent_match"),
                    "tool_potential_score": sub_data.get("tool_potential_score"),
                }
                # Preserve new evidence/scoring metadata in existing JSON fields for backward compatibility.
                trend_analysis_payload = trend_data.get("trend_analysis") or {}
                trend_analysis_payload["validation_state"] = sub_data.get("validation_state")
                trend_analysis_payload["seo_readiness_score"] = sub_data.get("seo_readiness_score")
                trend_analysis_payload["geo_readiness_score"] = sub_data.get("geo_readiness_score")
                trend_analysis_payload["editorial_value_score"] = sub_data.get("editorial_value_score")
                trend_data["trend_analysis"] = trend_analysis_payload

                monetization_payload = trend_data.get("monetization") or {}
                monetization_payload["keyword_evidence"] = sub_data.get("keyword_evidence", [])
                monetization_payload["primary_keyword"] = (
                    (sub_data.get("keyword_evidence") or [{}])[0].get("keyword")
                    if sub_data.get("keyword_evidence")
                    else None
                )
                trend_data["monetization"] = monetization_payload
                logger.info(f"DEBUG trend_data: {trend_data}")

                saved = await subtopics_service.create(
                    research_topic_id=topic_id,
                    name=sub_data.get("title"),
                    user_id=user_id,
                    trend_data=trend_data,
                )
                if saved:
                    logger.info(f"DEBUG saved subtopic: {saved.get('name')}, vol={saved.get('search_volume')}, cpc={saved.get('cpc')}")
                    # Persist keyword evidence into dedicated table when available.
                    keyword_evidence = sub_data.get("keyword_evidence") or []
                    if keyword_evidence:
                        try:
                            keyword_rows = []
                            for idx, kw in enumerate(keyword_evidence):
                                keyword_rows.append({
                                    "subtopic_id": saved.get("id"),
                                    "research_topic_id": topic_id,
                                    "user_id": user_id,
                                    "keyword": kw.get("keyword"),
                                    "variant_type": kw.get("source"),
                                    "search_volume": int(kw.get("search_volume") or 0),
                                    "cpc": float(kw.get("cpc") or 0.0),
                                    "keyword_difficulty": int(kw.get("keyword_difficulty") or 0),
                                    "competition": kw.get("competition"),
                                    "is_selected_primary": bool(idx == 0),
                                    "selection_reason": kw.get("selection_reason"),
                                })
                            supabase.table("subtopic_keyword_candidates").insert(keyword_rows).execute()
                        except Exception as keyword_evidence_err:
                            logger.warning(
                                "Keyword evidence persistence skipped topic_id=%s subtopic_id=%s error=%s",
                                topic_id,
                                saved.get("id"),
                                keyword_evidence_err,
                            )
                    saved_subtopics.append(saved)
                else:
                    failed_subtopics.append(sub_data.get("title"))

            logger.info(
                "Subtopic persistence summary request_id=%s attempted=%s saved=%s failed=%s failed_titles=%s persistence_ms=%.1f",
                request_id,
                len(enhanced_subtopics_data),
                len(saved_subtopics),
                len(failed_subtopics),
                failed_subtopics,
                (time.perf_counter() - persistence_started) * 1000,
            )

            return saved_subtopics, {
                "success": True,
                "message": result.get("message", "Subtopics generated"),
                "processing_time": result.get("processing_time"),
                "enhancement_methods": result.get("enhancement_methods", []),
                "debug": result.get("debug"),
            }

        saved_subtopics, result = asyncio.run(_run())
        logger.info(
            "Subtopic generation response request_id=%s total=%s success=%s message=%r total_ms=%.1f",
            request_id,
            len(saved_subtopics),
            result.get("success"),
            result.get("message"),
            (time.perf_counter() - request_started) * 1000,
        )

        return jsonify({
            "items": saved_subtopics,
            "total": len(saved_subtopics),
            "meta": {
                "success":            result.get("success", True),
                "message":            result.get("message"),
                "processing_time":    result.get("processing_time"),
                "enhancement_methods": result.get("enhancement_methods"),
                "debug": result.get("debug"),
            }
        }), 200

    except Exception as e:
        logger.error(
            "Error generating subtopics request_id=%s topic_id=%s elapsed_ms=%.1f error=%s",
            request_id,
            topic_id,
            (time.perf_counter() - request_started) * 1000,
            e,
            exc_info=True,
        )
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@research_topics_bp.route('/<topic_id>/keyword-research/run', methods=['POST'])
@require_api_key
def run_topic_keyword_research(topic_id):
    """Run the new topic-level keyword research pipeline for a single research topic."""
    import asyncio

    try:
        data = request.get_json(silent=True) or {}
        supabase = get_supabase_client()
        user_id = _resolve_user_id_from_request(supabase, data)
        if not user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401

        service = _get_topic_keyword_research_service(supabase)
        result = asyncio.run(
            service.run_topic_research(
                topic_id=topic_id,
                user_id=user_id,
                replace_existing=bool(data.get("replace_existing", False)),
                filters=data.get("filters") if isinstance(data.get("filters"), dict) else None,
                score_config=data.get("score_config") if isinstance(data.get("score_config"), dict) else None,
                manual_seed_keywords=data.get("manual_seed_keywords") if isinstance(data.get("manual_seed_keywords"), list) else None,
            )
        )

        return jsonify({
            "success": True,
            "run": result.get("run"),
            "summary": result.get("summary"),
            "keyword_count": len(result.get("keywords") or []),
            "cluster_count": len(result.get("clusters") or []),
            "top_clusters": (result.get("clusters") or [])[:5],
        }), 200
    except ValueError as err:
        message = str(err)
        error_code = "NOT_FOUND" if message == "Research topic not found" else "INVALID_REQUEST"
        status = 404 if message == "Research topic not found" else 400
        return jsonify(ErrorResponse(
            error="not_found" if status == 404 else "invalid_request",
            message=message,
            error_code=error_code,
            status=status
        ).dict()), status
    except Exception as err:
        logger.error("Error running topic keyword research topic_id=%s err=%s", topic_id, err, exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(err) or "Failed to run topic keyword research",
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@research_topics_bp.route('/<topic_id>/keyword-research/latest', methods=['GET'])
@require_api_key
def get_latest_topic_keyword_research(topic_id):
    """Fetch the latest topic-level keyword research run for a research topic."""
    try:
        supabase = get_supabase_client()
        user_id = _resolve_user_id_from_request(supabase)
        if not user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401

        service = _get_topic_keyword_research_service(supabase)
        run = service.get_latest_run(topic_id=topic_id, user_id=user_id)
        if not run:
            return jsonify(ErrorResponse(
                error="not_found",
                message="No topic keyword research run found for this topic",
                error_code="NOT_FOUND",
                status=404
            ).dict()), 404

        return jsonify(run), 200
    except Exception as err:
        logger.error("Error fetching latest topic keyword research topic_id=%s err=%s", topic_id, err, exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message="Failed to fetch topic keyword research",
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@research_topics_bp.route('/<topic_id>/keyword-research/runs/<run_id>', methods=['GET'])
@require_api_key
def get_topic_keyword_research_run(topic_id, run_id):
    """Fetch one topic-level keyword research run."""
    try:
        supabase = get_supabase_client()
        user_id = _resolve_user_id_from_request(supabase)
        if not user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401

        service = _get_topic_keyword_research_service(supabase)
        run = service.get_run(run_id=run_id, topic_id=topic_id, user_id=user_id)
        if not run:
            return jsonify(ErrorResponse(
                error="not_found",
                message="Topic keyword research run not found",
                error_code="NOT_FOUND",
                status=404
            ).dict()), 404

        return jsonify(run), 200
    except Exception as err:
        logger.error("Error fetching topic keyword research run topic_id=%s run_id=%s err=%s", topic_id, run_id, err, exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message="Failed to fetch topic keyword research run",
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@research_topics_bp.route('/<topic_id>/keyword-research/runs/<run_id>/keywords', methods=['GET'])
@require_api_key
def list_topic_keyword_research_keywords(topic_id, run_id):
    """List persisted keyword candidates for a topic-level research run."""
    try:
        supabase = get_supabase_client()
        user_id = _resolve_user_id_from_request(supabase)
        if not user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401

        include_filtered = request.args.get("include_filtered", "true").strip().lower() != "false"
        service = _get_topic_keyword_research_service(supabase)
        run = service.get_run(run_id=run_id, topic_id=topic_id, user_id=user_id)
        if not run:
            return jsonify(ErrorResponse(
                error="not_found",
                message="Topic keyword research run not found",
                error_code="NOT_FOUND",
                status=404
            ).dict()), 404

        rows = service.list_keywords(
            run_id=run_id,
            topic_id=topic_id,
            user_id=user_id,
            include_filtered=include_filtered,
        )
        return jsonify({
            "items": rows,
            "total": len(rows),
            "include_filtered": include_filtered,
        }), 200
    except Exception as err:
        logger.error("Error listing topic keyword research keywords topic_id=%s run_id=%s err=%s", topic_id, run_id, err, exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message="Failed to list topic keyword research keywords",
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@research_topics_bp.route('/<topic_id>/keyword-research/runs/<run_id>/clusters', methods=['GET'])
@require_api_key
def list_topic_keyword_research_clusters(topic_id, run_id):
    """List persisted keyword clusters for a topic-level research run."""
    try:
        supabase = get_supabase_client()
        user_id = _resolve_user_id_from_request(supabase)
        if not user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401

        service = _get_topic_keyword_research_service(supabase)
        run = service.get_run(run_id=run_id, topic_id=topic_id, user_id=user_id)
        if not run:
            return jsonify(ErrorResponse(
                error="not_found",
                message="Topic keyword research run not found",
                error_code="NOT_FOUND",
                status=404
            ).dict()), 404

        rows = service.list_clusters(
            run_id=run_id,
            topic_id=topic_id,
            user_id=user_id,
        )
        return jsonify({
            "items": rows,
            "total": len(rows),
        }), 200
    except Exception as err:
        logger.error("Error listing topic keyword research clusters topic_id=%s run_id=%s err=%s", topic_id, run_id, err, exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message="Failed to list topic keyword research clusters",
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@research_topics_bp.route('/<topic_id>/keyword-research/runs/<run_id>/generate-ideas', methods=['POST'])
@require_api_key
def generate_ideas_from_topic_keyword_clusters(topic_id, run_id):
    """Generate content ideas from selected topic keyword clusters and persist them to content_ideas."""
    try:
        if not request.is_json:
            return jsonify(ErrorResponse(
                error="invalid_content_type",
                message="Content-Type must be application/json",
                error_code="INVALID_CONTENT_TYPE",
                status=400
            ).dict()), 400

        data = request.get_json() or {}
        supabase = get_supabase_client()
        request_user_id = _resolve_user_id_from_request(supabase, data)
        if not request_user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401

        user_id = data.get("user_id") or request_user_id
        if user_id != request_user_id:
            return jsonify(ErrorResponse(
                error="forbidden",
                message="You do not have access to this user_id",
                error_code="FORBIDDEN",
                status=403
            ).dict()), 403

        service = _get_topic_keyword_research_service(supabase)
        run = service.get_run(run_id=run_id, topic_id=topic_id, user_id=user_id)
        if not run:
            return jsonify(ErrorResponse(
                error="not_found",
                message="Topic keyword research run not found",
                error_code="NOT_FOUND",
                status=404
            ).dict()), 404

        cluster_rows = service.list_clusters(run_id=run_id, topic_id=topic_id, user_id=user_id)
        cluster_ids = [str(item).strip() for item in (data.get("cluster_ids") or []) if str(item).strip()]
        selected_clusters = [
            row for row in cluster_rows
            if not cluster_ids or str(row.get("id") or "") in cluster_ids
        ]
        selected_clusters = selected_clusters[:3]
        if not selected_clusters:
            return jsonify(ErrorResponse(
                error="validation_error",
                message="No keyword clusters selected",
                error_code="VALIDATION_ERROR",
                status=400
            ).dict()), 400

        supabase_admin = _get_admin_supabase_client(supabase)
        topic_context_res = (
            supabase_admin
            .table('research_topics')
            .select(
                'title, description, project_id, primary_category_id, secondary_category_id, '
                'intent_bucket, decision_focus, angle_question, value_layer_tags, target_audience'
            )
            .eq('id', topic_id)
            .single()
            .execute()
        )
        topic_context = topic_context_res.data or {}
        idea_wp_context = _resolve_idea_wordpress_category_context(
            supabase_admin,
            project_id=topic_context.get("project_id"),
            user_id=user_id,
            primary_category_id=topic_context.get("primary_category_id"),
            secondary_category_id=topic_context.get("secondary_category_id"),
        )

        category_path = idea_wp_context.get("category_path") or "N/A"
        category_ids = [
            category_id
            for category_id in [topic_context.get('primary_category_id'), topic_context.get('secondary_category_id')]
            if category_id
        ]
        if category_ids and category_path == "N/A":
            try:
                category_response = (
                    supabase_admin
                    .table('project_categories')
                    .select('id, name')
                    .in_('id', category_ids)
                    .execute()
                )
                category_map = {item['id']: item.get('name') for item in (category_response.data or [])}
                primary_name = category_map.get(topic_context.get('primary_category_id'))
                secondary_name = category_map.get(topic_context.get('secondary_category_id'))
                category_path = " / ".join([part for part in [primary_name, secondary_name] if part]) or category_path
            except Exception:
                logger.warning("Could not load category names for cluster idea generation", exc_info=True)

        cluster_keyword_metrics_maps = {
            str(cluster.get("id")): _build_keyword_metrics_map(cluster.get("keyword_candidates_json") or [])
            for cluster in selected_clusters
        }
        _, clusters_text = _serialize_cluster_prompt_context(selected_clusters)

        effective_value_layer_tags = topic_context.get('value_layer_tags') or ["decision-support"]
        effective_intent_bucket = topic_context.get('intent_bucket') or "informational_decision"
        effective_decision_focus = topic_context.get('decision_focus') or f"Help users make a decision about {topic_context.get('title') or 'this topic'}"
        effective_angle_question = topic_context.get('angle_question') or f"What is the best way to approach {topic_context.get('title') or 'this topic'}?"
        try:
            effective_tool_potential_score = int(max([float(cluster.get("software_opportunity_score") or 0.0) for cluster in selected_clusters] or [50]))
        except Exception:
            effective_tool_potential_score = 50
        effective_serp_intent_match = "high" if any(
            (cluster.get("serp_validation_json") or {}).get("article_intent_confidence", 0) >= 75
            for cluster in selected_clusters
        ) else "medium"

        from supabase_client import LLM_ROLE_RESEARCH_IDEA_GENERATION
        from src.services.llm.llm_service import llm_service
        import asyncio

        async def generate_cluster_ideas():
            blog_prompt, software_prompt = _build_cluster_generation_prompts(
                topic_context=topic_context,
                category_path=category_path,
                effective_value_layer_tags=effective_value_layer_tags,
                effective_intent_bucket=effective_intent_bucket,
                effective_decision_focus=effective_decision_focus,
                effective_angle_question=effective_angle_question,
                effective_tool_potential_score=effective_tool_potential_score,
                selected_clusters=selected_clusters,
                clusters_text=clusters_text,
            )

            blog_response = await llm_service.generate_text(
                blog_prompt,
                task_role=LLM_ROLE_RESEARCH_IDEA_GENERATION,
                max_tokens=2200,
            )
            software_response = await llm_service.generate_text(
                software_prompt,
                task_role=LLM_ROLE_RESEARCH_IDEA_GENERATION,
                max_tokens=1800,
            )
            return blog_response.content, software_response.content

        blog_text, software_text = asyncio.run(generate_cluster_ideas())

        blog_ideas = _parse_cluster_idea_response_text(
            text=blog_text,
            content_type='blog',
            topic_id=topic_id,
            user_id=user_id,
            run_id=run_id,
            selected_clusters=selected_clusters,
        )
        software_ideas = _parse_cluster_idea_response_text(
            text=software_text,
            content_type='software',
            topic_id=topic_id,
            user_id=user_id,
            run_id=run_id,
            selected_clusters=selected_clusters,
        )

        def _cluster_context_for_idea(idea: dict) -> dict:
            cluster_meta = ((idea.get("idea_metadata") or {}).get("topic_keyword_research") or {})
            cluster_id = cluster_meta.get("keyword_cluster_id")
            return next((item for item in selected_clusters if str(item.get("id")) == str(cluster_id)), {}) or {}

        def _attach_cluster_keyword_metrics(ideas: list[dict]) -> list[dict]:
            enriched_ideas = []
            for idea in ideas or []:
                idea_copy = dict(idea)
                cluster_context = _cluster_context_for_idea(idea_copy)
                idea_metadata = idea_copy.get("idea_metadata") or {}
                cluster_id = str(((idea_metadata.get("topic_keyword_research") or {}).get("keyword_cluster_id")) or "")
                cluster_prompt_keywords = [
                    str(cluster_context.get("primary_keyword") or "").strip(),
                    *[str(k).strip() for k in (cluster_context.get("secondary_keywords_json") or []) if str(k).strip()],
                ]
                cluster_prompt_keywords = [kw for kw in cluster_prompt_keywords if kw]
                cluster_id = str(((idea_metadata.get("topic_keyword_research") or {}).get("keyword_cluster_id")) or "")
                keyword_metrics_map = cluster_keyword_metrics_maps.get(cluster_id, {})

                enriched_ideas.append(_apply_keyword_metrics_to_idea(
                    idea_copy,
                    keyword_metrics_map,
                    fallback_keywords=cluster_prompt_keywords,
                    fallback_metric_entries=cluster_context.get("keyword_candidates_json") or [],
                    exact_source="topic_keyword_cluster_metrics",
                    fallback_source="topic_keyword_cluster_fallback_metrics",
                ))
            return enriched_ideas

        blog_ideas, software_ideas = _rank_idea_groups(
            blog_ideas=_attach_cluster_keyword_metrics(blog_ideas),
            software_ideas=_attach_cluster_keyword_metrics(software_ideas),
            target_intent=effective_intent_bucket,
            tool_potential_score=effective_tool_potential_score,
            serp_intent_match=effective_serp_intent_match,
        )

        all_ideas = (blog_ideas or []) + (software_ideas or [])
        saved_count = 0
        persisted_idea_ids: list[str] = []
        if all_ideas:
            cluster_names = [str(cluster.get("cluster_name") or "").strip() for cluster in selected_clusters if str(cluster.get("cluster_name") or "").strip()]
            if cluster_names:
                try:
                    supabase_admin.table("content_ideas") \
                        .delete() \
                        .eq("user_id", user_id) \
                        .eq("topic_id", topic_id) \
                        .eq("status", "draft") \
                        .in_("subtopic", cluster_names) \
                        .execute()
                except Exception:
                    logger.warning("Skipping draft cleanup before cluster idea save", exc_info=True)

            persisted_rows = [
                _build_content_idea_persist_row(
                    idea={
                        **idea,
                        "subtopic": str(
                            (((idea.get("idea_metadata") or {}).get("topic_keyword_research") or {}).get("cluster_name")
                             or idea.get("subtopic")
                             or "Keyword Cluster")
                        ).strip(),
                    },
                    topic_id=topic_id,
                    user_id=user_id,
                    default_subtopic_name="Keyword Cluster",
                    idea_wp_context=idea_wp_context,
                    category_path=category_path,
                    category_context_project_id=topic_context.get("project_id"),
                    category_context_primary_category_id=topic_context.get("primary_category_id"),
                    category_context_secondary_category_id=topic_context.get("secondary_category_id"),
                    raw_dataforseo_output={
                        "topic_keyword_research_run_id": run_id,
                        "selected_cluster_ids": [cluster.get("id") for cluster in selected_clusters],
                    },
                )
                for idea in all_ideas
            ]
            for row in persisted_rows:
                if _insert_content_idea_with_schema_fallback(supabase_admin, row, log_label="Cluster idea"):
                    saved_count += 1
                    row_id = row.get("id")
                    if row_id:
                        persisted_idea_ids.append(str(row_id))
            try:
                supabase_admin.table("research_topics").update({
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", topic_id).eq("user_id", user_id).execute()
            except Exception:
                logger.warning("Could not update research topic timestamp after cluster idea generation", exc_info=True)

        return jsonify(_build_idea_generation_success_payload(
            blog_ideas=blog_ideas,
            software_ideas=software_ideas,
            persisted_count=saved_count,
            persisted_idea_ids=persisted_idea_ids,
            extra_fields={
                "selected_cluster_ids": [cluster.get("id") for cluster in selected_clusters],
            },
        )), 200
    except Exception as err:
        logger.error("Error generating ideas from topic keyword clusters: %s", err, exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message="Failed to generate ideas from keyword clusters",
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@research_topics_bp.route('/<topic_id>/editorial-ideas/generate', methods=['POST'])
@require_api_key
def generate_editorial_ideas_for_topic(topic_id):
    """Generate topic-level editorial ideas without requiring keyword clusters."""
    try:
        data = request.get_json() or {}
        supabase = get_supabase_client()
        request_user_id = _resolve_user_id_from_request(supabase, data)
        if not request_user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401

        user_id = data.get('user_id') or request_user_id
        if user_id != request_user_id:
            return jsonify(ErrorResponse(
                error="forbidden",
                message="You do not have access to this user_id",
                error_code="FORBIDDEN",
                status=403
            ).dict()), 403

        supabase_admin = _get_admin_supabase_client(supabase)
        topic_context_res = (
            supabase_admin
            .table('research_topics')
            .select(
                'id, title, description, project_id, primary_category_id, secondary_category_id, '
                'intent_bucket, decision_focus, angle_question, value_layer_tags, target_audience, '
                'related_terms, topic_mode, keyword_viability_score, keyword_viability_label, topic_generation_reasoning'
            )
            .eq('id', topic_id)
            .eq('user_id', user_id)
            .single()
            .execute()
        )
        topic_context = topic_context_res.data or {}
        if not topic_context:
            return jsonify(ErrorResponse(
                error="not_found",
                message="Research topic not found",
                error_code="NOT_FOUND",
                status=404
            ).dict()), 404

        category_path = "N/A"
        primary_name = None
        secondary_name = None
        category_ids = [
            category_id
            for category_id in [topic_context.get('primary_category_id'), topic_context.get('secondary_category_id')]
            if category_id
        ]
        if category_ids:
            try:
                category_response = (
                    supabase_admin
                    .table('project_categories')
                    .select('id, name, description')
                    .in_('id', category_ids)
                    .execute()
                )
                category_map = {item['id']: item for item in (category_response.data or [])}
                primary_name = (category_map.get(topic_context.get('primary_category_id')) or {}).get('name')
                secondary_name = (category_map.get(topic_context.get('secondary_category_id')) or {}).get('name')
                category_path = " / ".join([part for part in [primary_name, secondary_name] if part]) or category_path
            except Exception:
                logger.warning("Could not load category names for editorial topic idea context", exc_info=True)

        idea_wp_context = _resolve_idea_wordpress_category_context(
            supabase_admin,
            project_id=topic_context.get("project_id"),
            user_id=user_id,
            primary_category_id=topic_context.get("primary_category_id"),
            secondary_category_id=topic_context.get("secondary_category_id"),
        )

        topic_title = _safe_string(topic_context.get('title')) or 'Untitled Topic'
        topic_description = _safe_string(topic_context.get('description')) or ''
        related_terms = [
            _safe_string(term)
            for term in (topic_context.get('related_terms') or [])
            if _safe_string(term)
        ]
        topic_mode = _coerce_topic_mode(topic_context.get("topic_mode"))
        viability_score = _coerce_keyword_viability_score(topic_context.get("keyword_viability_score")) or 0
        viability_label = _coerce_keyword_viability_label(topic_context.get("keyword_viability_label"), viability_score)
        effective_intent_bucket = topic_context.get('intent_bucket') or 'informational_decision'
        effective_decision_focus = topic_context.get('decision_focus') or f"Help readers make a better decision about {topic_title}"
        effective_angle_question = topic_context.get('angle_question') or f"What practical question should this topic answer about {topic_title}?"
        effective_value_layer_tags = topic_context.get('value_layer_tags') or ['decision-support']
        effective_target_audience = topic_context.get('target_audience')

        tool_potential_score = 35
        if effective_intent_bucket == "solution_enablement" or "tool-builder" in effective_value_layer_tags:
            tool_potential_score = 72
        elif topic_mode == "keyword_first":
            tool_potential_score = 55

        compact_context_pack = (
            f"- Topic: {topic_title}\n"
            f"- Topic Mode: {topic_mode}\n"
            f"- Keyword Viability: {viability_label} ({int(viability_score) if viability_score else 0}/100)\n"
            f"- Category Path: {category_path}\n"
            f"- Decision Focus: {effective_decision_focus}\n"
            f"- Angle Question: {effective_angle_question}\n"
            f"- Target Audience: {effective_target_audience or 'General audience'}\n"
            f"- Value Tags: {', '.join([str(tag) for tag in effective_value_layer_tags if str(tag).strip()]) or 'decision-support'}\n"
            f"- Related Terms: {', '.join(related_terms[:8]) or 'None'}\n"
        )

        from supabase_client import LLM_ROLE_RESEARCH_IDEA_GENERATION
        from src.services.llm.llm_service import llm_service
        import asyncio

        async def generate_ideas():
            blog_prompt = f"""
You are a senior editorial strategist generating topic-level content ideas.

Current Year: 2026

Generate 5 BLOG ideas for this topic. These ideas do not need strong keyword demand to be valuable, but they should still be practical, concrete, and publishable.

Compact Context Pack:
{compact_context_pack}

Rules:
1. Keep titles practical and human.
2. Every idea must help with a different user decision, tradeoff, or recurring question.
3. Favor real reader utility over abstract thought pieces.
4. If the topic naturally supports search-shaped phrasing, use it. If not, keep the title editorial but concrete.
5. DESCRIPTION is required.
6. INPUT_KEYWORDS should be 3-5 short seed phrases we could later use for keyword expansion, even if the topic is mainly editorial.
7. Avoid near-duplicates and shallow paraphrases.

Output format:
BLOG_IDEA: [number]
TITLE: [title]
DESCRIPTION: [description]
SEARCH_PHRASE: [1-3 word query]
INPUT_KEYWORDS: [keyword1, keyword2, keyword3, keyword4]
INTENT: [informational/commercial/transactional]
FORMAT: [comparison/checklist/framework/case-study/how-to/calculator-guide]
USER_DECISION_HELPED: [decision]
INTERNAL_LINK_HOOK: [internal link strategy]
MONETIZATION: [monetization approach]
VIABILITY: [overall viability score 1-100]
END_IDEA
"""

            software_prompt = f"""
You are a product strategist generating companion software ideas only when the topic genuinely supports tool or workflow potential.

Current Year: 2026

Compact Context Pack:
{compact_context_pack}

Generate up to 3 SOFTWARE ideas.

Rules:
1. Only generate a software idea if it solves a repeated user task.
2. If the topic has weak tool potential, return 0-1 strong ideas instead of forcing generic software.
3. Keep tool names plain and practical.
4. DESCRIPTION is required whenever an idea is returned.
5. Each tool must solve a different repeated job.

Output format:
SOFTWARE_IDEA: [number]
TITLE: [tool name]
DESCRIPTION: [what the tool does and user interaction]
SEARCH_PHRASE: [1-3 word query]
INPUT_KEYWORDS: [keyword1, keyword2, keyword3, keyword4]
PRODUCT_TYPE: [calculator/planner/evaluator/comparison-tool/dashboard/workflow-helper]
USER_JOB: [job to be done]
KEY_INPUTS: [input1, input2, input3]
OUTPUT_RESULT: [result]
MONETIZATION: [how to monetize the tool]
BUILD_COMPLEXITY: [low/medium/high]
DISTRIBUTION_ANGLE: [distribution strategy]
VIABILITY: [overall viability score 1-100]
END_IDEA
"""

            blog_response = await llm_service.generate_text(
                blog_prompt,
                task_role=LLM_ROLE_RESEARCH_IDEA_GENERATION,
                max_tokens=1800,
            )
            software_response = await llm_service.generate_text(
                software_prompt,
                task_role=LLM_ROLE_RESEARCH_IDEA_GENERATION,
                max_tokens=1400,
            )
            return blog_response.content, software_response.content

        blog_text, software_text = asyncio.run(generate_ideas())
        blog_ideas = parse_idea_response(
            blog_text,
            'blog',
            topic_id,
            user_id,
            topic_title,
            primary_user_outcome=effective_decision_focus,
        )
        software_ideas = parse_idea_response(
            software_text,
            'software',
            topic_id,
            user_id,
            topic_title,
            primary_user_outcome=effective_decision_focus,
        )

        def _attach_editorial_generation_metadata(idea: dict) -> dict:
            enriched = dict(idea)
            metadata = dict(enriched.get("idea_metadata") or {})
            metadata["topic_editorial_generation"] = {
                "generation_origin": "topic_editorial_pipeline_v1",
                "topic_mode": topic_mode,
                "keyword_viability_score": viability_score,
                "keyword_viability_label": viability_label,
                "topic_generation_reasoning": topic_context.get("topic_generation_reasoning"),
                "related_terms": related_terms[:8],
            }
            enriched["idea_metadata"] = metadata
            return enriched

        blog_ideas, software_ideas = _rank_idea_groups(
            blog_ideas=[_attach_editorial_generation_metadata(idea) for idea in blog_ideas],
            software_ideas=[_attach_editorial_generation_metadata(idea) for idea in software_ideas],
            target_intent=effective_intent_bucket,
            tool_potential_score=tool_potential_score,
            serp_intent_match="medium",
        )

        try:
            supabase_admin.table("content_ideas") \
                .delete() \
                .eq("user_id", user_id) \
                .eq("topic_id", topic_id) \
                .eq("subtopic", topic_title) \
                .eq("status", "draft") \
                .execute()
        except Exception:
            logger.warning("Skipping draft cleanup before editorial topic idea save", exc_info=True)

        all_ideas = (blog_ideas or []) + (software_ideas or [])
        persisted_idea_ids: list[str] = []
        saved_count = 0
        for row in [
            _build_content_idea_persist_row(
                idea=idea,
                topic_id=topic_id,
                user_id=user_id,
                default_subtopic_name=topic_title,
                idea_wp_context=idea_wp_context,
                category_path=category_path,
                category_context_project_id=topic_context.get("project_id"),
                category_context_primary_category_id=topic_context.get("primary_category_id"),
                category_context_secondary_category_id=topic_context.get("secondary_category_id"),
            )
            for idea in all_ideas
        ]:
            if _insert_content_idea_with_schema_fallback(supabase_admin, row, log_label="Editorial topic ideas"):
                saved_count += 1
                if row.get("id"):
                    persisted_idea_ids.append(str(row.get("id")))

        try:
            supabase_admin.table("research_topics").update({
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", topic_id).eq("user_id", user_id).execute()
        except Exception:
            logger.warning("Could not update research topic timestamp after editorial idea generation", exc_info=True)

        return jsonify(_build_idea_generation_success_payload(
            blog_ideas=blog_ideas,
            software_ideas=software_ideas,
            persisted_count=saved_count,
            persisted_idea_ids=persisted_idea_ids,
        )), 200

    except Exception as e:
        logger.error("Error generating editorial ideas for topic: %s", e, exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@research_topics_bp.route('/idea-burst', methods=['POST'])
@require_api_key
def idea_burst():
    """
    Generate content ideas for a specific subtopic.
    Takes a subtopic with its keywords and generates blog and software/commercial content ideas.
    """
    try:
        if not request.is_json:
            return jsonify(ErrorResponse(
                error="invalid_content_type",
                message="Content-Type must be application/json",
                error_code="INVALID_CONTENT_TYPE",
                status=400
            ).dict()), 400

        data = request.get_json()
        supabase = get_supabase_client()
        request_user_id = _resolve_user_id_from_request(supabase, data)
        if not request_user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401

        user_id = data.get('user_id') or request_user_id
        topic_id = data.get('topic_id')
        subtopic_name = data.get('subtopic')
        raw_keywords = data.get('keywords', [])
        affiliate_offers = data.get('affiliate_offers', [])
        context_intent_bucket = data.get('intent_bucket')
        context_decision_focus = data.get('decision_focus')
        context_angle_question = data.get('angle_question')
        context_value_layer_tags = data.get('value_layer_tags') or []
        context_cluster_type = data.get('cluster_type')
        context_primary_user_outcome = data.get('primary_user_outcome')
        context_serp_intent_match = data.get('serp_intent_match')
        context_tool_potential_score = data.get('tool_potential_score')

        if user_id != request_user_id:
            return jsonify(ErrorResponse(
                error="forbidden",
                message="You do not have access to this user_id",
                error_code="FORBIDDEN",
                status=403
            ).dict()), 403

        if not all([user_id, topic_id, subtopic_name]):
            return jsonify(ErrorResponse(
                error="validation_error",
                message="user_id, topic_id, and subtopic are required",
                error_code="VALIDATION_ERROR",
                status=400
            ).dict()), 400

        topic_owner = supabase.table('research_topics').select('user_id').eq('id', topic_id).single().execute()
        if not topic_owner.data or topic_owner.data.get('user_id') != request_user_id:
            return jsonify(ErrorResponse(
                error="forbidden",
                message="You do not have access to this research topic",
                error_code="FORBIDDEN",
                status=403
            ).dict()), 403

        supabase_admin = _get_admin_supabase_client(supabase)

        topic_context_res = (
            supabase_admin
            .table('research_topics')
            .select(
                'title, description, project_id, primary_category_id, secondary_category_id, '
                'intent_bucket, decision_focus, angle_question, value_layer_tags, target_audience'
            )
            .eq('id', topic_id)
            .single()
            .execute()
        )
        topic_context = topic_context_res.data or {}
        idea_wp_context = _resolve_idea_wordpress_category_context(
            supabase_admin,
            project_id=topic_context.get("project_id"),
            user_id=user_id,
            primary_category_id=topic_context.get("primary_category_id"),
            secondary_category_id=topic_context.get("secondary_category_id"),
        )

        category_path = "N/A"
        category_ids = [
            category_id
            for category_id in [topic_context.get('primary_category_id'), topic_context.get('secondary_category_id')]
            if category_id
        ]
        if category_ids:
            try:
                category_response = (
                    supabase_admin
                    .table('project_categories')
                    .select('id, name')
                    .in_('id', category_ids)
                    .execute()
                )
                category_map = {item['id']: item.get('name') for item in (category_response.data or [])}
                primary_name = category_map.get(topic_context.get('primary_category_id'))
                secondary_name = category_map.get(topic_context.get('secondary_category_id'))
                category_path = " / ".join([part for part in [primary_name, secondary_name] if part]) or category_path
            except Exception:
                logger.warning("Could not load category names for Idea Burst context", exc_info=True)

        effective_intent_bucket = context_intent_bucket or topic_context.get('intent_bucket') or "informational_decision"
        effective_decision_focus = context_decision_focus or topic_context.get('decision_focus') or f"Help users make a decision about {subtopic_name}"
        effective_angle_question = context_angle_question or topic_context.get('angle_question') or f"What is the best way to approach {subtopic_name}?"
        effective_value_layer_tags = context_value_layer_tags or topic_context.get('value_layer_tags') or ["decision-support"]
        effective_cluster_type = context_cluster_type or "decision"
        effective_primary_user_outcome = context_primary_user_outcome or f"Choose a practical next action for {subtopic_name}"
        effective_serp_intent_match = context_serp_intent_match or "medium"
        try:
            effective_tool_potential_score = int(context_tool_potential_score) if context_tool_potential_score is not None else 50
        except Exception:
            effective_tool_potential_score = 50
        subtopic_keyword_metrics_map = _build_keyword_metrics_map(raw_keywords)
        prompt_keywords = [
            str(item.get("keyword") or "").strip()
            for item in _coerce_keyword_metric_entries(raw_keywords)
            if str(item.get("keyword") or "").strip()
        ]
        if not prompt_keywords and isinstance(raw_keywords, list):
            prompt_keywords = [str(item).strip() for item in raw_keywords if str(item).strip()]

        def _clip_prompt_text(value: str, limit: int = 180) -> str:
            text = " ".join(str(value or "").split())
            if len(text) <= limit:
                return text
            return f"{text[:limit - 3].rstrip()}..."

        def _build_compact_context_pack() -> str:
            value_tags = ", ".join(
                [str(tag).strip() for tag in (effective_value_layer_tags or []) if str(tag).strip()][:6]
            ) or "N/A"

            return (
                f"- Topic: {_clip_prompt_text(topic_context.get('title') or 'N/A', 120)}\n"
                f"- Category Path: {_clip_prompt_text(category_path or 'N/A', 120)}\n"
                f"- Subtopic: {_clip_prompt_text(subtopic_name, 120)}\n"
                f"- Intent Bucket: {_clip_prompt_text(effective_intent_bucket, 80)}\n"
                f"- Decision Focus: {_clip_prompt_text(effective_decision_focus, 220)}\n"
                f"- Primary Outcome: {_clip_prompt_text(effective_primary_user_outcome, 220)}\n"
                f"- Angle Question: {_clip_prompt_text(effective_angle_question, 220)}\n"
                f"- Cluster Type: {_clip_prompt_text(effective_cluster_type, 80)}\n"
                f"- Value Tags: {_clip_prompt_text(value_tags, 140)}\n"
            )

        compact_context_pack = _build_compact_context_pack()

        logger.info(f"Generating idea burst for subtopic: {subtopic_name}")

        # Generate ideas using LLM
        import asyncio
        from supabase_client import LLM_ROLE_RESEARCH_IDEA_GENERATION
        from src.services.llm.llm_service import llm_service

        async def generate_ideas():
            # Generate blog ideas
            blog_prompt = f"""
You are a veteran SEO content strategist. Generate article ideas in plain, human language that sounds like real Google searches, not consultant-speak.
Also act as a Search Intent Specialist: reverse complex concepts into short query terms users actually type.

Current Year: 2026
Topic Description: {_clip_prompt_text(topic_context.get('description') or 'N/A', 220)}
Affiliate Categories: {', '.join(affiliate_offers[:5]) if affiliate_offers else 'General'}
SERP Intent Match: {effective_serp_intent_match}
Tool Potential Score: {effective_tool_potential_score}/100

Compact Context Pack:
{compact_context_pack}

Generate 5 BLOG article ideas.

Hard constraints:
1. Use simple, practical language. Pass the "2 AM test": a stressed user should plausibly type these words.
2. Avoid consultant/corporate jargon in titles and search phrases (framework, paradigm, architecture, lens, methodology, optimization strategy).
3. Each idea MUST include SEARCH_PHRASE (1-3 words, lowercase) that looks like a real query and can be sent to keyword tools.
4. TITLE must include SEARCH_PHRASE verbatim.
5. Keep SEARCH_PHRASE and INPUT_KEYWORDS tightly aligned to subtopic + decision focus + primary outcome.
6. Avoid topic drift and generic listicles.
7. DESCRIPTION is required and cannot be empty.
8. Prioritize decision/action intent over abstract commentary.
9. INPUT_KEYWORDS must be short, human, and literal search language for DataForSEO related keyword mining.
10. INPUT_KEYWORDS must be 3-5 items, each 1-3 words max, no punctuation-heavy phrases, no jargon, and avoid connectors like "or", "vs", "and".
11. Every idea must target a meaningfully different user question or decision. Do not produce paraphrases of the same idea.
12. If two ideas would lead to mostly the same outline, keep only the stronger one.
13. Avoid near-duplicate variations such as cost vs pricing, best vs top, compare vs choose, checklist vs guide when the underlying topic is the same.
14. Each idea must have a distinct USER_DECISION_HELPED and a different core SEARCH_PHRASE stem.

For each idea, provide:
- Title: SEO-conscious title in plain language
- Description: 1-2 sentence angle summary
- Search Phrase: 1-3 words, plain language
- Input Keywords: 3-5 simple query-like seed phrases for keyword mining
- Intent: informational/commercial/transactional
- Format: comparison/checklist/framework/case-study/how-to/calculator-guide
- User Decision Helped: What choice this article helps make
- Internal Link Hook: Where this links in the topical graph
- Monetization Hook: How to monetize
- Viability: overall viability score 1-100

Output format (use exactly this format):
BLOG_IDEA: [number]
TITLE: [title]
DESCRIPTION: [description]
SEARCH_PHRASE: [1-3 word query]
INPUT_KEYWORDS: [keyword1, keyword2, keyword3, keyword4]
INTENT: [informational/commercial/transactional]
FORMAT: [comparison/checklist/framework/case-study/how-to/calculator-guide]
USER_DECISION_HELPED: [decision]
INTERNAL_LINK_HOOK: [internal link strategy]
MONETIZATION: [monetization approach]
VIABILITY: [overall viability score 1-100]
END_IDEA

Generate 5 blog ideas following this format.
"""

            # Generate software/commercial ideas
            software_prompt = f"""
You are a product strategist generating software tools users can discover through search. Use plain language and practical naming.
Also act as a Search Intent Specialist: reverse complex concepts into short query terms users actually type.

Current Year: 2026
Topic: {topic_context.get('title') or 'N/A'}
Affiliate Categories: {', '.join(affiliate_offers[:5]) if affiliate_offers else 'General'}
SERP Intent Match: {effective_serp_intent_match}
Tool Potential Score: {effective_tool_potential_score}/100

Compact Context Pack:
{compact_context_pack}

Generate 3 ACTUAL SOFTWARE TOOLS or FEATURES to BUILD for a website/app.

IMPORTANT: These are NOT articles. These are software products/features the user should develop.

Examples of what to generate:
- Interactive calculators (tax calculator, ROI calculator, comparison tool)
- Assessment tools (quiz, diagnostic, evaluator)
- Data visualization tools (chart builder, portfolio tracker, dashboard)
- Automation tools (planner, scheduler, optimizer)
- Utilities (converter, analyzer, generator)

Examples of what NOT to generate:
- "Best Software 2026" (this is a review article)
- "Top 5 Tools" (this is a listicle article)
- "Software Comparison" (this is a comparison article)

For each tool idea, provide:
- Title: Name of the tool/feature to build and include SEARCH_PHRASE verbatim
- Description: What the tool does and how users interact with it
- Search Phrase: 1-3 words users would type to find this tool
- Input Keywords: 3-5 simple query-like seed phrases for DataForSEO related keyword mining (each 1-3 words, no connector phrases like "x vs y")
- Product Type: calculator/planner/evaluator/comparison-tool/dashboard/workflow-helper
- User Job To Be Done: the repeated decision/action this solves
- Key Inputs: data users provide
- Output Result: what users get back
- Monetization Hook: How to monetize the tool (lead gen, freemium, affiliate integration, etc.)
- Build Complexity: low/medium/high
- Distribution Angle: how this gets discovered (SEO/interactive tool pages/etc.)
- Viability: overall viability score 1-100

Output format (use exactly this format):
SOFTWARE_IDEA: [number]
TITLE: [tool name - NOT a review article title]
DESCRIPTION: [what the tool does and user interaction]
SEARCH_PHRASE: [1-3 word query]
INPUT_KEYWORDS: [keyword1, keyword2, keyword3, keyword4]
PRODUCT_TYPE: [calculator/planner/evaluator/comparison-tool/dashboard/workflow-helper]
USER_JOB: [job to be done]
KEY_INPUTS: [input1, input2, input3]
OUTPUT_RESULT: [result]
MONETIZATION: [how to monetize the tool]
BUILD_COMPLEXITY: [low/medium/high]
DISTRIBUTION_ANGLE: [distribution strategy]
VIABILITY: [overall viability score 1-100]
END_IDEA

Generate 3 software tools/features to BUILD following this format.

Critical naming and language rules:
- Tool names must be plain-English and practical (what users would actually search)
- Apply the "2 AM test": if a stressed user would not search it, rewrite it
- Avoid consultant-speak and brochure language (no "framework", "paradigm", "value architecture", "strategic lens", "methodology")
- If a technical term is required, pair it with a simple phrase users understand
- DESCRIPTION is required and cannot be empty
- Every tool idea must solve a different repeated job. Do not rename the same tool concept three different ways.
- If two tool ideas would share nearly the same inputs, outputs, and user job, keep only the stronger one.
- Each tool must use a different SEARCH_PHRASE stem and a different USER_JOB.
"""

            # Generate both in parallel
            blog_response = await llm_service.generate_text(
                blog_prompt,
                task_role=LLM_ROLE_RESEARCH_IDEA_GENERATION,
                max_tokens=2000,
            )
            software_response = await llm_service.generate_text(
                software_prompt,
                task_role=LLM_ROLE_RESEARCH_IDEA_GENERATION,
                max_tokens=1500,
            )
            logger.info(
                "Idea burst LLM responses subtopic=%r blog_provider=%s blog_model=%s software_provider=%s software_model=%s blog_chars=%s software_chars=%s",
                subtopic_name,
                blog_response.provider,
                blog_response.model_name,
                software_response.provider,
                software_response.model_name,
                len(blog_response.content or ""),
                len(software_response.content or ""),
            )

            return blog_response.content, software_response.content

        blog_text, software_text = asyncio.run(generate_ideas())

        # Parse the responses
        blog_ideas = parse_idea_response(
            blog_text,
            'blog',
            topic_id,
            user_id,
            subtopic_name,
            primary_user_outcome=effective_primary_user_outcome,
        )
        software_ideas = parse_idea_response(
            software_text,
            'software',
            topic_id,
            user_id,
            subtopic_name,
            primary_user_outcome=effective_primary_user_outcome,
        )
        blog_ideas, software_ideas = _rank_idea_groups(
            blog_ideas=blog_ideas,
            software_ideas=software_ideas,
            target_intent=effective_intent_bucket,
            tool_potential_score=effective_tool_potential_score,
            serp_intent_match=effective_serp_intent_match,
        )

        def _attach_keyword_metrics(ideas: list[dict]) -> list[dict]:
            enriched_ideas = []
            for idea in ideas or []:
                enriched_ideas.append(_apply_keyword_metrics_to_idea(
                    dict(idea),
                    subtopic_keyword_metrics_map,
                    fallback_keywords=prompt_keywords,
                    exact_source="subtopic_keyword_metrics",
                ))
            return enriched_ideas

        blog_ideas, software_ideas = _rank_idea_groups(
            blog_ideas=_attach_keyword_metrics(blog_ideas),
            software_ideas=_attach_keyword_metrics(software_ideas),
            target_intent=effective_intent_bucket,
            tool_potential_score=effective_tool_potential_score,
            serp_intent_match=effective_serp_intent_match,
        )

        # Persist burst ideas so they appear in Content Library and Software Ideas screens.
        all_ideas = (blog_ideas or []) + (software_ideas or [])
        saved_count = 0
        persisted_idea_ids: list[str] = []
        if all_ideas:
            # Keep published ideas, replace only draft rows for this subtopic/topic/user.
            try:
                supabase_admin.table("content_ideas") \
                    .delete() \
                    .eq("user_id", user_id) \
                    .eq("topic_id", topic_id) \
                    .eq("subtopic", subtopic_name) \
                    .eq("status", "draft") \
                    .execute()
            except Exception:
                # Some schemas may not include status; skip cleanup in that case.
                logger.warning("Skipping draft cleanup before idea burst save", exc_info=True)

            persisted_rows = [
                _build_content_idea_persist_row(
                    idea=idea,
                    topic_id=topic_id,
                    user_id=user_id,
                    default_subtopic_name=subtopic_name,
                    idea_wp_context=idea_wp_context,
                    category_path=category_path,
                    category_context_project_id=topic_context.get("project_id"),
                    category_context_primary_category_id=topic_context.get("primary_category_id"),
                    category_context_secondary_category_id=topic_context.get("secondary_category_id"),
                )
                for idea in all_ideas
            ]
            for row in persisted_rows:
                if _insert_content_idea_with_schema_fallback(supabase_admin, row, log_label="Idea burst"):
                    saved_count += 1
                    row_id = row.get("id")
                    if row_id:
                        persisted_idea_ids.append(str(row_id))
            logger.info(
                "Idea burst persistence summary topic_id=%s subtopic=%s attempted=%s saved=%s",
                topic_id,
                subtopic_name,
                len(persisted_rows),
                saved_count,
            )

            # Touch the parent topic to mark progress recency.
            try:
                supabase_admin.table("research_topics").update({
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", topic_id).eq("user_id", user_id).execute()
            except Exception:
                logger.warning("Could not update research topic timestamp after idea burst", exc_info=True)

        return jsonify(_build_idea_generation_success_payload(
            blog_ideas=blog_ideas,
            software_ideas=software_ideas,
            persisted_count=saved_count,
            persisted_idea_ids=persisted_idea_ids,
        )), 200

    except Exception as e:
        logger.error(f"Error in idea burst: {e}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


def parse_idea_response(
    text: str,
    content_type: str,
    topic_id: str,
    user_id: str,
    subtopic_name: str,
    primary_user_outcome=None,
):
    """Parse LLM response into ContentIdea objects."""
    import re
    from uuid import uuid4

    IDEA_TOKEN_SYNONYMS = {
        "affordable": "price",
        "budget": "price",
        "cost": "price",
        "costs": "price",
        "pricing": "price",
        "compare": "comparison",
        "comparing": "comparison",
        "comparison": "comparison",
        "choose": "decision",
        "choosing": "decision",
        "decision": "decision",
        "evaluate": "decision",
        "evaluation": "decision",
        "picker": "tool",
        "planner": "tool",
        "tool": "tool",
        "tools": "tool",
        "tracker": "tool",
        "workflow": "tool",
    }

    IDEA_GENERIC_TOKENS = {
        "a", "an", "and", "article", "articles", "best", "better", "blog", "blogs",
        "build", "content", "decision", "decisions", "for", "guide", "guides", "help",
        "how", "idea", "ideas", "in", "of", "or", "plan", "plans", "software", "solution",
        "solutions", "the", "to", "tool", "tools", "using", "what", "with", "your",
    }
    IDEA_MODIFIER_TOKENS = {
        "best", "top", "vs", "versus", "checklist", "guide", "comparison", "compare",
        "cost", "price", "pricing", "framework", "playbook", "audit",
    }

    def _normalize_search_phrase(raw_phrase: str) -> str:
        return _normalize_search_phrase_text(raw_phrase)

    def _normalize_idea_title(raw_title: str) -> str:
        return _normalize_idea_title_text(raw_title)

    def _normalize_idea_token(token: str) -> str:
        token = re.sub(r"[^a-z0-9]", "", token.lower())
        if len(token) <= 2:
            return ""
        if token.endswith("ies") and len(token) > 4:
            token = f"{token[:-3]}y"
        elif token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("es") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        token = IDEA_TOKEN_SYNONYMS.get(token, token)
        if token in IDEA_GENERIC_TOKENS:
            return ""
        return token

    def _idea_concept_tokens(idea: dict) -> set[str]:
        text = " ".join(
            [
                str(idea.get("title") or ""),
                str(idea.get("search_phrase") or ""),
                str(idea.get("user_decision_helped") or ""),
                str(idea.get("user_job_to_be_done") or ""),
                str(idea.get("description") or ""),
            ]
        )
        tokens = set()
        for raw in re.findall(r"[a-zA-Z0-9]+", text.lower()):
            token = _normalize_idea_token(raw)
            if token:
                tokens.add(token)
        return tokens

    def _idea_core_tokens(idea: dict) -> set[str]:
        """Core concept tokens with weak modifiers removed for paraphrase detection."""
        return {token for token in _idea_concept_tokens(idea) if token not in IDEA_MODIFIER_TOKENS}

    def _is_near_duplicate_idea(candidate: dict, existing: dict) -> bool:
        candidate_title = re.sub(r"\s+", " ", str(candidate.get("title") or "").strip().lower())
        existing_title = re.sub(r"\s+", " ", str(existing.get("title") or "").strip().lower())
        if candidate_title and candidate_title == existing_title:
            return True

        candidate_phrase = re.sub(r"\s+", " ", str(candidate.get("search_phrase") or "").strip().lower())
        existing_phrase = re.sub(r"\s+", " ", str(existing.get("search_phrase") or "").strip().lower())
        if candidate_phrase and existing_phrase and candidate_phrase == existing_phrase:
            return True

        candidate_tokens = _idea_concept_tokens(candidate)
        existing_tokens = _idea_concept_tokens(existing)
        candidate_core_tokens = _idea_core_tokens(candidate)
        existing_core_tokens = _idea_core_tokens(existing)
        if not candidate_tokens or not existing_tokens:
            return False

        overlap = len(candidate_tokens & existing_tokens)
        coverage = overlap / max(1, min(len(candidate_tokens), len(existing_tokens)))
        jaccard = overlap / max(1, len(candidate_tokens | existing_tokens))
        core_overlap = len(candidate_core_tokens & existing_core_tokens)
        core_coverage = core_overlap / max(1, min(len(candidate_core_tokens), len(existing_core_tokens)))
        core_jaccard = core_overlap / max(1, len(candidate_core_tokens | existing_core_tokens))

        same_format = str(candidate.get("article_format") or candidate.get("product_type") or "").lower() == str(
            existing.get("article_format") or existing.get("product_type") or ""
        ).lower()
        same_intent = str(candidate.get("target_intent") or "").strip().lower() == str(
            existing.get("target_intent") or ""
        ).strip().lower()

        return (
            coverage >= 0.8
            or jaccard >= 0.65
            or core_coverage >= 0.85
            or core_jaccard >= 0.7
            or (same_intent and core_overlap >= 3 and core_coverage >= 0.6)
            or (same_format and overlap >= 3 and coverage >= 0.5)
        )

    def _dedupe_ideas(ideas: list[dict]) -> list[dict]:
        distinct: list[dict] = []
        for idea in ideas:
            if any(_is_near_duplicate_idea(idea, existing) for existing in distinct):
                logger.info(
                    "Dropping near-duplicate idea type=%s title=%r search_phrase=%r",
                    content_type,
                    idea.get("title"),
                    idea.get("search_phrase"),
                )
                continue
            distinct.append(idea)
        return distinct

    ideas = []
    current_idea = {}

    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for idea start
        if re.match(r'^(BLOG_IDEA|SOFTWARE_IDEA):', line, re.IGNORECASE):
            if current_idea and 'title' in current_idea:
                ideas.append(
                    create_idea_dict(
                        current_idea,
                        content_type,
                        topic_id,
                        user_id,
                        subtopic_name,
                        primary_user_outcome=primary_user_outcome,
                    )
                )
            current_idea = {'id': str(uuid4())}

        # Parse fields
        elif line.upper().startswith('TITLE:'):
            raw_title = line.split(':', 1)[1].strip()
            current_idea['title'] = _normalize_idea_title(raw_title)
        elif line.upper().startswith('DESCRIPTION:'):
            current_idea['description'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('SEARCH_PHRASE:'):
            current_idea['search_phrase'] = _normalize_search_phrase(line.split(':', 1)[1].strip())
        elif line.upper().startswith('KEYWORDS:'):
            kw_text = line.split(':', 1)[1].strip()
            current_idea['keywords'] = [k.strip() for k in kw_text.split(',') if k.strip()]
        elif line.upper().startswith('INPUT_KEYWORDS:'):
            kw_text = line.split(':', 1)[1].strip()
            current_idea['input_keywords'] = [k.strip() for k in kw_text.split(',') if k.strip()]
        elif line.upper().startswith('MONETIZATION:'):
            current_idea['monetization_hook'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('INTENT:'):
            current_idea['target_intent'] = line.split(':', 1)[1].strip().lower()
        elif line.upper().startswith('FORMAT:'):
            current_idea['article_format'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('USER_DECISION_HELPED:'):
            current_idea['user_decision_helped'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('INTERNAL_LINK_HOOK:'):
            current_idea['internal_link_hook'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('PRODUCT_TYPE:'):
            current_idea['product_type'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('USER_JOB:'):
            current_idea['user_job_to_be_done'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('KEY_INPUTS:'):
            inputs_text = line.split(':', 1)[1].strip()
            current_idea['key_inputs'] = [item.strip() for item in inputs_text.split(',') if item.strip()]
        elif line.upper().startswith('OUTPUT_RESULT:'):
            current_idea['output_result'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('BUILD_COMPLEXITY:'):
            current_idea['build_complexity'] = line.split(':', 1)[1].strip().lower()
        elif line.upper().startswith('DISTRIBUTION_ANGLE:'):
            current_idea['distribution_angle'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('VOLUME:'):
            try:
                vol_text = line.split(':', 1)[1].strip().replace(',', '')
                # Extract number from text
                vol_match = re.search(r'(\d+)', vol_text)
                current_idea['total_search_volume'] = int(vol_match.group(1)) if vol_match else 0
            except:
                current_idea['total_search_volume'] = 0
        elif line.upper().startswith('DIFFICULTY:'):
            try:
                diff_text = line.split(':', 1)[1].strip()
                diff_match = re.search(r'(\d+)', diff_text)
                current_idea['average_difficulty'] = int(diff_match.group(1)) if diff_match else 50
            except:
                current_idea['average_difficulty'] = 50
        elif line.upper().startswith('VIABILITY:'):
            try:
                via_text = line.split(':', 1)[1].strip()
                via_match = re.search(r'(\d+)', via_text)
                current_idea['viability_score'] = int(via_match.group(1)) if via_match else 50
            except:
                current_idea['viability_score'] = 50
        elif re.match(r'^END_IDEA', line, re.IGNORECASE):
            if current_idea and 'title' in current_idea:
                ideas.append(
                    create_idea_dict(
                        current_idea,
                        content_type,
                        topic_id,
                        user_id,
                        subtopic_name,
                        primary_user_outcome=primary_user_outcome,
                    )
                )
                current_idea = {}

    # Don't forget the last idea
    if current_idea and 'title' in current_idea:
        ideas.append(
            create_idea_dict(
                current_idea,
                content_type,
                topic_id,
                user_id,
                subtopic_name,
                primary_user_outcome=primary_user_outcome,
            )
        )

    deduped_ideas = _dedupe_ideas(ideas)
    logger.info(
        "Parsed idea response type=%s raw_count=%s deduped_count=%s",
        content_type,
        len(ideas),
        len(deduped_ideas),
    )
    return deduped_ideas


def create_idea_dict(
    idea_data: dict,
    content_type: str,
    topic_id: str,
    user_id: str,
    subtopic_name: str,
    primary_user_outcome=None,
) -> dict:
    """Create a standardized idea dictionary."""
    from datetime import datetime
    from uuid import uuid4
    import re

    def _normalize_search_phrase(raw_phrase: str) -> str:
        return _normalize_search_phrase_text(raw_phrase)

    def _derive_search_phrase(raw_title: str, raw_keywords: list[str]) -> str:
        for kw in raw_keywords:
            normalized_kw = _normalize_search_phrase(kw)
            if normalized_kw:
                return normalized_kw
        title_tokens = re.sub(r"[^a-zA-Z0-9\s\-]", " ", raw_title.lower()).split()
        if len(title_tokens) >= 2:
            return " ".join(title_tokens[: min(3, len(title_tokens))])
        if len(title_tokens) == 1 and len(title_tokens[0]) >= 3:
            return title_tokens[0]
        return ""

    def _title_contains_phrase(raw_title: str, phrase: str) -> bool:
        title_norm = re.sub(r"\s+", " ", str(raw_title or "").lower()).strip()
        phrase_norm = re.sub(r"\s+", " ", str(phrase or "").lower()).strip()
        return bool(title_norm and phrase_norm and phrase_norm in title_norm)

    def _normalize_outcome_text(raw_outcome: str) -> str:
        text = re.sub(r"\s+", " ", str(raw_outcome or "")).strip(" ,.-")
        if not text:
            return ""
        if len(text) > 120:
            text = f"{text[:117].rstrip()}..."
        return text

    SIMPLE_STOPWORDS = {
        "a", "an", "the", "to", "for", "of", "in", "on", "at", "with", "without",
        "from", "into", "by", "my", "your", "our", "their", "you", "is", "are", "be", "have", "has", "had",
        "too", "much", "more", "less", "first", "second", "third", "best", "better",
        "what", "how", "when", "why", "can", "should", "could", "would", "do",
        "does", "did", "or", "and", "vs", "versus",
    }
    JARGON_TOKENS = {
        "framework", "paradigm", "architecture", "methodology", "optimization",
        "strategic", "strategy", "lens", "playbook",
    }

    def _simplify_seed_phrase(raw_phrase: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9\s-]", " ", str(raw_phrase or "").lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
        if not cleaned:
            return ""

        # Prefer atomic user queries over comparison connectors.
        parts = re.split(r"\b(?:or|vs|versus|and)\b", cleaned)
        normalized_parts = []
        for part in parts:
            part = re.sub(r"\s+", " ", part).strip()
            if not part:
                continue
            tokens = [t for t in part.split(" ") if t and t not in SIMPLE_STOPWORDS and t not in JARGON_TOKENS]
            if len(tokens) < 2:
                if tokens and len(tokens[0]) >= 4:
                    normalized_parts.append(tokens[0])
                continue
            if len(tokens) > 3:
                tokens = tokens[:3]
            normalized_parts.append(" ".join(tokens))

        if normalized_parts:
            return normalized_parts[0]

        tokens = [t for t in cleaned.split(" ") if t and t not in SIMPLE_STOPWORDS and t not in JARGON_TOKENS]
        if len(tokens) < 2:
            return tokens[0] if tokens and len(tokens[0]) >= 4 else ""
        if len(tokens) > 3:
            tokens = tokens[:3]
        return " ".join(tokens)

    def _build_simple_keyword_seeds(title_value: str, subtopic_value: str, seeds: list[str]) -> list[str]:
        def _context_tokens(raw_text: str) -> set[str]:
            cleaned = re.sub(r"[^a-zA-Z0-9\s-]", " ", str(raw_text or "").lower())
            tokens = [t for t in re.sub(r"\s+", " ", cleaned).split(" ") if t]
            return {
                t for t in tokens
                if len(t) >= 3 and t not in SIMPLE_STOPWORDS and t not in JARGON_TOKENS
            }

        raw_candidates = list(seeds or [])
        raw_candidates.extend([title_value, subtopic_value])
        title_context = _context_tokens(title_value)
        subtopic_context = _context_tokens(subtopic_value)
        relevance_context = title_context | subtopic_context

        expanded = []
        for raw in raw_candidates:
            phrase = str(raw or "").strip()
            if not phrase:
                continue
            simple = _simplify_seed_phrase(phrase)
            if simple:
                expanded.append(simple)

            # Add short n-gram alternatives to increase DataForSEO hit-rate.
            tokens = [t for t in simple.split(" ") if t] if simple else []
            if len(tokens) >= 3:
                expanded.append(" ".join(tokens[:2]))
                expanded.append(" ".join(tokens[-2:]))

        seen = set()
        scored = []
        for item in expanded:
            normalized = re.sub(r"\s+", " ", item.lower()).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            token_count = len(normalized.split(" "))
            # Rank simple query-like terms first (2-3 words ideal).
            score = 0
            if token_count in (2, 3):
                score += 3
            elif token_count == 4:
                score += 1
            if any(conn in normalized for conn in (" or ", " vs ", " versus ", " and ")):
                score -= 3
            candidate_tokens = {
                t for t in normalized.split(" ")
                if t and t not in SIMPLE_STOPWORDS and t not in JARGON_TOKENS
            }
            overlap = len(candidate_tokens & relevance_context)
            if overlap > 0:
                score += overlap * 4
            else:
                # Strongly discourage seeds that don't match title/subtopic semantics.
                score -= 8
            scored.append((score, overlap, normalized))

        # Prefer semantically aligned terms when available.
        overlap_scored = [row for row in scored if row[1] > 0]
        target_rows = overlap_scored if overlap_scored else scored
        target_rows.sort(key=lambda x: (-x[0], -x[1], len(x[2])))
        selected = [item for _, _, item in target_rows[:5]]
        if selected:
            return selected

        # Last-resort fallback: derive simple n-grams from title/subtopic text.
        fallback_tokens = [t for t in (title_value or "").lower().split() if len(t) >= 3][:4]
        if len(fallback_tokens) >= 2:
            return [" ".join(fallback_tokens[:2]), " ".join(fallback_tokens[-2:])]
        return [str(subtopic_value or "").strip().lower()] if str(subtopic_value or "").strip() else []

    input_keywords = idea_data.get('input_keywords', [])
    if not isinstance(input_keywords, list):
        input_keywords = []
    input_keywords = [str(k).strip() for k in input_keywords if str(k).strip()]

    keywords = idea_data.get('keywords', [])
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(k).strip() for k in keywords if str(k).strip()]
    if not keywords:
        keywords = list(input_keywords)

    title = (idea_data.get('title') or 'Untitled Idea').strip()
    simple_seed_keywords = _build_simple_keyword_seeds(title, subtopic_name, input_keywords or keywords)
    if simple_seed_keywords:
        input_keywords = simple_seed_keywords
        keywords = simple_seed_keywords

    search_phrase = _normalize_search_phrase(idea_data.get('search_phrase', ''))
    if not search_phrase:
        search_phrase = _derive_search_phrase(title, input_keywords or keywords)
    simplified_search_phrase = _simplify_seed_phrase(search_phrase) or search_phrase
    if simplified_search_phrase:
        search_phrase = simplified_search_phrase

    # Keep first-pass titles natural; do not force prefixing with search phrase.

    if search_phrase:
        existing_norm = {re.sub(r"\s+", " ", k.lower()).strip() for k in keywords}
        if search_phrase not in existing_norm:
            keywords.insert(0, search_phrase)
    if input_keywords:
        seed_norm = {re.sub(r"\s+", " ", k.lower()).strip() for k in keywords}
        for seed in input_keywords:
            norm = re.sub(r"\s+", " ", seed.lower()).strip()
            if norm and norm not in seed_norm:
                keywords.append(seed)
                seed_norm.add(norm)
    keywords = [str(k).strip().lower() for k in keywords if str(k).strip()][:6]
    input_keywords = [str(k).strip().lower() for k in input_keywords if str(k).strip()][:5]

    # Do not force keyword prefixes in title; keep meaning and readability first.

    description = str(idea_data.get('description') or '').strip()
    preferred_outcome = _normalize_outcome_text(
        idea_data.get('user_decision_helped')
        or idea_data.get('user_job_to_be_done')
        or primary_user_outcome
    )
    if not description:
        if preferred_outcome:
            description = f"Outcome: {preferred_outcome}."
        elif keywords:
            description = (
                f"This article explains {keywords[0]} and helps the reader make a practical decision."
            )
        else:
            description = (
                f"This article gives a practical breakdown of {title.lower()} and what action the reader should take next."
            )
    elif preferred_outcome:
        description_norm = re.sub(r"\s+", " ", description.lower()).strip()
        outcome_norm = re.sub(r"\s+", " ", preferred_outcome.lower()).strip()
        # Outcome should lead the narrative weight for idea descriptions.
        if outcome_norm and outcome_norm not in description_norm:
            description = f"Outcome: {preferred_outcome}. {description}"
    description = re.sub(r"\bin plain language\b", "", description, flags=re.IGNORECASE)
    description = re.sub(r"\s{2,}", " ", description).strip(" ,.-")
    return {
        "id": idea_data.get('id', str(uuid4())),
        "title": title or 'Untitled Idea',
        "content_type": content_type,
        "description": description,
        "search_phrase": search_phrase,
        "primary_keywords": keywords,
        "secondary_keywords": [],
        "seo_optimization_score": 0,
        "traffic_potential_score": 0,
        "total_search_volume": idea_data.get('total_search_volume'),
        "average_difficulty": idea_data.get('average_difficulty'),
        "average_cpc": idea_data.get('average_cpc'),
        "created_at": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "topic_id": topic_id,
        "subtopic": subtopic_name,
        "monetization_hook": idea_data.get('monetization_hook', ''),
        "target_intent": idea_data.get('target_intent', ''),
        "article_format": idea_data.get('article_format', ''),
        "user_decision_helped": idea_data.get('user_decision_helped', ''),
        "internal_link_hook": idea_data.get('internal_link_hook', ''),
        "product_type": idea_data.get('product_type', ''),
        "user_job_to_be_done": idea_data.get('user_job_to_be_done', ''),
        "key_inputs": idea_data.get('key_inputs', []),
        "output_result": idea_data.get('output_result', ''),
        "build_complexity": idea_data.get('build_complexity', ''),
        "distribution_angle": idea_data.get('distribution_angle', ''),
        "idea_metadata": {
            "search_phrase": search_phrase,
            "input_keywords": input_keywords,
            "keyword_seed_pack": {
                "input_keywords": input_keywords,
                "normalization_version": "simple_queries_v1",
                "source": "idea_burst_first_pass",
            },
            "target_intent": idea_data.get('target_intent', ''),
            "article_format": idea_data.get('article_format', ''),
            "user_decision_helped": idea_data.get('user_decision_helped', ''),
            "internal_link_hook": idea_data.get('internal_link_hook', ''),
            "product_type": idea_data.get('product_type', ''),
            "user_job_to_be_done": idea_data.get('user_job_to_be_done', ''),
            "key_inputs": idea_data.get('key_inputs', []),
            "output_result": idea_data.get('output_result', ''),
            "build_complexity": idea_data.get('build_complexity', ''),
            "distribution_angle": idea_data.get('distribution_angle', ''),
        },
        "viability_score": idea_data.get('viability_score', 50),
        "trend_score": 0,
        "monetization_score": 0,
        "seo_ease_score": 0,
        "status": "draft"
    }


def _normalize_intent_value(value: str) -> str:
    if not value:
        return "informational"
    normalized = value.strip().lower()
    if "transactional" in normalized:
        return "transactional"
    if "commercial" in normalized:
        return "commercial"
    return "informational"


def _intent_match_score(target_intent: str, idea_intent: str) -> int:
    target = _normalize_intent_value(target_intent)
    actual = _normalize_intent_value(idea_intent)
    if target == actual:
        return 100
    if {"commercial", "transactional"} == {target, actual}:
        return 75
    return 50


def _normalize_build_complexity(value: str) -> str:
    normalized = (value or "").strip().lower()
    if "high" in normalized:
        return "high"
    if "medium" in normalized:
        return "medium"
    return "low"


def _build_complexity_score(value: str) -> int:
    level = _normalize_build_complexity(value)
    if level == "low":
        return 90
    if level == "medium":
        return 70
    return 45


def _compute_opportunity_score(
    idea: dict,
    content_type: str,
    context_target_intent: str,
    context_tool_potential_score: int,
    context_serp_intent_match: str,
) -> tuple[int, dict]:
    viability = int(idea.get("viability_score") or 0)
    search_opportunity = max(0, min(100, int((idea.get("total_search_volume") or 0) / 100)))
    difficulty = max(0, min(100, int(idea.get("average_difficulty") or 0)))
    seo_ease = 100 - difficulty
    intent_match = _intent_match_score(context_target_intent, idea.get("target_intent", ""))
    serp_intent_match_score = {"high": 95, "medium": 75, "low": 55}.get((context_serp_intent_match or "medium").lower(), 75)
    tool_potential = max(0, min(100, int(context_tool_potential_score or 0)))

    if content_type == "software":
        complexity_score = _build_complexity_score(idea.get("build_complexity", ""))
        score = (
            viability * 0.30
            + tool_potential * 0.25
            + intent_match * 0.15
            + search_opportunity * 0.10
            + serp_intent_match_score * 0.10
            + complexity_score * 0.10
        )
        breakdown = {
            "viability": viability,
            "tool_potential": tool_potential,
            "intent_match": intent_match,
            "search_opportunity": search_opportunity,
            "serp_intent_match": serp_intent_match_score,
            "build_complexity_score": complexity_score,
        }
    else:
        score = (
            viability * 0.30
            + intent_match * 0.20
            + search_opportunity * 0.20
            + seo_ease * 0.15
            + serp_intent_match_score * 0.15
        )
        breakdown = {
            "viability": viability,
            "intent_match": intent_match,
            "search_opportunity": search_opportunity,
            "seo_ease": seo_ease,
            "serp_intent_match": serp_intent_match_score,
        }

    final_score = int(round(max(0, min(100, score))))
    return final_score, breakdown


def _rank_ideas(
    ideas: list[dict],
    content_type: str,
    context_target_intent: str,
    context_tool_potential_score: int,
    context_serp_intent_match: str,
) -> list[dict]:
    ranked = []
    for idea in ideas:
        score, breakdown = _compute_opportunity_score(
            idea=idea,
            content_type=content_type,
            context_target_intent=context_target_intent,
            context_tool_potential_score=context_tool_potential_score,
            context_serp_intent_match=context_serp_intent_match,
        )
        enriched = dict(idea)
        enriched["opportunity_score"] = score
        enriched["ranking_breakdown"] = breakdown
        ranked.append(enriched)
        logger.info(
            "idea_ranking_detail type=%s title=%r score=%s breakdown=%s",
            content_type,
            enriched.get("title", "")[:120],
            score,
            breakdown,
        )

    ranked.sort(
        key=lambda idea: (
            idea.get("opportunity_score", 0),
            idea.get("viability_score", 0),
            idea.get("total_search_volume", 0),
        ),
        reverse=True,
    )

    top_ranked = [
        {
            "rank": idx + 1,
            "title": (idea.get("title") or "")[:120],
            "score": idea.get("opportunity_score", 0),
            "viability": idea.get("viability_score", 0),
            "volume": idea.get("total_search_volume", 0),
            "breakdown": idea.get("ranking_breakdown", {}),
        }
        for idx, idea in enumerate(ranked[:3])
    ]
    logger.info(
        "idea_ranking_summary type=%s count=%s target_intent=%s serp_intent=%s tool_potential=%s top=%s",
        content_type,
        len(ranked),
        _normalize_intent_value(context_target_intent),
        (context_serp_intent_match or "medium").lower(),
        int(context_tool_potential_score or 0),
        top_ranked,
    )
    return ranked
