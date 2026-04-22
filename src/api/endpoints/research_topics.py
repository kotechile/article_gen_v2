"""
Research Topics API endpoints.

This module provides endpoints for managing research topics.
"""

import logging
import json
import re
from datetime import datetime
from uuid import uuid4
from flask import Blueprint, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from ...core.models.errors import ErrorResponse, ValidationErrorResponse
from ...api.middleware.auth import require_api_key

from ...core.models.topic_analysis import Subtopic

# Import supabase client
try:
    from supabase_client import get_supabase_client
except ImportError:
    # Fallback for when running in different context
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
            logger.warning(f"Failed to validate token or get user: {auth_error}")

    if not user_id and data and data.get('user_id'):
        user_id = data['user_id']

    return user_id


def _get_admin_supabase_client(default_client):
    """Return a service-role Supabase client when available."""
    from supabase import create_client
    import os
    import httpx

    sb_url = os.environ.get('SUPABASE_URL')
    sb_key = os.environ.get('SUPABASE_SERVICE_KEY')

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

        try:
            search_volume = int(float(item.get("search_volume") or item.get("volume") or 0))
        except Exception:
            search_volume = 0
        try:
            cpc = float(item.get("cpc") or 0.0)
        except Exception:
            cpc = 0.0
        try:
            keyword_difficulty = float(
                item.get("keyword_difficulty")
                or item.get("difficulty")
                or item.get("seo_difficulty")
                or 0.0
            )
        except Exception:
            keyword_difficulty = 0.0

        entries.append({
            "keyword": keyword,
            "search_volume": max(0, search_volume),
            "cpc": max(0.0, cpc),
            "keyword_difficulty": max(0.0, keyword_difficulty),
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
            "search_volume": int(item.get("search_volume") or 0),
            "cpc": round(float(item.get("cpc") or 0.0), 2),
            "keyword_difficulty": round(float(item.get("keyword_difficulty") or 0.0), 1),
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

        search_volume = int(row.get("search_volume") or 0)
        cpc = float(row.get("cpc") or 0.0)
        keyword_difficulty = float(row.get("keyword_difficulty") or 0.0)
        keyword_metrics[keyword] = {
            "search_volume": search_volume,
            "cpc": round(cpc, 2),
            "keyword_difficulty": round(keyword_difficulty, 1),
        }
        if search_volume > 0:
            volumes.append(search_volume)
        if cpc > 0:
            cpcs.append(cpc)
        if keyword_difficulty > 0:
            difficulties.append(keyword_difficulty)

    aggregates = {
        "total_search_volume": int(sum(volumes)) if volumes else 0,
        "average_cpc": round((sum(cpcs) / len(cpcs)) if cpcs else 0.0, 2),
        "average_difficulty": round((sum(difficulties) / len(difficulties)) if difficulties else 0.0, 1),
        "keywords_used": list(keyword_metrics.keys()),
    }
    return keyword_metrics, aggregates


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


def _safe_string(value):
    """Normalize optional values into compact strings."""
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = " ".join(value.split()).strip()
        return cleaned or None
    return str(value)


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
            "updated_at": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "project_id": data.get('project_id'),
            "primary_category_id": data.get('primary_category_id'),
            "secondary_category_id": data.get('secondary_category_id'),
            "topic_source": data.get('topic_source'),
            "source_topic_id": data.get('source_topic_id'),
        }
        insert_data.update(_extract_angle_metadata(data))
        hydrated = _hydrate_angle_metadata_for_payloads(supabase, [insert_data])
        insert_data = hydrated[0] if hydrated else insert_data

        supabase_admin = _get_admin_supabase_client(supabase)
        response = supabase_admin.table('research_topics').insert(insert_data).execute()
            
        if not response or not response.data:
            raise Exception("Failed to insert record")

        enriched = _enrich_research_topics(supabase, response.data)
        return jsonify(enriched[0]), 201

    except Exception as e:
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
                'project_id',
                'primary_category_id',
                'secondary_category_id',
                'topic_source',
                'source_topic_id',
                *ANGLE_METADATA_FIELDS,
            ]
        }
        update_data['updated_at'] = datetime.utcnow().isoformat()
        
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
                "updated_at": now,
                "user_id": user_id,
                "project_id": item.get('project_id'),
                "primary_category_id": item.get('primary_category_id'),
                "secondary_category_id": item.get('secondary_category_id'),
                "topic_source": item.get('topic_source'),
                "source_topic_id": item.get('source_topic_id'),
            }
            item_payload.update(_extract_angle_metadata(item))
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
        try:
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
            "Topic delete cascade completed topic_id=%s user_id=%s deleted_topics=%s deleted_subtopics=%s deleted_content_ideas=%s",
            topic_id,
            user_id,
            deleted_topics,
            deleted_subtopics,
            deleted_content_ideas,
        )
        return jsonify({
            "message": "Topic deleted successfully",
            "deleted_topics": deleted_topics,
            "deleted_subtopics": deleted_subtopics,
            "deleted_content_ideas": deleted_content_ideas,
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

        # Delete content ideas linked by topic + subtopic name.
        if subtopic_name:
            try:
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
            "Subtopic delete cascade completed topic_id=%s subtopic_id=%s subtopic_name=%r user_id=%s deleted_subtopics=%s deleted_content_ideas=%s",
            topic_id,
            subtopic_id,
            subtopic_name,
            user_id,
            deleted_subtopics,
            deleted_content_ideas,
        )
        return jsonify({
            "message": "Subtopic deleted successfully",
            "deleted_subtopics": deleted_subtopics,
            "deleted_content_ideas": deleted_content_ideas,
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
from src.core.models.enhanced_subtopic import EnhancedSubtopic

# Instantiate services
enhanced_decomposition_service = EnhancedTopicDecompositionService()
subtopics_service = SubtopicsService()

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
    try:
        supabase = get_supabase_client()
        request_user_id = _resolve_user_id_from_request(supabase)
        if not request_user_id:
            return jsonify(ErrorResponse(
                error="authentication_required",
                message="Authorization bearer token is required",
                error_code="AUTHENTICATION_REQUIRED",
                status=401
            ).dict()), 401

        # 1. Fetch topic metadata
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
            "Subtopic generation started request_id=%s topic_id=%s request_user_id=%s owner_user_id=%s title=%r",
            request_id,
            topic_id,
            request_user_id,
            user_id,
            topic_title
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
            result = await enhanced_decomposition_service.decompose_topic_enhanced(
                query=topic_title,
                user_id=user_id,
                max_subtopics=12,
                decomposition_context=decomposition_context,
            )
            logger.info(
                "Enhanced decomposition finished request_id=%s success=%s subtopic_count=%s message=%r methods=%s",
                request_id,
                result.get("success"),
                len(result.get("subtopics") or []),
                result.get("message"),
                result.get("enhancement_methods")
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
                }

            enhanced_subtopics_data = result.get("subtopics", [])
            saved_subtopics = []
            failed_subtopics = []
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
                "Subtopic persistence summary request_id=%s attempted=%s saved=%s failed=%s failed_titles=%s",
                request_id,
                len(enhanced_subtopics_data),
                len(saved_subtopics),
                len(failed_subtopics),
                failed_subtopics
            )

            return saved_subtopics, {
                "success": True,
                "message": result.get("message", "Subtopics generated"),
                "processing_time": result.get("processing_time"),
                "enhancement_methods": result.get("enhancement_methods", []),
            }

        saved_subtopics, result = asyncio.run(_run())
        logger.info(
            "Subtopic generation response request_id=%s total=%s success=%s message=%r",
            request_id,
            len(saved_subtopics),
            result.get("success"),
            result.get("message")
        )

        return jsonify({
            "items": saved_subtopics,
            "total": len(saved_subtopics),
            "meta": {
                "success":            result.get("success", True),
                "message":            result.get("message"),
                "processing_time":    result.get("processing_time"),
                "enhancement_methods": result.get("enhancement_methods"),
            }
        }), 200

    except Exception as e:
        logger.error(f"Error generating subtopics request_id={request_id}: {e}", exc_info=True)
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

        topic_context_res = (
            supabase
            .table('research_topics')
            .select('title, description, primary_category_id, secondary_category_id, intent_bucket, decision_focus, angle_question, value_layer_tags, target_audience')
            .eq('id', topic_id)
            .single()
            .execute()
        )
        topic_context = topic_context_res.data or {}

        category_path = "N/A"
        category_ids = [
            category_id
            for category_id in [topic_context.get('primary_category_id'), topic_context.get('secondary_category_id')]
            if category_id
        ]
        if category_ids:
            try:
                category_response = (
                    supabase
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
        from src.services.llm.llm_service import llm_service

        async def generate_ideas():
            # Generate blog ideas
            blog_prompt = f"""
You are a veteran SEO content strategist. Generate article ideas in plain, human language that sounds like real Google searches, not consultant-speak.

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
3. Each idea MUST include SEARCH_PHRASE (2-5 words, lowercase) that looks like a real query and can be sent to keyword tools.
4. TITLE must include SEARCH_PHRASE verbatim.
5. Keep SEARCH_PHRASE and INPUT_KEYWORDS tightly aligned to subtopic + decision focus + primary outcome.
6. Avoid topic drift and generic listicles.
7. DESCRIPTION is required and cannot be empty.
8. Prioritize decision/action intent over abstract commentary.
9. INPUT_KEYWORDS must be short, human, and literal search language for DataForSEO related keyword mining.
10. INPUT_KEYWORDS must be 3-5 items, each 2-4 words max, no punctuation-heavy phrases, no jargon.

For each idea, provide:
- Title: SEO-conscious title in plain language
- Description: 1-2 sentence angle summary
- Search Phrase: 2-5 words, plain language
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
SEARCH_PHRASE: [2-5 word query]
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
- Search Phrase: 2-5 words users would type to find this tool
- Input Keywords: 3-5 simple query-like seed phrases for DataForSEO related keyword mining
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
SEARCH_PHRASE: [2-5 word query]
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
"""

            # Generate both in parallel
            blog_response = await llm_service.generate_text(blog_prompt, max_tokens=2000)
            software_response = await llm_service.generate_text(software_prompt, max_tokens=1500)

            return blog_response.content, software_response.content

        blog_text, software_text = asyncio.run(generate_ideas())

        # Parse the responses
        blog_ideas = parse_idea_response(blog_text, 'blog', topic_id, user_id, subtopic_name)
        software_ideas = parse_idea_response(software_text, 'software', topic_id, user_id, subtopic_name)
        blog_ideas = _rank_ideas(
            ideas=blog_ideas,
            content_type="blog",
            context_target_intent=effective_intent_bucket,
            context_tool_potential_score=effective_tool_potential_score,
            context_serp_intent_match=effective_serp_intent_match,
        )
        software_ideas = _rank_ideas(
            ideas=software_ideas,
            content_type="software",
            context_target_intent=effective_intent_bucket,
            context_tool_potential_score=effective_tool_potential_score,
            context_serp_intent_match=effective_serp_intent_match,
        )

        def _attach_keyword_metrics(ideas: list[dict]) -> list[dict]:
            enriched_ideas = []
            for idea in ideas or []:
                idea_copy = dict(idea)
                idea_metadata = idea_copy.get("idea_metadata") or {}
                if not isinstance(idea_metadata, dict):
                    idea_metadata = {}

                keyword_metrics, aggregates = _build_idea_keyword_metrics_payload(
                    idea_copy,
                    subtopic_keyword_metrics_map,
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
                        "source": "subtopic_keyword_metrics",
                        "enriched_at": datetime.utcnow().isoformat(),
                    }
                elif prompt_keywords:
                    idea_metadata["seo_offer_enrichment"] = {
                        **(idea_metadata.get("seo_offer_enrichment") or {}),
                        "keywords_used": idea_copy.get("primary_keywords") or idea_copy.get("keywords") or prompt_keywords,
                        "source": "llm_estimate_only",
                    }

                idea_copy["idea_metadata"] = idea_metadata
                enriched_ideas.append(idea_copy)
            return enriched_ideas

        blog_ideas = _rank_ideas(
            ideas=_attach_keyword_metrics(blog_ideas),
            content_type="blog",
            context_target_intent=effective_intent_bucket,
            context_tool_potential_score=effective_tool_potential_score,
            context_serp_intent_match=effective_serp_intent_match,
        )
        software_ideas = _rank_ideas(
            ideas=_attach_keyword_metrics(software_ideas),
            content_type="software",
            context_target_intent=effective_intent_bucket,
            context_tool_potential_score=effective_tool_potential_score,
            context_serp_intent_match=effective_serp_intent_match,
        )

        # Persist burst ideas so they appear in Content Library and Software Ideas screens.
        all_ideas = (blog_ideas or []) + (software_ideas or [])
        if all_ideas:
            # Keep published ideas, replace only draft rows for this subtopic/topic/user.
            try:
                supabase.table("content_ideas") \
                    .delete() \
                    .eq("user_id", user_id) \
                    .eq("topic_id", topic_id) \
                    .eq("subtopic", subtopic_name) \
                    .eq("status", "draft") \
                    .execute()
            except Exception:
                # Some schemas may not include status; skip cleanup in that case.
                logger.warning("Skipping draft cleanup before idea burst save", exc_info=True)

            def _build_persist_row(idea: dict) -> dict:
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
                idea_metadata = idea.get("idea_metadata") or {}
                if not isinstance(idea_metadata, dict):
                    idea_metadata = {}

                return {
                    "id": idea.get("id"),
                    "title": idea.get("title") or "Untitled Idea",
                    "description": idea.get("description") or "",
                    "content_type": content_type,
                    "category": category,
                    "subtopic": subtopic_name,
                    "topic_id": topic_id,
                    "user_id": user_id,
                    "keywords": keywords,
                    "primary_keywords": keywords,
                    "secondary_keywords": secondary_keywords,
                    "search_phrase": idea.get("search_phrase") or "",
                    "total_search_volume": int(idea.get("total_search_volume") or 0),
                    "average_difficulty": float(idea.get("average_difficulty") or 0.0),
                    "average_cpc": float(idea.get("average_cpc") or 0.0),
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
                    "idea_metadata": idea_metadata,
                    "status": idea.get("status") or "draft",
                    "created_at": idea.get("created_at") or datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                }

            def _insert_with_schema_fallback(row: dict) -> bool:
                payload = dict(row)
                for _ in range(6):
                    try:
                        supabase.table("content_ideas").insert(payload).execute()
                        return True
                    except Exception as insert_error:
                        err = str(insert_error)
                        missing_cols = re.findall(r"Could not find the '([^']+)' column", err)
                        if not missing_cols:
                            logger.warning("Idea burst insert failed without recoverable schema hint", exc_info=True)
                            return False
                        for col in missing_cols:
                            payload.pop(col, None)
                        if not payload:
                            return False
                return False

            persisted_rows = [_build_persist_row(idea) for idea in all_ideas]
            saved_count = 0
            for row in persisted_rows:
                if _insert_with_schema_fallback(row):
                    saved_count += 1
            logger.info(
                "Idea burst persistence summary topic_id=%s subtopic=%s attempted=%s saved=%s",
                topic_id,
                subtopic_name,
                len(persisted_rows),
                saved_count,
            )

            # Touch the parent topic to mark progress recency.
            try:
                supabase.table("research_topics").update({
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", topic_id).eq("user_id", user_id).execute()
            except Exception:
                logger.warning("Could not update research topic timestamp after idea burst", exc_info=True)

        return jsonify({
            "success": True,
            "blog_ideas": [idea.to_dict() if hasattr(idea, 'to_dict') else idea for idea in blog_ideas],
            "software_ideas": [idea.to_dict() if hasattr(idea, 'to_dict') else idea for idea in software_ideas],
            "persisted_count": len(all_ideas),
        }), 200

    except Exception as e:
        logger.error(f"Error in idea burst: {e}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


def parse_idea_response(text: str, content_type: str, topic_id: str, user_id: str, subtopic_name: str):
    """Parse LLM response into ContentIdea objects."""
    import re
    from uuid import uuid4

    def _normalize_search_phrase(raw_phrase: str) -> str:
        phrase = re.sub(r"[^a-zA-Z0-9\s\-]", " ", str(raw_phrase or "")).lower()
        phrase = re.sub(r"\s+", " ", phrase).strip(" -")
        if not phrase:
            return ""
        tokens = [token for token in phrase.split(" ") if token]
        if len(tokens) < 2:
            return ""
        if len(tokens) > 5:
            tokens = tokens[:5]
        return " ".join(tokens)

    def _normalize_idea_title(raw_title: str) -> str:
        """Normalize LLM titles to plain language and reduce consultant-speak drift."""
        if not raw_title:
            return raw_title

        title = re.sub(r"\s+", " ", raw_title.strip())

        # Replace jargon-heavy words with simpler alternatives.
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

        # Keep punctuation readable and avoid repeated separators.
        title = re.sub(r"\s*[-–—]{2,}\s*", " - ", title)
        title = re.sub(r"\s{2,}", " ", title).strip(" -")
        return title

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
                ideas.append(create_idea_dict(current_idea, content_type, topic_id, user_id, subtopic_name))
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
                ideas.append(create_idea_dict(current_idea, content_type, topic_id, user_id, subtopic_name))
                current_idea = {}

    # Don't forget the last idea
    if current_idea and 'title' in current_idea:
        ideas.append(create_idea_dict(current_idea, content_type, topic_id, user_id, subtopic_name))

    return ideas


def create_idea_dict(idea_data: dict, content_type: str, topic_id: str, user_id: str, subtopic_name: str) -> dict:
    """Create a standardized idea dictionary."""
    from datetime import datetime
    from uuid import uuid4
    import re

    def _normalize_search_phrase(raw_phrase: str) -> str:
        phrase = re.sub(r"[^a-zA-Z0-9\s\-]", " ", str(raw_phrase or "")).lower()
        phrase = re.sub(r"\s+", " ", phrase).strip(" -")
        if not phrase:
            return ""
        tokens = [token for token in phrase.split(" ") if token]
        if len(tokens) < 2:
            return ""
        if len(tokens) > 5:
            tokens = tokens[:5]
        return " ".join(tokens)

    def _derive_search_phrase(raw_title: str, raw_keywords: list[str]) -> str:
        for kw in raw_keywords:
            normalized_kw = _normalize_search_phrase(kw)
            if normalized_kw:
                return normalized_kw
        title_tokens = re.sub(r"[^a-zA-Z0-9\s\-]", " ", raw_title.lower()).split()
        if len(title_tokens) >= 2:
            return " ".join(title_tokens[: min(4, len(title_tokens))])
        return ""

    def _title_contains_phrase(raw_title: str, phrase: str) -> bool:
        title_norm = re.sub(r"\s+", " ", str(raw_title or "").lower()).strip()
        phrase_norm = re.sub(r"\s+", " ", str(phrase or "").lower()).strip()
        return bool(title_norm and phrase_norm and phrase_norm in title_norm)

    SIMPLE_STOPWORDS = {
        "a", "an", "the", "to", "for", "of", "in", "on", "at", "with", "without",
        "from", "into", "by", "my", "your", "our", "their", "is", "are", "be",
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
                continue
            if len(tokens) > 4:
                tokens = tokens[:4]
            normalized_parts.append(" ".join(tokens))

        if normalized_parts:
            return normalized_parts[0]

        tokens = [t for t in cleaned.split(" ") if t and t not in SIMPLE_STOPWORDS and t not in JARGON_TOKENS]
        if len(tokens) < 2:
            return tokens[0] if tokens and len(tokens[0]) >= 4 else ""
        if len(tokens) > 4:
            tokens = tokens[:4]
        return " ".join(tokens)

    def _build_simple_keyword_seeds(title_value: str, subtopic_value: str, seeds: list[str]) -> list[str]:
        raw_candidates = list(seeds or [])
        raw_candidates.extend([title_value, subtopic_value])

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
            scored.append((score, normalized))

        scored.sort(key=lambda x: (-x[0], len(x[1])))
        return [item for _, item in scored[:5]]

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

    if search_phrase and not _title_contains_phrase(title, search_phrase):
        logger.info("Rewriting idea title to include search phrase: '%s'", search_phrase)
        title = f"{search_phrase}: {title}"

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

    title_lower = title.lower()
    contains_keyword = any(kw.lower() in title_lower for kw in keywords)
    if keywords and not contains_keyword:
        primary_kw = keywords[0]
        clean_primary_kw = re.sub(r"\s+", " ", primary_kw).strip()
        if clean_primary_kw:
            logger.info("Prefixing idea title with keyword fallback: '%s'", clean_primary_kw)
            title = f"{clean_primary_kw}: {title}"

    description = str(idea_data.get('description') or '').strip()
    if not description:
        if keywords:
            description = (
                f"This article explains {keywords[0]} in plain language and helps the reader make a practical decision."
            )
        else:
            description = (
                f"This article gives a practical breakdown of {title.lower()} and what action the reader should take next."
            )
    # Keep description short and UI-friendly.
    if len(description) > 220:
        description = description[:217].rstrip() + "..."

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
        "total_search_volume": idea_data.get('total_search_volume', 0),
        "average_difficulty": idea_data.get('average_difficulty', 50),
        "average_cpc": 0.0,
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
