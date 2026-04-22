"""
Content Ideas API endpoints.

Provides list, publish, and delete actions used by the frontend Idea Burst flow.
"""

import asyncio
import json
import logging
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


def _ensure_short_description(raw_description, title: str = "", keywords: list | None = None, subtopic: str = "") -> str:
    """Guarantee a short non-empty description for idea cards."""
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

    if len(description) > 220:
        description = description[:217].rstrip() + "..."
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


def _coerce_numeric(value, cast, default):
    try:
        if value is None:
            return default
        return cast(value)
    except Exception:
        return default


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
    total_search_volume = _coerce_numeric(row_copy.get("total_search_volume"), int, 0)
    average_difficulty = _coerce_numeric(row_copy.get("average_difficulty"), float, 0.0)
    average_cpc = _coerce_numeric(row_copy.get("average_cpc"), float, 0.0)

    if total_search_volume <= 0:
        total_search_volume = _coerce_numeric(seo_offer.get("total_search_volume"), int, 0)
    if average_difficulty <= 0:
        average_difficulty = _coerce_numeric(seo_offer.get("average_difficulty"), float, 0.0)
    if average_cpc <= 0:
        average_cpc = _coerce_numeric(seo_offer.get("average_cpc"), float, 0.0)

    if (total_search_volume <= 0 or average_difficulty <= 0 or average_cpc <= 0) and keyword_metrics:
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

        if total_search_volume <= 0 and volumes:
            total_search_volume = int(sum(volumes))
        if average_difficulty <= 0 and difficulties:
            average_difficulty = round(sum(difficulties) / len(difficulties), 1)
        if average_cpc <= 0 and cpcs:
            average_cpc = round(sum(cpcs) / len(cpcs), 2)

    row_copy["total_search_volume"] = max(0, int(total_search_volume or 0))
    row_copy["average_difficulty"] = round(float(average_difficulty or 0.0), 1)
    row_copy["average_cpc"] = round(float(average_cpc or 0.0), 2)
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


def _select_primary_seed_keyword(keywords: list[str], title: str = "", search_phrase: str = "") -> str:
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
    for raw in ordered_inputs:
        seed = _simplify_seed_keyword(raw)
        if not seed or seed in seen:
            continue
        seen.add(seed)
        if _looks_human_seed(seed):
            return seed

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
        if diagnostics is not None:
            diagnostics["related_rows_returned"] = len(related_rows or [])
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


def _restate_title_with_primary_keyword(title: str, primary_keyword: str) -> tuple[str, bool]:
    base_title = str(title or "").strip()
    keyword = str(primary_keyword or "").strip().lower()
    if not base_title or not keyword:
        return base_title, False

    normalized_title = _normalize_title_keyword_term(base_title)
    normalized_keyword = _normalize_title_keyword_term(keyword)
    if normalized_keyword and normalized_keyword in normalized_title:
        return base_title, False

    return f"{keyword}: {base_title}", True


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
    primary_keywords = idea.get("primary_keywords") or idea.get("keywords") or []
    secondary_keywords = idea.get("secondary_keywords") or []
    if isinstance(primary_keywords, str):
        primary_keywords = [k.strip() for k in primary_keywords.split(",") if k.strip()]
    if isinstance(secondary_keywords, str):
        secondary_keywords = [k.strip() for k in secondary_keywords.split(",") if k.strip()]
    if not secondary_keywords and isinstance(primary_keywords, list) and len(primary_keywords) > 1:
        secondary_keywords = primary_keywords[1:]

    primary_keyword = primary_keywords[0] if primary_keywords else ""
    exact_keyword_metrics = idea.get("keyword_metrics") or {}
    if not exact_keyword_metrics:
        seo_offer_enrichment = (idea.get("idea_metadata") or {}).get("seo_offer_enrichment") or {}
        exact_keyword_metrics = seo_offer_enrichment.get("keyword_metrics") or {}
    selected_keyword_metrics_json = _build_keyword_metrics_payload(
        primary_keyword=primary_keyword,
        secondary_keywords=secondary_keywords,
        metrics_map=exact_keyword_metrics,
        source="dataforseo" if idea.get("total_search_volume") else "llm_fallback",
        target_intent=idea.get("target_intent") or "informational",
    )
    primary_metric = (selected_keyword_metrics_json.get("primary") or {}) if primary_keyword else {}

    update_payload = {
        "Keywords": ", ".join(primary_keywords),
        "keyword_candidates_json": primary_keywords + [k for k in secondary_keywords if k not in primary_keywords],
        "keyword_research_status": "ready" if primary_keywords else "fallback",
        "keyword_research_source": "dataforseo" if idea.get("total_search_volume") else "llm_fallback",
        "keyword_research_confidence": 0.85 if idea.get("total_search_volume") else 0.35,
        "keyword_research_generated_at": now_iso,
        "primary_keyword": primary_keyword,
        "secondary_keywords_json": secondary_keywords,
        "selected_keyword_search_volume": int(primary_metric.get("search_volume") or idea.get("total_search_volume") or 0),
        "selected_keyword_difficulty": float(primary_metric.get("keyword_difficulty") or idea.get("average_difficulty") or 0.0),
        "selected_keyword_intent": idea.get("target_intent") or "informational",
        "selected_keyword_metrics_json": selected_keyword_metrics_json,
        "keyword_selection_reason": "Refreshed from content_ideas keyword enrichment.",
        "keyword_strategy_version": "phase1_v3",
        "keyword_selection_source": "re-ranked_with_dataforseo",
    }
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
        search_phrase=str(idea.get("search_phrase") or ""),
    )
    working_candidates = [primary_seed_keyword] if primary_seed_keyword else list(candidates[:1])
    metrics_map: dict = {}
    ranked_candidates: list[dict] = []
    tier_diagnostics: list[dict] = []
    dataforseo_calls = 0

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
    ranked_pool = list(dict.fromkeys(
        [primary_seed_keyword] +
        [str(k).strip().lower() for k in metrics_map.keys() if str(k).strip()] +
        [str(k).strip().lower() for k in (keywords or []) if str(k).strip()]
    ))
    ranked_candidates = _rank_keywords_by_opportunity(ranked_pool, metrics_map)
    summary = _keyword_quality_summary(ranked_candidates)
    tier_diag["quality"] = summary
    tier_diag["calls"] = 1
    tier_diagnostics.append(tier_diag)
    dataforseo_calls = 1

    # Persist exact raw response from DataForSEO when present.
    raw_dataforseo_output: dict = (
        (tier_diag.get("raw_dataforseo") or {}).get("related_keywords_live")
        or {
            "idea_id": idea_id,
            "captured_at": datetime.utcnow().isoformat(),
            "tiers": [],
        }
    )

    selected_keywords = [row["keyword"] for row in ranked_candidates[:5]]
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
        affiliate_offers_preview = [
            {
                "name": program.get("name"),
                "network": program.get("network"),
                "commission_rate": program.get("commission_rate"),
            }
            for program in programs[:5]
        ]
    except Exception:
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
    safe_raw = raw_output or {}
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
            "idea_metadata": idea_metadata,
            "updated_at": now_iso,
        },
        {
            "raw_dataforseo_output": safe_raw,
            "updated_at": now_iso,
        },
        {
            "idea_metadata": idea_metadata,
            "updated_at": now_iso,
        },
        {"updated_at": now_iso},
    ):
        try:
            supabase.table("content_ideas").update(payload).eq("id", idea.get("id")).eq("user_id", user_id).execute()
            return
        except Exception:
            continue


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

            # 2) For blog ideas, publish into Titles table directly.
            if (idea.get("content_type") or "").lower() != "software":
                primary_keywords = idea.get("primary_keywords") or idea.get("keywords") or []
                secondary_keywords = idea.get("secondary_keywords") or []
                if isinstance(primary_keywords, str):
                    primary_keywords = [k.strip() for k in primary_keywords.split(",") if k.strip()]
                if isinstance(secondary_keywords, str):
                    secondary_keywords = [k.strip() for k in secondary_keywords.split(",") if k.strip()]
                if not secondary_keywords and isinstance(primary_keywords, list) and len(primary_keywords) > 1:
                    secondary_keywords = primary_keywords[1:]
                primary_keyword = primary_keywords[0] if primary_keywords else ""
                # Prefer explicit keyword_metrics column in content_ideas schema.
                exact_keyword_metrics = idea.get("keyword_metrics") or {}
                if not exact_keyword_metrics:
                    seo_offer_enrichment = (idea.get("idea_metadata") or {}).get("seo_offer_enrichment") or {}
                    exact_keyword_metrics = seo_offer_enrichment.get("keyword_metrics") or {}
                selected_keyword_metrics_json = _build_keyword_metrics_payload(
                    primary_keyword=primary_keyword,
                    secondary_keywords=secondary_keywords,
                    metrics_map=exact_keyword_metrics,
                    source="dataforseo" if idea.get("total_search_volume") else "llm_fallback",
                    target_intent=idea.get("target_intent") or "informational",
                )
                primary_metric = (selected_keyword_metrics_json.get("primary") or {}) if primary_keyword else {}
                title_payload = {
                    "id": str(uuid4()),
                    "user_id": user_id,
                    "Title": idea.get("title") or "Untitled Article",
                    "userDescription": idea.get("description") or "",
                    "Keywords": ", ".join(primary_keywords),
                    # This record is queued for writing, not published to an external CMS.
                    "status": "New",
                    "published": False,
                    "dateCreatedOn": now,
                    "source_idea_id": idea.get("id"),
                    # Phase 1 keyword handoff defaults (Research -> Content Generation)
                    "keyword_candidates_json": primary_keywords + [k for k in secondary_keywords if k not in primary_keywords],
                    "keyword_clusters_json": [],
                    "keyword_research_status": "ready" if primary_keywords else "fallback",
                    "keyword_research_source": "dataforseo" if idea.get("total_search_volume") else "llm_fallback",
                    "keyword_research_confidence": 0.85 if idea.get("total_search_volume") else 0.35,
                    "keyword_research_generated_at": now,
                    "primary_keyword": primary_keyword,
                    "secondary_keywords_json": secondary_keywords,
                    "selected_keyword_search_volume": int(primary_metric.get("search_volume") or idea.get("total_search_volume") or 0),
                    "selected_keyword_difficulty": float(primary_metric.get("keyword_difficulty") or idea.get("average_difficulty") or 0.0),
                    "selected_keyword_intent": idea.get("target_intent") or "informational",
                    "selected_keyword_metrics_json": selected_keyword_metrics_json,
                    "keyword_selection_reason": "Initialized from research idea publish payload.",
                    "keyword_strategy_version": "phase1_v1",
                    "keyword_selection_source": "research_dossier_reused",
                }
                try:
                    supabase.table("Titles").insert(title_payload).execute()
                    published_to_titles_count += 1
                except Exception as insert_error:
                    err = str(insert_error)
                    missing_cols = re.findall(r"Could not find the '([^']+)' column", err)
                    if missing_cols:
                        fallback_payload = dict(title_payload)
                        for col in missing_cols:
                            fallback_payload.pop(col, None)
                        try:
                            supabase.table("Titles").insert(fallback_payload).execute()
                            published_to_titles_count += 1
                            logger.warning(
                                "Inserted Titles row for idea_id=%s after dropping missing columns: %s",
                                idea_id,
                                ", ".join(missing_cols),
                            )
                        except Exception:
                            logger.warning("Could not insert Titles row for idea_id=%s", idea_id, exc_info=True)
                    else:
                        logger.warning("Could not insert Titles row for idea_id=%s", idea_id, exc_info=True)

            # 3) Best-effort status update on content_ideas with progressive fallbacks.
            try:
                supabase.table("content_ideas").update({
                    "status": "published",
                    "published": True,
                    "published_to_titles": True,
                    "published_at": now,
                    "updated_at": now,
                }).eq("id", idea_id).eq("user_id", user_id).execute()
                updated_count += 1
                continue
            except Exception:
                pass

            try:
                supabase.table("content_ideas").update({
                    "status": "published",
                    "updated_at": now,
                }).eq("id", idea_id).eq("user_id", user_id).execute()
                updated_count += 1
                continue
            except Exception:
                pass

            try:
                supabase.table("content_ideas").update({
                    "updated_at": now,
                }).eq("id", idea_id).eq("user_id", user_id).execute()
                updated_count += 1
            except Exception:
                # Last fallback for minimal schemas without updated_at.
                result = (
                    supabase
                    .table("content_ideas")
                    .update({"description": idea.get("description") or ""})
                    .eq("id", idea_id)
                    .eq("user_id", user_id)
                    .execute()
                )
                if result.data:
                    updated_count += 1

        success = (updated_count > 0) or (published_to_titles_count > 0)
        status_code = 200 if success else 400
        return jsonify({
            "success": success,
            "published_count": updated_count,
            "published_to_titles_count": published_to_titles_count,
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
                supabase
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
                    supabase=supabase,
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

            # If all enrichment metrics are zero, report this as a failed enrichment
            # so UI does not show a misleading "success" with no visible SEO stats.
            if (
                not bool(enrichment.get("has_exact_keyword_metrics"))
                and int(enrichment.get("total_search_volume") or 0) == 0
                and float(enrichment.get("average_cpc") or 0.0) == 0.0
                and float(enrichment.get("average_difficulty") or 0.0) == 0.0
            ):
                logger.warning(
                    "Enrichment produced zero metrics for idea_id=%s keywords=%s",
                    idea_id,
                    enrichment.get("keywords_used") or [],
                )
                _persist_raw_trace_for_idea(
                    supabase=supabase,
                    idea=idea,
                    user_id=user_id,
                    now_iso=now,
                    raw_output=enrichment.get("raw_dataforseo_output") or {},
                    reason="zero_metrics",
                )
                results.append({
                    "idea_id": idea_id,
                    "status": "failed",
                    "reason": "No SEO/offer metrics returned for this idea",
                })
                continue

            all_keywords_found = [
                str(keyword).strip()
                for keyword in (enrichment.get("all_keywords_found") or [])
                if str(keyword).strip()
            ]
            keywords_used = all_keywords_found or [
                str(keyword).strip()
                for keyword in (enrichment.get("keywords_used") or [])
                if str(keyword).strip()
            ]
            selected_primary_keyword = keywords_used[0] if keywords_used else ""
            original_title = str(idea.get("title") or "").strip()
            restated_title, title_restated = _restate_title_with_primary_keyword(
                original_title,
                selected_primary_keyword,
            )
            keyword_projection_payload = {
                "keywords": keywords_used,
                "primary_keywords": keywords_used,
                "secondary_keywords": keywords_used[1:],
                "search_phrase": selected_primary_keyword,
            }
            update_payload = {
                "total_search_volume": enrichment["total_search_volume"],
                "average_cpc": enrichment["average_cpc"],
                "average_difficulty": enrichment["average_difficulty"],
                "affiliate_offer_count": enrichment["affiliate_offer_count"],
                "raw_dataforseo_output": enrichment.get("raw_dataforseo_output") or {},
                "status": "in_progress",
                "updated_at": now,
                "title": restated_title or original_title,
            }
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
                    "affiliate_offers_preview": enrichment["affiliate_offers"],
                    "enriched_at": now,
                },
                "keyword_pass_2": {
                    "selected_primary_keyword": selected_primary_keyword,
                    "selection_rule_version": "kd_asc_volume_desc_v1",
                    "keyword_ranked_candidates": enrichment.get("keyword_ranked_candidates") or [],
                    "keyword_quality_summary": enrichment.get("keyword_quality_summary") or {},
                    "title_restated": title_restated,
                    "restated_title": restated_title or original_title,
                    "original_title": original_title,
                    "restated_at": now,
                },
            }

            updated = False
            # Try richest payload first; gracefully degrade for older schemas.
            for payload in (
                {
                    **update_payload,
                    **keyword_projection_payload,
                    "keyword_metrics": enrichment.get("keyword_metrics_map") or {},
                    "idea_metadata": enrichment_metadata,
                },
                {
                    **update_payload,
                    **keyword_projection_payload,
                    "idea_metadata": enrichment_metadata,
                },
                {
                    **update_payload,
                    **keyword_projection_payload,
                    "keyword_metrics": enrichment.get("keyword_metrics_map") or {},
                },
                {
                    **update_payload,
                    **keyword_projection_payload,
                },
                {
                    **update_payload,
                    "keyword_metrics": enrichment.get("keyword_metrics_map") or {},
                    "idea_metadata": enrichment_metadata,
                },
                {
                    **update_payload,
                    "idea_metadata": enrichment_metadata,
                },
                {
                    **update_payload,
                    "keyword_metrics": enrichment.get("keyword_metrics_map") or {},
                },
                update_payload,
                {"updated_at": now},
            ):
                try:
                    supabase.table("content_ideas").update(payload).eq("id", idea_id).eq("user_id", user_id).execute()
                    updated = True
                    break
                except Exception:
                    continue

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
                    "selected_primary_keyword": selected_primary_keyword,
                    "restated_title": restated_title or original_title,
                    "title_restated": title_restated,
                    "metrics": {
                        "total_search_volume": enrichment["total_search_volume"],
                        "average_cpc": enrichment["average_cpc"],
                        "average_difficulty": enrichment["average_difficulty"],
                        "affiliate_offer_count": enrichment["affiliate_offer_count"],
                    },
                    "keywords_used": enrichment["keywords_used"],
                    "keyword_metrics_map": enrichment.get("keyword_metrics_map") or {},
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
                supabase.table("content_ideas")
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
                    supabase=supabase,
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
                "primary_keywords": keywords_used,
                "secondary_keywords": keywords_used[1:],
            }
            update_payload = {
                "total_search_volume": enrichment["total_search_volume"],
                "average_cpc": enrichment["average_cpc"],
                "average_difficulty": enrichment["average_difficulty"],
                "affiliate_offer_count": enrichment["affiliate_offer_count"],
                "raw_dataforseo_output": enrichment.get("raw_dataforseo_output") or {},
                "status": "in_progress",
                "updated_at": now,
            }
            for payload in (
                {
                    **update_payload,
                    **keyword_projection_payload,
                    "keyword_metrics": enrichment.get("keyword_metrics_map") or {},
                    "idea_metadata": {
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
                            "affiliate_offers_preview": enrichment["affiliate_offers"],
                            "enriched_at": now,
                        },
                    },
                },
                {
                    **update_payload,
                    **keyword_projection_payload,
                    "keyword_metrics": enrichment.get("keyword_metrics_map") or {},
                },
                {
                    **update_payload,
                    **keyword_projection_payload,
                },
                {
                    **update_payload,
                    "keyword_metrics": enrichment.get("keyword_metrics_map") or {},
                    "idea_metadata": {
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
                            "affiliate_offers_preview": enrichment["affiliate_offers"],
                            "enriched_at": now,
                        },
                    },
                },
                update_payload,
                {"updated_at": now},
            ):
                try:
                    supabase.table("content_ideas").update(payload).eq("id", idea_id).eq("user_id", user_id).execute()
                    break
                except Exception:
                    continue

            # Re-fetch updated row and sync Titles projection.
            refreshed_idea_resp = (
                supabase.table("content_ideas")
                .select("*")
                .eq("id", idea_id)
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            refreshed_idea = (refreshed_idea_resp.data or [idea])[0]
            synced = _sync_titles_keyword_fields_from_idea(
                supabase=supabase,
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
                    "average_difficulty": enrichment["average_difficulty"],
                },
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

        return jsonify({"success": True, "id": idea_id}), 200

    except Exception as e:
        logger.error(f"Error deleting content idea {idea_id}: {e}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500,
        ).dict()), 500


@content_ideas_bp.route("/keyword-lab/metrics", methods=["POST"])
@require_api_key
def keyword_lab_metrics():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        data = request.get_json() or {}
        raw_keywords = data.get("keywords") or []
        if isinstance(raw_keywords, str):
            raw_keywords = [part.strip() for part in re.split(r"[\n,]+", raw_keywords) if part.strip()]

        keywords = []
        seen = set()
        for kw in raw_keywords:
            norm = _normalize_keyword_term(str(kw))
            if not norm or norm in seen:
                continue
            seen.add(norm)
            keywords.append(norm)
        if not keywords:
            return jsonify({"error": "No valid keywords provided"}), 400

        supabase = get_supabase_client()
        request_user_id = _resolve_user_id_from_request(supabase, data)
        if not request_user_id:
            return jsonify({"error": "Authorization bearer token is required"}), 401

        metrics_map = asyncio.run(
            asyncio.wait_for(
                _fetch_metrics_map_for_keywords(keywords, max_keywords_for_metrics=max(15, len(keywords))),
                timeout=PER_IDEA_ENRICH_TIMEOUT_SECONDS,
            )
        )
        ranked = _rank_keywords_by_opportunity(keywords, metrics_map)
        return jsonify({
            "success": True,
            "keywords": ranked,
            "quality": _keyword_quality_summary(ranked),
        }), 200
    except Exception as e:
        logger.error("keyword_lab_metrics failed: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@content_ideas_bp.route("/keyword-lab/related", methods=["POST"])
@require_api_key
def keyword_lab_related():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        data = request.get_json() or {}
        seed = _normalize_keyword_term(data.get("seed_keyword") or "")
        limit = int(data.get("limit") or 12)
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
                dataforseo_api.get_related_keywords_labs_live(
                    [seed],
                    limit_per_seed=max(5, min(limit, 25)),
                    return_raw=True,
                ),
                timeout=DATAFORSEO_BULK_TIMEOUT_SECONDS,
            )
        )
        if isinstance(related_rows_response, dict):
            related_rows = related_rows_response.get("items") or []
        else:
            related_rows = related_rows_response or []
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
        metrics_map = asyncio.run(
            asyncio.wait_for(
                _fetch_metrics_map_for_keywords(candidate_keywords, max_keywords_for_metrics=max(15, len(candidate_keywords))),
                timeout=PER_IDEA_ENRICH_TIMEOUT_SECONDS,
            )
        )
        ranked = _rank_keywords_by_opportunity(candidate_keywords, metrics_map)
        return jsonify({
            "success": True,
            "seed_keyword": seed,
            "keywords": ranked,
            "quality": _keyword_quality_summary(ranked),
        }), 200
    except Exception as e:
        logger.error("keyword_lab_related failed: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@content_ideas_bp.route("/keyword-lab/apply", methods=["POST"])
@require_api_key
def keyword_lab_apply():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        data = request.get_json() or {}
        title_id = str(data.get("title_id") or "").strip()
        primary_keyword = _normalize_keyword_term(data.get("primary_keyword") or "")
        secondary_keywords = data.get("secondary_keywords") or []
        if isinstance(secondary_keywords, str):
            secondary_keywords = [part.strip() for part in re.split(r"[\n,]+", secondary_keywords) if part.strip()]
        secondary_keywords = [_normalize_keyword_term(k) for k in secondary_keywords if _normalize_keyword_term(k)]
        if not title_id or not primary_keyword:
            return jsonify({"error": "title_id and primary_keyword are required"}), 400

        supabase = get_supabase_client()
        request_user_id = _resolve_user_id_from_request(supabase, data)
        if not request_user_id:
            return jsonify({"error": "Authorization bearer token is required"}), 401
        user_id = data.get("user_id") or request_user_id
        if user_id != request_user_id:
            return jsonify({"error": "forbidden"}), 403

        all_keywords = [primary_keyword] + [k for k in secondary_keywords if k != primary_keyword]
        metrics_map = asyncio.run(
            asyncio.wait_for(
                _fetch_metrics_map_for_keywords(all_keywords, max_keywords_for_metrics=max(15, len(all_keywords))),
                timeout=PER_IDEA_ENRICH_TIMEOUT_SECONDS,
            )
        )
        payload_json = _build_keyword_metrics_payload(
            primary_keyword=primary_keyword,
            secondary_keywords=[k for k in secondary_keywords if k != primary_keyword],
            metrics_map=metrics_map,
            source="manual_verified",
            target_intent="informational",
        )
        primary_metric = payload_json.get("primary") or {}
        update_payload = {
            "Keywords": ", ".join(all_keywords),
            "keyword_candidates_json": all_keywords,
            "keyword_research_status": "ready",
            "keyword_research_source": "manual_verified",
            "keyword_research_confidence": 0.95,
            "keyword_research_generated_at": datetime.utcnow().isoformat(),
            "primary_keyword": primary_keyword,
            "secondary_keywords_json": [k for k in secondary_keywords if k != primary_keyword],
            "selected_keyword_search_volume": int(primary_metric.get("search_volume") or 0),
            "selected_keyword_difficulty": float(primary_metric.get("keyword_difficulty") or 0.0),
            "selected_keyword_intent": "informational",
            "selected_keyword_metrics_json": payload_json,
            "keyword_selection_reason": "Manually selected in Keyword Lab.",
            "keyword_strategy_version": "manual_v1",
            "keyword_selection_source": "manual_keyword_lab",
        }
        response = (
            supabase.table("Titles")
            .update(update_payload)
            .eq("id", title_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not response.data:
            return jsonify({"error": "Title not found or not accessible"}), 404
        return jsonify({"success": True, "title_id": title_id, "keyword_metrics": payload_json}), 200
    except Exception as e:
        logger.error("keyword_lab_apply failed: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500
