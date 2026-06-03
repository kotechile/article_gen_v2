"""
Research API endpoints for Content Generator V2.

This module provides the main research endpoints for creating,
monitoring, and retrieving research tasks.
"""

import logging
import json
import os
import re
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from ...core.models.research import (
    ResearchRequest,
    ResearchResponse,
    ResearchProgress,
    ResearchStatus
)
from ...core.models.errors import (
    ErrorResponse,
    ValidationErrorResponse,
    ValidationError
)
from ...api.middleware.auth import require_api_key
from ...api.schemas.research import ResearchRequestSchema, ResearchResponseSchema
from llm_client import create_llm_client
from supabase_client import (
    LLM_ROLE_ARTICLE_GENERATION,
    LLM_ROLE_FINAL_REVIEW,
    get_llm_api_key,
    get_llm_provider_for_role,
    resolve_llm_provider,
)
# Import tasks when needed to avoid circular imports


logger = logging.getLogger(__name__)

_SOURCE_STRATEGIES = {
    "rag_only",
    "live_web_only",
    "rag_plus_live_web",
    "none",
}


def _normalize_source_strategy_payload(data: dict) -> None:
    """
    Normalize source strategy and keep legacy booleans in sync.

    During rollout we keep both representations:
    - `source_strategy` (new source of truth when flag enabled)
    - `rag_enabled` / `claims_research_enabled` (legacy compatibility)
    """
    if not isinstance(data, dict):
        return

    requested_strategy = str(data.get("source_strategy") or "").strip().lower()
    rag_enabled = bool(data.get("rag_enabled", False))
    claims_enabled = bool(data.get("claims_research_enabled", True))

    # Map legacy strategies to new options for backward compatibility
    if requested_strategy == "dossier_only":
        requested_strategy = "live_web_only"
    elif requested_strategy in ("dossier_plus_rag", "dossier_plus_rag_plus_live_web"):
        requested_strategy = "rag_plus_live_web"

    if requested_strategy in _SOURCE_STRATEGIES:
        strategy = requested_strategy
    else:
        # Infer strategy from legacy flags if requested_strategy is missing or unknown
        if rag_enabled and claims_enabled:
            strategy = "rag_plus_live_web"
        elif rag_enabled:
            strategy = "rag_only"
        elif claims_enabled:
            strategy = "live_web_only"
        else:
            strategy = "none"

    # Keep strategy as normalized canonical field.
    data["source_strategy"] = strategy

    # Keep legacy booleans aligned to avoid regressions in old code paths.
    if strategy == "live_web_only":
        data["rag_enabled"] = False
        data["claims_research_enabled"] = True
    elif strategy == "rag_plus_live_web":
        data["rag_enabled"] = True
        data["claims_research_enabled"] = True
    elif strategy == "rag_only":
        data["rag_enabled"] = True
        data["claims_research_enabled"] = False
    elif strategy == "none":
        data["rag_enabled"] = False
        data["claims_research_enabled"] = False

# Create blueprint
research_bp = Blueprint('research', __name__, url_prefix='/api/v1')

# Create rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1000 per hour", "60 per minute"],
    storage_uri="memory://"
)


def _extract_refined_metadata_from_response(raw: str, fallback_title: str, fallback_description: str) -> dict:
    cleaned = str(raw or "").strip()
    if not cleaned:
        return {
            "refined_title": fallback_title,
            "refined_description": fallback_description,
            "rationale": "",
        }

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and ("refined_title" in parsed or "refined_description" in parsed):
            return {
                "refined_title": str(parsed.get("refined_title") or fallback_title).strip(),
                "refined_description": str(parsed.get("refined_description") or fallback_description).strip(),
                "rationale": str(parsed.get("rationale") or "").strip(),
            }
    except Exception:
        pass

    # Fallback: extract JSON from deepthink/reasoning text via regex.
    # Matches {"refined_title": "...", ...} even when buried in model reasoning.
    import re
    for match in re.finditer(r'\{[^{}]*"refined_title"[^{}]*\}', cleaned):
        try:
            candidate = json.loads(match.group())
            if isinstance(candidate, dict):
                return {
                    "refined_title": str(candidate.get("refined_title") or fallback_title).strip(),
                    "refined_description": str(candidate.get("refined_description") or fallback_description).strip(),
                    "rationale": str(candidate.get("rationale") or "").strip(),
                }
        except Exception:
            continue

    # Fallback heuristics if model didn't return strict JSON.
    refined_title = fallback_title
    refined_description = fallback_description
    for line in cleaned.splitlines():
        low = line.lower()
        if "title:" in low and refined_title == fallback_title:
            refined_title = line.split(":", 1)[1].strip() or fallback_title
        elif "description:" in low and refined_description == fallback_description:
            refined_description = line.split(":", 1)[1].strip() or fallback_description

    return {
        "refined_title": refined_title,
        "refined_description": refined_description,
        "rationale": "",
    }


def _extract_refined_metadata_options_from_response(raw: str, fallback_title: str, fallback_description: str) -> list[dict]:
    cleaned = str(raw or "").strip()
    if not cleaned:
        return []

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    def _normalize_option(candidate: dict) -> dict | None:
        if not isinstance(candidate, dict):
            return None
        refined_title = str(candidate.get("refined_title") or "").strip()
        refined_description = str(candidate.get("refined_description") or "").strip()
        if not refined_title or not refined_description:
            return None
        return {
            "refined_title": refined_title,
            "refined_description": refined_description,
            "rationale": str(candidate.get("rationale") or "").strip(),
        }

    options: list[dict] = []
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            parsed_options = parsed.get("options")
            if isinstance(parsed_options, list):
                for item in parsed_options:
                    normalized = _normalize_option(item)
                    if normalized:
                        options.append(normalized)
        elif isinstance(parsed, list):
            for item in parsed:
                normalized = _normalize_option(item)
                if normalized:
                    options.append(normalized)
    except Exception:
        pass

    deduped: list[dict] = []
    seen = set()
    for opt in options:
        key = (
            re.sub(r"\s+", " ", opt.get("refined_title", "").strip().lower()),
            re.sub(r"\s+", " ", opt.get("refined_description", "").strip().lower()),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(opt)
    return deduped


def _contains_keyword(text: str, keyword: str) -> bool:
    normalized_text = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    normalized_keyword = re.sub(r"\s+", " ", str(keyword or "").lower()).strip()
    return bool(normalized_keyword and normalized_keyword in normalized_text)


def _sanitize_keyword_value(value) -> str:
    current = str(value or "").strip()
    if not current:
        return ""

    for _ in range(5):
        if not current:
            return ""
        try:
            parsed = json.loads(current)
        except Exception:
            break

        if isinstance(parsed, list):
            for item in parsed:
                candidate = _sanitize_keyword_value(item)
                if candidate:
                    return candidate
            return ""
        if isinstance(parsed, dict):
            candidate = _sanitize_keyword_value(parsed.get("keyword"))
            if candidate:
                return candidate
            break
        if isinstance(parsed, str):
            next_value = parsed.strip()
            if next_value == current:
                break
            current = next_value
            continue
        break

    while (
        (current.startswith("[") and current.endswith("]")) or
        (current.startswith('"') and current.endswith('"')) or
        (current.startswith("'") and current.endswith("'"))
    ):
        current = current[1:-1].strip()

    parts = [
        re.sub(r"\s+", " ", piece.strip().strip("\"'")).strip()
        for piece in current.split(",")
    ]
    parts = [piece for piece in parts if piece]
    return parts[0] if parts else ""


def _sanitize_refined_title(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return ""
    text = re.sub(r'^\[\s*["\']?(.*?)["\']?\s*\]$', r"\1", text).strip()
    text = re.sub(r'\[\s*["\']([^"\']+)["\']\s*\]', r"\1", text).strip()
    text = re.sub(r"\s+", " ", text).strip(" -:;,")
    return text


_TITLE_CONTEXT_STOPWORDS = {
    "about", "above", "after", "again", "against", "all", "also", "among", "and", "are", "because",
    "before", "being", "between", "both", "build", "complex", "could", "deciding", "does", "down",
    "each", "from", "have", "helps", "holdings", "how", "into", "just", "keep", "lack", "many",
    "more", "most", "much", "need", "only", "onto", "other", "over", "portfolio", "practical",
    "same", "sell", "should", "show", "specific", "than", "that", "their", "them", "then", "there",
    "these", "they", "this", "through", "too", "under", "very", "what", "when", "which", "while",
    "with", "without", "your", "guide", "strategy", "approach", "outcome",
}


def _extract_context_fragments(title: str, description: str, keyword: str, limit: int = 4) -> list[str]:
    keyword_terms = {
        token
        for token in re.findall(r"[a-z0-9]+", str(keyword or "").lower())
        if len(token) >= 3
    }
    fragments: list[str] = []
    seen = set()
    source_text = f"{title}. {description}"
    for raw_phrase in re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*(?:\s+[A-Za-z0-9][A-Za-z0-9'/-]*){1,2}", source_text):
        cleaned_words = []
        for word in re.findall(r"[a-z0-9]+", raw_phrase.lower()):
            if len(word) < 4:
                continue
            if word in _TITLE_CONTEXT_STOPWORDS:
                continue
            if word in keyword_terms:
                continue
            cleaned_words.append(word)
        if len(cleaned_words) < 2:
            continue
        phrase = " ".join(cleaned_words[:3])
        if phrase in seen:
            continue
        seen.add(phrase)
        fragments.append(phrase)
        if len(fragments) >= limit:
            break
    return fragments


def _fit_keyword_title(keyword: str, fragment: str | None = None, connector: str = "for") -> str:
    clean_keyword = _sanitize_keyword_value(keyword)
    clean_fragment = re.sub(r"\s+", " ", str(fragment or "").strip())
    if not clean_keyword:
        return clean_fragment[:60].strip()
    if not clean_fragment:
        return clean_keyword[:60].strip()
    candidate = f"{clean_keyword} {connector} {clean_fragment}".strip()
    if len(candidate) <= 60:
        return candidate
    shorter_fragment = " ".join(clean_fragment.split()[:2]).strip()
    candidate = f"{clean_keyword} {connector} {shorter_fragment}".strip()
    if len(candidate) <= 60:
        return candidate
    return clean_keyword[:60].strip()


def _normalize_title_with_keyword(title: str, keyword: str) -> str:
    clean_title = _sanitize_refined_title(title)
    clean_keyword = _sanitize_keyword_value(keyword)
    if not clean_keyword:
        return clean_title
    if _contains_keyword(clean_title, clean_keyword):
        return clean_title[:60].strip()
    candidate = f"{clean_keyword}: {clean_title}".strip(" :")
    if len(candidate) <= 60:
        return candidate
    compact = f"{clean_keyword} guide".strip()
    return compact[:60].strip()


def _normalize_description_with_keyword(description: str, keyword: str) -> str:
    clean_description = re.sub(r"\s+", " ", str(description or "").strip())
    clean_description = re.sub(r'\[\s*["\']([^"\']+)["\']\s*\]', r"\1", clean_description).strip()
    clean_keyword = _sanitize_keyword_value(keyword)
    if not clean_keyword:
        return clean_description[:320].strip()
    if _contains_keyword(clean_description, clean_keyword):
        return clean_description[:320].strip()
    prefixed = f"{clean_keyword}: {clean_description}".strip(" :")
    return prefixed[:320].strip()


def _build_default_refinement_options(
    title: str,
    description: str,
    primary_keyword: str,
    decision_focus: str = "",
    angle_question: str = "",
    primary_user_outcome: str = "",
) -> list[dict]:
    base_title = re.sub(r"\s+", " ", str(title or "").strip())
    base_description = re.sub(r"\s+", " ", str(description or "").strip())
    keyword = _sanitize_keyword_value(primary_keyword)

    if not keyword:
        return [{
            "refined_title": base_title[:60].strip(),
            "refined_description": base_description[:320].strip(),
            "rationale": "Original metadata kept because no primary keyword was provided.",
        }]

    context_description = " ".join(
        part for part in [primary_user_outcome, decision_focus, angle_question, base_description] if str(part or "").strip()
    ).strip()
    fragments = _extract_context_fragments(base_title, context_description, keyword, limit=4)
    intro_sentence = _safe_context_string(primary_user_outcome or decision_focus or base_description, 180)
    if intro_sentence and not intro_sentence.endswith("."):
        intro_sentence = intro_sentence.rstrip(" .") + "."

    options: list[dict] = []
    candidates = [
        (
            _normalize_title_with_keyword(base_title, keyword),
            _normalize_description_with_keyword(base_description, keyword),
            "Keyword-forward conservative rewrite.",
        ),
        (
            _fit_keyword_title(keyword, fragments[0] if len(fragments) > 0 else None, "for"),
            _normalize_description_with_keyword(
                f"{intro_sentence} Practical guidance on {keyword} with concrete next-step framing.",
                keyword,
            ),
            "Context-aware rewrite anchored to the original article angle.",
        ),
        (
            _fit_keyword_title(keyword, fragments[1] if len(fragments) > 1 else fragments[0] if fragments else None, "and"),
            _normalize_description_with_keyword(
                f"Decision-focused explanation of {keyword} tied to {fragments[2] if len(fragments) > 2 else 'the original user problem'} and practical trade-offs.",
                keyword,
            ),
            "Fallback variant that preserves the article's problem framing.",
        ),
    ]
    for candidate_title, candidate_description, rationale in candidates:
        options.append({
            "refined_title": candidate_title[:60].strip(),
            "refined_description": candidate_description[:320].strip(),
            "rationale": rationale,
        })

    deduped: list[dict] = []
    seen = set()
    for opt in options:
        key = (
            re.sub(r"\s+", " ", opt["refined_title"].lower()).strip(),
            re.sub(r"\s+", " ", opt["refined_description"].lower()).strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(opt)

    if len(deduped) >= 3:
        return deduped[:3]

    filler_fragments = _extract_context_fragments(base_title, context_description, keyword, limit=6)
    filler_connectors = ["for", "with", "and"]
    for idx, connector in enumerate(filler_connectors):
        fragment = filler_fragments[idx] if idx < len(filler_fragments) else None
        fallback_title = _fit_keyword_title(keyword, fragment, connector)
        fallback_description = _normalize_description_with_keyword(
            f"{keyword} guidance tailored to {fragment or 'the original article intent'} with a clear reader takeaway.",
            keyword,
        )
        key = (
            re.sub(r"\s+", " ", fallback_title.lower()).strip(),
            re.sub(r"\s+", " ", fallback_description.lower()).strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append({
            "refined_title": fallback_title[:60].strip(),
            "refined_description": fallback_description[:320].strip(),
            "rationale": "Guaranteed fallback variant to keep three distinct keyword-aligned choices.",
        })
        if len(deduped) >= 3:
            break

    return deduped[:3]


def _safe_context_string(value, limit: int = 220) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _safe_context_list(value, limit_items: int = 6, item_limit: int = 80) -> list[str]:
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = _safe_context_string(item, item_limit)
        if text:
            cleaned.append(text)
        if len(cleaned) >= limit_items:
            break
    return cleaned


@research_bp.route('/research', methods=['POST'])
@require_api_key
@limiter.limit("10 per minute")
def create_research_task():
    """
    Create a new research task.
    
    Expected JSON body:
    {
        "brief": "Research topic or brief",
        "keywords": "Comma-separated keywords",
        "provider": "LLM provider (e.g., 'openai', 'anthropic')",
        "model": "Model name (e.g., 'gpt-4', 'claude-3.5-sonnet')",
        "api_key": "LLM API key",
        "depth": "Research depth (standard, comprehensive, deep)",
        "tone": "Article tone (academic, journalistic, casual, technical, persuasive)",
        "target_word_count": 2000,
        "claims_research_enabled": true,
        "rag_enabled": true,
        "include_in_text_citations": true,
        "rag_collection": "RAG collection name (optional)",
        "rag_endpoint": "RAG endpoint URL (optional)",
        "rag_llm_provider": "RAG LLM provider (optional)"
    }
    """
    try:
        # Validate content type
        if not request.is_json:
            return jsonify(ErrorResponse(
                error="invalid_content_type",
                message="Content-Type must be application/json",
                error_code="INVALID_CONTENT_TYPE",
                status=400
            ).dict()), 400
        
        # Get and validate JSON data
        data = request.get_json()
        # Normalize minimal-app parameter names to full-app schema
        # - Map llm_model -> provider/model (split on first '/')
        # - Map llm_key   -> api_key
        # - Map rag_collection_name -> rag_collection
        if isinstance(data, dict):
            # Map llm_key to api_key if api_key not provided
            if 'api_key' not in data and 'llm_key' in data and data.get('llm_key'):
                data['api_key'] = data['llm_key']
            
            # Map llm_model to provider/model if either missing
            llm_model = data.get('llm_model')
            provider = data.get('provider')
            model = data.get('model')
            if llm_model and (not provider or not model):
                if isinstance(llm_model, str) and '/' in llm_model:
                    split_provider, split_model = llm_model.split('/', 1)
                    data['provider'] = provider or split_provider
                    data['model'] = model or split_model
                else:
                    # Fallback: assume openai if not specified
                    data['provider'] = provider or 'openai'
                    data['model'] = model or llm_model
            
            # Map rag_collection_name to rag_collection if rag_collection not provided
            if 'rag_collection' not in data and 'rag_collection_name' in data and data.get('rag_collection_name'):
                data['rag_collection'] = data['rag_collection_name']

            # Normalize source strategy only when rollout is enabled or explicitly provided.
            source_strategy_rollout = (
                os.environ.get("SOURCE_STRATEGY_REFACTOR_ENABLED", "false").strip().lower() == "true"
            )
            if source_strategy_rollout or data.get("source_strategy"):
                _normalize_source_strategy_payload(data)
        
        if not data:
            return jsonify(ErrorResponse(
                error="invalid_request",
                message="Request body is required",
                error_code="INVALID_REQUEST",
                status=400
            ).dict()), 400
        
        # Validate request data
        try:
            research_request = ResearchRequest(**data)
        except Exception as e:
            return jsonify(ValidationErrorResponse(
                validation_errors=[{
                    "field": "request_data",
                    "message": str(e)
                }]
            ).dict()), 400
        
        # Import here to avoid circular imports
        # Use the main pipeline tasks (top-level `tasks.py`) to keep full functionality
        from tasks import process_research_task
        
        # Prepare task data - merge validated request with original data to preserve
        # fields not in Pydantic model (e.g., rag_collection_name, use_verbalized_sampling, etc.)
        task_data = research_request.dict()
        # Add additional fields from original data that tasks.py expects
        extra_fields = [
            'rag_collection_name',
            'use_verbalized_sampling',
            'rag_balance_emphasis',
            'draft_title',
            'article_id',
            'source_strategy',
            'seo_primary_keyword',
            'seo_secondary_keywords',
            'writer_notes',
        ]
        for field in extra_fields:
            if field in data and field not in task_data:
                task_data[field] = data[field]
        
        # Resolve the article-generation model in the backend unless a valid explicit
        # provider/model pair is still supplied by a legacy caller.
        provider = str(task_data.get('provider') or '').strip().lower()
        model = str(task_data.get('model') or '').strip()
        final_api_key = task_data.get('api_key')

        if provider and model:
            resolved_key = None
            if not final_api_key or final_api_key == 'development':
                logger.info("Resolving API key from DB for explicit model %s/%s...", provider, model)
                resolved_key = get_llm_api_key(provider, model)
            if resolved_key:
                task_data['api_key'] = resolved_key
            task_data['provider'] = provider
            task_data['model'] = model
        else:
            resolved_provider, resolved_model, resolved_key = get_llm_provider_for_role(LLM_ROLE_ARTICLE_GENERATION)
            if not resolved_provider or not resolved_model:
                return jsonify(ErrorResponse(
                    error="llm_configuration_error",
                    message="No active article_generation LLM is configured in llm_used_for or llm_providers.used_for",
                    error_code="LLM_CONFIGURATION_ERROR",
                    status=500
                ).dict()), 500
            task_data['provider'] = resolved_provider
            task_data['model'] = resolved_model
            if resolved_key:
                task_data['api_key'] = resolved_key

        provider = task_data.get('provider')
        model = task_data.get('model')

        # Create research task
        task = process_research_task.delay(task_data)
        
        # Calculate estimated completion time
        depth_multipliers = {
            "standard": 1,
            "comprehensive": 2.5,
            "deep": 5
        }
        
        base_time_minutes = 5
        # depth is already a string (enum value) due to use_enum_values=True
        depth_str = research_request.depth if isinstance(research_request.depth, str) else research_request.depth.value
        estimated_minutes = base_time_minutes * depth_multipliers.get(depth_str, 1)
        estimated_completion = datetime.utcnow() + timedelta(minutes=estimated_minutes)
        
        # Create response
        response = ResearchResponse(
            research_id=task.id,
            status=ResearchStatus.PENDING,
            brief=research_request.brief,
            model=f"{provider}/{model}",
            depth=research_request.depth,
            tone=research_request.tone,
            target_word_count=research_request.target_word_count,
            created_at=datetime.utcnow(),
            estimated_completion=estimated_completion
        )
        
        logger.info(f"Research task created: {task.id} for {request.remote_addr}")
        
        return jsonify(response.dict()), 202
        
    except Exception as e:
        logger.error(f"Error creating research task: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message="An internal error occurred while processing your request",
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@research_bp.route('/research/refine-metadata', methods=['POST'])
@require_api_key
@limiter.limit("30 per minute")
def refine_research_metadata():
    """
    Generate GEO-refined title/description preview for user approval before full generation.
    """
    try:
        if not request.is_json:
            return jsonify({
                "success": False,
                "message": "Content-Type must be application/json",
            }), 400

        data = request.get_json() or {}
        title = str(data.get("title") or "").strip()
        description = str(data.get("description") or "").strip()
        provider = str(data.get("provider") or "").strip().lower()
        model = str(data.get("model") or "").strip()
        llm_model = str(data.get("llm_model") or "").strip()
        api_key = str(data.get("api_key") or data.get("llm_key") or "").strip()
        primary_keyword = _sanitize_keyword_value(data.get("primary_keyword"))
        secondary_keywords = data.get("secondary_keywords") or []
        if isinstance(secondary_keywords, str):
            secondary_keywords = [k.strip() for k in secondary_keywords.split(",") if k.strip()]
        if not isinstance(secondary_keywords, list):
            secondary_keywords = []
        secondary_keywords = [str(k).strip() for k in secondary_keywords if str(k).strip()][:8]
        domain = str(data.get("domain") or "").strip()
        context = data.get("context") or {}
        if not isinstance(context, dict):
            context = {}

        target_audience = _safe_context_string(context.get("target_audience"), 120)
        article_length = _safe_context_string(context.get("article_length"), 40)
        tone = _safe_context_string(context.get("tone"), 80)
        keyword_intent = _safe_context_string(context.get("keyword_intent"), 80)
        keyword_search_volume = _safe_context_string(context.get("keyword_search_volume"), 40)
        keyword_difficulty = _safe_context_string(context.get("keyword_difficulty"), 40)
        supporting_entities = _safe_context_list(context.get("supporting_entities"), limit_items=8, item_limit=60)
        priority_questions = _safe_context_list(context.get("priority_questions"), limit_items=6, item_limit=120)
        decision_focus = _safe_context_string(context.get("decision_focus"), 220)
        angle_question = _safe_context_string(context.get("angle_question"), 220)
        primary_user_outcome = _safe_context_string(context.get("primary_user_outcome"), 220)
        internal_link_hook = _safe_context_string(context.get("internal_link_hook"), 180)
        affiliate_offer_names = _safe_context_list(context.get("affiliate_offer_names"), limit_items=6, item_limit=80)

        if not title or not description:
            return jsonify({
                "success": False,
                "message": "Both title and description are required.",
            }), 400

        if (not provider or not model) and llm_model:
            if "/" in llm_model:
                split_provider, split_model = llm_model.split("/", 1)
                provider = provider or split_provider.strip().lower()
                model = model or split_model.strip()
            else:
                provider = provider or "openai"
                model = model or llm_model

        if provider and model:
            if not api_key or api_key == "development":
                resolved_key = get_llm_api_key(provider, model)
                if resolved_key:
                    api_key = resolved_key
        else:
            resolved = resolve_llm_provider(task_role=LLM_ROLE_FINAL_REVIEW)
            provider = str(resolved.get("provider") or "").strip().lower()
            model = str(resolved.get("model") or "").strip()
            api_key = str(resolved.get("api_key") or "").strip()

        if not provider or not model:
            return jsonify({
                "success": False,
                "message": "No active final_review LLM is configured in llm_used_for or llm_providers.used_for.",
            }), 500

        if not api_key:
            return jsonify({
                "success": False,
                "message": "Could not resolve API key for the backend-selected review model.",
            }), 500

        llm_client = create_llm_client(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=0.2,
            timeout=45,
            max_retries=1,
            max_tokens=600,
        )

        secondary_part = ", ".join(secondary_keywords) if secondary_keywords else "none"
        context_lines = []
        if target_audience:
            context_lines.append(f"Target audience: {target_audience}")
        if article_length:
            context_lines.append(f"Requested article length: {article_length}")
        if tone:
            context_lines.append(f"Tone: {tone}")
        if keyword_intent:
            metrics_suffix = []
            if keyword_search_volume:
                metrics_suffix.append(f"search volume {keyword_search_volume}")
            if keyword_difficulty:
                metrics_suffix.append(f"difficulty {keyword_difficulty}")
            keyword_line = f"Selected keyword intent: {keyword_intent}"
            if metrics_suffix:
                keyword_line += f" ({', '.join(metrics_suffix)})"
            context_lines.append(keyword_line)
        if decision_focus:
            context_lines.append(f"Decision focus: {decision_focus}")
        if angle_question:
            context_lines.append(f"Angle question: {angle_question}")
        if primary_user_outcome:
            context_lines.append(f"Primary user outcome: {primary_user_outcome}")
        if internal_link_hook:
            context_lines.append(f"Internal link hook: {internal_link_hook}")
        if supporting_entities:
            context_lines.append(f"Supporting entities to preserve when relevant: {', '.join(supporting_entities)}")
        if priority_questions:
            context_lines.append("Priority questions the article should help answer: " + " | ".join(priority_questions))
        if affiliate_offer_names:
            context_lines.append(f"Relevant affiliate/commercial context: {', '.join(affiliate_offer_names)}")
        context_block = "\n".join(context_lines) if context_lines else "No additional article context provided."
        prompt = f"""
You are a GEO + SEO editorial optimizer.
Rewrite title + description to improve AI-search discoverability while preserving original intent and the article's editorial brief.

Rules:
- Keep the title <= 60 characters.
- Keep the description <= 320 characters.
- Prioritize information density and keyword relevance.
- Include primary keyword naturally when provided.
- Keep tone authoritative and data-driven.
- Generate exactly 3 distinct options.
- Treat the original description plus additional article context as the source brief.
- Preserve the core audience, decision frame, and promised outcome unless the keyword requires tighter phrasing.
- Use the angle question, supporting entities, and priority questions to keep the new metadata directionally aligned with the original article.
- Return STRICT JSON only with this schema:
{{
  "options": [
    {{"refined_title":"...","refined_description":"...","rationale":"..."}},
    {{"refined_title":"...","refined_description":"...","rationale":"..."}},
    {{"refined_title":"...","refined_description":"...","rationale":"..."}}
  ],
  "selected_index": 0,
  "rationale": "optional overall note"
}}

Original title: {title}
Original description: {description}
Primary keyword: {primary_keyword or "none"}
Secondary keywords: {secondary_part}
Domain context: {domain or "none"}
Additional article context:
{context_block}
""".strip()

        fallback_reason = ""
        try:
            response = llm_client.generate([
                {
                    "role": "system",
                    "content": "You optimize metadata for GEO. Output strict JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ])
            parsed = _extract_refined_metadata_from_response(response.content, title, description)
            options = _extract_refined_metadata_options_from_response(response.content, title, description)
            if not options:
                options = [{
                    "refined_title": parsed.get("refined_title") or title,
                    "refined_description": parsed.get("refined_description") or description,
                    "rationale": parsed.get("rationale") or "",
                }]

            default_options = _build_default_refinement_options(
                title=title,
                description=description,
                primary_keyword=primary_keyword,
                decision_focus=decision_focus,
                angle_question=angle_question,
                primary_user_outcome=primary_user_outcome,
            )

            normalized_options: list[dict] = []
            seen = set()
            for opt in options + default_options:
                refined_title = _normalize_title_with_keyword(opt.get("refined_title") or title, primary_keyword)
                refined_description = _normalize_description_with_keyword(opt.get("refined_description") or description, primary_keyword)
                key = (
                    re.sub(r"\s+", " ", refined_title.lower()).strip(),
                    re.sub(r"\s+", " ", refined_description.lower()).strip(),
                )
                if key in seen:
                    continue
                seen.add(key)
                normalized_options.append({
                    "refined_title": refined_title,
                    "refined_description": refined_description,
                    "rationale": str(opt.get("rationale") or "").strip(),
                })
                if len(normalized_options) >= 3:
                    break

            options = normalized_options if normalized_options else default_options
            refined_title = str(options[0].get("refined_title") or title).strip()
            refined_description = str(options[0].get("refined_description") or description).strip()
            rationale = parsed.get("rationale") or options[0].get("rationale") or ""
            changed = (refined_title != title) or (refined_description != description)
        except Exception as llm_error:
            logger.warning(
                "Metadata refinement LLM call failed; returning original metadata for manual approval. provider=%s model=%s error=%s",
                provider,
                model,
                llm_error,
            )
            refined_title = title
            refined_description = description
            changed = False
            fallback_reason = (
                "Automatic refinement is temporarily unavailable for this model/key. "
                "Review the original metadata and approve to continue."
            )
            rationale = fallback_reason
            options = _build_default_refinement_options(
                title=refined_title,
                description=refined_description,
                primary_keyword=primary_keyword,
                decision_focus=decision_focus,
                angle_question=angle_question,
                primary_user_outcome=primary_user_outcome,
            )

        return jsonify({
            "success": True,
            "data": {
                "refined_title": refined_title,
                "refined_description": refined_description,
                "rationale": rationale,
                "changed": changed,
                "fallback_used": bool(fallback_reason),
                "options": options,
            }
        }), 200

    except Exception as exc:
        logger.error("Metadata refinement preview failed", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Failed to refine metadata: {str(exc)}",
        }), 500


@research_bp.route('/research/analyze-competitors', methods=['POST'])
@require_api_key
@limiter.limit("30 per minute")
def analyze_competitors():
    """
    Perform a competitor search and LLM analysis synchronously to extract must-haves & competitive edge.
    Saves the results under idea_metadata.competitor_analysis in Supabase.
    """
    try:
        if not request.is_json:
            return jsonify({
                "success": False,
                "message": "Content-Type must be application/json",
            }), 400

        data = request.get_json() or {}
        article_id = str(data.get("article_id") or "").strip()
        primary_keyword = str(data.get("primary_keyword") or "").strip()
        brief = str(data.get("brief") or "").strip()

        if not article_id:
            return jsonify({
                "success": False,
                "message": "article_id is required.",
            }), 400

        from supabase_client import get_supabase_client
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({
                "success": False,
                "message": "Supabase client not available",
            }), 500

        # Fetch the Titles row
        response = supabase.table('Titles').select('*').eq('id', article_id).limit(1).execute()
        if not response.data:
            return jsonify({
                "success": False,
                "message": f"Article title row not found for ID {article_id}",
            }), 404

        title_row = response.data[0]
        
        # Fallbacks
        if not primary_keyword:
            primary_keyword = str(title_row.get('primary_keyword') or title_row.get('search_phrase') or '').strip()
            if not primary_keyword:
                candidates = title_row.get('keyword_candidates_json') or []
                if candidates and isinstance(candidates, list):
                    primary_keyword = candidates[0]
        
        if not primary_keyword:
            return jsonify({
                "success": False,
                "message": "No primary keyword found or provided for this article.",
            }), 400

        if not brief:
            brief = str(title_row.get('userDescription') or '').strip()

        # Resolve LLM Provider
        resolved = resolve_llm_provider(task_role=LLM_ROLE_FINAL_REVIEW)
        provider = str(resolved.get("provider") or "").strip().lower()
        model = str(resolved.get("model") or "").strip()
        api_key = str(resolved.get("api_key") or "").strip()

        if not provider or not model:
            resolved = resolve_llm_provider(task_role=LLM_ROLE_ARTICLE_GENERATION)
            provider = str(resolved.get("provider") or "").strip().lower()
            model = str(resolved.get("model") or "").strip()
            api_key = str(resolved.get("api_key") or "").strip()

        if not provider or not model or not api_key:
            return jsonify({
                "success": False,
                "message": "Could not resolve LLM provider or API key for competitor analysis.",
            }), 500

        llm_client = create_llm_client(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=0.2,
            timeout=60,
        )

        from tasks import run_competitor_analysis_sync
        analysis = run_competitor_analysis_sync(
            primary_keyword=primary_keyword,
            brief=brief,
            llm_client=llm_client
        )

        # Merge competitor analysis into the existing idea_metadata
        existing_metadata = title_row.get('idea_metadata') or {}
        if isinstance(existing_metadata, str):
            try:
                existing_metadata = json.loads(existing_metadata)
            except Exception:
                existing_metadata = {}
        if not isinstance(existing_metadata, dict):
            existing_metadata = {}

        existing_metadata["competitor_analysis"] = analysis

        # Save back to database
        supabase.table('Titles').update({'idea_metadata': existing_metadata}).eq('id', article_id).execute()

        return jsonify({
            "success": True,
            "data": analysis
        }), 200

    except Exception as exc:
        logger.error("Competitor analysis endpoint failed", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Failed to analyze competitors: {str(exc)}",
        }), 500


@research_bp.route('/research/<task_id>', methods=['GET'])
@require_api_key
@limiter.limit("1000 per hour")
def get_research_status(task_id):
    """
    Get the status of a research task.
    
    Args:
        task_id: Research task ID
        
    Returns:
        Task status and progress information
    """
    try:
        # Import here to avoid circular imports
        # Use main pipeline task status from top-level `tasks.py`
        from tasks import get_task_status
        
        # Get task status
        task_status = get_task_status(task_id)
        
        if not task_status:
            return jsonify(ErrorResponse(
                error="task_not_found",
                message="Research task not found",
                error_code="TASK_NOT_FOUND",
                status=404
            ).dict()), 404
        
        # Build response
        response = {
            "task_id": task_id,
            "status": task_status.get("status", "unknown"),
            "progress_percent": task_status.get("progress_percent", 0),
            "current_step": task_status.get("current_step", ""),
            "message": task_status.get("message", ""),
            "stage": task_status.get("stage", ""),
            "eta": task_status.get("eta"),
            "timestamp": datetime.utcnow().isoformat(),
            "info": {
                "progress": task_status.get("progress_percent", 0),
                "message": task_status.get("message", ""),
                "stage": task_status.get("stage", ""),
                "current_step": task_status.get("current_step", "")
            }
        }
        
        # Add result if task is completed
        if task_status.get("status") == "SUCCESS":
            result = task_status.get("result")
            if result:
                response["result"] = result
        
        # Add error if task failed
        if task_status.get("status") == "FAILURE":
            error = task_status.get("error")
            if error:
                response["error"] = error
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message="An error occurred while retrieving task status",
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@research_bp.route('/research/<task_id>/result', methods=['GET'])
@require_api_key
def get_research_result(task_id):
    """
    Get the result of a completed research task.
    
    Args:
        task_id: Research task ID
        
    Returns:
        Research result (article, citations, etc.)
    """
    try:
        # Import here to avoid circular imports
        from tasks import get_task_status
        
        # Get task status
        task_status = get_task_status(task_id)
        
        if not task_status:
            return jsonify(ErrorResponse(
                error="task_not_found",
                message="Research task not found",
                error_code="TASK_NOT_FOUND",
                status=404
            ).dict()), 404
        
        # Check if task is completed
        if task_status.get("status") != "SUCCESS":
            return jsonify(ErrorResponse(
                error="task_not_completed",
                message="Research task is not completed yet",
                error_code="TASK_NOT_COMPLETED",
                status=202
            ).dict()), 202
        
        # Get result
        result = task_status.get("result")
        if not result:
            return jsonify(ErrorResponse(
                error="no_result",
                message="No result available for this task",
                error_code="NO_RESULT",
                status=404
            ).dict()), 404
        
        # Extract final article and format for Noodl compatibility
        final_article = result.get('final_article', {})
        
        # Create response with top-level fields for Noodl
        response_data = {
            'research_id': task_id,
            'status': 'completed',
            'message': 'Task completed successfully',
            'result': final_article,  # Keep nested structure for compatibility
            # Top-level fields for Noodl
            'title': final_article.get('title', ''),
            'hook': final_article.get('hook', ''),
            'excerpt': final_article.get('excerpt', ''),
            'thesis': final_article.get('thesis', ''),
            'content': final_article.get('content', ''),
            'html_content': final_article.get('html_content', ''),
            'html_content_in_text_citations': final_article.get('html_content_in_text_citations', ''),
            'citations': final_article.get('citations', []),
            'sections': final_article.get('sections', []),
            'metadata': final_article.get('metadata', {})
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error getting task result: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message="An error occurred while retrieving task result",
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@research_bp.route('/research/<task_id>/cancel', methods=['POST'])
@require_api_key
def cancel_research_task(task_id):
    """
    Cancel a running research task.
    
    Args:
        task_id: Research task ID
        
    Returns:
        Cancellation confirmation
    """
    try:
        # Import here to avoid circular imports
        from tasks import cancel_task
        
        # Cancel task
        success = cancel_task(task_id)
        
        if not success:
            return jsonify(ErrorResponse(
                error="task_not_found",
                message="Research task not found or already completed",
                error_code="TASK_NOT_FOUND",
                status=404
            ).dict()), 404
        
        return jsonify({
            "task_id": task_id,
            "status": "cancelled",
            "message": "Task has been cancelled successfully",
            "timestamp": datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error cancelling task: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message="An error occurred while cancelling the task",
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@research_bp.route('/research', methods=['GET'])
@require_api_key
def get_research_info():
    """Get information about the research endpoint."""
    return jsonify({
        "endpoint": "/api/v1/research",
        "methods": ["POST"],
        "description": "Create a new research task",
        "endpoints": {
            "create_task": {
                "method": "POST",
                "path": "/api/v1/research",
                "description": "Create a new research task"
            },
            "get_status": {
                "method": "GET",
                "path": "/api/v1/research/{task_id}",
                "description": "Get task status and progress"
            },
            "get_result": {
                "method": "GET",
                "path": "/api/v1/research/{task_id}/result",
                "description": "Get completed task result"
            },
            "cancel_task": {
                "method": "POST",
                "path": "/api/v1/research/{task_id}/cancel",
                "description": "Cancel a running task"
            }
        },
        "request_schema": {
            "brief": "string (required) - Research brief or topic",
            "keywords": "string (required) - Comma-separated keywords",
            "provider": "string (optional) - explicit LLM provider override; otherwise resolved by backend task role",
            "model": "string (optional) - explicit model override; otherwise resolved by backend task role",
            "api_key": "string (optional) - explicit API key override; otherwise resolved from llm_providers/api_keys",
            "depth": "string (optional) - Research depth: standard, comprehensive, deep (default: standard)",
            "tone": "string (optional) - Article tone: academic, journalistic, casual, technical, persuasive (default: journalistic)",
            "target_word_count": "integer (optional) - Target article length in words (default: 2000, range: 500-10000)",
            "claims_research_enabled": "boolean (optional) - Enable claims research (default: true)",
            "rag_enabled": "boolean (optional) - Enable RAG evidence collection (default: true)",
            "include_in_text_citations": "boolean (optional) - Include in-text citation references like [^1], [^2] in the content (default: true)",
            "rag_collection": "string (optional) - RAG collection name",
            "rag_endpoint": "string (optional) - RAG endpoint URL",
            "rag_llm_provider": "string (optional) - RAG LLM provider"
        },
        "authentication": "API Key required (X-API-Key header)",
        "async_processing": True,
        "background_processing": "Tasks are processed asynchronously by Celery workers"
    }), 200
