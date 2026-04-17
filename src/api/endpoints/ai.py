"""
AI Utility Endpoints.

Provides AI-powered helpers that don't fit into the main research pipeline,
such as niche topic proposals.
"""

import json
import logging
import os
import re
from flask import Blueprint, request, jsonify
from ...api.middleware.auth import require_api_key

logger = logging.getLogger(__name__)

ai_bp = Blueprint('ai', __name__, url_prefix='/api/ai')

def _get_user_id_from_request():
    """Extract and validate user_id from Bearer token."""
    try:
        from supabase_client import get_supabase_client
    except ImportError:
        import sys
        sys.path.append(os.getcwd())
        from supabase_client import get_supabase_client

    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return None

    token = auth_header.split('Bearer ')[1]
    supabase = get_supabase_client()
    try:
        user_response = supabase.auth.get_user(token)
        if user_response and user_response.user:
            return user_response.user.id
    except Exception as e:
        logger.warning(f"Token validation failed: {e}")
    return None


# ─── POST /api/ai/propose-topics ─────────────────────────────────────────────

@ai_bp.route('/propose-topics', methods=['POST'])
@require_api_key
def propose_topics():
    """
    Use an LLM to propose N broad SEO topics for a given niche.

    Request body:
        niche_description (str): Description of the niche/website context.
        primary_category (str, optional): Selected primary category name.
        secondary_category (str, optional): Selected sub-category name.
        count (int, optional): Number of topics to propose (default: 5).

    Response:
        { "topics": [{ "title": str, "rationale": str }] }
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    niche_description = (data.get('niche_description') or '').strip()
    primary_category = (data.get('primary_category') or '').strip() or None
    secondary_category = (data.get('secondary_category') or '').strip() or None
    count = min(int(data.get('count', 5)), 10)  # cap at 10

    if not niche_description:
        return jsonify({"error": "niche_description is required"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    # ── Build prompt ──────────────────────────────────────────────────────────
    system_prompt = (
        "You are an expert content strategist helping brainstorm BROAD SEED TOPICS for a research workflow. "
        "Your job is to propose topical themes/content pillars that can be expanded into many specific articles later. "
        "At this stage, do NOT optimize for competitive head terms or write clickbait titles."
    )

    selected_category_line = None
    if primary_category and secondary_category:
        selected_category_line = f"Selected category: {primary_category} / {secondary_category}"
    elif primary_category:
        selected_category_line = f"Selected category: {primary_category}"

    user_prompt = f"""I run a content website with the following niche:

NICHE DESCRIPTION:
{niche_description}
{selected_category_line or ""}

Please propose exactly {count} BROAD SEED TOPICS for this niche.

For each topic, provide:
1. A short theme/title (NOT an article headline; avoid keyword stuffing)
2. A brief 1–2 sentence rationale explaining why this theme fits the niche and can expand into multiple articles

Respond ONLY with a valid JSON array in this exact format:
[
  {{
    "title": "Topic Title Here",
    "rationale": "Brief rationale here."
  }}
]

Do not include any text before or after the JSON array."""

    # ── Call LLM — ALWAYS use default provider/model/key from DB ─────────────
    # Requirement:
    # 1) llm_providers where is_default=true → provider, model_name, api_keys_id
    # 2) api_keys where id=api_keys_id → key_value
    try:
        from supabase_client import get_default_llm_provider as _get_default_llm_provider
    except ImportError:
        import sys as _sys
        _sys.path.append(os.getcwd())
        from supabase_client import get_default_llm_provider as _get_default_llm_provider

    provider, model, api_key = _get_default_llm_provider()
    if not provider or not model or not api_key:
        return jsonify({
            "error": "No default LLM configured. Set llm_providers.is_default=true and attach api_keys.key_value via llm_providers.api_keys_id."
        }), 503

    import sys as _sys2
    _sys2.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ))))
    from llm_client_direct import create_llm_client

    try:
        if provider == 'perplexity':
            import openai as _oai
            _client = _oai.OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
            _resp = _client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            raw_content = _resp.choices[0].message.content
        else:
            llm = create_llm_client(
                provider=provider,
                model=model,
                api_key=api_key,
                temperature=0.7,
                max_tokens=1500,
                timeout=60,
                max_retries=0
            )
            raw_content = llm.generate([
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ]).content

        logger.info(f"propose-topics: succeeded with provider={provider} model={model}")
    except Exception as e:
        err_str = str(e)
        is_key_issue = any(kw in err_str.lower() for kw in [
            'api key', 'api_key', 'invalid_argument', 'expired',
            'authentication', 'unauthorized', 'permission', 'quota'
        ])
        logger.error(f"propose-topics: failed with provider={provider} model={model}: {e}", exc_info=True)
        if is_key_issue:
            return jsonify({
                "error": "Default LLM API key is invalid/expired. Update api_keys.key_value for the default llm_providers row and retry."
            }), 503
        return jsonify({"error": f"LLM generation failed: {err_str}"}), 500

    # ── Parse LLM response ────────────────────────────────────────────────────
    try:
        # Strip markdown code fences if present
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        topics = json.loads(cleaned)

        # Validate structure
        if not isinstance(topics, list):
            raise ValueError("Expected a JSON array")

        validated = []
        for t in topics:
            if isinstance(t, dict) and t.get('title'):
                validated.append({
                    "title": str(t.get('title', '')).strip(),
                    "rationale": str(t.get('rationale', '')).strip()
                })

        if not validated:
            raise ValueError("No valid topics parsed from LLM response")

        return jsonify({"topics": validated[:count]}), 200

    except (json.JSONDecodeError, ValueError) as parse_err:
        logger.error(f"Failed to parse LLM JSON response: {parse_err}\nRaw: {raw_content[:500]}")
        # Graceful fallback — try to extract titles manually with regex
        fallback_titles = re.findall(r'"title"\s*:\s*"([^"]+)"', raw_content)
        fallback_rationales = re.findall(r'"rationale"\s*:\s*"([^"]+)"', raw_content)
        if fallback_titles:
            fallback = []
            for i, title in enumerate(fallback_titles[:count]):
                fallback.append({
                    "title": title,
                    "rationale": fallback_rationales[i] if i < len(fallback_rationales) else ""
                })
            return jsonify({"topics": fallback}), 200

        return jsonify({
            "error": "LLM returned an unexpected response format. Please try again.",
            "raw_preview": raw_content[:200]
        }), 500
