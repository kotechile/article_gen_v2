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

# ─── Helper: resolve LLM config from Supabase application_settings ───────────

def _get_llm_config():
    """
    Fetch the best available LLM config from the application_settings table.
    Priority: Gemini → OpenAI → Anthropic → Perplexity
    Returns (provider, model, api_key) or raises RuntimeError.
    """
    try:
        from supabase_client import get_supabase_client
    except ImportError:
        import sys
        sys.path.append(os.getcwd())
        from supabase_client import get_supabase_client

    supabase = get_supabase_client()
    res = supabase.table('application_settings').select(
        'geminiKey, geminiModel, openAIKey, openAIModel, perplexityAI_key, perplexityModel, claudeKey'
    ).eq('id', 1).single().execute()

    s = res.data or {}

    if s.get('geminiKey'):
        return ('gemini', s.get('geminiModel') or 'gemini-1.5-flash', s['geminiKey'])
    if s.get('openAIKey'):
        return ('openai', s.get('openAIModel') or 'gpt-4o-mini', s['openAIKey'])
    if s.get('claudeKey'):
        return ('anthropic', 'claude-3-haiku-20240307', s['claudeKey'])
    if s.get('perplexityAI_key'):
        # Perplexity is OpenAI-compatible
        return ('perplexity', s.get('perplexityModel') or 'llama-3.1-sonar-small-128k-online', s['perplexityAI_key'])

    raise RuntimeError(
        "No LLM API key configured. Please add a key in Settings → Content Generation."
    )


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
        count (int, optional): Number of topics to propose (default: 5).

    Response:
        { "topics": [{ "title": str, "rationale": str }] }
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    niche_description = (data.get('niche_description') or '').strip()
    count = min(int(data.get('count', 5)), 10)  # cap at 10

    if not niche_description:
        return jsonify({"error": "niche_description is required"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    # ── Build prompt ──────────────────────────────────────────────────────────
    system_prompt = (
        "You are an expert SEO content strategist. "
        "Your job is to propose broad, evergreen content topics that have high search demand "
        "and are directly aligned with a specific niche. "
        "Each topic should be broad enough to expand into multiple articles, "
        "but specific enough to be relevant to the niche audience."
    )

    user_prompt = f"""I run a content website with the following niche:

NICHE DESCRIPTION:
{niche_description}

Please propose exactly {count} broad SEO content topics for this niche.

For each topic, provide:
1. A clear, SEO-friendly topic title (not a specific article title — think of it as a content pillar)
2. A brief 1–2 sentence rationale explaining why this topic fits the niche and has search demand

Respond ONLY with a valid JSON array in this exact format:
[
  {{
    "title": "Topic Title Here",
    "rationale": "Brief rationale here."
  }}
]

Do not include any text before or after the JSON array."""

    # ── Call LLM — try providers in priority order, skip on auth errors ──────
    PROVIDER_PRIORITY = [
        lambda s: ('gemini',     s.get('geminiModel') or 'gemini-1.5-flash',  s['geminiKey'])      if s.get('geminiKey')         else None,
        lambda s: ('openai',     s.get('openAIModel') or 'gpt-4o-mini',       s['openAIKey'])      if s.get('openAIKey')         else None,
        lambda s: ('anthropic',  'claude-3-haiku-20240307',                   s['claudeKey'])      if s.get('claudeKey')         else None,
        lambda s: ('perplexity', s.get('perplexityModel') or 'llama-3.1-sonar-small-128k-online', s['perplexityAI_key']) if s.get('perplexityAI_key') else None,
    ]

    # Fetch settings once
    try:
        from supabase_client import get_supabase_client as _gsb
    except ImportError:
        import sys as _sys
        _sys.path.append(os.getcwd())
        from supabase_client import get_supabase_client as _gsb

    _sb = _gsb()
    _res = _sb.table('application_settings').select(
        'geminiKey, geminiModel, openAIKey, openAIModel, perplexityAI_key, perplexityModel, claudeKey'
    ).eq('id', 1).single().execute()
    _s = _res.data or {}

    import sys as _sys2
    _sys2.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ))))
    from llm_client_direct import create_llm_client

    raw_content = None
    last_error = None

    for provider_fn in PROVIDER_PRIORITY:
        creds = provider_fn(_s)
        if not creds:
            continue
        provider, model, api_key = creds

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
                # Single attempt — no retries for auth-type errors
                llm = create_llm_client(
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    temperature=0.7,
                    max_tokens=1500,
                    timeout=60,
                    max_retries=0   # Don't retry on first provider; move to next on failure
                )
                raw_content = llm.generate([
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ]).content

            logger.info(f"propose-topics: succeeded with provider={provider} model={model}")
            break  # success — stop trying other providers

        except Exception as e:
            err_str = str(e).lower()
            is_auth_err = any(kw in err_str for kw in [
                'api key', 'api_key', 'invalid_argument', 'expired',
                'authentication', 'unauthorized', 'permission', 'quota'
            ])
            last_error = e
            if is_auth_err:
                logger.warning(f"propose-topics: auth/key error for provider={provider} ({e.__class__.__name__}), trying next provider")
            else:
                logger.error(f"propose-topics: non-auth error for provider={provider}: {e}", exc_info=True)
            continue

    if raw_content is None:
        # All providers failed
        err_msg = str(last_error) if last_error else "Unknown error"
        is_key_issue = last_error and any(kw in str(last_error).lower() for kw in [
            'api key', 'expired', 'invalid_argument', 'unauthorized', 'quota'
        ])
        if is_key_issue:
            return jsonify({
                "error": "Your LLM API key is expired or invalid. Please update it in Settings → Content Generation and try again."
            }), 503

        logger.error(f"propose-topics: all providers failed. Last error: {err_msg}", exc_info=True)
        return jsonify({"error": f"LLM generation failed: {err_msg}"}), 500

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
