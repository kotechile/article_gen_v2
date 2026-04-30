"""
Supabase client utility for fetching API keys and configuration.

This module provides functions to interact with Supabase database
to retrieve API keys and other configuration stored in the database.
"""

import os
import logging
from typing import Any, Optional
from supabase import create_client, Client

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:  # pragma: no cover - optional runtime fallback
    psycopg2 = None
    RealDictCursor = None

logger = logging.getLogger(__name__)

# Cache for Supabase client
_supabase_client: Optional[Client] = None

LLM_ROLE_ARTICLE_GENERATION = "article_generation"
LLM_ROLE_DEEP_RESEARCH = "deep_research"
LLM_ROLE_SVG = "svg"
LLM_ROLE_FINAL_REVIEW = "final_review"
LLM_ROLE_TOC = "toc"
LLM_ROLE_RESEARCH = "research"
LLM_ROLE_RESEARCH_TOPIC_GENERATION = "research_topic_generation"
LLM_ROLE_RESEARCH_SUBTOPIC_GENERATION = "research_subtopic_generation"
LLM_ROLE_RESEARCH_IDEA_GENERATION = "research_idea_generation"

_LLM_ROLE_ALIASES = {
    "all_other": LLM_ROLE_ARTICLE_GENERATION,
    "article_generation": LLM_ROLE_ARTICLE_GENERATION,
    "default_generation": LLM_ROLE_ARTICLE_GENERATION,
    "deep_research": LLM_ROLE_DEEP_RESEARCH,
    "deep research": LLM_ROLE_DEEP_RESEARCH,
    "research": LLM_ROLE_RESEARCH,
    "research_topic_generation": LLM_ROLE_RESEARCH_TOPIC_GENERATION,
    "research_subtopic_generation": LLM_ROLE_RESEARCH_SUBTOPIC_GENERATION,
    "research_idea_generation": LLM_ROLE_RESEARCH_IDEA_GENERATION,
    "svg": LLM_ROLE_SVG,
    "final_review": LLM_ROLE_FINAL_REVIEW,
    "final review": LLM_ROLE_FINAL_REVIEW,
    "toc": LLM_ROLE_TOC,
}


def _get_database_url() -> Optional[str]:
    return (
        os.environ.get('DATABASE_URL')
        or os.environ.get('DB_URL')
        or None
    )


def _fetch_rows_via_postgres(query: str, params: tuple = ()) -> list[dict]:
    if psycopg2 is None or RealDictCursor is None:
        return []

    database_url = _get_database_url()
    if not database_url:
        return []

    connection = None
    try:
        connection = psycopg2.connect(database_url, cursor_factory=RealDictCursor)
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall() or []
            return [dict(row) for row in rows]
    except Exception as exc:
        logger.warning("Postgres fallback query failed: %s", exc)
        return []
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def get_supabase_client() -> Optional[Client]:
    """
    Get or create Supabase client instance.
    
    Returns:
        Supabase client instance or None if credentials are not configured
    """
    global _supabase_client
    
    if _supabase_client is not None:
        return _supabase_client
    
    supabase_url = os.environ.get('SUPABASE_URL')
    
    # Prioritize Service Role Key (Bypass RLS) -> Service Key -> Standard Key -> Anon Key
    service_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY') or os.environ.get('SUPABASE_SERVICE_KEY')
    anon_key = os.environ.get('SUPABASE_ANON_KEY')
    common_key = os.environ.get('SUPABASE_KEY')
    
    # Use the most privileged key available
    supabase_key = service_key or common_key or anon_key
    
    auth_type = "Service Role (Bypass RLS)" if service_key else "Standard/Anon (RLS Restricted)"
    
    if not supabase_url or not supabase_key:
        logger.warning("Supabase credentials not found (SUPABASE_URL and SUPABASE_KEY/SUPABASE_SERVICE_KEY required)")
        return None
    
    try:
        # Verify=False patch for self-hosted Supabase with self-signed certs
        import httpx
        original_init = httpx.Client.__init__
        
        def new_init(self, *args, **kwargs):
            kwargs['verify'] = False
            original_init(self, *args, **kwargs)
            
        httpx.Client.__init__ = new_init
        
        _supabase_client = create_client(supabase_url, supabase_key)
        logger.info(f"Supabase client initialized successfully using {auth_type}")
        return _supabase_client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {str(e)}")
        return None


def _normalize_llm_role(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return _LLM_ROLE_ALIASES.get(normalized, normalized)


def _normalize_used_for(raw_value: Any) -> list[str]:
    if isinstance(raw_value, list):
        tokens = raw_value
    elif isinstance(raw_value, str):
        value = raw_value.strip()
        if value.startswith("{") and value.endswith("}"):
            value = value[1:-1]
        tokens = value.split(",")
    else:
        tokens = []

    normalized: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        cleaned_token = str(token or "").strip().strip('"').strip("'").strip("{}")
        role = _normalize_llm_role(cleaned_token)
        if not role or role in seen:
            continue
        normalized.append(role)
        seen.add(role)
    return normalized


def _normalize_model_name(provider: Any, model_name: Any) -> str:
    normalized_provider = str(provider or "").strip().lower()
    normalized_model = str(model_name or "").strip()
    if not normalized_model:
        return ""

    if "deepseek" in normalized_provider:
        compact_model = normalized_model.strip().strip('"').strip("'")
        lowered_model = compact_model.lower().replace("deepdeek", "deepseek")
        if lowered_model in {"deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"}:
            return lowered_model
        return lowered_model

    return normalized_model


def _fetch_api_key_value_by_id(client: Client, api_key_id: Any) -> Optional[str]:
    if not api_key_id:
        return None
    try:
        key_resp = client.table('api_keys').select('key_value').eq('id', api_key_id).limit(1).execute()
        if key_resp.data and len(key_resp.data) > 0:
            api_key = key_resp.data[0].get('key_value')
            if isinstance(api_key, str):
                api_key = api_key.strip().strip('"').strip("'")
            if api_key:
                return api_key
    except Exception as exc:
        logger.warning("Failed to fetch API key by id %s: %s", api_key_id, exc)
        rows = _fetch_rows_via_postgres(
            'SELECT key_value FROM api_keys WHERE id = %s LIMIT 1',
            (str(api_key_id),),
        )
        if rows:
            api_key = rows[0].get('key_value')
            if isinstance(api_key, str):
                api_key = api_key.strip().strip('"').strip("'")
            if api_key:
                return api_key
    return None


def _normalize_llm_provider_row(row: dict) -> Optional[dict]:
    if not isinstance(row, dict):
        return None
    provider_name = str(row.get('provider') or '').strip().lower()
    model_name = _normalize_model_name(provider_name, row.get('model_name'))
    if not model_name:
        return None
    return {
        'id': row.get('id'),
        'name': str(row.get('name') or model_name).strip(),
        'provider': provider_name,
        'model_name': model_name,
        'api_keys_id': row.get('api_keys_id') or row.get('api_key_id'),
        'base_url': row.get('base_url'),
        'is_default': row.get('is_default') if isinstance(row.get('is_default'), bool) else False,
        'is_active': row.get('is_active') if isinstance(row.get('is_active'), bool) else None,
        'used_for': _normalize_used_for(row.get('used_for')),
    }


def _fetch_llm_provider_rows(client: Client) -> list[dict]:
    attempts = [
        ("role-aware", "id,name,provider,model_name,api_keys_id,base_url,is_default,is_active,used_for"),
        ("active-default", "id,name,provider,model_name,api_keys_id,base_url,is_default,is_active"),
        ("legacy-default", "id,name,provider,model_name,api_keys_id,base_url,is_default"),
        ("legacy-core", "id,name,provider,model_name,api_keys_id,base_url"),
    ]

    for label, select_fields in attempts:
        try:
            response = client.table('llm_providers').select(select_fields).execute()
            rows = []
            for raw_row in response.data or []:
                normalized = _normalize_llm_provider_row(raw_row)
                if normalized:
                    rows.append(normalized)
            if rows:
                return rows
        except Exception as exc:
            logger.warning("LLM provider query attempt failed: %s (%s)", label, exc)
    rows = _fetch_rows_via_postgres(
        """
        SELECT
            id::text AS id,
            name,
            provider,
            model_name,
            api_keys_id::text AS api_keys_id,
            base_url,
            is_default,
            is_active,
            used_for
        FROM llm_providers
        """
    )
    normalized_rows = []
    for raw_row in rows:
        normalized = _normalize_llm_provider_row(raw_row)
        if normalized:
            normalized_rows.append(normalized)
    return normalized_rows


def _fetch_llm_role_assignments(client: Client) -> dict[str, str]:
    attempts = [
        ("role-map", "llm_provider_id,used_for"),
        ("role-map-legacy", "llm_provider_id,used_for"),
    ]

    for label, select_fields in attempts:
        try:
            response = client.table('llm_used_for').select(select_fields).execute()
            assignment_map: dict[str, str] = {}
            for row in response.data or []:
                provider_id = str(row.get('llm_provider_id') or '').strip()
                if not provider_id:
                    continue
                normalized_roles = _normalize_used_for(row.get('used_for'))
                if len(normalized_roles) != 1:
                    continue
                assignment_map[normalized_roles[0]] = provider_id
            if assignment_map:
                return assignment_map
        except Exception as exc:
            logger.warning("LLM used_for query attempt failed: %s (%s)", label, exc)
    rows = _fetch_rows_via_postgres(
        """
        SELECT
            llm_provider_id::text AS llm_provider_id,
            used_for
        FROM llm_used_for
        """
    )
    assignment_map: dict[str, str] = {}
    for row in rows:
        provider_id = str(row.get('llm_provider_id') or '').strip()
        if not provider_id:
            continue
        normalized_roles = _normalize_used_for(row.get('used_for'))
        if len(normalized_roles) != 1:
            continue
        assignment_map[normalized_roles[0]] = provider_id
    return assignment_map


def _sort_llm_provider_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            not bool(row.get('is_default')),
            str(row.get('name') or row.get('model_name') or '').lower(),
        ),
    )


def resolve_llm_provider(task_role: Optional[str] = None, provider: Optional[str] = None, model: Optional[str] = None) -> dict:
    """
    Resolve an LLM provider/model/api key combination.

    Priority:
    1. Explicit provider/model if supplied
    2. Active provider mapped to task_role via llm_used_for
    3. Default provider (is_default=true)
    4. First active provider
    """
    client = get_supabase_client()
    if not client:
        return {
            "provider": provider or None,
            "model": model or None,
            "api_key": None,
            "source": "no_supabase",
        }

    rows = _fetch_llm_provider_rows(client)
    if not rows:
        return {
            "provider": provider or None,
            "model": model or None,
            "api_key": None,
            "source": "no_rows",
        }

    role_assignments = _fetch_llm_role_assignments(client)
    if role_assignments:
        for row in rows:
            provider_id = str(row.get('id') or '').strip()
            matched_roles = [role for role, mapped_provider_id in role_assignments.items() if mapped_provider_id == provider_id]
            row['used_for'] = matched_roles

    active_rows = [row for row in rows if row.get('is_active') is not False]
    candidate_rows = active_rows or rows

    explicit_provider = str(provider or "").strip().lower()
    explicit_model = str(model or "").strip()
    explicit_match = None
    if explicit_model:
        for row in candidate_rows:
            if row.get('model_name') != explicit_model:
                continue
            if explicit_provider and row.get('provider') != explicit_provider:
                continue
            explicit_match = row
            break

    role = _normalize_llm_role(task_role)
    fallback_role = None
    if role and role != LLM_ROLE_RESEARCH and role.startswith("research_"):
        fallback_role = LLM_ROLE_RESEARCH
    role_match = None
    if not explicit_match and role:
        if role_assignments and role in role_assignments:
            target_provider_id = role_assignments[role]
            role_match = next(
                (row for row in candidate_rows if str(row.get('id') or '').strip() == target_provider_id),
                None,
            )
        if not role_match:
            role_candidates = [row for row in candidate_rows if role in row.get('used_for', [])]
            sorted_role_candidates = _sort_llm_provider_rows(role_candidates)
            if sorted_role_candidates:
                role_match = sorted_role_candidates[0]

    if not explicit_match and not role_match and fallback_role:
        if role_assignments and fallback_role in role_assignments:
            fallback_provider_id = role_assignments[fallback_role]
            role_match = next(
                (row for row in candidate_rows if str(row.get('id') or '').strip() == fallback_provider_id),
                None,
            )
        if not role_match:
            fallback_candidates = [row for row in candidate_rows if fallback_role in row.get('used_for', [])]
            sorted_fallback_candidates = _sort_llm_provider_rows(fallback_candidates)
            if sorted_fallback_candidates:
                role_match = sorted_fallback_candidates[0]

    default_candidates = [row for row in candidate_rows if row.get('is_default')]
    ordered_candidates: list[tuple[dict, str]] = []

    def _append_candidate(row: Optional[dict], source: str) -> None:
        if not row:
            return
        row_id = str(row.get("id") or "").strip()
        if not row_id:
            return
        if any(str(existing.get("id") or "").strip() == row_id for existing, _ in ordered_candidates):
            return
        ordered_candidates.append((row, source))

    _append_candidate(explicit_match, "explicit")
    _append_candidate(role_match, "task_role")

    for row in _sort_llm_provider_rows(default_candidates):
        _append_candidate(row, "default")
    for row in _sort_llm_provider_rows(candidate_rows):
        _append_candidate(row, "default")

    selected: Optional[dict] = None
    selected_source = "default"
    resolved_key: Optional[str] = None
    for row, source in ordered_candidates:
        key_value = _fetch_api_key_value_by_id(client, row.get('api_keys_id'))
        if key_value:
            selected = row
            selected_source = source
            resolved_key = key_value
            break

    if not selected:
        selected = ordered_candidates[0][0] if ordered_candidates else None
        selected_source = ordered_candidates[0][1] if ordered_candidates else "default"

    if not selected:
        return {
            "provider": provider or None,
            "model": model or None,
            "api_key": None,
            "base_url": None,
            "name": None,
            "used_for": [],
            "is_default": False,
            "source": "no_candidates",
        }

    return {
        "provider": selected.get('provider') or explicit_provider or None,
        "model": selected.get('model_name') or explicit_model or None,
        "api_key": resolved_key,
        "base_url": selected.get('base_url'),
        "name": selected.get('name'),
        "used_for": selected.get('used_for', []),
        "is_default": bool(selected.get('is_default')),
        "source": selected_source,
    }


def get_llm_provider_for_role(task_role: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    resolved = resolve_llm_provider(task_role=task_role)
    return resolved.get("provider"), resolved.get("model"), resolved.get("api_key")


def get_api_key_from_supabase(provider: str) -> Optional[str]:
    """
    Fetch API key from Supabase api_keys table.
    
    Args:
        provider: Provider name (e.g., 'linkup', 'openai', etc.)
        
    Returns:
        API key value or None if not found or error occurred
    """
    try:
        client = get_supabase_client()
        if not client:
            logger.warning(f"Cannot fetch {provider} API key: Supabase client not available")
            return None
        
        # Query Supabase for the API key
        response = client.table('api_keys').select('key_value').eq('provider', provider).execute()
        
        if response.data and len(response.data) > 0:
            api_key = response.data[0].get('key_value')
            if api_key:
                logger.info(f"Successfully fetched {provider} API key from Supabase")
                return api_key
            else:
                logger.warning(f"{provider} API key found in Supabase but key_value is empty")
        
        # Fallback: Try case-insensitive search
        logger.info(f"Exact match failed for {provider}, trying case-insensitive search")
        response = client.table('api_keys').select('key_value').ilike('provider', provider).execute()
        
        if response.data and len(response.data) > 0:
            api_key = response.data[0].get('key_value')
            if api_key:
                logger.info(f"Successfully fetched {provider} API key via case-insensitive search")
                return api_key
        
        logger.warning(f"{provider} API key not found in Supabase api_keys table")
        
        return None
        
    except Exception as e:
        logger.error(f"Error fetching {provider} API key from Supabase: {str(e)}")
        return None


def get_linkup_api_key() -> Optional[str]:
    """
    Get Linkup API key from Supabase api_keys table.
    
    All API keys are stored in Supabase, not in environment variables.
    Only Supabase credentials (SUPABASE_URL, SUPABASE_KEY) should be in .env.
    
    Returns:
        Linkup API key or None if not found
    """
    api_key = get_api_key_from_supabase('linkup')
    if not api_key:
        logger.warning("Linkup API key not found in Supabase api_keys table (provider='linkup')")
    return api_key


def get_api_key(provider: str) -> Optional[str]:
    """
    Generic function to get any API key from Supabase.
    
    All API keys are stored in Supabase api_keys table, not in environment variables.
    Only Supabase credentials (SUPABASE_URL, SUPABASE_KEY) should be in .env.
    
    Args:
        provider: Provider name (e.g., 'linkup', 'openai', 'gemini', 'anthropic', etc.)
        
    Returns:
        API key value or None if not found
    """
    return get_api_key_from_supabase(provider)


def get_llm_api_key(provider: str, model: str) -> Optional[str]:
    """
    Fetch API key for a specific LLM provider and model.
    
    Logic:
    1. Query 'llm_providers' with provider_type and model_name to get api_key (UUID).
    2. Query 'api_keys' with the UUID to get key_value.
    
    Args:
        provider: Provider type (e.g., 'google', 'openai')
        model: Model name (e.g., 'gemini-1.5-pro', 'gpt-4')
        
    Returns:
        API key value or None if not found
    """
    try:
        client = get_supabase_client()
        if not client:
            logger.warning(f"Cannot fetch API key for {provider}/{model}: Supabase client not available")
            return None
        
        # 1. Find the provider record
        # Note: provider_type in DB might match 'provider' arg, valid check required
        # Some mappings might be needed if frontend sends 'openai' but DB has 'OpenAI'
        
        # Try exact match first
        provider_query = client.table('llm_providers').select('api_keys_id').eq('model_name', model)
        # Optional: also filter by provider if duplication exists across providers (rare for models)
        if provider:
            provider_query = provider_query.eq('provider', provider)
            
        provider_resp = provider_query.execute()
        
        if not provider_resp.data or len(provider_resp.data) == 0:
            logger.warning(f"LLM Provider record not found for {provider}/{model}")
            # Fallback: Try identifying by model name only if provider didn't match
            if provider:
                fallback_resp = client.table('llm_providers').select('api_keys_id').eq('model_name', model).execute()
                if fallback_resp.data and len(fallback_resp.data) > 0:
                    provider_resp = fallback_resp
                    logger.info(f"Found LLM record by model name '{model}' ignoring provider '{provider}'")
                else:
                    return None
            else:
                return None
                
        api_key_id = provider_resp.data[0].get('api_keys_id')
        if not api_key_id:
            logger.warning(f"LLM Provider record found for {model} but has no linked api_key")
            return None
            
        # 2. Get the actual key
        key_resp = client.table('api_keys').select('key_value').eq('id', api_key_id).execute()
        
        if key_resp.data and len(key_resp.data) > 0:
            key_value = key_resp.data[0].get('key_value')
            if isinstance(key_value, str):
                key_value = key_value.strip().strip('"').strip("'")
            if key_value:
                return key_value
                
        logger.warning(f"API Key linked to {model} (ID: {api_key_id}) not found or empty")
        return None
        
    except Exception as e:
        logger.error(f"Error fetching LLM API key for {provider}/{model}: {str(e)}")
        return None


def get_default_llm_provider() -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Fetch the default LLM provider and model configuration from Supabase.
    
    Query:
    1. Table 'llm_providers' where is_default = true
    2. Retrieve model_name, provider
    3. Use api_keys_id to fetch the key_value from 'api_keys' table
    
    Returns:
        Tuple of (provider_name, model_name, api_key_value)
        Returns (None, None, None) if not found.
    """
    try:
        resolved = resolve_llm_provider()
        return resolved.get("provider"), resolved.get("model"), resolved.get("api_key")
    except Exception as e:
        logger.error(f"Error fetching default LLM provider: {str(e)}")
        return None, None, None
