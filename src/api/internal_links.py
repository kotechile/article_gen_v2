from flask import Blueprint, jsonify, request
from supabase_client import get_supabase_client, get_api_key
from llm_client import create_llm_client
import logging
import json

internal_links_bp = Blueprint('internal_links', __name__)
logger = logging.getLogger(__name__)

@internal_links_bp.route('/api/internal-links/suggest', methods=['POST'])
def suggest_internal_links():
    """
    Suggest internal links for the given content using LLM.
    Expected JSON body:
    {
        "content": "Article text...",
        "user_id": "user_uuid"
    }
    """
    try:
        data = request.get_json()
        content = data.get('content')
        user_id = data.get('user_id')

        if not content or not user_id:
            return jsonify({'error': 'Missing content or user_id'}), 400

        # Limit content length to avoid token limits (approx 2000 words)
        content_snippet = content[:10000]

        # 1. Fetch imported posts for candidates
        supabase = get_supabase_client()
        response = supabase.table("wordpress_imported_posts").select("id, title, link").eq("user_id", user_id).execute()
        posts = response.data
        
        if not posts:
             return jsonify({'matches': []}), 200

        # Prepare candidates list for prompt
        candidates = [{"id": p['id'], "title": p['title'], "link": p['link']} for p in posts]
        candidates_str = json.dumps([{"title": c["title"], "link": c["link"]} for c in candidates], indent=2)

        # 2. Initialize LLM Client
        # Fetch default provider from DB as per user request
        api_key = None
        provider = 'openai'
        model = 'gpt-4o-mini'
        
        try:
            # Check for default provider
            # Note: is_default might be boolean or string depending on DB, assume boolean first
            provider_res = supabase.table("llm_providers").select("provider, model_name, api_keys_id").eq("is_default", True).execute()
            if provider_res.data and len(provider_res.data) > 0:
                default_config = provider_res.data[0]
                db_provider = default_config.get('provider')
                db_model = default_config.get('model_name')
                api_keys_id = default_config.get('api_keys_id')
                
                if db_provider and db_model and api_keys_id:
                    # Fetch key using api_keys_id
                    # We query by 'id' assuming api_keys table has an id column or api_keys_id refers to the PK
                    key_res = supabase.table("api_keys").select("key_value").eq("id", api_keys_id).execute()
                    if key_res.data and len(key_res.data) > 0:
                        api_key = key_res.data[0].get('key_value')
                        provider = db_provider
                        model = db_model
                        logger.info(f"Using default LLM provider from DB: {provider}/{model}")

        except Exception as db_err:
            logger.warning(f"Failed to fetch default LLM provider from DB: {db_err}")

        # Fallback if DB lookup failed
        if not api_key:
             logger.info("Falling back to hardcoded API key check")
             api_key = get_api_key('openai')
             provider = 'openai'
             model = 'gpt-4o-mini' 
             
             if not api_key:
                 api_key = get_api_key('gemini')
                 provider = 'gemini'
                 model = 'gemini-1.5-flash'
        
        if not api_key:
             return jsonify({'error': 'No configured LLM API key found'}), 500

        llm = create_llm_client(provider=provider, model=model, api_key=api_key)

        # 3. Construct Prompt
        prompt = f"""
        You are an SEO expert. Your task is to identify phrases in the provided Input Content that can be hyperlinked to the provided Candidate Articles.
        
        Rules:
        1. Find semantically relevant matches. The phrase in the text doesn't need to match the Title exactly, but must refer to the same concept.
        2. Identify the EXACT substring in the Input Content that should be linked.
        3. Do not link generic words like "the", "article", "this". Link meaningful phrases.
        4. Select top 1-3 best matches.
        5. Return ONLY a JSON array of objects. No markdown formatting.
        
        Output Format:
        [
            {{
                "matched_text": "text substring from input",
                "link": "url from candidate",
                "title": "title from candidate",
                "relevance_score": 0.9
            }}
        ]

        Input Content:
        {content_snippet}

        Candidate Articles:
        {candidates_str}
        """

        messages = [
            {"role": "system", "content": "You are a helpful SEO assistant that suggests internal links."},
            {"role": "user", "content": prompt}
        ]

        # 4. Generate Response
        llm_response = llm.generate(messages)
        content_response = llm_response.content
        
        # Clean response if it contains markdown code blocks
        if "```json" in content_response:
             content_response = content_response.split("```json")[1].split("```")[0].strip()
        elif "```" in content_response:
             content_response = content_response.split("```")[1].split("```")[0].strip()

        try:
            matches = json.loads(content_response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response: {content_response}")
            return jsonify({'error': 'Failed to parse suggestions'}), 500

        return jsonify({'matches': matches}), 200

    except Exception as e:
        logger.error(f"Error suggesting internal links: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
