"""
AI Utility Endpoints.

Provides AI-powered helpers that don't fit into the main research pipeline,
such as editorial topic proposals.
"""

import asyncio
import logging
import os
from flask import Blueprint, request, jsonify
from ...api.middleware.auth import require_api_key
from ...services.topic_generation_brief_service import topic_generation_brief_service
from ...services.editorial_topic_generation_service import editorial_topic_generation_service

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
    Generate category-aware editorial topic candidates for a research workflow.

    Request body:
        niche_description (str, optional): Fallback project description.
        project_name (str, optional): Website or project name.
        project_description (str, optional): Website description.
        primary_category (str, optional): Selected primary category name.
        primary_category_description (str, optional): Primary category description.
        secondary_category (str, optional): Selected sub-category name.
        secondary_category_description (str, optional): Secondary category description.
        trend_titles (list[str], optional): Recent trend themes for freshness context.
        count (int, optional): Number of topics to propose.
        generation_mode (str, optional): keyword_first, editorial_first, or mixed.

    Response:
        { "topics": [{ "title": str, "rationale": str, ... }] }
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    niche_description = (data.get('niche_description') or '').strip()
    project_name = (data.get('project_name') or '').strip()
    project_description = (data.get('project_description') or '').strip()
    primary_category = (data.get('primary_category') or '').strip() or None
    primary_category_description = (data.get('primary_category_description') or '').strip() or None
    secondary_category = (data.get('secondary_category') or '').strip() or None
    secondary_category_description = (data.get('secondary_category_description') or '').strip() or None
    trend_titles = data.get('trend_titles') if isinstance(data.get('trend_titles'), list) else []
    count = min(int(data.get('count', 10)), 20)
    generation_mode = (data.get('generation_mode') or 'mixed').strip().lower()
    if generation_mode not in {'keyword_first', 'editorial_first', 'mixed'}:
        generation_mode = 'mixed'

    if not any([niche_description, project_description, project_name]):
        return jsonify({"error": "project_description or niche_description is required"}), 400

    user_id = _get_user_id_from_request()
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    try:
        brief = topic_generation_brief_service.build(
            project={
                "project_name": project_name,
                "project_description": project_description,
            },
            primary_category={
                "name": primary_category,
                "description": primary_category_description,
            },
            secondary_category={
                "name": secondary_category,
                "description": secondary_category_description,
            },
            trend_titles=trend_titles,
            fallback_niche_description=niche_description,
            count=count,
        )
        topics = asyncio.run(editorial_topic_generation_service.generate(brief, generation_mode=generation_mode))
        logger.info(
            "propose-topics: generated count=%s category_path=%r project=%r generation_mode=%s",
            len(topics),
            brief.get("category_path"),
            brief.get("project_name"),
            generation_mode,
        )
        return jsonify({"topics": topics[:count]}), 200
    except Exception as e:
        logger.error("propose-topics: failed err=%s", e, exc_info=True)
        return jsonify({"error": f"Topic generation failed: {str(e)}"}), 500
