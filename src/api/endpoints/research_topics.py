"""
Research Topics API endpoints.

This module provides endpoints for managing research topics.
"""

import logging
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
        
        update_data = {k: v for k, v in data.items() if k in ['title', 'description', 'status', 'project_id', 'primary_category_id', 'secondary_category_id', 'topic_source', 'source_topic_id']}
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

            insert_payload.append({
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
            })

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
    """Delete a research topic."""
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
            .delete()
            .eq('id', topic_id)
            .eq('user_id', user_id)
            .execute()
        )
        
        # Note: delete() might return empty list if not found, but we can consider it success (idempotent)
        return jsonify({"message": "Topic deleted successfully"}), 200

    except Exception as e:
        logger.error(f"Error deleting research topic: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message="An error occurred",
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

        # Fetch directly from DB if possible
        response = (
            supabase
            .table('subtopics')
            .select('*')
            .eq('research_topic_id', topic_id)
            .eq('user_id', user_id)
            .execute()
        )
        
        return jsonify({
            "items": response.data,
            "total": len(response.data)
        }), 200

    except Exception as e:
        logger.error(f"Error getting subtopics: {str(e)}", exc_info=True)
        # Return empty list instead of 500 to avoid breaking UI if table missing
        return jsonify({"items": [], "total": 0}), 200

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
        topic_res = supabase.table('research_topics').select('title, user_id').eq('id', topic_id).single().execute()

        if not topic_res.data:
            return jsonify(ErrorResponse(
                error="not_found",
                message="Research topic not found",
                error_code="NOT_FOUND",
                status=404
            ).dict()), 404

        topic_title = topic_res.data['title']
        user_id     = topic_res.data['user_id']
        if user_id != request_user_id:
            return jsonify(ErrorResponse(
                error="forbidden",
                message="You do not have access to this research topic",
                error_code="FORBIDDEN",
                status=403
            ).dict()), 403

        # 2. Run the async decomposition pipeline synchronously
        async def _run():
            result = await enhanced_decomposition_service.decompose_topic_enhanced(
                query=topic_title,
                user_id=user_id,
                max_subtopics=12
            )

            if not result.get("success"):
                raise Exception(result.get("message", "Decomposition failed"))

            enhanced_subtopics_data = result.get("subtopics", [])
            saved_subtopics = []

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
                }
                logger.info(f"DEBUG trend_data: {trend_data}")

                saved = await subtopics_service.create(
                    research_topic_id=topic_id,
                    name=sub_data.get("title"),
                    user_id=user_id,
                    trend_data=trend_data,
                )
                if saved:
                    logger.info(f"DEBUG saved subtopic: {saved.get('name')}, vol={saved.get('search_volume')}, cpc={saved.get('cpc')}")
                    saved_subtopics.append(saved)

            return saved_subtopics, result

        saved_subtopics, result = asyncio.run(_run())

        return jsonify({
            "items": saved_subtopics,
            "total": len(saved_subtopics),
            "meta": {
                "processing_time":    result.get("processing_time"),
                "enhancement_methods": result.get("enhancement_methods"),
            }
        }), 200

    except Exception as e:
        logger.error(f"Error generating subtopics: {e}", exc_info=True)
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
        keywords = data.get('keywords', [])
        affiliate_offers = data.get('affiliate_offers', [])

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

        logger.info(f"Generating idea burst for subtopic: {subtopic_name}")

        # Generate ideas using LLM
        import asyncio
        from src.services.llm.llm_service import llm_service

        async def generate_ideas():
            # Generate blog ideas
            blog_prompt = f"""
You are a content strategist specializing in SEO-optimized blog content.

Current Year: 2026
Subtopic: {subtopic_name}
Keywords: {', '.join(keywords[:10])}
Affiliate Categories: {', '.join(affiliate_offers[:5]) if affiliate_offers else 'General'}

Generate 5 blog article ideas that:
1. Target specific long-tail keywords
2. Have clear search intent (informational, commercial, or transactional)
3. Include monetization opportunities
4. Are specific and actionable (not generic)

For each idea, provide:
- Title: A compelling, SEO-optimized title
- Description: 1-2 sentences describing the angle
- Primary Keywords: 2-3 main keywords to target
- Monetization Hook: How to monetize (affiliate product, service, etc.)
- Estimated Metrics: Search volume (low/medium/high), Difficulty (1-100), Viability (1-100)

Output format (use exactly this format):
BLOG_IDEA: [number]
TITLE: [title]
DESCRIPTION: [description]
KEYWORDS: [keyword1, keyword2, keyword3]
MONETIZATION: [monetization approach]
VOLUME: [estimated monthly searches as number]
DIFFICULTY: [SEO difficulty 1-100]
VIABILITY: [overall viability score 1-100]
END_IDEA

Generate 5 blog ideas following this format.
"""

            # Generate software/commercial ideas
            software_prompt = f"""
You are a product strategist specializing in identifying software tools and features to BUILD (not review).

Current Year: 2026
Subtopic: {subtopic_name}
Keywords: {', '.join(keywords[:10])}
Affiliate Categories: {', '.join(affiliate_offers[:5]) if affiliate_offers else 'General'}

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
- Title: Name of the tool/feature to build (e.g., "RSU Tax Calculator", "Portfolio Rebalancing Tool")
- Description: What the tool does and how users interact with it
- Primary Keywords: Keywords people would search to find this tool
- Monetization Hook: How to monetize the tool (lead gen, freemium, affiliate integration, etc.)
- Estimated Metrics: Search volume, Difficulty, Viability

Output format (use exactly this format):
SOFTWARE_IDEA: [number]
TITLE: [tool name - NOT a review article title]
DESCRIPTION: [what the tool does and user interaction]
KEYWORDS: [keyword1, keyword2, keyword3]
MONETIZATION: [how to monetize the tool]
VOLUME: [estimated monthly searches as number]
DIFFICULTY: [SEO difficulty 1-100]
VIABILITY: [overall viability score 1-100]
END_IDEA

Generate 3 software tools/features to BUILD following this format.
"""

            # Generate both in parallel
            blog_response = await llm_service.generate_text(blog_prompt, max_tokens=2000)
            software_response = await llm_service.generate_text(software_prompt, max_tokens=1500)

            return blog_response.content, software_response.content

        blog_text, software_text = asyncio.run(generate_ideas())

        # Parse the responses
        blog_ideas = parse_idea_response(blog_text, 'blog', topic_id, user_id, subtopic_name)
        software_ideas = parse_idea_response(software_text, 'software', topic_id, user_id, subtopic_name)

        return jsonify({
            "success": True,
            "blog_ideas": [idea.to_dict() if hasattr(idea, 'to_dict') else idea for idea in blog_ideas],
            "software_ideas": [idea.to_dict() if hasattr(idea, 'to_dict') else idea for idea in software_ideas]
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
            current_idea['title'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('DESCRIPTION:'):
            current_idea['description'] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('KEYWORDS:'):
            kw_text = line.split(':', 1)[1].strip()
            current_idea['keywords'] = [k.strip() for k in kw_text.split(',') if k.strip()]
        elif line.upper().startswith('MONETIZATION:'):
            current_idea['monetization_hook'] = line.split(':', 1)[1].strip()
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

    return {
        "id": idea_data.get('id', str(uuid4())),
        "title": idea_data.get('title', 'Untitled Idea'),
        "content_type": content_type,
        "description": idea_data.get('description', ''),
        "primary_keywords": idea_data.get('keywords', []),
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
        "viability_score": idea_data.get('viability_score', 50),
        "trend_score": 0,
        "monetization_score": 0,
        "seo_ease_score": 0,
        "status": "draft"
    }
