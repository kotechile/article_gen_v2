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

        # Calculate range
        start = (page - 1) * size
        end = start + size - 1

        # Build query
        query = supabase.table('research_topics').select('*', count='exact')

        if status:
            query = query.eq('status', status)
        
        # Apply sorting
        query = query.order(order_by, desc=(order_direction == 'desc'))
        
        # Apply pagination
        query = query.range(start, end)

        # Execute
        response = query.execute()

        return jsonify({
            "items": response.data,
            "total": response.count,
            "page": page,
            "size": size,
            "has_next": (start + len(response.data)) < response.count,
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
        
        # Prepare data for insertion
        # Ensure we don't try to insert unknown columns if possible, but for now we trust exact match or allow db to error
        
        # Resolve User ID from Authorization Header
        import os
        service_key_env = os.environ.get('SUPABASE_SERVICE_KEY')
        logger.info(f"Checking Env: SUPABASE_SERVICE_KEY Present: {bool(service_key_env)}")
        if service_key_env:
             logger.info(f"Service Key Start: {service_key_env[:10]}...")
        else:
             logger.warning("SUPABASE_SERVICE_KEY is NOT set in environment!")

        auth_header = request.headers.get('Authorization')
        logger.info(f"Received Authorization header: {'Present' if auth_header else 'Missing'}")
        if auth_header:
            logger.info(f"Auth Header Start: {auth_header[:15]}...")
            
        user_id = None
        
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split('Bearer ')[1]
            try:
                logger.info(f"Attempting to validate token: {token[:10]}...")
                user_response = supabase.auth.get_user(token)
                if user_response and user_response.user:
                    user_id = user_response.user.id
                    logger.info(f"Successfully resolved user_id: {user_id}")
                else:
                    logger.warning("Supabase get_user returned no user")
            except Exception as auth_error:
                logger.warning(f"Failed to validate token or get user: {auth_error}")
        else:
             logger.warning("Authorization header missing or invalid format")
        
        # Fallback: Check if user_id is in the body (not recommended for production but keeps backward compatibility if needed)
        if not user_id:
             # Try to get from body
             if request.json and 'user_id' in request.json:
                 user_id = request.json['user_id']
                 
        if not user_id:
             return jsonify(ErrorResponse(
                error="authentication_required", 
                message="Could not resolve a valid user ID. Please ensure you are logged in."
             ).dict()), 401

        insert_data = {
            "title": data.get('title'),
            "description": data.get('description', ''),
            "status": data.get('status', 'active'),
            "updated_at": datetime.utcnow().isoformat(),
            "user_id": user_id
        }
        
        # Use a fresh Service Role client for this operation to ensure RLS bypass
        from supabase import create_client
        import os
        
        sb_url = os.environ.get('SUPABASE_URL')
        sb_key = os.environ.get('SUPABASE_SERVICE_KEY')
        
        response = None
        if sb_url and sb_key:
            # Initialize with verify=False for self-hosted
            import httpx
            original_init = httpx.Client.__init__
            def new_init(self, *args, **kwargs):
                kwargs['verify'] = False
                original_init(self, *args, **kwargs)
            httpx.Client.__init__ = new_init
            
            try:
                supabase_admin = create_client(sb_url, sb_key)
                logger.info("Using dedicated Service Role client for insert")
                response = supabase_admin.table('research_topics').insert(insert_data).execute()
            except Exception as admin_err:
                logger.error(f"Admin insert failed: {admin_err}")
                # Fallback to global client
                response = supabase.table('research_topics').insert(insert_data).execute()
        else:
            logger.warning("Falling back to global client (Service Key missing?)")
            response = supabase.table('research_topics').insert(insert_data).execute()
            
        if not response or not response.data:
            raise Exception("Failed to insert record")

        return jsonify(response.data[0]), 201

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
        response = supabase.table('research_topics').select('*').eq('id', topic_id).single().execute()
        
        if not response.data: # Should throw error from single() usually if not found, but safe check
             return jsonify(ErrorResponse(
                error="not_found",
                message="Research topic not found",
                error_code="NOT_FOUND",
                status=404
            ).dict()), 404

        return jsonify(response.data), 200

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
        
        update_data = {k: v for k, v in data.items() if k in ['title', 'description', 'status']}
        update_data['updated_at'] = datetime.utcnow().isoformat()
        
        response = supabase.table('research_topics').update(update_data).eq('id', topic_id).execute()
        
        if not response.data:
             return jsonify(ErrorResponse(
                error="not_found",
                message="Research topic not found",
                error_code="NOT_FOUND",
                status=404
            ).dict()), 404

        return jsonify(response.data[0]), 200

    except Exception as e:
        logger.error(f"Error updating research topic: {str(e)}", exc_info=True)
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
        response = supabase.table('research_topics').delete().eq('id', topic_id).execute()
        
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
        # Fetch directly from DB if possible
        response = supabase.table('subtopics').select('*').eq('project_id', topic_id).execute()
        
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
        user_id = data.get('user_id')
        topic_id = data.get('topic_id')
        subtopic_name = data.get('subtopic')
        keywords = data.get('keywords', [])
        affiliate_offers = data.get('affiliate_offers', [])

        if not all([user_id, topic_id, subtopic_name]):
            return jsonify(ErrorResponse(
                error="validation_error",
                message="user_id, topic_id, and subtopic are required",
                error_code="VALIDATION_ERROR",
                status=400
            ).dict()), 400

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
