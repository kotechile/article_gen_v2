"""
Research Topics API endpoints.

This module provides endpoints for managing research topics.
"""

import logging
from datetime import datetime
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
        
        # Fallback: Check if user_id is in the body (not recommended for production but keeps backward compatibility if needed)
        if not user_id:
             user_id = data.get('user_id')

        if not user_id:
             return jsonify(ErrorResponse(
                error="authentication_required",
                message="Could not resolve a valid user ID. Please ensure you are logged in and your token is valid.",
                error_code="USER_ID_REQUIRED",
                status=401
            ).dict()), 401

        insert_data = {
            "title": data.get('title'),
            "description": data.get('description', ''),
            "status": data.get('status', 'active'),
            "updated_at": datetime.utcnow().isoformat(),
            "user_id": user_id
            # created_at is usually auto-generated
        }
        
        response = supabase.table('research_topics').insert(insert_data).execute()
        
        if not response.data:
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
async def generate_subtopics(topic_id):
    """
    Generate subtopics for a research topic using the Enhanced Decomposition Pipeline.
    
    This uses the 6-phase process:
    1. Semantic Explosion (LLM)
    2. Bulk Data Retrieval (DataForSEO)
    3. Profitability Mathematical Filtering
    4. SEO Difficulty Enrichment
    5. Semantic Clustering
    6. Multi-dimensional Verification
    """
    try:
        supabase = get_supabase_client()
        
        # 1. Get Topic Title and User ID
        # We need the user_id to correctly attribute the subtopics and check permissions
        topic_res = supabase.table('research_topics').select('title, user_id').eq('id', topic_id).single().execute()
        
        if not topic_res.data:
            return jsonify(ErrorResponse(
                error="not_found",
                message="Research topic not found",
                error_code="NOT_FOUND",
                status=404
            ).dict()), 404
            
        topic_title = topic_res.data['title']
        user_id = topic_res.data['user_id']
        
        # 2. Trigger Enhanced Decomposition Service
        # This is an async call that performs the heavy lifting
        result = await enhanced_decomposition_service.decompose_topic_enhanced(
            query=topic_title,
            user_id=user_id,
            max_subtopics=12  # Generate a good number of options
        )
        
        if not result.get("success"):
            raise Exception(result.get("message", "Decomposition failed"))
            
        enhanced_subtopics_data = result.get("subtopics", [])
        
        # 3. Save Verified Subtopics to Database
        saved_subtopics = []
        
        for sub_data in enhanced_subtopics_data:
            # Map EnhancedSubtopic dictionary to what SubtopicsService expects
            # The enhanced service returns a dict representation of EnhancedSubtopic
            
            # Construct trend_data for rich persistence
            trend_data = {
                "trend_score": 80,  # Default high score for verified items, or extract if available
                "seo_difficulty": sub_data.get("keyword_difficulty", 50),
                "search_volume": sub_data.get("search_volume", 0),
                "cpc": sub_data.get("cpc", 0.0),
                "keywords": sub_data.get("seed_keywords", []),
                "rationale": sub_data.get("rationale"),
                "target_audience": sub_data.get("target_audience"),
                "trend_analysis": sub_data.get("trend_analysis"),
                "monetization": sub_data.get("monetization_data")
            }
            
            # Create subtopic record
            saved_subtopic = await subtopics_service.create(
                research_topic_id=topic_id,
                name=sub_data.get("title"),
                user_id=user_id,
                trend_data=trend_data
            )
            
            if saved_subtopic:
                saved_subtopics.append(saved_subtopic)
                
        # 4. Return results (Mapped to frontend expectations)
        # Frontend expects { items: Subtopic[], total: number }
        return jsonify({
            "items": saved_subtopics,
            "total": len(saved_subtopics),
            "meta": {
                "processing_time": result.get("processing_time"),
                "enhancement_methods": result.get("enhancement_methods")
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

