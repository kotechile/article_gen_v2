"""
Research API endpoints for Content Generator V2.

This module provides the main research endpoints for creating,
monitoring, and retrieving research tasks.
"""

import logging
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
# Import tasks when needed to avoid circular imports


logger = logging.getLogger(__name__)

# Create blueprint
research_bp = Blueprint('research', __name__, url_prefix='/api/v1')

# Create rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1000 per hour", "60 per minute"],
    storage_uri="memory://"
)


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
        extra_fields = ['rag_collection_name', 'use_verbalized_sampling', 'rag_balance_emphasis', 'draft_title']
        for field in extra_fields:
            if field in data and field not in task_data:
                task_data[field] = data[field]
        
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
            model=f"{research_request.provider}/{research_request.model}",
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
            "timestamp": datetime.utcnow().isoformat()
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
            "provider": "string (required) - LLM provider (e.g., 'openai', 'anthropic')",
            "model": "string (required) - Model name (e.g., 'gpt-4', 'claude-3.5-sonnet')",
            "api_key": "string (required) - LLM API key",
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
