"""
Research API endpoints for Content Generator V2.

This module provides the main research endpoints for creating,
monitoring, and retrieving research tasks.
"""

import logging
import json
import os
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
# Import tasks when needed to avoid circular imports


logger = logging.getLogger(__name__)

_SOURCE_STRATEGIES = {
    "dossier_only",
    "dossier_plus_rag",
    "dossier_plus_rag_plus_live_web",
    "rag_only",
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

    if requested_strategy in _SOURCE_STRATEGIES:
        strategy = requested_strategy
    else:
        if rag_enabled and claims_enabled:
            strategy = "dossier_plus_rag_plus_live_web"
        elif rag_enabled:
            strategy = "dossier_plus_rag"
        else:
            strategy = "dossier_only"

    # Keep strategy as normalized canonical field.
    data["source_strategy"] = strategy

    # Keep legacy booleans aligned to avoid regressions in old code paths.
    if strategy == "dossier_only":
        data["rag_enabled"] = False
        data["claims_research_enabled"] = False
    elif strategy == "dossier_plus_rag":
        data["rag_enabled"] = True
        data["claims_research_enabled"] = False
    elif strategy == "dossier_plus_rag_plus_live_web":
        data["rag_enabled"] = True
        data["claims_research_enabled"] = True
    elif strategy == "rag_only":
        data["rag_enabled"] = True
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
        return {
            "refined_title": str(parsed.get("refined_title") or fallback_title).strip(),
            "refined_description": str(parsed.get("refined_description") or fallback_description).strip(),
            "rationale": str(parsed.get("rationale") or "").strip(),
        }
    except Exception:
        pass

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
        ]
        for field in extra_fields:
            if field in data and field not in task_data:
                task_data[field] = data[field]
        
        # Helper to resolve API Key from DB if not provided or dummy
        final_api_key = task_data.get('api_key')
        
        # Check if we need to resolve the key from DB
        # Conditions: Key is missing, OR Key is 'development' (frontend fallback)
        if not final_api_key or final_api_key == 'development':
            from supabase_client import get_llm_api_key
            
            provider = task_data.get('provider')
            model = task_data.get('model')
            
            logger.info(f"Resolving API key from DB for {provider}/{model}...")
            db_key = get_llm_api_key(provider, model)
            
            if db_key:
                task_data['api_key'] = db_key
                logger.info("Successfully resolved API key from database")
            elif final_api_key == 'development':
                 # If we have a dummy key and failed to resolve, warn but proceed
                 # (Task might fail later if key is truly required by LLM client)
                 logger.warning("Could not resolve real API key from DB, using 'development' placeholder")
            else:
                 # No key at all and failed to resolve
                 logger.warning("No API key provided and failed to resolve from DB")
        
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
        primary_keyword = str(data.get("primary_keyword") or "").strip()
        secondary_keywords = data.get("secondary_keywords") or []
        if isinstance(secondary_keywords, str):
            secondary_keywords = [k.strip() for k in secondary_keywords.split(",") if k.strip()]
        if not isinstance(secondary_keywords, list):
            secondary_keywords = []
        secondary_keywords = [str(k).strip() for k in secondary_keywords if str(k).strip()][:8]
        domain = str(data.get("domain") or "").strip()

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

        if not provider or not model:
            return jsonify({
                "success": False,
                "message": "LLM provider/model is required.",
            }), 400

        if not api_key or api_key == "development":
            from supabase_client import get_llm_api_key
            resolved_key = get_llm_api_key(provider, model)
            if resolved_key:
                api_key = resolved_key

        if not api_key:
            return jsonify({
                "success": False,
                "message": "Could not resolve API key for selected model.",
            }), 400

        llm_client = create_llm_client(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=0.2,
            timeout=45,
            max_retries=1,
            max_tokens=450,
        )

        secondary_part = ", ".join(secondary_keywords) if secondary_keywords else "none"
        prompt = f"""
You are a GEO + SEO editorial optimizer.
Rewrite title + description to improve AI-search discoverability while preserving original intent.

Rules:
- Keep the title <= 60 characters.
- Keep the description <= 320 characters.
- Prioritize information density and keyword relevance.
- Include primary keyword naturally when provided.
- Keep tone authoritative and data-driven.
- Return STRICT JSON only with keys: refined_title, refined_description, rationale.

Original title: {title}
Original description: {description}
Primary keyword: {primary_keyword or "none"}
Secondary keywords: {secondary_part}
Domain context: {domain or "none"}
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
            refined_title = parsed.get("refined_title") or title
            refined_description = parsed.get("refined_description") or description
            rationale = parsed.get("rationale") or ""
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

        return jsonify({
            "success": True,
            "data": {
                "refined_title": refined_title,
                "refined_description": refined_description,
                "rationale": rationale,
                "changed": changed,
                "fallback_used": bool(fallback_reason),
            }
        }), 200

    except Exception as exc:
        logger.error("Metadata refinement preview failed", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Failed to refine metadata: {str(exc)}",
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
