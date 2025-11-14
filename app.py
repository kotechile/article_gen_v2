#!/usr/bin/env python3
"""
Minimal working Flask app for Content Generator V2.
This version has basic functionality without complex imports.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from datetime import datetime
from celery_config import celery_app
from tasks import process_research_task, get_task_status, cancel_task

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Create rate limiter
limiter = Limiter(
    app,
    default_limits=["1000 per hour", "60 per minute"],
    storage_uri="memory://"
)

# Simple API key check (for testing)
def check_api_key():
    """Simple API key check for testing."""
    api_key = request.headers.get('X-API-Key')
    if not api_key:
        return False
    # For testing, accept any non-empty API key
    return len(api_key) > 0

@app.route('/api/v1/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Content Generator V2 is running',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0'
    })

@app.route('/api/v1/research', methods=['POST'])
@app.route('/api/v1//research', methods=['POST'])  # Handle double slash from Noodl
@limiter.limit("10 per minute")
def create_research_task():
    """
    Create a new research task (minimal version).
    
    Expected JSON body:
    {
        "brief": "Research topic or brief",
        "keywords": "Comma-separated keywords", 
        "draft_title": "Optional draft title to guide article generation and ensure focus",
        "llm_model": "LLM model (e.g., 'openai/gpt-4', 'gemini/gemini-1.5-pro')",
        "llm_key": "LLM API key",
        "depth": "Research depth (standard, comprehensive, deep)",
        "tone": "Article tone (academic, journalistic, casual, technical, persuasive)",
        "target_word_count": 2000,
        "claims_research_enabled": true,  // Optional: Enable web search (default: true)
        "rag_enabled": false,             // Optional: Enable RAG search (default: false)
        "rag_collection": "RAG collection name (optional)",
        "rag_endpoint": "RAG endpoint URL (optional)",
        "rag_balance_emphasis": "auto"    // Optional: RAG emphasis mode (default: auto)
    }
    
    Note: Both claims_research_enabled and rag_enabled are optional flags.
    - claims_research_enabled: Enables web search via Linkup API (requires LINKUP_API_KEY)
    - rag_enabled: Enables RAG search (requires rag_endpoint to be provided)
    - rag_balance_emphasis: Controls RAG search emphasis mode:
      * "news_focused": For current events and recent developments
      * "balanced": For general queries (default)
      * "comprehensive": For detailed research and analysis
      * "auto": Let the system decide (recommended)
    """
    try:
        # Check API key
        if not check_api_key():
            return jsonify({
                'error': 'unauthorized',
                'message': 'API key required'
            }), 401
        
        # Validate content type
        if not request.is_json:
            return jsonify({
                'error': 'invalid_content_type',
                'message': 'Content-Type must be application/json'
            }), 400
        
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'invalid_request',
                'message': 'Request body is required'
            }), 400
        
        # Basic validation
        required_fields = ['brief', 'keywords', 'llm_model', 'llm_key']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': 'validation_error',
                    'message': f'Missing required field: {field}'
                }), 400
        
        # Extract provider and model from llm_model (format: "provider/model")
        llm_model = data.get('llm_model', 'openai/gpt-4')
        if '/' in llm_model:
            provider, model = llm_model.split('/', 1)
        else:
            provider, model = 'openai', llm_model
        
        # Prepare research data for Celery task
        research_data = {
            'brief': data['brief'],
            'keywords': data['keywords'],
            'draft_title': data.get('draft_title'),
            'llm_model': llm_model,
            'provider': provider,
            'model': model,
            'llm_key': data['llm_key'],
            'depth': data.get('depth', 'standard'),
            'tone': data.get('tone', 'journalistic'),
            'target_word_count': data.get('target_word_count', 2000),
            'claims_research_enabled': data.get('claims_research_enabled', True),
            'rag_enabled': data.get('rag_enabled', False),
            # Support both 'rag_collection' and 'rag_collection_name' field names
            # Frontend sends 'rag_collection_name', but API also accepts 'rag_collection'
            'rag_collection': data.get('rag_collection') or data.get('rag_collection_name'),
            'rag_endpoint': data.get('rag_endpoint'),
            'rag_balance_emphasis': data.get('rag_balance_emphasis', 'auto'),
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Submit task to Celery
        task = process_research_task.delay(research_data)
        task_id = task.id
        
        response = {
            'research_id': task_id,
            'status': 'pending',
            'brief': data['brief'],
            'keywords': data['keywords'],
            'llm_model': llm_model,
            'provider': provider,
            'model': model,
            'depth': data.get('depth', 'standard'),
            'tone': data.get('tone', 'journalistic'),
            'target_word_count': data.get('target_word_count', 2000),
            'claims_research_enabled': data.get('claims_research_enabled', True),
            'rag_enabled': data.get('rag_enabled', False),
            'rag_collection': data.get('rag_collection'),
            'rag_endpoint': data.get('rag_endpoint'),
            'rag_balance_emphasis': data.get('rag_balance_emphasis', 'auto'),
            'created_at': datetime.utcnow().isoformat(),
            'message': 'Research task created successfully and queued for processing',
            'debug_info': {
                'received_parameters': list(data.keys()),
                'rag_status': 'enabled' if data.get('rag_enabled') else 'disabled',
                'rag_collection': data.get('rag_collection'),
                'rag_endpoint': data.get('rag_endpoint'),
                'claims_research': 'enabled' if data.get('claims_research_enabled') else 'disabled',
                'celery_task_id': task_id
            }
        }
        
        logger.info(f"Research task created and queued: {task_id}")
        return jsonify(response), 202
        
    except Exception as e:
        logger.error(f"Error creating research task: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'internal_error',
            'message': 'Failed to create research task'
        }), 500

@app.route('/api/v1/research/<task_id>', methods=['GET'])
@app.route('/api/v1//research/<task_id>', methods=['GET'])  # Handle double slash from Noodl
@limiter.limit("1000 per hour")
def get_research_status(task_id):
    """Get research task status."""
    try:
        # Check API key
        if not check_api_key():
            return jsonify({
                'error': 'unauthorized',
                'message': 'API key required'
            }), 401
        
        # Get task status from Celery
        status_info = get_task_status(task_id)
        
        if status_info is None:
            return jsonify({
                'error': 'not_found',
                'message': 'Task not found'
            }), 404
        
        return jsonify(status_info)
        
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'internal_error',
            'message': 'Failed to get task status'
        }), 500

@app.route('/api/v1/research/<task_id>/result', methods=['GET'])
def get_research_result(task_id):
    """Get research result."""
    try:
        # Check API key
        if not check_api_key():
            return jsonify({
                'error': 'unauthorized',
                'message': 'API key required'
            }), 401
        
        # Get task status from Celery
        status_info = get_task_status(task_id)
        
        if status_info is None:
            return jsonify({
                'error': 'not_found',
                'message': 'Task not found'
            }), 404
        
        # Check if task is completed
        if status_info['status'] != 'SUCCESS':
            return jsonify({
                'error': 'not_ready',
                'message': 'Task is not completed yet',
                'status': status_info['status'],
                'progress': status_info.get('progress', 0)
            }), 202
        
        # Return the result
        result = status_info.get('result', {})
        final_article = result.get('final_article', {})
        
        # Extract fields for top-level access (Noodl compatibility)
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
            'meta_description': final_article.get('meta_description', ''),
            'content': final_article.get('content', ''),
            'html_content': final_article.get('html_content', ''),
            'html_content_in_text_citations': final_article.get('html_content_in_text_citations', ''),
            'citations': final_article.get('citations', []),
            'sections': final_article.get('sections', []),
            'metadata': final_article.get('metadata', {}),
            # SEO fields for Titles table
            'seo_title_optimized': final_article.get('seo_title_optimized', ''),
            'metaTitle': final_article.get('metaTitle', ''),
            'metaDescription': final_article.get('metaDescription', ''),
            'seo_meta_desc_optimized': final_article.get('seo_meta_desc_optimized', ''),
            'focus_keyword': final_article.get('focus_keyword', ''),
            'breadcrumb_title': final_article.get('breadcrumb_title', ''),
            # Content fields
            'articleText': final_article.get('articleText', ''),
            'htmlArticle': final_article.get('htmlArticle', ''),
            # WordPress fields
            'wp_slug': final_article.get('wp_slug', ''),
            'wp_tag_ids': final_article.get('wp_tag_ids', []),
            'wp_excerpt_auto_generated': final_article.get('wp_excerpt_auto_generated', ''),
            'wp_custom_fields': final_article.get('wp_custom_fields', {}),
            # Engagement and scoring fields
            'engagement_hooks': final_article.get('engagement_hooks', []),
            'call_to_action_text': final_article.get('call_to_action_text', ''),
            'viral_potential_score': final_article.get('viral_potential_score', 0.0),
            'seo_optimization_score': final_article.get('seo_optimization_score', 0.0),
            'external_links_suggested': final_article.get('external_links_suggested', [])
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error getting task result: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'internal_error',
            'message': 'Failed to get task result'
        }), 500

@app.route('/api/v1/research/<task_id>/cancel', methods=['POST'])
def cancel_research_task(task_id):
    """Cancel research task."""
    try:
        # Check API key
        if not check_api_key():
            return jsonify({
                'error': 'unauthorized',
                'message': 'API key required'
            }), 401
        
        # Cancel task using Celery
        success = cancel_task(task_id)
        
        if success:
            response = {
                'research_id': task_id,
                'status': 'cancelled',
                'message': 'Task cancelled successfully'
            }
        else:
            response = {
                'research_id': task_id,
                'status': 'error',
                'message': 'Failed to cancel task'
            }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error cancelling task: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'internal_error',
            'message': 'Failed to cancel task'
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Root endpoint."""
    return jsonify({
        'message': 'Content Generator V2 API',
        'version': '2.0.0',
        'status': 'running',
        'endpoints': {
            'health': '/api/v1/health',
            'create_research': '/api/v1/research',
            'get_status': '/api/v1/research/{task_id}',
            'get_result': '/api/v1/research/{task_id}/result',
            'cancel_task': '/api/v1/research/{task_id}/cancel'
        }
    })

if __name__ == '__main__':
    print("Starting Content Generator V2 (Minimal Version)...")
    print("API available at: http://localhost:5001")
    print("Health check: http://localhost:5001/api/v1/health")
    app.run(host='0.0.0.0', port=5001, debug=True)
