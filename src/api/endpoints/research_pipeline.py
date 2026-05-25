from flask import Blueprint, request, jsonify
import logging
from src.services.research_pipeline_service import research_pipeline_service

logger = logging.getLogger(__name__)

research_pipeline_bp = Blueprint('research_pipeline', __name__, url_prefix='/api/research-pipeline')

import asyncio

@research_pipeline_bp.route('', methods=['POST'])
def run_pipeline():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'invalid_request', 'message': 'JSON body required'}), 400

        query_text = data.get('query_text')
        if not query_text:
            return jsonify({'error': 'validation_error', 'message': 'query_text is required'}), 400

        user_id = data.get('user_id', 'anonymous')

        clusters = asyncio.run(
            research_pipeline_service.run_pipeline(
                seed_keyword=query_text,
                user_id=user_id
            )
        )

        return jsonify({'clusters': clusters})

    except Exception as e:
        logger.error(f"Error running research pipeline: {str(e)}", exc_info=True)
        return jsonify({'error': 'internal_error', 'message': str(e)}), 500
