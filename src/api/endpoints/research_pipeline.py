from flask import Blueprint, request, jsonify
import logging
import asyncio
from src.services.research_pipeline_service import research_pipeline_service

logger = logging.getLogger(__name__)

research_pipeline_bp = Blueprint('research_pipeline', __name__, url_prefix='/api/research-pipeline')

@research_pipeline_bp.route('/extract', methods=['POST'])
def extract_pipeline():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'invalid_request', 'message': 'JSON body required'}), 400

        query_text = data.get('query_text')
        if not query_text:
            return jsonify({'error': 'validation_error', 'message': 'query_text is required'}), 400

        user_id = data.get('user_id', 'anonymous')

        result = asyncio.run(
            research_pipeline_service.extract_and_persist(
                seed_keyword=query_text,
                user_id=user_id
            )
        )

        return jsonify(result)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Error running extraction pipeline: {str(e)}\n{tb}", exc_info=True)
        return jsonify({'error': 'internal_error', 'message': f"{str(e)}\n{tb}"}), 500

@research_pipeline_bp.route('/cluster', methods=['POST'])
def cluster_pipeline():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'invalid_request', 'message': 'JSON body required'}), 400

        keywords = data.get('keywords')
        if not keywords or not isinstance(keywords, list):
            return jsonify({'error': 'validation_error', 'message': 'keywords list is required'}), 400

        clusters = asyncio.run(
            research_pipeline_service.cluster_detailed_keywords(keywords)
        )

        return jsonify({'clusters': clusters})

    except Exception as e:
        logger.error(f"Error running clustering pipeline: {str(e)}", exc_info=True)
        return jsonify({'error': 'internal_error', 'message': str(e)}), 500

@research_pipeline_bp.route('', methods=['POST'])
def run_full_pipeline():
    # Backwards compatibility
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'invalid_request', 'message': 'JSON body required'}), 400

        query_text = data.get('query_text')
        user_id = data.get('user_id', 'anonymous')

        clusters = asyncio.run(
            research_pipeline_service.run_pipeline(
                seed_keyword=query_text,
                user_id=user_id
            )
        )

        return jsonify({'clusters': clusters})

    except Exception as e:
        logger.error(f"Error running full pipeline: {str(e)}", exc_info=True)
        return jsonify({'error': 'internal_error', 'message': str(e)}), 500
