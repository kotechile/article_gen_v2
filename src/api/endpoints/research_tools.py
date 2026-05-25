from flask import Blueprint, request, jsonify
import logging
import asyncio
from src.integrations.dataforseo import DataForSEOAPI

logger = logging.getLogger(__name__)

research_tools_bp = Blueprint('research_tools', __name__, url_prefix='/api/research-tools')

# Instantiate the API client
dataforseo_api = DataForSEOAPI()

@research_tools_bp.route('/bulk-metrics', methods=['POST'])
def bulk_metrics():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'invalid_request', 'message': 'JSON body required'}), 400

        keywords = data.get('keywords')
        if not keywords or not isinstance(keywords, list):
            return jsonify({'error': 'validation_error', 'message': 'keywords list is required'}), 400

        # Optional: location & language
        location_code = data.get('location_code', 2840)
        language_code = data.get('language_code', "en")

        result = asyncio.run(
            dataforseo_api.get_bulk_metrics_standard(
                keywords=keywords,
                location_code=location_code,
                language_code=language_code
            )
        )

        return jsonify({'keywords': result or []})

    except Exception as e:
        logger.error(f"Error fetching bulk metrics: {str(e)}", exc_info=True)
        return jsonify({'error': 'internal_error', 'message': str(e)}), 500


@research_tools_bp.route('/website-keywords', methods=['POST'])
def website_keywords():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'invalid_request', 'message': 'JSON body required'}), 400

        domain = data.get('domain')
        if not domain:
            return jsonify({'error': 'validation_error', 'message': 'domain is required'}), 400

        limit = data.get('limit', 100)

        result = asyncio.run(
            dataforseo_api.get_ranked_keywords_live(
                target=domain,
                limit=limit
            )
        )

        return jsonify({'keywords': result or []})

    except Exception as e:
        logger.error(f"Error fetching website keywords: {str(e)}", exc_info=True)
        return jsonify({'error': 'internal_error', 'message': str(e)}), 500


@research_tools_bp.route('/related-keywords', methods=['POST'])
def related_keywords():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'invalid_request', 'message': 'JSON body required'}), 400

        seed_keyword = data.get('seed_keyword')
        if not seed_keyword:
            return jsonify({'error': 'validation_error', 'message': 'seed_keyword is required'}), 400

        limit = data.get('limit', 100)

        result = asyncio.run(
            dataforseo_api.get_keyword_ideas_labs_live(
                keywords=[seed_keyword],
                limit=limit
            )
        )

        return jsonify({'keywords': result or []})

    except Exception as e:
        logger.error(f"Error fetching related keywords: {str(e)}", exc_info=True)
        return jsonify({'error': 'internal_error', 'message': str(e)}), 500
