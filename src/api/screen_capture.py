from flask import Blueprint, request, jsonify, Response
import logging
from src.services.screen_capture import ScreenCaptureService

screen_capture_bp = Blueprint('screen_capture', __name__)
logger = logging.getLogger(__name__)
service = ScreenCaptureService()

@screen_capture_bp.route('/api/v1/generate-image', methods=['POST'])
def generate_image():
    """
    Generate an image from HTML/CSS.
    Expected JSON:
    {
        "html": "<div>...</div>",
        "css": ".class { ... }",
        "width": 1920,
        "height": 1080,
        "clip": {"x": 0, "y": 0, "width": 100, "height": 100}  // Optional
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request', 'message': 'JSON body required'}), 400

        html = data.get('html')
        css = data.get('css', '')
        width = data.get('width', 1920)
        height = data.get('height', 1080)
        clip = data.get('clip')

        if not html:
            return jsonify({'error': 'Validation error', 'message': 'HTML content is required'}), 400

        image_bytes = service.generate_screenshot(html, css, int(width), int(height), clip)

        return Response(image_bytes, mimetype='image/png')

    except Exception as e:
        logger.error(f"Generate image failed: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal error', 'message': str(e)}), 500
