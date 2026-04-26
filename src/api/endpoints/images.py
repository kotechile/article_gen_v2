"""
Image API endpoints for Content Generator V2.

This module provides endpoints for image generation, stock image search,
upload, and infographic generation.
"""

import logging
import os
import io
import base64
import asyncio
import re
import html
import requests
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename

from supabase_client import get_supabase_client, get_api_key, resolve_llm_provider, LLM_ROLE_SVG
from ...core.models.errors import ErrorResponse, ValidationErrorResponse
from ...services.llm.providers import get_provider_class

logger = logging.getLogger(__name__)

# Create blueprint
images_bp = Blueprint('images', __name__, url_prefix='/api/v1/images')

# Create rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1000 per hour", "60 per minute"],
    storage_uri="memory://"
)


def _query_llm_provider_rows(client, filters, require_active=True):
    query = client.table('llm_providers').select('*')
    for field, value in filters:
        query = query.eq(field, value)

    if require_active:
        query = query.eq('is_active', True)

    try:
        return query.execute()
    except Exception as exc:
        message = str(exc)
        if require_active and "llm_providers.is_active" in message and "does not exist" in message:
            fallback_query = client.table('llm_providers').select('*')
            for field, value in filters:
                fallback_query = fallback_query.eq(field, value)
            return fallback_query.execute()
        raise


def _resolve_infographic_llm(client, llm_model=None):
    raw_model = (llm_model or '').strip()
    provider_hint = None
    model_hint = raw_model

    if '/' in raw_model:
        provider_hint, model_hint = [part.strip() for part in raw_model.split('/', 1)]

    provider_row = None

    if model_hint:
        candidate_results = []
        filters = [('model_name', model_hint)]
        if provider_hint:
            filters.insert(0, ('provider', provider_hint))
        candidate_results.append(_query_llm_provider_rows(client, filters, require_active=True))

        if not candidate_results[-1].data:
            try:
                candidate_results.append(_query_llm_provider_rows(client, [('name', raw_model)], require_active=True))
            except Exception as exc:
                if "llm_providers.name" not in str(exc):
                    raise

        if not candidate_results[-1].data and provider_hint:
            candidate_results.append(_query_llm_provider_rows(client, [('provider', provider_hint)], require_active=True))

        for result in candidate_results:
            if result.data:
                provider_row = result.data[0]
                break

    api_key = None
    base_url = None

    if not provider_row:
        resolved = resolve_llm_provider(task_role=LLM_ROLE_SVG)
        provider_name = str(resolved.get('provider') or '').strip().lower()
        model_name = str(resolved.get('model') or '').strip()
        api_key = resolved.get('api_key')
        if not provider_name or not model_name:
            raise ValueError("No active svg LLM is configured in llm_providers.used_for")
    else:
        key_id = provider_row.get('api_keys_id') or provider_row.get('api_key_id')
        base_url = provider_row.get('base_url')

        if key_id:
            key_result = client.table('api_keys').select('*').eq('id', key_id).execute()
            if key_result.data:
                key_row = key_result.data[0]
                api_key = key_row.get('key_value')
                if not base_url:
                    base_url = key_row.get('base_url')

        provider_name = (provider_row.get('provider') or provider_row.get('provider_name') or 'google').strip().lower()
        model_name = (provider_row.get('model_name') or model_hint).strip()

    if not api_key:
        api_key = current_app.config.get('LITELLM_API_KEY')

    if not api_key:
        raise ValueError("Selected LLM provider is missing an API key")

    if not model_name:
        raise ValueError("No SVG model is configured in llm_providers")

    return {
        'api_key': api_key,
        'base_url': base_url,
        'model_name': model_name,
        'provider_name': provider_name,
    }


def _extract_svg_markup(content):
    if not content:
        return ""

    cleaned = str(content).strip()
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r'^```(?:svg|xml)?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned)

    match = re.search(r'(<svg[\s\S]*?</svg>)', cleaned, flags=re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
    else:
        escaped_match = re.search(r'(&lt;svg[\s\S]*?&lt;/svg&gt;)', str(content), flags=re.IGNORECASE)
        if escaped_match:
            cleaned = html.unescape(escaped_match.group(1)).strip()

    if cleaned.lower().startswith('<?xml'):
        xml_end = cleaned.find('?>')
        if xml_end != -1:
            cleaned = cleaned[xml_end + 2:].strip()

    return cleaned


def _build_svg_infographic_prompt(text, accent_color, text_color, secondary_color=None, neutral_color=None):
    secondary = secondary_color or accent_color
    neutral = neutral_color or "#94a3b8"
    return f"""System instruction:
Role: Senior SVG Architect & Creative Coder.

Objective:
Transform the input text into a high-end, custom-illustrated SVG infographic.
Analyze the provided text, extract the most important data points, steps, tradeoffs, or takeaways, and visualize them with original geometry rather than generic UI cards.

Output contract:
- Return ONLY raw <svg> code.
- Do not use markdown fences.
- Do not include any explanation before or after the SVG.
- The output must be exactly one <svg>...</svg> document.
- Ensure all major shapes, panels, icons, and data markers are filled with color.
- Do not return a wireframe, grayscale mockup, or outline-only composition.

Visual design rules:
- Use a modern, vibrant SaaS-style visual language with rich color, depth, contrast, and polish.
- Favor bold editorial compositions, sculptural geometry, layered forms, and custom vector illustration.
- Use viewBox="0 0 800 450" and ensure the SVG is responsive with width="100%" and height="auto".
- Keep the background transparent overall; do not draw a flat full-canvas solid background rectangle. It is fine to use soft gradient glows, tinted panels, and translucent cards.
- Use clean, bold sans-serif typography with strong contrast and a clear headline hierarchy.
- Ensure all text is large enough and high-contrast enough to be legible inside an article body on desktop and mobile.
- Never place light text on a light card or dark text on a dark card. Maintain strong contrast in every section.
- The main title and subtitle must always sit on a clearly contrasting area and remain immediately readable at first glance.
- Do not place headline text over blobs, gradients, or illustrations unless you add a solid/tinted backing shape or another clear contrast layer behind the text.
- Do not use low-opacity text for primary information. Headings, subtitles, key metrics, and labels must be fully legible foreground elements, not decorative background text.
- If a background shape passes behind text, either move the text, recolor the text, or add a dedicated text container so contrast stays strong.
- Use this palette consistently:
  - Primary text: {text_color}
  - Accent: {accent_color}
  - Secondary accent: {secondary}
  - Neutral/supporting color: {neutral}
- Expand that base palette into a vivid full-color system. Add 1 or 2 complementary highlight hues when helpful, such as electric blue, indigo, coral, amber, teal, or magenta.
- Avoid dull monochrome results, washed-out navy slabs, or near-identical cards. The infographic should feel colorful and intentionally art-directed.
- Use saturated fills, tinted surfaces, contrast-rich callouts, colored badges, and vivid section differentiation.
- Include soft gradients using <linearGradient> where helpful for cards, highlights, or backgrounds.
- Include <radialGradient> where helpful for focal glows, atmosphere, or dimensional depth.
- Include subtle depth using <filter> effects such as soft drop shadows or glows.
- Use colorful vector icons, custom geometry, or simple geometric illustrations built from SVG primitives to represent the data points visually.
- Apply layering and overlapping translucent shapes to create a sense of depth and three-dimensional space.

Geometric requirements (strict):
- Do NOT default to standard rectangles, rounded rectangles, or square cards as the main composition.
- Use organic, fluid shapes and custom vector paths.
- Include at least one organic blob background shape using a complex Bezier-curve <path d="M...">.
- Use path variety: combine blobs, arcs, wedges, ribbons, rings, tapered bands, cutaway panels, or irregular containers.
- Instead of generic cards, represent ideas using isometric 3D shapes, sculptural callouts, hand-drawn vector outlines, or content-relevant custom illustrated modules.
- Connect ideas with curved lines, tapered lines, orbital arcs, or S-curves rather than straight divider lines.
- Avoid any layout that reads like three standard boxes placed side by side.

- Choose the infographic structure that best matches the source text. Valid structures include:
  - hero stat plus supporting callouts
  - comparison matrix
  - timeline or step flow
  - decision framework
  - split-panel pros/cons or tradeoff analysis
  - dashboard with 2 to 4 distinct modules
  - custom illustrated spatial composition
  - card-based layout only as a fallback when the content genuinely cannot support a more expressive structure
- Do not default to three equal-width cards. Use asymmetry, nesting, varied module sizes, curved sections, radial arrangements, or a non-card layout unless the content strongly calls for a simple comparison.
- Vary the composition between generations when possible so outputs do not feel templated.
- Create clear hierarchy with a concise title, strong focal area, and only as many sections as the content genuinely needs.
- Favor balanced spacing, alignment, and visual clarity over decoration.

Layout safety rules:
- Build the composition on a disciplined grid with generous padding and gutters.
- Reserve clear zones for title/subtitle, content modules, and footer/callout areas so sections never overlap.
- Every card, panel, or chart module must have consistent internal padding before any text begins.
- Do not let text overflow outside a card, panel, badge, label, or boundary.
- Keep decorative background geometry away from headline text unless contrast is explicitly protected.
- Wrap all multi-line copy intentionally using separate <tspan> lines instead of a single long <text> line.
- Keep titles short enough to fit cleanly. Section titles should be at most 2 lines.
- Keep body copy brief and summarized. Supporting copy should be at most 2 to 3 short lines per section.
- If the source text is too long, summarize harder rather than shrinking text to an unreadable size.
- Use shorter phrases, not sentences copied verbatim from the source paragraph.
- Avoid any text collisions, clipping, or cross-module overlap.
- Before finalizing, perform a contrast check mentally: if any text risks blending into its immediate background, adjust the fill colors or add a backing layer.
- Before finalizing, verify that all labels fit within the 800x450 canvas with comfortable margins.

Technical rules:
- Avoid scripts, animation, foreignObject, external assets, and embedded raster images.
- Keep the SVG self-contained.
- Prefer straightforward shapes, labels, data markers, and simple icons built from SVG primitives.
- Make the SVG production-ready and visually complete without relying on external CSS.
- Use explicit x/y positioning and tspans for text layout instead of relying on automatic wrapping.
- All styles must be inline or within a <style> block inside the SVG.

Text to transform:
\"\"\"{text.strip()}\"\"\"
"""


def upload_to_supabase_storage(file_data: bytes, filename: str, user_id: str, content_type: str = 'image/jpeg') -> str:
    """
    Upload image to Supabase storage.
    
    Args:
        file_data: Image file bytes
        filename: Target filename
        user_id: User's Supabase auth ID
        content_type: MIME type of the image
        
    Returns:
        Public URL of uploaded image
    """
    try:
        client = get_supabase_client()
        if not client:
            raise Exception("Supabase client not available")
        
        # Construct path: articleImages/user_id/filename
        storage_path = f"articleImages/{user_id}/{secure_filename(filename)}"
        
        # Upload to User Files bucket
        response = client.storage.from_('User Files').upload(
            path=storage_path,
            file=file_data,
            file_options={"content-type": content_type}
        )
        
        # Get public URL
        public_url = client.storage.from_('User Files').get_public_url(storage_path)
        
        logger.info(f"Uploaded image to Supabase: {storage_path}")
        return public_url
        
    except Exception as e:
        logger.error(f"Error uploading to Supabase storage: {str(e)}")
        raise


def generate_stable_diffusion_image(prompt: str, api_key: str, aspect_ratio: str = "1:1", 
                                   model: str = "sd3", reference_image: bytes = None, 
                                   strength: float = 0.7) -> bytes:
    """Generate image using Stable Diffusion API."""
    try:
        url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
        
        # Build form data
        files = {}
        data = {
            "prompt": prompt,
            "output_format": "jpeg",
            "model": model,
            "negative_prompt": "blurry, pixelated, bad anatomy, poor lighting, dullness, ugly, boring"
        }
        
        # Add aspect ratio or reference image
        if reference_image:
            files['image'] = ('reference.jpg', io.BytesIO(reference_image), 'image/jpeg')
            data['mode'] = 'image-to-image'
            data['strength'] = strength
        else:
            data['aspect_ratio'] = aspect_ratio
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Accept': 'image/*'
        }
        
        response = requests.post(url, headers=headers, data=data, files=files)
        response.raise_for_status()
        
        return response.content
        
    except Exception as e:
        logger.error(f"Stable Diffusion API error: {str(e)}")
        raise


def generate_google_imagen(prompt: str, api_key: str, model: str = "imagen-4.0-generate-001", 
                           aspect_ratio: str = "1:1") -> bytes:
    """Generate image using Google Imagen API."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateImages"
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        body = {
            "prompt": prompt,
            "config": {
                "number_of_images": 1,
                "aspect_ratio": aspect_ratio
            }
        }
        
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('generated_images') and len(data['generated_images']) > 0:
            image_bytes_b64 = data['generated_images'][0]['image']['image_bytes']
            return base64.b64decode(image_bytes_b64)
        else:
            raise Exception("No image generated in response")
            
    except Exception as e:
        logger.error(f"Google Imagen API error: {str(e)}")
        raise


def generate_flux_image(prompt: str, api_key: str, model: str = "flux-kontext-pro", 
                       aspect_ratio: str = "1:1") -> bytes:
    """Generate image using Flux API with polling."""
    try:
        # Start generation
        generate_url = 'https://api.fluxapi.ai/api/v1/flux/kontext/generate'
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        body = {
            "prompt": prompt,
            "enableTranslation": True,
            "aspectRatio": aspect_ratio,
            "outputFormat": "jpeg",
            "model": model
        }
        
        response = requests.post(generate_url, headers=headers, json=body)
        response.raise_for_status()
        
        initial_data = response.json()
        logger.info(f"Flux API initial response: {initial_data}")
        
        # Defensive check for taskId in different possible structures
        data_obj = initial_data.get('data')
        if data_obj is None:
            data_obj = {}
            
        task_id = data_obj.get('taskId') or initial_data.get('taskId')
        
        if not task_id:
            logger.error(f"No task ID in Flux API response: {initial_data}")
            raise Exception(f"No task ID in Flux API response: {initial_data}")
        
        # Poll for completion
        polling_url = f"https://api.fluxapi.ai/api/v1/flux/kontext/record-info?taskId={task_id}"
        
        import time
        max_attempts = 150  # 5 minutes max
        for _ in range(max_attempts):
            time.sleep(2)
            poll_resp = requests.get(polling_url, headers={'Authorization': f'Bearer {api_key}'})
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            
            success_flag = poll_data.get('data', {}).get('successFlag')
            
            if success_flag == 1:  # Success
                image_url = poll_data.get('data', {}).get('response', {}).get('resultImageUrl')
                if not image_url:
                    raise Exception("No image URL in completed Flux task")
                
                img_resp = requests.get(image_url)
                img_resp.raise_for_status()
                return img_resp.content
            elif success_flag in [2, 3]:  # Failed
                raise Exception(f"Flux generation failed with code {success_flag}")
        
        raise Exception("Flux generation timed out")
        
    except Exception as e:
        logger.error(f"Flux API error: {str(e)}")
        raise


@images_bp.route('/generate-ai', methods=['POST'])
@limiter.limit("20 per minute")
def generate_ai_image():
    """
    Generate an AI image using configured providers.
    
    Expected JSON body:
    {
        "prompt": "Image description",
        "model": "model_technical_name",
        "aspectRatio": "16:9",
        "referenceImage": "base64_encoded_image" (optional)
    }
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
        if not data:
            return jsonify(ErrorResponse(
                error="invalid_request",
                message="Request body is empty",
                error_code="INVALID_REQUEST",
                status=400
            ).dict()), 400
            
        prompt = data.get('prompt')
        model = data.get('model')
        aspect_ratio = data.get('aspectRatio', '1:1')
        reference_image_b64 = data.get('referenceImage')
        user_id = data.get('user_id')  # Should come from auth middleware
        
        if not prompt or not model or not user_id:
            return jsonify(ErrorResponse(
                error="missing_parameters",
                message="prompt, model, and user_id are required",
                error_code="MISSING_PARAMETERS",
                status=400
            ).dict()), 400
        
        # Get model info from llm_providers_image table
        client = get_supabase_client()
        if not client:
            return jsonify(ErrorResponse(
                error="database_error",
                message="Database connection failed",
                error_code="DATABASE_ERROR",
                status=500
            ).dict()), 500
        
        # Strip potential whitespace from model name (handles cases like '\nkontext')
        search_model = model.strip()
        logger.info(f"Searching for model: '{search_model}' (original: '{model}')")
        
        # 1. Fetch model info
        # We try exact match first, then stripped match if needed
        model_query = client.table('llm_providers_image')\
            .select('*')\
            .eq('model_name', model)\
            .execute()
            
        if not model_query.data and search_model != model:
            model_query = client.table('llm_providers_image')\
                .select('*')\
                .eq('model_name', search_model)\
                .execute()
        
        if not model_query.data:
            # Try one more: search by display_name if model_name failed
            model_query = client.table('llm_providers_image')\
                .select('*')\
                .eq('display_name', model)\
                .execute()

        if not model_query.data:
            logger.error(f"Model not found in DB: {model}")
            return jsonify(ErrorResponse(
                error="model_not_found",
                message=f"Model '{model}' not found in database",
                error_code="MODEL_NOT_FOUND",
                status=404
            ).dict()), 404
        
        row = model_query.data[0]
        if not row:
            logger.error(f"Model {model} found in DB but row is empty")
            return jsonify(ErrorResponse(
                error="invalid_model_config",
                message=f"Model '{model}' configuration is invalid",
                error_code="INVALID_MODEL_CONFIG",
                status=500
            ).dict()), 500
            
        model_name_actual = row.get('model_name')
        provider = row.get('provider', '').lower()
        api_keys_id = row.get('api_keys_id')
        
        logger.info(f"Found model row: id={row.get('id')}, provider={provider}, api_keys_id={api_keys_id}")
        
        if not api_keys_id:
            logger.error(f"Model {model} has no API key linked in llm_providers_image table")
            return jsonify(ErrorResponse(
                error="api_key_missing",
                message="This model is not correctly configured (missing API key link)",
                error_code="API_KEY_MISSING",
                status=500
            ).dict()), 500
            
        # 2. Fetch the actual API key value
        key_query = client.table('api_keys')\
            .select('key_value')\
            .eq('id', api_keys_id)\
            .execute()
            
        if not key_query.data:
            logger.error(f"API key ID {api_keys_id} not found in api_keys table")
            return jsonify(ErrorResponse(
                error="api_key_not_found",
                message="Linked API key not found in configuration",
                error_code="API_KEY_NOT_FOUND",
                status=500
            ).dict()), 500
            
        key_row = key_query.data[0]
        if not key_row:
            logger.error(f"API key record for ID {api_keys_id} is null")
            return jsonify(ErrorResponse(
                error="api_key_not_found",
                message="Linked API key is null or missing",
                error_code="API_KEY_NOT_FOUND",
                status=500
            ).dict()), 500
            
        api_key = key_row.get('key_value')
        
        if not api_key:
            logger.error(f"API key ID {api_keys_id} found but key_value is empty")
            return jsonify(ErrorResponse(
                error="api_key_empty",
                message="API key is empty. Please check database configuration.",
                error_code="API_KEY_EMPTY",
                status=500
            ).dict()), 500
        
        logger.info(f"Successfully retrieved API key (len={len(api_key)}) for provider {provider}")
        
        # Use the possibly stripped model name for the actual API call
        model_to_use = model_name_actual.strip() if model_name_actual else search_model
        
        # Parse reference image if provided
        reference_image = None
        if reference_image_b64:
            reference_image = base64.b64decode(reference_image_b64)
        
        # Generate image based on provider
        image_data = None
        if 'stable' in provider or 'stability' in provider:
            image_data = generate_stable_diffusion_image(
                prompt, api_key, aspect_ratio, model_to_use, reference_image
            )
        elif 'google' in provider or 'imagen' in provider:
            image_data = generate_google_imagen(prompt, api_key, model_to_use, aspect_ratio)
        elif 'flux' in provider:
            image_data = generate_flux_image(prompt, api_key, model_to_use, aspect_ratio)
        else:
            return jsonify(ErrorResponse(
                error="unsupported_provider",
                message=f"Provider {provider} not supported",
                error_code="UNSUPPORTED_PROVIDER",
                status=400
            ).dict()), 400
        
        # Upload to Supabase
        filename = f"ai_{datetime.utcnow().timestamp()}.jpg"
        image_url = upload_to_supabase_storage(image_data, filename, user_id)
        
        # Prepare metadata
        metadata = {
            "ImageUrl": image_url,
            "ImageAuthor": f"AI - {model}",
            "MediaAltText": prompt[:200],  # Use prompt as alt text
            "mediaTitle": prompt[:100],
            "mediaCaption": ""
        }
        
        return jsonify({
            "imageUrl": image_url,
            "metadata": metadata
        }), 200
        
    except Exception as e:
        logger.error(f"Error generating AI image: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@images_bp.route('/stock-search', methods=['GET'])
@limiter.limit("30 per minute")
def search_stock_images():
    """
    Search stock images from Pexels or Unsplash.
    
    Query params:
        provider: 'pexels' or 'unsplash'
        query: search term
        page: page number (default: 1)
        perPage: results per page (default: 10)
    """
    try:
        provider = request.args.get('provider', 'unsplash').lower()
        query = request.args.get('query', '')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('perPage', 10))
        
        if not query:
            return jsonify(ErrorResponse(
                error="missing_query",
                message="query parameter is required",
                error_code="MISSING_QUERY",
                status=400
            ).dict()), 400
        
        # Get API key from database
        api_key = get_api_key(provider)
        if not api_key:
            return jsonify(ErrorResponse(
                error="api_key_not_found",
                message=f"API key for {provider} not found",
                error_code="API_KEY_NOT_FOUND",
                status=500
            ).dict()), 500
        
        if provider == 'unsplash':
            url = f"https://api.unsplash.com/search/photos?client_id={api_key}&query={query}&page={page}&per_page={per_page}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            images = [{
                "id": img['id'],
                "url": img['urls']['regular'],
                "thumbnail": img['urls']['thumb'],
                "author": f"{img['user'].get('first_name', '')} {img['user'].get('last_name', '')}".strip(),
                "description": img.get('description', ''),
                "downloadUrl": img['links']['download']
            } for img in data['results']]
            
            return jsonify({
                "images": images,
                "totalPages": data['total_pages']
            }), 200
            
        elif provider == 'pexels':
            url = f"https://api.pexels.com/v1/search?query={query}&page={page}&per_page={per_page}"
            headers = {'Authorization': api_key}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            images = [{
                "id": img['id'],
                "url": img['src']['large'],
                "thumbnail": img['src']['small'],
                "author": img['photographer'],
                "description": img.get('alt', ''),
                "downloadUrl": img['src']['original']
            } for img in data['photos']]
            
            return jsonify({
                "images": images,
                "totalPages": data['total_results'] // per_page + 1
            }), 200
        else:
            return jsonify(ErrorResponse(
                error="unsupported_provider",
                message=f"Provider {provider} not supported",
                error_code="UNSUPPORTED_PROVIDER",
                status=400
            ).dict()), 400
            
    except Exception as e:
        logger.error(f"Error searching stock images: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500



@images_bp.route('/download-stock', methods=['POST'])
@limiter.limit("20 per minute")
def download_stock_image():
    """
    Download stock image server-side (to bypass CORS) and upload to Supabase.
    
    Expected JSON body:
    {
        "url": "http://...",
        "user_id": "uuid"
    }
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
        image_url = data.get('url')
        user_id = data.get('user_id')
        
        if not image_url or not user_id:
            return jsonify(ErrorResponse(
                error="missing_parameters",
                message="url and user_id are required",
                error_code="MISSING_PARAMETERS",
                status=400
            ).dict()), 400
            
        # Download image content
        # Note: Some stock APIs restrict User-Agent, so we might need a generic one
        headers = {'User-Agent': 'ContentGenerator/2.0'}
        response = requests.get(image_url, headers=headers, stream=True)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', 'image/jpeg')
        file_data = response.content
        
        # Generate filename from URL or timestamp
        # Stock URLs often don't have nice filenames, so we generate one
        ext = 'jpg'
        if 'png' in content_type: ext = 'png'
        elif 'webp' in content_type: ext = 'webp'
        
        filename = f"stock_{datetime.utcnow().timestamp()}.{ext}"
        
        # Upload to Supabase
        storage_url = upload_to_supabase_storage(file_data, filename, user_id, content_type)
        
        return jsonify({"imageUrl": storage_url}), 200
        
    except Exception as e:
        logger.error(f"Error downloading stock image: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="download_failed",
            message=str(e),
            error_code="DOWNLOAD_FAILED",
            status=500
        ).dict()), 500

@images_bp.route('/upload', methods=['POST'])
@limiter.limit("30 per minute")
def upload_image():
    """
    Upload image from local file.
    
    Expects multipart/form-data with 'image' file and 'user_id' field.
    """
    try:
        if 'image' not in request.files:
            return jsonify(ErrorResponse(
                error="no_file",
                message="No image file provided",
                error_code="NO_FILE",
                status=400
            ).dict()), 400
        
        file = request.files['image']
        user_id = request.form.get('user_id')
        
        if not user_id:
            return jsonify(ErrorResponse(
                error="missing_user_id",
                message="user_id is required",
                error_code="MISSING_USER_ID",
                status=400
            ).dict()), 400
        
        if file.filename == '':
            return jsonify(ErrorResponse(
                error="empty_filename",
                message="No file selected",
                error_code="EMPTY_FILENAME",
                status=400
            ).dict()), 400
        
        # Read file data
        file_data = file.read()
        filename = f"upload_{datetime.utcnow().timestamp()}_{file.filename}"
        
        # Upload to Supabase
        image_url = upload_to_supabase_storage(
            file_data, 
            filename, 
            user_id,
            file.content_type or 'image/jpeg'
        )
        
        return jsonify({"imageUrl": image_url}), 200
        
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@images_bp.route('/infographic/templates', methods=['GET'])
def get_infographic_templates():
    """Get list of available infographic templates."""
    try:
        client = get_supabase_client()
        if not client:
            return jsonify(ErrorResponse(
                error="database_error",
                message="Database connection failed",
                error_code="DATABASE_ERROR",
                status=500
            ).dict()), 500
        
        templates = client.table('infographic')\
            .select('*')\
            .execute()
        
        return jsonify({"templates": templates.data}), 200
        
    except Exception as e:
        logger.error(f"Error fetching infographic templates: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@images_bp.route('/infographic/generate-svg', methods=['POST'])
@limiter.limit("10 per minute")
def generate_infographic_svg():
    """Generate raw SVG infographic markup from selected article text."""
    try:
        if not request.is_json:
            return jsonify(ErrorResponse(
                error="invalid_content_type",
                message="Content-Type must be application/json",
                error_code="INVALID_CONTENT_TYPE",
                status=400
            ).dict()), 400

        data = request.get_json() or {}
        text = (data.get('text') or '').strip()
        user_id = data.get('user_id')
        llm_model = data.get('llmModel')
        theme = data.get('theme') or {}
        accent_color = (theme.get('accent') or '#3b82f6').strip()
        text_color = (theme.get('text') or '#1e293b').strip()
        secondary_color = (theme.get('secondary') or '').strip() or None
        neutral_color = (theme.get('neutral') or '').strip() or None

        if not text or not user_id:
            return jsonify(ErrorResponse(
                error="missing_parameters",
                message="text and user_id are required",
                error_code="MISSING_PARAMETERS",
                status=400
            ).dict()), 400

        client = get_supabase_client()
        if not client:
            return jsonify(ErrorResponse(
                error="database_error",
                message="Database connection failed",
                error_code="DATABASE_ERROR",
                status=500
            ).dict()), 500

        selected_llm_config = _resolve_infographic_llm(client, llm_model)
        default_llm_config = _resolve_infographic_llm(client, None)

        llm_attempts = [selected_llm_config]
        if (
            default_llm_config.get('provider_name') != selected_llm_config.get('provider_name')
            or default_llm_config.get('model_name') != selected_llm_config.get('model_name')
        ):
            llm_attempts.append(default_llm_config)

        # If the explicitly selected model is a reasoning profile, try default first.
        selected_model_name = str(selected_llm_config.get('model_name') or '').lower()
        if 'reasoner' in selected_model_name and len(llm_attempts) > 1:
            llm_attempts = [default_llm_config, selected_llm_config]

        prompt = _build_svg_infographic_prompt(text, accent_color, text_color, secondary_color, neutral_color)
        svg_markup = ""
        used_llm_config = selected_llm_config
        for candidate in llm_attempts:
            ProviderClass = get_provider_class(candidate['provider_name'])
            llm = ProviderClass(
                api_key=candidate['api_key'],
                model_name=candidate['model_name'],
                base_url=candidate['base_url']
            )
            response = asyncio.run(llm.generate(
                prompt,
                temperature=0.2,
                max_tokens=1400,
                top_p=0.9,
            ))
            svg_markup = _extract_svg_markup(response.content)
            if svg_markup.lower().startswith('<svg'):
                used_llm_config = candidate
                break

        if not svg_markup.lower().startswith('<svg'):
            raise ValueError("LLM did not return a valid SVG document")

        return jsonify({
            "svg": svg_markup,
            "provider": used_llm_config['provider_name'],
            "model": used_llm_config['model_name'],
        }), 200

    except Exception as e:
        logger.error(f"Error generating infographic SVG: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@images_bp.route('/infographic/generate', methods=['POST'])
@limiter.limit("10 per minute")
def generate_infographic():
    """
    Generate infographic from text using LLM and template.
    
    Expected JSON body:
    {
        "templateId": 1,
        "storyText": "Content to visualize",
        "user_id": "user_uuid"
    }
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
        if not data:
            return jsonify(ErrorResponse(
                error="invalid_request",
                message="Request body is empty",
                error_code="INVALID_REQUEST",
                status=400
            ).dict()), 400
            
        template_id = data.get('templateId')
        story_text = data.get('storyText')
        user_id = data.get('user_id')
        
        if not template_id or not story_text or not user_id:
            return jsonify(ErrorResponse(
                error="missing_parameters",
                message="templateId, storyText, and user_id are required",
                error_code="MISSING_PARAMETERS",
                status=400
            ).dict()), 400
        
        # Get template from database
        client = get_supabase_client()
        if not client:
            return jsonify(ErrorResponse(
                error="database_error",
                message="Database connection failed",
                error_code="DATABASE_ERROR",
                status=500
            ).dict()), 500
        
        template = client.table('infographic')\
            .select('*')\
            .eq('id', template_id)\
            .single()\
            .execute()
        
        if not template.data:
            return jsonify(ErrorResponse(
                error="template_not_found",
                message=f"Template {template_id} not found",
                error_code="TEMPLATE_NOT_FOUND",
                status=404
            ).dict()), 404
        
        # Safe access to Label/label
        lbl = template.data.get('Label') or template.data.get('label') or template.data.get('name') or "Infographic"
        
        # Extract HTML/CSS/Dimensions from template
        html_content = template.data.get('HTML') or template.data.get('html') or ""
        css_content = template.data.get('CSS') or template.data.get('css') or ""
        
        # Default dimensions if not present
        width = template.data.get('width') or 800
        height = template.data.get('height') or 600
        
        # --- LLM Content Generation ---
        
        # 1. Get Default LLM Provider and its API Key
        provider_name = "google" # Fallback
        model_name = "gemini-1.5-flash" # Fallback
        api_key_val = current_app.config.get('LITELLM_API_KEY')
        
        logger.info(f"Starting LLM config fetch. Initial api_key_val from env: {'present' if api_key_val else 'missing'}")
        
        # Fetch default provider from DB
        try:
            llm_provider_query = client.table('llm_providers')\
                .select('*')\
                .eq('is_default', True)\
                .execute()
            
            if llm_provider_query.data and len(llm_provider_query.data) > 0:
                provider_rec = llm_provider_query.data[0]
                logger.info(f"Found {len(llm_provider_query.data)} default providers. Using the first one: {provider_rec.get('name')}")
                
                provider_name = provider_rec.get('provider_name') or provider_rec.get('provider') or "google"
                model_name = provider_rec.get('model_name') or "gemini-1.5-flash"
                key_id = provider_rec.get('api_keys_id')
                
                logger.info(f"Default provider: {provider_name}, model: {model_name}, key_id: {key_id}")
                
                if key_id:
                    key_query = client.table('api_keys').select('key_value').eq('id', key_id).execute()
                    if key_query.data and len(key_query.data) > 0:
                        api_key_val = key_query.data[0].get('key_value')
                        logger.info(f"Successfully retrieved API key from DB for key_id {key_id}")
                    else:
                        logger.warning(f"Key ID {key_id} found in provider but no record in api_keys table")
                else:
                    logger.warning(f"Default provider {provider_rec.get('name')} has no api_keys_id linked")
            else:
                logger.warning("No default LLM provider found in DB (is_default=True)")
        except Exception as db_err:
            logger.error(f"Error querying LLM configuration from DB: {db_err}")

        # Fallback to Config/Env if DB lookup failed
        if not api_key_val:
             api_key_val = current_app.config.get('LITELLM_API_KEY')
             if api_key_val:
                 logger.info("Using LITELLM_API_KEY from environment as fallback.")
                 if not provider_name or provider_name == 'None':
                     provider_name = 'google'
                 if not model_name or model_name == 'None':
                     model_name = 'gemini-1.5-flash'

        # 2. Call LLM
        if not api_key_val:
             logger.error("Final check: No API key found for default LLM (DB or Env)")
             return jsonify(ErrorResponse(
                error="api_key_missing",
                message="Default LLM is missing API key configuration",
                error_code="API_KEY_MISSING",
                status=500
            ).dict()), 500
        else:
            try:
                import asyncio
                import json
                from ...integrations.llm.litellm_client import LiteLLMClient
                from ...core.models.llm import LLMConfig, LLMModel, LLMProvider

                llm_client = LiteLLMClient(api_key=api_key_val)
                
                # Construct Prompt
                template_prompt = template.data.get('prompt') or "Generate a JSON for this infographic."
                full_prompt = f"{template_prompt}\n\nContent/Story:\n{story_text}\n\nOutput strictly valid JSON."
                
                # Map provider string to Enum
                try:
                    provider_enum = LLMProvider(provider_name.lower())
                except ValueError:
                    provider_enum = LLMProvider.GOOGLE # Default fallback
                    
                config = LLMConfig(
                    model=LLMModel(
                        provider=provider_enum,
                        model_name=model_name,
                        api_key=api_key_val
                    ),
                    user_prompt=full_prompt,
                    system_prompt="You are a JSON generator for infographics. Output ONLY valid JSON."
                )
                
                # Execute Async Call
                response = asyncio.run(llm_client.generate(config))
                generated_json_str = response.content
                
                # Clean JSON
                if "```json" in generated_json_str:
                    generated_json_str = generated_json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in generated_json_str:
                     generated_json_str = generated_json_str.split("```")[1].split("```")[0].strip()
                     
                generated_data = json.loads(generated_json_str)
                logger.info(f"LLM Generated Data: {generated_data}")
                
                # 3. Flatten JSON & Replace Placeholders
                replacements = {}
                
                # Top level keys
                for k, v in generated_data.items():
                    if not isinstance(v, (list, dict)):
                        replacements[k] = str(v)
                        
                # Items array
                if 'items' in generated_data and isinstance(generated_data['items'], list):
                    for item in generated_data['items']:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                replacements[k] = str(v)
                                
                # Perform Replacement in HTML
                for key, val in replacements.items():
                     html_content = html_content.replace(f"+{key}+", val)
                     html_content = html_content.replace(f"+{key.lower()}+", val)
                
            except Exception as e:
                logger.error(f"LLM Generation failed: {e}", exc_info=True)
                # We could potentially continue with non-replaced template if that makes sense,
                # but usually infographic without data is useless.
                # For now, let's proceed and see if the render service handles it.
        
        # --- End LLM Content Generation ---

        # Build clip parameters
        clip = {
            "x": template.data.get('clipX') or 0,
            "y": template.data.get('clipY') or 0,
            "width": template.data.get('clipWidth') or width,
            "height": template.data.get('clipHeight') or height
        }
        
        # Call Screen Capture Service
        render_url = current_app.config.get('RENDER_SERVICE_URL', 'http://localhost:8082/generate-image')
        payload = {
            "html": f'<div class="full-screen">{html_content}</div>',
            "css": css_content,
            "width": width,
            "height": height,
            "clip": clip
        }
        
        logger.info(f"Calling render service at {render_url} for template {template_id}")
        
        # TODO: Inject story_text into HTML before rendering if needed (e.g. replacing {{text}} placeholder)
        # For now, we render the template as-is to verify the pipeline
        
        render_response = requests.post(render_url, json=payload, timeout=30)
        
        if render_response.status_code != 200:
            logger.error(f"Render service failed: {render_response.text}")
            raise Exception(f"Render service failed with status {render_response.status_code}")
            
        image_data = render_response.content
        filename = f"infographic_{template_id}_{datetime.utcnow().timestamp()}.png"
        
        # Upload to Supabase
        image_url = upload_to_supabase_storage(
            image_data, 
            filename, 
            user_id,
            'image/png'
        )
        
        metadata = {
            "ImageUrl": image_url,
            "ImageAuthor": f"Infographic - {lbl}",
            "MediaAltText": f"Infographic: {story_text[:100]}",
            "mediaTitle": lbl,
            "mediaCaption": ""
        }
        
        return jsonify({
            "imageUrl": image_url,
            "metadata": metadata
        }), 200
        
    except Exception as e:
        logger.error(f"Error generating infographic: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@images_bp.route('/', methods=['POST'])
def save_image_metadata():
    """
    Save image metadata to Images table.
    
    Expected JSON body:
    {
        "user_id": "uuid",
        "ImageUrl": "url",
        "ImageAuthor": "author",
        "MediaAltText": "alt text",
        "mediaTitle": "title",
        "mediaCaption": "caption"
    }
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
        if not data:
            return jsonify(ErrorResponse(
                error="invalid_request",
                message="Request body is empty",
                error_code="INVALID_REQUEST",
                status=400
            ).dict()), 400
            
        # Validate required fields
        if not data.get('user_id') or not data.get('ImageUrl'):
            return jsonify(ErrorResponse(
                error="missing_parameters",
                message="user_id and ImageUrl are required",
                error_code="MISSING_PARAMETERS",
                status=400
            ).dict()), 400
        
        # Save to database
        client = get_supabase_client()
        if not client:
            return jsonify(ErrorResponse(
                error="database_error",
                message="Database connection failed",
                error_code="DATABASE_ERROR",
                status=500
            ).dict()), 500
        
        result = client.table('images').insert({
            "user_id": data['user_id'],
            "imageurl": data['ImageUrl'],
            "imageauthor": data.get('ImageAuthor'),
            "mediaalttext": data.get('MediaAltText'),
            "mediatitle": data.get('mediaTitle'),
            "mediacaption": data.get('mediaCaption')
        }).execute()
        
        return jsonify(result.data[0]), 201
        
    except Exception as e:
        logger.error(f"Error saving image metadata: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@images_bp.route('/<image_id>', methods=['GET'])
def get_image_metadata(image_id):
    """Get image metadata by ID."""
    try:
        client = get_supabase_client()
        if not client:
            return jsonify(ErrorResponse(
                error="database_error",
                message="Database connection failed",
                error_code="DATABASE_ERROR",
                status=500
            ).dict()), 500
        
        result = client.table('images')\
            .select('*')\
            .eq('id', image_id)\
            .single()\
            .execute()
        
        if not result.data:
            return jsonify(ErrorResponse(
                error="not_found",
                message="Image not found",
                error_code="NOT_FOUND",
                status=404
            ).dict()), 404
        
        return jsonify(result.data), 200
        
    except Exception as e:
        logger.error(f"Error fetching image metadata: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@images_bp.route('/<image_id>', methods=['PUT'])
def update_image_metadata(image_id):
    """Update image metadata."""
    try:
        if not request.is_json:
            return jsonify(ErrorResponse(
                error="invalid_content_type",
                message="Content-Type must be application/json",
                error_code="INVALID_CONTENT_TYPE",
                status=400
            ).dict()), 400
        
        data = request.get_json()
        if not data:
            return jsonify(ErrorResponse(
                error="invalid_request",
                message="Request body is empty",
                error_code="INVALID_REQUEST",
                status=400
            ).dict()), 400
            
        # Remove fields that shouldn't be updated
        data.pop('id', None)
        data.pop('user_id', None)
        data.pop('created_at', None)
        
        client = get_supabase_client()
        if not client:
            return jsonify(ErrorResponse(
                error="database_error",
                message="Database connection failed",
                error_code="DATABASE_ERROR",
                status=500
            ).dict()), 500
        
        # Map camelCase keys to snake_case columns
        db_data = {}
        if 'ImageUrl' in data: db_data['imageurl'] = data['ImageUrl']
        if 'ImageAuthor' in data: db_data['imageauthor'] = data['ImageAuthor']
        if 'MediaAltText' in data: db_data['mediaalttext'] = data['MediaAltText']
        if 'mediaTitle' in data: db_data['mediatitle'] = data['mediaTitle']
        if 'mediaCaption' in data: db_data['mediacaption'] = data['mediaCaption']
        
        result = client.table('images')\
            .update(db_data)\
            .eq('id', image_id)\
            .execute()
        
        if not result.data:
            return jsonify(ErrorResponse(
                error="not_found",
                message="Image not found",
                error_code="NOT_FOUND",
                status=404
            ).dict()), 404
        
        return jsonify(result.data[0]), 200
        
    except Exception as e:
        logger.error(f"Error updating image metadata: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500
