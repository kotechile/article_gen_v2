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
import json
import re
import html
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename

from supabase_client import (
    get_supabase_client,
    get_api_key,
    resolve_llm_provider,
    resolve_image_provider,
    get_image_applications_config,
    IMAGE_APP_ARTICLE_IMAGE,
    IMAGE_APP_INFOGRAPHICS,
)
from ...core.models.errors import ErrorResponse, ValidationErrorResponse
from ...services.llm.providers import get_provider_class
from ...services.infographic_llm import (
    apply_icon_markup_to_html,
    append_infographic_llm_log,
    inject_fontawesome_icon_styles,
    normalize_infographic_payload_icons,
    write_infographic_render_debug_artifacts,
)

logger = logging.getLogger(__name__)

KIE_FLUX_RESOLUTION = "1K"

# Create blueprint
images_bp = Blueprint('images', __name__, url_prefix='/api/v1/images')

# Create rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1000 per hour", "60 per minute"],
    storage_uri="memory://"
)

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

        logger.info(
            "SUPABASE UPLOAD: starting upload to bucket='User Files' path='%s' size_bytes=%s content_type='%s'",
            storage_path,
            len(file_data),
            content_type,
        )

        # Upload to User Files bucket
        response = client.storage.from_('User Files').upload(
            path=storage_path,
            file=file_data,
            file_options={"content-type": content_type}
        )

        logger.info(
            "SUPABASE UPLOAD: upload() returned response=%s",
            response,
        )

        # Get public URL
        public_url = client.storage.from_('User Files').get_public_url(storage_path)

        logger.info(
            "SUPABASE UPLOAD: get_public_url() returned url='%s'",
            public_url,
        )

        logger.info(f"Uploaded image to Supabase: {storage_path}")
        return public_url

    except Exception as e:
        logger.error(f"Error uploading to Supabase storage: {str(e)}", exc_info=True)
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


def generate_google_imagen(
    prompt: str,
    api_key: str,
    model: str = "imagen-4.0-generate-001", 
    aspect_ratio: str = "1:1",
    resolution: str = "1K",
    reference_image: bytes = None,
) -> bytes:
    """
    Generate image using Google Imagen / Gemini API (including Nano Banana Pro / gemini-3-pro-image-preview).
    Supports optional reference image (image-to-image or text-to-image), aspect ratio, and resolution.
    """
    try:
        model_clean = str(model or "imagen-4.0-generate-001").strip()
        # Normalize Nano Banana Pro alias to Google model if needed
        if "banana" in model_clean.lower() or "gemini" in model_clean.lower():
            if "banana" in model_clean.lower() and "gemini" not in model_clean.lower():
                model_clean = "gemini-3-pro-image-preview"

        headers = {
            'x-goog-api-key': api_key,
            'Content-Type': 'application/json'
        }

        # 1. For Gemini native image generation models (e.g. gemini-3-pro-image-preview, nano banana pro)
        if "gemini" in model_clean.lower():
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_clean}:generateContent"
            parts = [{"text": prompt}]
            if reference_image:
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(reference_image).decode('utf-8')
                    }
                })
            
            body = {
                "contents": [
                    {
                        "role": "user",
                        "parts": parts
                    }
                ],
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"],
                    "imageConfig": {
                        "aspectRatio": aspect_ratio,
                        "imageSize": resolution or "1K"
                    }
                }
            }
            response = requests.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()

            candidates = data.get('candidates') or []
            for candidate in candidates:
                content_parts = (candidate.get('content') or {}).get('parts') or []
                for part in content_parts:
                    inline_data = part.get('inline_data') or part.get('inlineData')
                    if inline_data and inline_data.get('data'):
                        return base64.b64decode(inline_data['data'])

            raise Exception(f"Gemini image response contained no image parts: {data}")

        # 2. For Imagen models (e.g. imagen-4.0-generate-001, imagen-3.0-generate-002)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_clean}:predict"
        instance = {"prompt": prompt}
        if reference_image:
            instance["image"] = {
                "bytesBase64Encoded": base64.b64encode(reference_image).decode('utf-8')
            }

        body = {
            "instances": [instance],
            "parameters": {
                "sampleCount": 1,
                "aspectRatio": aspect_ratio,
                "imageSize": resolution or "1K"
            }
        }

        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()

        # New REST response shape.
        predictions = data.get('predictions') if isinstance(data.get('predictions'), list) else []
        if predictions:
            image_bytes_b64 = predictions[0].get('bytesBase64Encoded')
            if image_bytes_b64:
                return base64.b64decode(image_bytes_b64)

        # Backward compatibility for older response shapes.
        if data.get('generated_images') and len(data['generated_images']) > 0:
            image_bytes_b64 = data['generated_images'][0].get('image', {}).get('image_bytes')
            if not image_bytes_b64:
                raise Exception(f"Imagen response missing image bytes: {data}")
            return base64.b64decode(image_bytes_b64)

        raise Exception(f"No image generated in response: {data}")

    except Exception as e:
        logger.error(f"Google Imagen / Gemini API error: {str(e)}")
        raise


def generate_kie_flux_image(
    prompt: str,
    api_key: str,
    model: str,
    aspect_ratio: str = "1:1",
    reference_image_urls=None,
    resolution: str = "1K",
) -> bytes:
    """Generate image through KIE Market API task endpoints."""
    try:
        create_url = "https://api.kie.ai/api/v1/jobs/createTask"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        input_payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution or "1K",
            "nsfw_checker": False,
        }

        model_name = str(model or "").strip().lower()
        image_urls = [
            str(url).strip()
            for url in (reference_image_urls or [])
            if isinstance(url, str) and str(url).strip()
        ]

        target_model = model
        # If reference images are present: provide image URLs and use image-to-image mode
        if image_urls:
            input_payload["input_urls"] = image_urls
            input_payload["image_urls"] = image_urls
            if "flux-2" in model_name and not model_name.endswith("-image-to-image"):
                target_model = "flux-2/flex-image-to-image"
        else:
            # If reference images are absent: if model is flux-2/flex-image-to-image, route to text-to-image
            if model_name == "flux-2/flex-image-to-image":
                target_model = "flux-2/flex"

        create_payload = {
            "model": target_model,
            "input": input_payload,
        }

        create_resp = requests.post(create_url, headers=headers, json=create_payload)
        create_resp.raise_for_status()
        create_data = create_resp.json()
        logger.info("KIE Flux createTask response: %s", create_data)

        task_id = ((create_data.get("data") or {}).get("taskId") or "").strip()
        if not task_id:
            raise Exception(f"KIE did not return taskId: {create_data}")

        poll_url = "https://api.kie.ai/api/v1/jobs/recordInfo"
        import time
        max_attempts = 150  # 5 minutes
        for _ in range(max_attempts):
            time.sleep(2)
            poll_resp = requests.get(
                poll_url,
                headers={"Authorization": f"Bearer {api_key}"},
                params={"taskId": task_id},
            )
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            data = poll_data.get("data") if isinstance(poll_data.get("data"), dict) else {}
            state = str(data.get("state") or "").strip().lower()

            if state == "success":
                result_json = data.get("resultJson")
                parsed_result = {}
                if isinstance(result_json, dict):
                    parsed_result = result_json
                elif isinstance(result_json, str) and result_json.strip():
                    try:
                        parsed_result = json.loads(result_json)
                    except Exception:
                        logger.warning("Failed to parse KIE resultJson for task_id=%s: %s", task_id, result_json)

                result_urls = parsed_result.get("resultUrls") if isinstance(parsed_result, dict) else None
                image_url = result_urls[0] if isinstance(result_urls, list) and result_urls else None
                if not image_url:
                    raise Exception(f"KIE task completed but no result URL found. task_id={task_id} payload={poll_data}")

                image_resp = requests.get(image_url)
                image_resp.raise_for_status()
                return image_resp.content

            if state == "fail":
                fail_code = str(data.get("failCode") or "").strip()
                fail_msg = str(data.get("failMsg") or "").strip()
                raise Exception(
                    f"KIE Flux task failed. task_id={task_id} fail_code={fail_code or 'n/a'} "
                    f"fail_msg={fail_msg or 'no provider message'}"
                )

            # still processing: waiting / queuing / generating / empty
            continue

        raise Exception("KIE Flux generation timed out")
    except Exception as e:
        logger.error(f"KIE Flux API error: {str(e)}")
        raise


def generate_fluxapi_image(prompt: str, api_key: str, model: str = "flux-kontext-pro", 
                          aspect_ratio: str = "1:1") -> bytes:
    """Generate image using fluxapi.ai endpoints with polling."""
    try:
        def _extract_flux_failure_reason(payload):
            candidates = []
            ignored_values = {"success", "ok", "done", "completed", "complete"}

            if isinstance(payload, dict):
                data = payload.get('data') if isinstance(payload.get('data'), dict) else {}
                response = data.get('response') if isinstance(data.get('response'), dict) else {}
                candidates.extend([
                    data.get('failReason'),
                    data.get('reason'),
                    data.get('error'),
                    data.get('errorCode'),
                    data.get('errorMessage'),
                    response.get('msg'),
                    response.get('message'),
                    response.get('error'),
                    response.get('errorCode'),
                    response.get('errorMessage'),
                    response.get('reason'),
                    payload.get('message'),
                ])

            for candidate in candidates:
                text = str(candidate or '').strip()
                if text and text.lower() not in ignored_values:
                    return text

            return ""

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
                failure_reason = _extract_flux_failure_reason(poll_data)
                logger.error(
                    "Flux task failed. task_id=%s model=%s aspect_ratio=%s success_flag=%s poll_data=%s",
                    task_id,
                    model,
                    aspect_ratio,
                    success_flag,
                    poll_data,
                )
                if failure_reason:
                    raise Exception(f"Flux generation failed with code {success_flag}: {failure_reason}")
                raise Exception(
                    f"Flux generation failed with code {success_flag}. "
                    "The provider rejected or dropped the task without returning a detailed reason. "
                    "Try a shorter prompt, remove dense quoted text or long keyword lists, switch aspect ratio, or use another model."
                )
        
        raise Exception("Flux generation timed out")
        
    except Exception as e:
        logger.error(f"Flux API error: {str(e)}")
        raise


def generate_flux_image(
    prompt: str,
    api_key: str,
    model: str = "flux-kontext-pro",
    aspect_ratio: str = "1:1",
    provider: str = "",
    reference_image_urls=None,
    resolution: str = "1K",
) -> bytes:
    """Route Flux generation to provider-specific implementation."""
    provider_name = str(provider or "").strip().lower()
    model_name = str(model or "").strip().lower()

    if "kie.ai" in provider_name or model_name.startswith("flux-2/") or "flux" in provider_name or "flux-2" in model_name:
        return generate_kie_flux_image(
            prompt,
            api_key,
            model,
            aspect_ratio,
            reference_image_urls=reference_image_urls,
            resolution=resolution,
        )

    return generate_fluxapi_image(prompt, api_key, model, aspect_ratio)


@images_bp.route('/application-config', methods=['GET'])
def get_application_config():
    """
    Get current image model assignments for applications ('article_image', 'infographics')
    configured in Supabase 'used_for' linking to 'llm_providers_image' and 'api_keys'.
    """
    try:
        config = get_image_applications_config()
        return jsonify({"applications": config}), 200
    except Exception as e:
        logger.error(f"Error fetching image application config: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@images_bp.route('/generate-ai', methods=['POST'])
@limiter.limit("20 per minute")
def generate_ai_image():
    """
    Generate an AI image using configured providers (Nano Banana Pro / Google, Flux 2 / KIE, etc.).
    
    Expected JSON body:
    {
        "prompt": "Image description",
        "model": "model_technical_name" (optional; if omitted, resolves via application),
        "application": "article_image" or "infographics" (optional; defaults to article_image),
        "aspectRatio": "16:9" (optional; defaults to 1:1),
        "resolution": "1K" or "2K" (optional; defaults to 1K),
        "referenceImage": "base64_encoded_image" (optional; image can be present or absent),
        "referenceImageUrls": ["https://..."] (optional),
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
        if not data:
            return jsonify(ErrorResponse(
                error="invalid_request",
                message="Request body is empty",
                error_code="INVALID_REQUEST",
                status=400
            ).dict()), 400
            
        prompt = data.get('prompt')
        model = str(data.get('model') or '').strip()
        application = str(data.get('application') or '').strip()
        aspect_ratio = str(data.get('aspectRatio') or '1:1').strip()
        resolution = str(data.get('resolution') or '1K').strip().upper()
        reference_image_b64 = data.get('referenceImage')
        raw_ref_urls = data.get('referenceImageUrls') or []
        if isinstance(raw_ref_urls, str):
            raw_ref_urls = [raw_ref_urls]
        reference_image_urls = [
            str(url).strip() for url in raw_ref_urls if isinstance(url, str) and str(url).strip()
        ]
        user_id = data.get('user_id')  # Should come from auth middleware
        
        if not prompt or not user_id:
            return jsonify(ErrorResponse(
                error="missing_parameters",
                message="prompt and user_id are required",
                error_code="MISSING_PARAMETERS",
                status=400
            ).dict()), 400
        
        # If neither model nor application was provided, default application to article_image
        if not model and not application:
            application = IMAGE_APP_ARTICLE_IMAGE

        # Resolve image model and API key:
        # 1. Via 'used_for' table (using 'llm_image_id' -> 'llm_providers_image' -> 'api_keys')
        # 2. Or explicit model name in 'llm_providers_image' -> 'api_keys'
        resolved = resolve_image_provider(application=application or None, model=model or None)
        
        model_to_use = resolved.get('model')
        provider = str(resolved.get('provider') or '').lower()
        api_key = resolved.get('api_key')
        display_name = resolved.get('display_name') or model_to_use
        
        if not model_to_use:
            logger.error(f"Image model could not be resolved for application='{application}', model='{model}'")
            return jsonify(ErrorResponse(
                error="model_not_found",
                message=f"No image model configured for application '{application}' or model '{model}'",
                error_code="MODEL_NOT_FOUND",
                status=404
            ).dict()), 404
        
        if not api_key:
            logger.error(f"API key missing for model '{model_to_use}' ({provider})")
            return jsonify(ErrorResponse(
                error="api_key_missing",
                message=f"API key is missing or not configured for model '{model_to_use}' ({provider})",
                error_code="API_KEY_MISSING",
                status=500
            ).dict()), 500
            
        logger.info(
            f"Resolved image model '{model_to_use}' (provider: '{provider}', source: '{resolved.get('source')}') "
            f"with API key (len={len(api_key)})"
        )
        
        # Parse reference image bytes if provided
        reference_image = None
        if reference_image_b64:
            try:
                clean_b64 = reference_image_b64
                if ',' in clean_b64:
                    clean_b64 = clean_b64.split(',', 1)[1]
                reference_image = base64.b64decode(clean_b64)
            except Exception as e:
                logger.warning(f"Failed to decode referenceImage base64: {e}")

        # If reference image was provided as base64 but no public URL was provided,
        # upload it to Supabase Storage so URL-based providers (like KIE Flux) have a valid URL.
        if reference_image and not reference_image_urls and user_id:
            try:
                ref_filename = f"ref_{int(datetime.utcnow().timestamp())}.jpg"
                ref_url = upload_to_supabase_storage(reference_image, ref_filename, user_id)
                if ref_url:
                    reference_image_urls.append(ref_url)
            except Exception as e:
                logger.warning(f"Could not upload reference image to storage for URL-based provider: {e}")
        
        # Generate image based on provider
        image_data = None
        if 'stable' in provider or 'stability' in provider:
            image_data = generate_stable_diffusion_image(
                prompt, api_key, aspect_ratio, model_to_use, reference_image
            )
        elif 'google' in provider or 'imagen' in provider or 'gemini' in provider or 'banana' in model_to_use.lower():
            image_data = generate_google_imagen(
                prompt,
                api_key,
                model_to_use,
                aspect_ratio,
                resolution=resolution,
                reference_image=reference_image,
            )
        elif 'flux' in provider or 'kie.ai' in provider or model_to_use.lower().startswith("flux-2/") or "flux" in model_to_use.lower():
            image_data = generate_flux_image(
                prompt,
                api_key,
                model_to_use,
                aspect_ratio,
                provider,
                reference_image_urls=reference_image_urls,
                resolution=resolution,
            )
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
            "ImageAuthor": f"AI - {display_name}",
            "MediaAltText": prompt[:200],  # Use prompt as alt text
            "mediaTitle": prompt[:100],
            "mediaCaption": ""
        }
        
        return jsonify({
            "imageUrl": image_url,
            "metadata": metadata,
            "model": model_to_use,
            "provider": provider,
            "application": resolved.get("application"),
            "aspectRatio": aspect_ratio,
            "resolution": resolution
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
        start_time = datetime.utcnow()
        request_received_time = start_time

        logger.info(
            "Upload request RECEIVED: remote_addr=%s content_type=%s content_length=%s",
            request.remote_addr,
            request.content_type,
            request.content_length,
        )

        if 'image' not in request.files:
            logger.warning("Upload request rejected: no 'image' file in request.files. Keys=%s", list(request.files.keys()))
            return jsonify(ErrorResponse(
                error="no_file",
                message="No image file provided",
                error_code="NO_FILE",
                status=400
            ).dict()), 400

        file = request.files['image']
        user_id = request.form.get('user_id')

        logger.info(
            "Upload request VALIDATED: user_id=%s filename='%s' content_type=%s size=%s",
            user_id,
            file.filename,
            file.content_type,
            request.content_length,
        )

        if not user_id:
            logger.warning("Upload request rejected: missing user_id")
            return jsonify(ErrorResponse(
                error="missing_user_id",
                message="user_id is required",
                error_code="MISSING_USER_ID",
                status=400
            ).dict()), 400

        if file.filename == '':
            logger.warning("Upload request rejected: empty filename")
            return jsonify(ErrorResponse(
                error="empty_filename",
                message="No file selected",
                error_code="EMPTY_FILENAME",
                status=400
            ).dict()), 400

        # Read file data
        file_data = file.read()
        file_size = len(file_data)
        read_elapsed_ms = int((datetime.utcnow() - request_received_time).total_seconds() * 1000)

        logger.info(
            "File READ complete: user_id=%s filename='%s' size_bytes=%s read_elapsed_ms=%s",
            user_id,
            file.filename,
            file_size,
            read_elapsed_ms,
        )

        filename = f"upload_{datetime.utcnow().timestamp()}_{file.filename}"

        # Log before Supabase upload
        upload_start_time = datetime.utcnow()
        logger.info(
            "SUPABASE UPLOAD STARTING: user_id=%s filename='%s' storage_filename=%s size_bytes=%s",
            user_id,
            file.filename,
            filename,
            file_size,
        )

        # Upload to Supabase
        image_url = upload_to_supabase_storage(
            file_data,
            filename,
            user_id,
            file.content_type or 'image/jpeg'
        )

        upload_elapsed_ms = int((datetime.utcnow() - upload_start_time).total_seconds() * 1000)
        total_elapsed_ms = int((datetime.utcnow() - request_received_time).total_seconds() * 1000)

        logger.info(
            "Upload request COMPLETED: user_id=%s storage_filename=%s image_url=%s upload_elapsed_ms=%s total_elapsed_ms=%s",
            user_id,
            filename,
            image_url,
            upload_elapsed_ms,
            total_elapsed_ms,
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
        
        def _coerce_dimension(value, fallback):
            try:
                if value is None or value == "":
                    return fallback
                return max(1, int(float(value)))
            except (TypeError, ValueError):
                return fallback

        def _extract_css_dimension(css_value: str, selector: str, prop: str):
            if not css_value:
                return None

            block_start = css_value.find(selector)
            if block_start == -1:
                return None

            brace_start = css_value.find('{', block_start)
            brace_end = css_value.find('}', brace_start)
            if brace_start == -1 or brace_end == -1:
                return None

            block = css_value[brace_start + 1:brace_end]
            match = re.search(rf'{re.escape(prop)}\s*:\s*([0-9]+(?:\.[0-9]+)?)px', block, re.IGNORECASE)
            if not match:
                return None

            return _coerce_dimension(match.group(1), None)

        # Prefer dimensions declared in the template itself. If the template
        # relies on a background asset for sizing, the render service will
        # derive the final canvas from that image rather than DB metadata.
        width = _extract_css_dimension(css_content, '.infographic-template', 'width') or 1920
        height = _extract_css_dimension(css_content, '.infographic-template', 'height') or 1080
        
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

        request_debug_stamp = datetime.utcnow().strftime('%Y%m%dT%H%M%S_%f')
        llm_log_path = current_app.config.get(
            'INFOGRAPHIC_LLM_LOG_FILE',
            os.path.join('logs', 'infographic_llm_responses.jsonl')
        )
        render_log_path = current_app.config.get(
            'INFOGRAPHIC_RENDER_LOG_FILE',
            os.path.join('logs', 'infographic_render_payloads.jsonl')
        )
        render_debug_dir = current_app.config.get(
            'INFOGRAPHIC_RENDER_DEBUG_DIR',
            os.path.join('logs', 'infographic_render_debug')
        )
        llm_log_context = {
            "template_id": template_id,
            "template_label": lbl,
            "user_id": user_id,
            "provider": provider_name,
            "model": model_name,
            "story_text": story_text,
            "request_debug_stamp": request_debug_stamp,
        }

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
                raw_generated_json_str = response.content or ""
                generated_json_str = raw_generated_json_str
                
                # Clean JSON
                if "```json" in generated_json_str:
                    generated_json_str = generated_json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in generated_json_str:
                     generated_json_str = generated_json_str.split("```")[1].split("```")[0].strip()
                     
                generated_data = json.loads(generated_json_str)
                generated_data, icon_audit = normalize_infographic_payload_icons(generated_data)
                logger.info(f"LLM Generated Data: {generated_data}")
                if icon_audit:
                    logger.info("Infographic icon normalization audit: %s", icon_audit)
                append_infographic_llm_log(
                    llm_log_path,
                    {
                        **llm_log_context,
                        "status": "success",
                        "template_prompt": template_prompt,
                        "full_prompt": full_prompt,
                        "raw_response": raw_generated_json_str,
                        "cleaned_response": generated_json_str,
                        "normalized_payload": generated_data,
                        "icon_audit": icon_audit,
                    }
                )
                
                # 3. Flatten JSON & Replace Placeholders
                replacements = {}
                icon_replacements = {}
                
                # Top level keys
                for k, v in generated_data.items():
                    if not isinstance(v, (list, dict)):
                        replacements[k] = str(v)
                        
                # Items array
                if 'items' in generated_data and isinstance(generated_data['items'], list):
                    for item in generated_data['items']:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                if k.startswith('icon'):
                                    icon_replacements[k] = str(v)
                                    continue
                                replacements[k] = str(v)
                                
                # Perform Replacement in HTML
                for key, val in replacements.items():
                     html_content = html_content.replace(f"+{key}+", val)
                     html_content = html_content.replace(f"+{key.lower()}+", val)

                if icon_replacements:
                    html_content, icon_render_audit = apply_icon_markup_to_html(
                        html_content,
                        icon_replacements,
                    )
                else:
                    icon_render_audit = []
                css_content = inject_fontawesome_icon_styles(css_content)
                
            except Exception as e:
                append_infographic_llm_log(
                    llm_log_path,
                    {
                        **llm_log_context,
                        "status": "error",
                        "template_prompt": template.data.get('prompt') or "Generate a JSON for this infographic.",
                        "error": str(e),
                        "raw_response": locals().get("raw_generated_json_str"),
                        "cleaned_response": locals().get("generated_json_str"),
                    }
                )
                logger.error(f"LLM Generation failed: {e}", exc_info=True)
                # We could potentially continue with non-replaced template if that makes sense,
                # but usually infographic without data is useless.
                # For now, let's proceed and see if the render service handles it.
                icon_render_audit = []
        
        # --- End LLM Content Generation ---

        render_debug_files = write_infographic_render_debug_artifacts(
            render_debug_dir,
            template_id=template_id,
            request_timestamp=request_debug_stamp,
            html_content=html_content,
            css_content=css_content,
        )
        append_infographic_llm_log(
            render_log_path,
            {
                "template_id": template_id,
                "template_label": lbl,
                "user_id": user_id,
                "request_debug_stamp": request_debug_stamp,
                "html_path": render_debug_files["html_path"],
                "css_path": render_debug_files["css_path"],
                "icon_render_audit": icon_render_audit,
                "html_excerpt": html_content[:4000],
                "css_excerpt": css_content[:4000],
            }
        )

        # Build clip parameters only if explicitly defined
        clip_data = {
            "x": template.data.get('clipX'),
            "y": template.data.get('clipY'),
            "width": template.data.get('clipWidth'),
            "height": template.data.get('clipHeight')
        }
        
        # If any clip parameters are missing, prefer element capture over full-page capture.
        use_clip = all(v is not None for v in clip_data.values())
        
        root_selectors = []
        if 'class="infographic-template"' in html_content or "class='infographic-template'" in html_content:
            root_selectors.append('.infographic-template')
        if 'id="infographic-template"' in html_content or "id='infographic-template'" in html_content:
            root_selectors.append('#infographic-template')

        root_selectors.extend([
            '[data-infographic-root="true"]',
            'section',
            'main',
            'body > *:first-child'
        ])

        # Preserve order while removing duplicates.
        root_selectors = list(dict.fromkeys(root_selectors))

        # Call Screen Capture Service
        render_url = current_app.config.get('RENDER_SERVICE_URL', 'http://localhost:8082/generate-image')
        payload = {
            "html": html_content,
            "css": css_content,
            "width": width,
            "height": height,
            "rootSelectors": root_selectors
        }
        
        if use_clip:
            payload["clip"] = clip_data
        
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


@images_bp.route('/context-analyze', methods=['POST'])
@limiter.limit("30 per minute")
def analyze_image_context():
    """
    Analyze article text excerpt to extract the target entity,
    craft a web search query, synthesize a generation prompt,
    and retrieve candidate reference images via Linkup & Tavily.
    """
    try:
        data = request.get_json() or {}
        text = data.get('text', '').strip()
        user_instructions = data.get('user_instructions', '').strip()
        max_reference_images = int(data.get('max_reference_images', 6))

        if not text:
            return jsonify(ErrorResponse(
                error="validation_error",
                message="Text excerpt is required for context analysis",
                error_code="VALIDATION_ERROR",
                status=400
            ).dict()), 400

        from src.services.context_image import ContextImagePipeline
        pipeline = ContextImagePipeline()
        analysis = pipeline.analyze_context(
            text=text,
            user_instructions=user_instructions if user_instructions else None,
            max_reference_images=max_reference_images
        )
        return jsonify({"status": "success", "data": analysis}), 200

    except Exception as e:
        logger.error(f"Error in context-analyze endpoint: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@images_bp.route('/context-generate', methods=['POST'])
@limiter.limit("20 per minute")
def generate_context_image_endpoint():
    """
    Generate a new scene featuring the target entity conditioned on
    a reference image retrieved from the web or provided by user.
    """
    try:
        data = request.get_json() or {}
        text = data.get('text', '').strip()
        prompt = data.get('prompt', '').strip()
        reference_image_url = data.get('reference_image_url', '').strip()
        model = data.get('model', '').strip()
        aspect_ratio = data.get('aspectRatio') or data.get('aspect_ratio') or '16:9'
        resolution = data.get('resolution') or '1K'
        user_id = data.get('user_id')
        application = data.get('application') or IMAGE_APP_ARTICLE_IMAGE
        isolate_bg = bool(data.get('isolate_background', False))

        from src.services.context_image import ContextImagePipeline
        pipeline = ContextImagePipeline()

        # If prompt or reference is not supplied, auto-analyze from text
        analysis = None
        if not prompt or not reference_image_url:
            if not text:
                return jsonify(ErrorResponse(
                    error="validation_error",
                    message="Either prompt or text excerpt must be provided",
                    error_code="VALIDATION_ERROR",
                    status=400
                ).dict()), 400

            analysis = pipeline.analyze_context(text)
            if not prompt:
                prompt = analysis.get('generation_prompt')
            if not reference_image_url and analysis.get('candidate_references'):
                reference_image_url = analysis['candidate_references'][0]['url']

        if not prompt:
            return jsonify(ErrorResponse(
                error="validation_error",
                message="Unable to synthesize or find a prompt for image generation",
                error_code="VALIDATION_ERROR",
                status=400
            ).dict()), 400

        # Prepare reference image
        ref_bytes = None
        ref_http_url = None
        if reference_image_url:
            ref_bytes, ref_http_url = pipeline.prepare_reference_asset(
                reference_url=reference_image_url,
                isolate_bg=isolate_bg,
                user_id=user_id
            )

        # Resolve provider/model from Supabase used_for
        resolved = resolve_image_provider(application=application, model=model)
        provider = resolved.get("provider") or "google"
        model_to_use = resolved.get("model") or model or "gemini-3-pro-image-preview"
        api_key = resolved.get("api_key")
        display_name = resolved.get("display_name") or model_to_use

        if not api_key:
            return jsonify(ErrorResponse(
                error="missing_api_key",
                message=f"No API key found for image provider {provider} (model {model_to_use})",
                error_code="MISSING_API_KEY",
                status=400
            ).dict()), 400

        # Conditioned generation
        image_data = None
        ref_urls = [ref_http_url] if ref_http_url else ([reference_image_url] if reference_image_url else [])

        if 'flux' in provider or 'kie.ai' in provider or model_to_use.lower().startswith("flux-2/") or "flux" in model_to_use.lower():
            image_data = generate_flux_image(
                prompt,
                api_key,
                model_to_use,
                aspect_ratio,
                provider,
                reference_image_urls=ref_urls if ref_urls else None,
                resolution=resolution,
            )
        elif 'google' in provider or 'imagen' in provider or 'gemini' in provider or 'banana' in model_to_use.lower():
            image_data = generate_google_imagen(
                prompt,
                api_key,
                model_to_use,
                aspect_ratio,
                resolution=resolution,
                reference_image=ref_bytes,
            )
        elif 'stable' in provider or 'stability' in provider:
            image_data = generate_stable_diffusion_image(
                prompt,
                api_key,
                aspect_ratio,
                model_to_use,
                reference_image=ref_bytes,
            )
        else:
            return jsonify(ErrorResponse(
                error="unsupported_provider",
                message=f"Provider {provider} not supported",
                error_code="UNSUPPORTED_PROVIDER",
                status=400
            ).dict()), 400

        # Upload generated image to Supabase Storage
        filename = f"context_ai_{int(datetime.utcnow().timestamp())}.jpg"
        image_url = upload_to_supabase_storage(image_data, filename, user_id)

        metadata = {
            "ImageUrl": image_url,
            "ImageAuthor": f"AI - {display_name}",
            "MediaAltText": prompt[:200],
            "mediaTitle": prompt[:100],
            "mediaCaption": f"Reference: {reference_image_url[:80]}" if reference_image_url else ""
        }

        return jsonify({
            "imageUrl": image_url,
            "metadata": metadata,
            "model": model_to_use,
            "provider": provider,
            "application": application,
            "aspectRatio": aspect_ratio,
            "resolution": resolution,
            "referenceUsed": reference_image_url,
            "extractedAnalysis": analysis
        }), 200

    except Exception as e:
        logger.error(f"Error generating context-aware AI image: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


@images_bp.route('/generate-ai-infographic', methods=['POST'])
@limiter.limit("20 per minute")
def generate_ai_infographic_endpoint():
    """
    Generate an AI infographic image using the model assigned to 'infographics'
    in table 'used_for' (typically Nano Banana Pro / gemini-3-pro-image-preview),
    supporting 7 distinct visual archetypes.
    """
    try:
        data = request.get_json() or {}
        text = data.get('text') or data.get('storyText') or ''
        text = text.strip()
        archetype = data.get('archetype', 'auto')
        user_instructions = data.get('user_instructions') or ''
        aspect_ratio = data.get('aspectRatio') or data.get('aspect_ratio') or '16:9'
        resolution = data.get('resolution') or '1K'
        user_id = data.get('user_id')
        application = IMAGE_APP_INFOGRAPHICS

        if not text:
            return jsonify(ErrorResponse(
                error="validation_error",
                message="Text or storyText is required to generate an infographic",
                error_code="VALIDATION_ERROR",
                status=400
            ).dict()), 400

        from src.services.infographic_ai_service import InfographicAIService
        prompt, effective_archetype = InfographicAIService.synthesize_prompt(
            text=text,
            archetype=archetype,
            user_instructions=user_instructions
        )

        # Resolve provider/model from Supabase used_for for 'infographics'
        resolved = resolve_image_provider(application=application)
        provider = resolved.get("provider") or "google"
        model_to_use = resolved.get("model") or "gemini-3-pro-image-preview"
        api_key = resolved.get("api_key")
        display_name = resolved.get("display_name") or model_to_use

        if not api_key:
            return jsonify(ErrorResponse(
                error="missing_api_key",
                message=f"No API key found for image provider {provider} (model {model_to_use})",
                error_code="MISSING_API_KEY",
                status=400
            ).dict()), 400

        # Dispatch generation
        image_data = None
        if 'google' in provider or 'imagen' in provider or 'gemini' in provider or 'banana' in model_to_use.lower():
            image_data = generate_google_imagen(
                prompt,
                api_key,
                model_to_use,
                aspect_ratio,
                resolution=resolution,
                reference_image=None
            )
        elif 'flux' in provider or 'kie.ai' in provider or model_to_use.lower().startswith("flux-2/") or "flux" in model_to_use.lower():
            image_data = generate_flux_image(
                prompt,
                api_key,
                model_to_use,
                aspect_ratio,
                provider,
                reference_image_urls=None,
                resolution=resolution
            )
        elif 'stable' in provider or 'stability' in provider:
            image_data = generate_stable_diffusion_image(
                prompt,
                api_key,
                aspect_ratio,
                model_to_use,
                reference_image=None
            )
        else:
            return jsonify(ErrorResponse(
                error="unsupported_provider",
                message=f"Provider {provider} not supported for infographics",
                error_code="UNSUPPORTED_PROVIDER",
                status=400
            ).dict()), 400

        # Upload generated infographic to Supabase Storage
        filename = f"infographic_ai_{int(datetime.utcnow().timestamp())}.jpg"
        image_url = upload_to_supabase_storage(image_data, filename, user_id)

        metadata = {
            "ImageUrl": image_url,
            "ImageAuthor": f"AI Infographic - {display_name}",
            "MediaAltText": f"Infographic: {effective_archetype.replace('_', ' ').title()} - {text[:150]}",
            "mediaTitle": f"Infographic: {effective_archetype.replace('_', ' ').title()}",
            "mediaCaption": f"Archetype: {effective_archetype}"
        }

        return jsonify({
            "imageUrl": image_url,
            "metadata": metadata,
            "archetype": effective_archetype,
            "model": model_to_use,
            "provider": provider,
            "application": application,
            "aspectRatio": aspect_ratio,
            "resolution": resolution,
            "prompt": prompt
        }), 200

    except Exception as e:
        logger.error(f"Error generating AI infographic: {str(e)}", exc_info=True)
        return jsonify(ErrorResponse(
            error="internal_error",
            message=str(e),
            error_code="INTERNAL_ERROR",
            status=500
        ).dict()), 500


