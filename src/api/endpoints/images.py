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

from supabase_client import get_supabase_client, get_api_key, resolve_llm_provider
from ...core.models.errors import ErrorResponse, ValidationErrorResponse
from ...services.llm.providers import get_provider_class

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


def generate_google_imagen(prompt: str, api_key: str, model: str = "imagen-4.0-generate-001", 
                           aspect_ratio: str = "1:1") -> bytes:
    """Generate image using Google Imagen API."""
    try:
        # Gemini Developer API REST contract for Imagen models uses :predict.
        # Ref: https://ai.google.dev/gemini-api/docs/imagen (REST example)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:predict"

        headers = {
            'x-goog-api-key': api_key,
            'Content-Type': 'application/json'
        }

        body = {
            "instances": [
                {
                    "prompt": prompt
                }
            ],
            "parameters": {
                "sampleCount": 1,
                "aspectRatio": aspect_ratio,
                "imageSize": "1K"
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
        logger.error(f"Google Imagen API error: {str(e)}")
        raise


def generate_kie_flux_image(
    prompt: str,
    api_key: str,
    model: str,
    aspect_ratio: str = "1:1",
    reference_image_urls=None,
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
            # Hard-lock KIE Flux resolution for this application.
            "resolution": KIE_FLUX_RESOLUTION,
            "nsfw_checker": False,
        }

        model_name = str(model or "").strip().lower()
        if model_name == "flux-2/flex-image-to-image":
            image_urls = [
                str(url).strip()
                for url in (reference_image_urls or [])
                if isinstance(url, str) and str(url).strip()
            ]
            if not image_urls:
                raise Exception(
                    "Model flux-2/flex-image-to-image requires at least one reference image URL "
                    "in request field referenceImageUrls"
                )
            input_payload["input_urls"] = image_urls

        create_payload = {
            "model": model,
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
) -> bytes:
    """Route Flux generation to provider-specific implementation."""
    provider_name = str(provider or "").strip().lower()
    model_name = str(model or "").strip().lower()

    if "kie.ai" in provider_name or model_name.startswith("flux-2/"):
        return generate_kie_flux_image(
            prompt,
            api_key,
            model,
            aspect_ratio,
            reference_image_urls=reference_image_urls,
        )

    return generate_fluxapi_image(prompt, api_key, model, aspect_ratio)


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
        "referenceImage": "base64_encoded_image" (optional, legacy),
        "referenceImageUrls": ["https://..."] (optional, used by flux-2/flex-image-to-image)
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
        reference_image_urls = data.get('referenceImageUrls') or []
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
        elif 'flux' in provider or 'kie.ai' in provider or model_to_use.lower().startswith("flux-2/"):
            image_data = generate_flux_image(
                prompt,
                api_key,
                model_to_use,
                aspect_ratio,
                provider,
                reference_image_urls=reference_image_urls,
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

        # Use the template canvas dimensions when available.
        width = _coerce_dimension(template.data.get('width'), 1254)
        height = _coerce_dimension(template.data.get('height'), 1254)

        if html_content:
            if 'class="infographic-template"' in html_content or "class='infographic-template'" in html_content:
                css_content = f"""{css_content}

.infographic-template {{
  width: {width}px !important;
  height: {height}px !important;
  max-width: none !important;
}}
"""
        
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
