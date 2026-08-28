"""
Unit tests for Context-Aware Image API endpoints:
- POST /api/v1/images/context-analyze
- POST /api/v1/images/context-generate
"""

import importlib.util
import json
import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock heavy/external framework dependencies so unit tests run purely in stdlib
def passthrough_decorator(*args, **kwargs):
    return lambda f: f

mock_flask = MagicMock()
mock_bp = MagicMock()
mock_bp.route.side_effect = passthrough_decorator
mock_flask.Blueprint.return_value = mock_bp
sys.modules["flask"] = mock_flask

mock_limiter = MagicMock()
mock_limiter.limit.side_effect = passthrough_decorator
mock_limiter_cls = MagicMock(return_value=mock_limiter)

sys.modules["flask_limiter"] = MagicMock(Limiter=mock_limiter_cls)
sys.modules["flask_limiter.util"] = MagicMock()

for mod in [
    "werkzeug", "werkzeug.utils",
    "src.services.llm.providers", "src.services.infographic_llm",
    "src.core.models.errors"
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

spec = importlib.util.spec_from_file_location("src.api.endpoints.images", "src/api/endpoints/images.py")
images_mod = importlib.util.module_from_spec(spec)
sys.modules["src.api.endpoints.images"] = images_mod
spec.loader.exec_module(images_mod)

analyze_image_context = images_mod.analyze_image_context
generate_context_image_endpoint = images_mod.generate_context_image_endpoint


class TestContextImageEndpoints(unittest.TestCase):
    @patch.object(images_mod, "request")
    @patch("src.services.context_image.ContextImagePipeline.analyze_context")
    def test_context_analyze_endpoint_success(self, mock_analyze, mock_request):
        mock_request.get_json.return_value = {
            "text": "The vintage 1984 Macintosh 128k with beige case.",
            "user_instructions": "Studio lighting",
            "max_reference_images": 4
        }
        mock_analyze.return_value = {
            "has_physical_entity": True,
            "main_object": "1984 Macintosh 128k",
            "search_query": "1984 Macintosh 128k studio photo",
            "generation_prompt": "A vintage 1984 Macintosh 128k on a minimalist desk",
            "candidate_references": [
                {"url": "https://example.com/mac.jpg", "provider": "linkup"}
            ]
        }

        with patch.object(images_mod, "jsonify", side_effect=lambda x: x):
            res, status = analyze_image_context()
            self.assertEqual(status, 200)
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["data"]["main_object"], "1984 Macintosh 128k")

    @patch.object(images_mod, "request")
    def test_context_analyze_missing_text_error(self, mock_request):
        mock_request.get_json.return_value = {"text": ""}

        with patch.object(images_mod, "jsonify", side_effect=lambda x: x):
            res, status = analyze_image_context()
            self.assertEqual(status, 400)

    @patch.object(images_mod, "request")
    @patch.object(images_mod, "resolve_image_provider")
    @patch.object(images_mod, "generate_google_imagen")
    @patch.object(images_mod, "upload_to_supabase_storage")
    @patch("src.services.context_image.ContextImagePipeline.prepare_reference_asset")
    def test_context_generate_endpoint_success(
        self,
        mock_prepare,
        mock_upload,
        mock_generate,
        mock_resolve,
        mock_request
    ):
        mock_request.get_json.return_value = {
            "text": "Apple Watch Ultra on wrist",
            "prompt": "An Apple Watch Ultra on a swimmer wrist",
            "reference_image_url": "https://example.com/apple_watch.jpg",
            "model": "nano banana pro",
            "aspectRatio": "16:9",
            "resolution": "1K",
            "user_id": "user-123"
        }

        mock_prepare.return_value = (b"ref-bytes", "https://example.com/ref.jpg")
        mock_resolve.return_value = {
            "provider": "google",
            "model": "gemini-3-pro-image-preview",
            "api_key": "fake-google-key",
            "display_name": "Nano Banana Pro"
        }
        mock_generate.return_value = b"generated-scene-bytes"
        mock_upload.return_value = "https://storage.supabase.co/scene.jpg"

        with patch.object(images_mod, "jsonify", side_effect=lambda x: x):
            res, status = generate_context_image_endpoint()
            self.assertEqual(status, 200)
            self.assertEqual(res["imageUrl"], "https://storage.supabase.co/scene.jpg")
            self.assertEqual(res["resolution"], "1K")
            self.assertEqual(res["aspectRatio"], "16:9")
            self.assertEqual(res["referenceUsed"], "https://example.com/apple_watch.jpg")


if __name__ == "__main__":
    unittest.main()
