"""
Unit tests for AI Infographic Generation Endpoint:
POST /api/v1/images/generate-ai-infographic
"""

import importlib.util
import json
import sys
import unittest
from unittest.mock import MagicMock, patch

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

generate_ai_infographic_endpoint = images_mod.generate_ai_infographic_endpoint


class TestAIInfographicEndpoint(unittest.TestCase):
    @patch.object(images_mod, "request")
    def test_missing_text_returns_400(self, mock_request):
        mock_request.get_json.return_value = {"text": ""}
        with patch.object(images_mod, "jsonify", side_effect=lambda x: x):
            res, status = generate_ai_infographic_endpoint()
            self.assertEqual(status, 400)

    @patch.object(images_mod, "request")
    @patch.object(images_mod, "resolve_image_provider")
    @patch.object(images_mod, "generate_google_imagen")
    @patch.object(images_mod, "upload_to_supabase_storage")
    def test_successful_infographic_generation(
        self,
        mock_upload,
        mock_generate,
        mock_resolve,
        mock_request
    ):
        mock_request.get_json.return_value = {
            "text": "Kubernetes pods communicate via services and ingress controllers.",
            "archetype": "technical_scientific",
            "user_instructions": "blueprint background",
            "aspectRatio": "16:9",
            "resolution": "1K",
            "user_id": "user-456"
        }

        mock_resolve.return_value = {
            "provider": "google",
            "model": "gemini-3-pro-image-preview",
            "api_key": "fake-key",
            "display_name": "Nano Banana Pro"
        }
        mock_generate.return_value = b"infographic-bytes"
        mock_upload.return_value = "https://storage.supabase.co/infographic_ai.jpg"

        with patch.object(images_mod, "jsonify", side_effect=lambda x: x):
            res, status = generate_ai_infographic_endpoint()
            self.assertEqual(status, 200)
            self.assertEqual(res["imageUrl"], "https://storage.supabase.co/infographic_ai.jpg")
            self.assertEqual(res["archetype"], "technical_scientific")
            self.assertEqual(res["aspectRatio"], "16:9")
            self.assertEqual(res["resolution"], "1K")
            self.assertEqual(res["model"], "gemini-3-pro-image-preview")
            self.assertIn("metadata", res)
            self.assertIn("AI Infographic - Nano Banana Pro", res["metadata"]["ImageAuthor"])


if __name__ == "__main__":
    unittest.main()
