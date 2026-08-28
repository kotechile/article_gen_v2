"""
Unit tests for images API endpoints with image application resolution.
"""

import unittest
from unittest.mock import MagicMock, patch
from flask import Flask

from src.api.endpoints.images import images_bp


class TestImagesEndpoints(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.app.register_blueprint(images_bp)
        self.client = self.app.test_client()

    @patch("src.api.endpoints.images.get_image_applications_config")
    def test_get_application_config(self, mock_get_config):
        mock_get_config.return_value = {
            "article_image": {
                "application": "article_image",
                "provider": "flux",
                "model_name": "flux-kontext-pro",
                "display_name": "Flux Kontext Pro",
                "llm_image_id": "uuid-flux-1",
                "has_api_key": True,
                "source": "used_for",
            },
            "infographics": {
                "application": "infographics",
                "provider": "stability",
                "model_name": "sd3",
                "display_name": "Stable Diffusion 3",
                "llm_image_id": "uuid-sd3-1",
                "has_api_key": True,
                "source": "used_for",
            }
        }

        resp = self.client.get("/api/v1/images/application-config")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("applications", data)
        self.assertEqual(data["applications"]["article_image"]["model_name"], "flux-kontext-pro")
        self.assertEqual(data["applications"]["infographics"]["model_name"], "sd3")

    @patch("src.api.endpoints.images.upload_to_supabase_storage")
    @patch("src.api.endpoints.images.generate_flux_image")
    @patch("src.api.endpoints.images.resolve_image_provider")
    def test_generate_ai_image_with_application(
        self,
        mock_resolve,
        mock_flux,
        mock_upload,
    ):
        mock_resolve.return_value = {
            "provider": "flux",
            "model": "flux-kontext-pro",
            "api_key": "test-secret-key",
            "display_name": "Flux Kontext Pro",
            "llm_image_id": "uuid-1",
            "application": "article_image",
            "source": "used_for",
        }
        mock_flux.return_value = b"fake-image-bytes"
        mock_upload.return_value = "https://example.com/storage/article_img.jpg"

        payload = {
            "prompt": "Modern minimalist living room with houseplants",
            "application": "article_image",
            "user_id": "test-user-uuid",
        }

        resp = self.client.post("/api/v1/images/generate-ai", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["imageUrl"], "https://example.com/storage/article_img.jpg")
        self.assertEqual(data["model"], "flux-kontext-pro")
        self.assertEqual(data["provider"], "flux")
        self.assertEqual(data["application"], "article_image")

    @patch("src.api.endpoints.images.resolve_image_provider")
    def test_generate_ai_image_missing_key(self, mock_resolve):
        mock_resolve.return_value = {
            "provider": "flux",
            "model": "flux-kontext-pro",
            "api_key": None,  # Missing key
            "display_name": "Flux Kontext Pro",
            "llm_image_id": "uuid-1",
            "application": "article_image",
            "source": "used_for",
        }

        payload = {
            "prompt": "Modern minimalist living room",
            "application": "article_image",
            "user_id": "test-user-uuid",
        }

        resp = self.client.post("/api/v1/images/generate-ai", json=payload)
        self.assertEqual(resp.status_code, 500)
        data = resp.get_json()
        self.assertEqual(data["error"], "api_key_missing")


if __name__ == "__main__":
    unittest.main()
