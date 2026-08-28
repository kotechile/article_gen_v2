"""
Unit tests for Nano Banana Pro and Flux 2 image generation features:
- Resolution control (1K, 2K, 4K)
- Aspect ratio control (1:1, 16:9, etc.)
- Reference image optionality (present or absent)
"""

import base64
import json
import sys
import unittest
from unittest.mock import MagicMock, patch

import importlib.util

# Mock heavy/external framework dependencies so unit tests run purely in stdlib
for mod in [
    "flask", "flask_limiter", "flask_limiter.util", "werkzeug", "werkzeug.utils",
    "src.services.llm.providers", "src.services.infographic_llm",
    "src.core.models.errors"
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

spec = importlib.util.spec_from_file_location("src.api.endpoints.images", "src/api/endpoints/images.py")
images_mod = importlib.util.module_from_spec(spec)
sys.modules["src.api.endpoints.images"] = images_mod
spec.loader.exec_module(images_mod)

generate_google_imagen = images_mod.generate_google_imagen
generate_kie_flux_image = images_mod.generate_kie_flux_image
generate_flux_image = images_mod.generate_flux_image


class TestNanoBananaProGeneration(unittest.TestCase):
    @patch("requests.post")
    def test_gemini_text_to_image_without_reference(self, mock_post):
        fake_image_b64 = base64.b64encode(b"generated-gemini-image-bytes").decode("utf-8")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": fake_image_b64
                                }
                            }
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        image_bytes = generate_google_imagen(
            prompt="A cozy mountain cabin at sunrise",
            api_key="fake-key",
            model="gemini-3-pro-image-preview",
            aspect_ratio="16:9",
            resolution="2K",
            reference_image=None
        )

        self.assertEqual(image_bytes, b"generated-gemini-image-bytes")
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        body = call_kwargs["json"]

        # Assert no reference image in parts
        self.assertEqual(len(body["contents"][0]["parts"]), 1)
        self.assertEqual(body["contents"][0]["parts"][0]["text"], "A cozy mountain cabin at sunrise")
        # Assert aspect ratio and resolution
        self.assertEqual(body["generationConfig"]["imageConfig"]["aspectRatio"], "16:9")
        self.assertEqual(body["generationConfig"]["imageConfig"]["imageSize"], "2K")
        self.assertEqual(body["generationConfig"]["responseModalities"], ["TEXT", "IMAGE"])

    @patch("requests.post")
    def test_gemini_image_to_image_with_reference(self, mock_post):
        fake_image_b64 = base64.b64encode(b"edited-gemini-image-bytes").decode("utf-8")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": fake_image_b64
                                }
                            }
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        ref_image_bytes = b"input-reference-photo"

        image_bytes = generate_google_imagen(
            prompt="Transform this into a watercolor painting",
            api_key="fake-key",
            model="nano banana pro",  # Normalized alias
            aspect_ratio="4:3",
            resolution="1K",
            reference_image=ref_image_bytes
        )

        self.assertEqual(image_bytes, b"edited-gemini-image-bytes")
        mock_post.assert_called_once()
        body = mock_post.call_args[1]["json"]

        # Assert reference image is included in parts
        parts = body["contents"][0]["parts"]
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0]["text"], "Transform this into a watercolor painting")
        self.assertIn("inline_data", parts[1])
        self.assertEqual(parts[1]["inline_data"]["data"], base64.b64encode(ref_image_bytes).decode("utf-8"))


class TestFlux2Generation(unittest.TestCase):
    @patch("requests.get")
    @patch("requests.post")
    def test_flux2_text_to_image_without_reference(self, mock_post, mock_get):
        # Create Task response
        create_resp = MagicMock()
        create_resp.status_code = 200
        create_resp.json.return_value = {"code": 200, "data": {"taskId": "task-uuid-123"}}
        mock_post.return_value = create_resp

        # Poll task status
        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.json.return_value = {
            "code": 200,
            "data": {
                "state": "success",
                "resultJson": json.dumps({"resultUrls": ["https://cdn.example.com/output.jpg"]})
            }
        }
        # Image fetch response
        img_resp = MagicMock()
        img_resp.status_code = 200
        img_resp.content = b"flux2-image-bytes"

        mock_get.side_effect = [poll_resp, img_resp]

        image_bytes = generate_kie_flux_image(
            prompt="Futuristic city with flying cars",
            api_key="fake-kie-key",
            model="flux-2/flex-image-to-image",
            aspect_ratio="16:9",
            reference_image_urls=None,  # Reference image is absent!
            resolution="2K"
        )

        self.assertEqual(image_bytes, b"flux2-image-bytes")
        create_payload = mock_post.call_args[1]["json"]
        # Model should switch to text-to-image flux-2/flex when no reference image is provided
        self.assertEqual(create_payload["model"], "flux-2/flex")
        self.assertNotIn("input_urls", create_payload["input"])
        self.assertEqual(create_payload["input"]["aspect_ratio"], "16:9")
        self.assertEqual(create_payload["input"]["resolution"], "2K")

    @patch("requests.get")
    @patch("requests.post")
    def test_flux2_image_to_image_with_reference(self, mock_post, mock_get):
        create_resp = MagicMock()
        create_resp.status_code = 200
        create_resp.json.return_value = {"code": 200, "data": {"taskId": "task-uuid-456"}}
        mock_post.return_value = create_resp

        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.json.return_value = {
            "code": 200,
            "data": {
                "state": "success",
                "resultJson": json.dumps({"resultUrls": ["https://cdn.example.com/output2.jpg"]})
            }
        }
        img_resp = MagicMock()
        img_resp.status_code = 200
        img_resp.content = b"flux2-i2i-image-bytes"

        mock_get.side_effect = [poll_resp, img_resp]

        image_bytes = generate_kie_flux_image(
            prompt="Add neon lights to this street",
            api_key="fake-kie-key",
            model="flux-2/flex",
            aspect_ratio="3:2",
            reference_image_urls=["https://example.com/reference.jpg"],  # Reference image present
            resolution="1K"
        )

        self.assertEqual(image_bytes, b"flux2-i2i-image-bytes")
        create_payload = mock_post.call_args[1]["json"]
        # Model switches to flex-image-to-image when reference image URLs are provided
        self.assertEqual(create_payload["model"], "flux-2/flex-image-to-image")
        self.assertEqual(create_payload["input"]["input_urls"], ["https://example.com/reference.jpg"])
        self.assertEqual(create_payload["input"]["aspect_ratio"], "3:2")
        self.assertEqual(create_payload["input"]["resolution"], "1K")

    @patch.object(images_mod, "generate_kie_flux_image")
    def test_generate_flux_image_routing(self, mock_kie):
        mock_kie.return_value = b"routed-bytes"

        res = generate_flux_image(
            prompt="Abstract geometric background",
            api_key="test-key",
            model="flux-2/flex",
            aspect_ratio="1:1",
            provider="kie.ai",
            reference_image_urls=None,
            resolution="4K"
        )

        self.assertEqual(res, b"routed-bytes")
        mock_kie.assert_called_once_with(
            "Abstract geometric background",
            "test-key",
            "flux-2/flex",
            "1:1",
            reference_image_urls=None,
            resolution="4K"
        )


if __name__ == "__main__":
    unittest.main()
