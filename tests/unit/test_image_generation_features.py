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
generate_kie_image = images_mod.generate_kie_image
generate_kie_flux_image = images_mod.generate_kie_flux_image
generate_flux_image = images_mod.generate_flux_image
generate_image_with_provider = images_mod.generate_image_with_provider


class TestKieBananaProGeneration(unittest.TestCase):
    @patch("requests.get")
    @patch("requests.post")
    def test_kie_banana_pro_generation_without_reference(self, mock_post, mock_get):
        create_resp = MagicMock()
        create_resp.status_code = 200
        create_resp.json.return_value = {"code": 200, "data": {"taskId": "task-banana-123"}}
        mock_post.return_value = create_resp

        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.json.return_value = {
            "code": 200,
            "data": {
                "state": "success",
                "resultJson": json.dumps({"resultUrls": ["https://cdn.kie.ai/output-banana.png"]})
            }
        }
        img_resp = MagicMock()
        img_resp.status_code = 200
        img_resp.content = b"kie-banana-image-bytes"

        mock_get.side_effect = [poll_resp, img_resp]

        image_bytes = generate_kie_image(
            prompt="Comic poster of cool banana hero in shades",
            api_key="sk-kie-secret",
            model="nano-banana-pro",
            aspect_ratio="1:1",
            resolution="1K",
            reference_image_urls=None
        )

        self.assertEqual(image_bytes, b"kie-banana-image-bytes")
        mock_post.assert_called_once()
        create_url = mock_post.call_args[0][0]
        self.assertEqual(create_url, "https://api.kie.ai/api/v1/jobs/createTask")
        headers = mock_post.call_args[1]["headers"]
        self.assertEqual(headers["Authorization"], "Bearer sk-kie-secret")
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["model"], "nano-banana-pro")
        self.assertEqual(payload["input"]["prompt"], "Comic poster of cool banana hero in shades")
        self.assertEqual(payload["input"]["aspect_ratio"], "1:1")
        self.assertEqual(payload["input"]["resolution"], "1K")
        self.assertEqual(payload["input"]["output_format"], "png")
        self.assertEqual(payload["input"]["image_input"], [])


class TestGenerateImageWithProviderRouting(unittest.TestCase):
    @patch.object(images_mod, "generate_kie_image")
    def test_routing_to_kie_for_nano_banana_pro(self, mock_kie):
        mock_kie.return_value = b"kie-bytes"
        res = generate_image_with_provider(
            prompt="A sample prompt",
            provider="kie.ai",
            model="nano-banana-pro",
            api_key="kie-key",
            aspect_ratio="16:9",
            resolution="1K"
        )
        self.assertEqual(res, b"kie-bytes")
        mock_kie.assert_called_once_with(
            prompt="A sample prompt",
            api_key="kie-key",
            model="nano-banana-pro",
            aspect_ratio="16:9",
            reference_image_urls=None,
            resolution="1K"
        )

    @patch.object(images_mod, "generate_google_imagen")
    def test_routing_to_google_for_gemini(self, mock_google):
        mock_google.return_value = b"google-bytes"
        res = generate_image_with_provider(
            prompt="A sample prompt",
            provider="google",
            model="gemini-3-pro-image-preview",
            api_key="goog-key",
            aspect_ratio="16:9",
            resolution="1K"
        )
        self.assertEqual(res, b"google-bytes")
        mock_google.assert_called_once_with(
            prompt="A sample prompt",
            api_key="goog-key",
            model="gemini-3-pro-image-preview",
            aspect_ratio="16:9",
            resolution="1K",
            reference_image=None
        )


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
        # Model should normalize to text-to-image flux-2/flex-text-to-image when no reference image is provided
        self.assertEqual(create_payload["model"], "flux-2/flex-text-to-image")
        self.assertNotIn("input_urls", create_payload["input"])
        self.assertEqual(create_payload["input"]["aspect_ratio"], "16:9")
        self.assertEqual(create_payload["input"]["resolution"], "2K")

    @patch("requests.get")
    @patch("requests.post")
    def test_flux2_pro_text_to_image_normalization(self, mock_post, mock_get):
        create_resp = MagicMock()
        create_resp.status_code = 200
        create_resp.json.return_value = {"code": 200, "data": {"taskId": "task-uuid-pro"}}
        mock_post.return_value = create_resp

        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.json.return_value = {
            "code": 200,
            "data": {
                "state": "success",
                "resultJson": json.dumps({"resultUrls": ["https://cdn.example.com/output-pro.jpg"]})
            }
        }
        img_resp = MagicMock()
        img_resp.status_code = 200
        img_resp.content = b"flux2-pro-image-bytes"

        mock_get.side_effect = [poll_resp, img_resp]

        image_bytes = generate_kie_flux_image(
            prompt="Hyperrealistic photo of an ancient temple",
            api_key="fake-kie-key",
            model="flux-2/pro",
            aspect_ratio="1:1",
            reference_image_urls=None,
            resolution="1K"
        )

        self.assertEqual(image_bytes, b"flux2-pro-image-bytes")
        create_payload = mock_post.call_args[1]["json"]
        self.assertEqual(create_payload["model"], "flux-2/pro-text-to-image")

    @patch("requests.get")
    @patch("requests.post")
    def test_flux2_pro_500_fallback_to_flex(self, mock_post, mock_get):
        # Attempt 1 (Pro) creates task 101, but poll returns fail with 500 Internal Error
        create_resp_pro = MagicMock()
        create_resp_pro.status_code = 200
        create_resp_pro.json.return_value = {"code": 200, "data": {"taskId": "task-pro-failed"}}

        # Attempt 2 (Flex fallback) creates task 102, poll returns success
        create_resp_flex = MagicMock()
        create_resp_flex.status_code = 200
        create_resp_flex.json.return_value = {"code": 200, "data": {"taskId": "task-flex-success"}}

        mock_post.side_effect = [create_resp_pro, create_resp_flex]

        # Poll 1 (Pro): fail with 500 Internal Error
        poll_fail = MagicMock()
        poll_fail.status_code = 200
        poll_fail.json.return_value = {
            "code": 200,
            "data": {
                "state": "fail",
                "failCode": "500",
                "failMsg": "Internal Error"
            }
        }

        # Poll 2 (Flex): success
        poll_success = MagicMock()
        poll_success.status_code = 200
        poll_success.json.return_value = {
            "code": 200,
            "data": {
                "state": "success",
                "resultJson": json.dumps({"resultUrls": ["https://cdn.example.com/output-flex.jpg"]})
            }
        }

        # Image download
        img_resp = MagicMock()
        img_resp.status_code = 200
        img_resp.content = b"flux2-flex-fallback-bytes"

        mock_get.side_effect = [poll_fail, poll_success, img_resp]

        image_bytes = generate_kie_flux_image(
            prompt="High tech dashboard with data visualizations",
            api_key="fake-kie-key",
            model="flux-2/pro-text-to-image",
            aspect_ratio="16:9",
            reference_image_urls=None,
            resolution="1K"
        )

        self.assertEqual(image_bytes, b"flux2-flex-fallback-bytes")
        self.assertEqual(mock_post.call_count, 2)
        # First call was pro
        self.assertEqual(mock_post.call_args_list[0][1]["json"]["model"], "flux-2/pro-text-to-image")
        # Second call was flex fallback
        self.assertEqual(mock_post.call_args_list[1][1]["json"]["model"], "flux-2/flex-text-to-image")

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

    @patch.object(images_mod, "generate_kie_image")
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

