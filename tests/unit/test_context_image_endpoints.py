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
synthesize_prompt_endpoint = images_mod.synthesize_image_prompt_endpoint
generate_overlay_copy_endpoint = images_mod.generate_overlay_copy_endpoint
apply_image_overlay_endpoint = images_mod.apply_image_overlay_endpoint


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

    @patch.object(images_mod, "request")
    @patch.object(images_mod, "resolve_image_provider")
    @patch.object(images_mod, "generate_google_imagen")
    @patch.object(images_mod, "upload_to_supabase_storage")
    @patch("src.services.context_image.ContextImagePipeline.prepare_reference_asset")
    def test_context_generate_endpoint_with_uploaded_base64_reference(
        self,
        mock_prepare,
        mock_upload,
        mock_generate,
        mock_resolve,
        mock_request
    ):
        mock_request.get_json.return_value = {
            "text": "Residential heat pump unit installed outside a home",
            "prompt": "A modern heat pump unit glowing with soft blue light outside a home",
            "reference_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "model": "nano banana pro",
            "aspectRatio": "16:9",
            "resolution": "1K",
            "user_id": "user-456"
        }

        mock_prepare.return_value = (b"uploaded-ref-bytes", "https://storage.supabase.co/user-456/context_refs/ref.jpg")
        mock_resolve.return_value = {
            "provider": "google",
            "model": "gemini-3-pro-image-preview",
            "api_key": "fake-google-key",
            "display_name": "Nano Banana Pro"
        }
        mock_generate.return_value = b"generated-heat-pump-image"
        mock_upload.return_value = "https://storage.supabase.co/heat_pump_scene.jpg"

        with patch.object(images_mod, "jsonify", side_effect=lambda x: x):
            res, status = generate_context_image_endpoint()
            self.assertEqual(status, 200)
            self.assertEqual(res["imageUrl"], "https://storage.supabase.co/heat_pump_scene.jpg")
            mock_prepare.assert_called_once()
            self.assertIn("reference_base64", str(mock_prepare.call_args))

    @patch.object(images_mod, "request")
    @patch("src.services.context_image.entity_extractor.EntityExtractor.extract")
    def test_synthesize_prompt_endpoint_with_art_direction(self, mock_extract, mock_request):
        from src.services.context_image.entity_extractor import EntityExtractionResult
        mock_request.get_json.return_value = {
            "text": "Tesla gigafactory battery cell bottleneck causing delays in Model Y ramp.",
            "style": "Cinematic Still",
            "style_id": "cinematic_still",
            "article_title": "The Scaling Ceiling",
            "article_context": {"thesis": "Battery scaling limits EV velocity", "vertical": "EV / Energy"}
        }

        mock_extract.return_value = EntityExtractionResult(
            has_physical_entity=True,
            main_object="Battery cell packaging line",
            search_query="cylindrical battery cell assembly line",
            generation_prompt="A 35mm photograph of cylindrical lithium cells moving down an automated inspection conveyor.",
            entity_type="industrial_machinery",
            object_fidelity_weight=0.92,
            is_metaphorical=False,
            hero_subject="Automated cylindrical battery cell assembly conveyor",
            core_thesis="Battery scaling limits EV velocity",
            core_conflict="Precision manufacturing cadence versus supply-chain latency",
            composition="Low-angle perspective, leading lines of conveyor receding to upper right",
            style_id="cinematic_still",
            style_label="Cinematic Still",
            alt_text="Battery cell conveyor line with cylindrical cells moving under inspection lights",
            caption="Cylindrical cells moving through high-speed automated sorting.",
            title="Battery Line Velocity",
            negative_prompt="Do not include: text, lettering, numbers, logos, watermarks, UI."
        )

        with patch.object(images_mod, "jsonify", side_effect=lambda x: x):
            res, status = synthesize_prompt_endpoint()
            self.assertEqual(status, 200)
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["hero_subject"], "Automated cylindrical battery cell assembly conveyor")
            self.assertEqual(res["core_thesis"], "Battery scaling limits EV velocity")
            self.assertEqual(res["core_conflict"], "Precision manufacturing cadence versus supply-chain latency")
            self.assertEqual(res["composition"], "Low-angle perspective, leading lines of conveyor receding to upper right")
            self.assertEqual(res["style_id"], "cinematic_still")
            self.assertEqual(res["alt_text"], "Battery cell conveyor line with cylindrical cells moving under inspection lights")
            self.assertIn("A 35mm photograph", res["prompt"])

    @patch.object(images_mod, "request")
    @patch("src.services.image_typography_overlay.generate_overlay_copy")
    def test_generate_overlay_copy_endpoint(self, mock_copy, mock_request):
        from src.services.image_typography_overlay import OverlayCopy
        mock_request.get_json.return_value = {
            "text": "NVIDIA Blackwell B200 packaging delays cause hyperscale allocation queue.",
            "article_title": "AI Accelerator Constraints",
            "article_context": {"vertical": "AI Hardware", "thesis": "Packaging latency limits throughput"}
        }
        mock_copy.return_value = OverlayCopy(
            kicker="AI HARDWARE // PACKAGING BOTTLENECK",
            title="ACCELERATOR CONSTRAINTS",
            hook="Packaging delays constrain hyperscale throughput."
        )

        with patch.object(images_mod, "jsonify", side_effect=lambda x: x):
            res, status = generate_overlay_copy_endpoint()
            self.assertEqual(status, 200)
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["kicker"], "AI HARDWARE // PACKAGING BOTTLENECK")
            self.assertEqual(res["title"], "ACCELERATOR CONSTRAINTS")
            self.assertEqual(res["hook"], "Packaging delays constrain hyperscale throughput.")

    @patch.object(images_mod, "request")
    @patch.object(images_mod, "upload_to_supabase_storage")
    def test_apply_image_overlay_endpoint_with_base64(self, mock_upload, mock_request):
        # 100x100 base64 jpeg
        fake_b64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCABkAGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDxuiiitjIKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA//9k="
        mock_request.get_json.return_value = {
            "image_base64": fake_b64,
            "kicker": "ENERGY // STORAGE",
            "title": "GRID POWER",
            "hook": "Utility batteries stabilize frequency fluctuations.",
            "corner": "top-left",
            "user_id": "test-user"
        }
        mock_upload.return_value = "https://storage.supabase.co/overlay_123.jpg"

        with patch.object(images_mod, "jsonify", side_effect=lambda x: x):
            res, status = apply_image_overlay_endpoint()
            self.assertEqual(status, 200)
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["imageUrl"], "https://storage.supabase.co/overlay_123.jpg")
            self.assertIn("overlayDetails", res)
            self.assertEqual(res["overlayDetails"]["kicker"], "ENERGY // STORAGE")
            self.assertEqual(res["overlayDetails"]["title"], "GRID POWER")


if __name__ == "__main__":
    unittest.main()
