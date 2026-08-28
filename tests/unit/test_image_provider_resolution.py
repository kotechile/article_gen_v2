"""
Unit tests for image model and API key resolution via used_for, llm_providers_image, and api_keys.
"""

import unittest
from unittest.mock import MagicMock, patch

from supabase_client import (
    IMAGE_APP_ARTICLE_IMAGE,
    IMAGE_APP_INFOGRAPHICS,
    _normalize_image_provider_row,
    _fetch_image_application_assignments,
    resolve_image_provider,
    get_image_provider_for_application,
    get_image_applications_config,
)


class TestImageProviderNormalization(unittest.TestCase):
    def test_normalize_valid_row(self):
        raw = {
            "id": "img-model-1",
            "model_name": "flux-kontext-pro",
            "provider": "FLUX",
            "display_name": "Flux Kontext Pro",
            "api_keys_id": "key-uuid-1",
            "is_active": True,
        }
        normalized = _normalize_image_provider_row(raw)
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["id"], "img-model-1")
        self.assertEqual(normalized["model_name"], "flux-kontext-pro")
        self.assertEqual(normalized["provider"], "flux")
        self.assertEqual(normalized["display_name"], "Flux Kontext Pro")
        self.assertEqual(normalized["api_keys_id"], "key-uuid-1")
        self.assertTrue(normalized["is_active"])

    def test_normalize_fallback_api_key_id(self):
        raw = {
            "id": "img-model-2",
            "model_name": "sd3",
            "provider": "stability",
            "api_key_id": "key-uuid-2",
        }
        normalized = _normalize_image_provider_row(raw)
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["api_keys_id"], "key-uuid-2")
        self.assertEqual(normalized["display_name"], "sd3")

    def test_normalize_empty_model_returns_none(self):
        self.assertIsNone(_normalize_image_provider_row({}))
        self.assertIsNone(_normalize_image_provider_row({"model_name": ""}))


class TestFetchImageApplicationAssignments(unittest.TestCase):
    def test_fetch_from_used_for_table(self):
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_select = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.select.return_value = mock_select

        mock_select.execute.return_value = MagicMock(
            data=[
                {"application": "article_image", "llm_image_id": "img-uuid-1"},
                {"used_for": "infographics", "llm_image_id": "img-uuid-2"},
            ]
        )

        assignments = _fetch_image_application_assignments(mock_client)
        self.assertEqual(assignments[IMAGE_APP_ARTICLE_IMAGE], "img-uuid-1")
        self.assertEqual(assignments[IMAGE_APP_INFOGRAPHICS], "img-uuid-2")

    def test_fetch_handles_aliases(self):
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_select = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.select.return_value = mock_select

        mock_select.execute.return_value = MagicMock(
            data=[
                {"application": "article-image", "llm_image_id": "img-uuid-article"},
                {"name": "infographic", "llm_image_id": "img-uuid-infographic"},
            ]
        )

        assignments = _fetch_image_application_assignments(mock_client)
        self.assertEqual(assignments[IMAGE_APP_ARTICLE_IMAGE], "img-uuid-article")
        self.assertEqual(assignments[IMAGE_APP_INFOGRAPHICS], "img-uuid-infographic")


class TestResolveImageProvider(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()

        self.sample_image_rows = [
            {
                "id": "model-id-flux",
                "model_name": "flux-kontext-pro",
                "provider": "flux",
                "display_name": "Flux Kontext Pro",
                "api_keys_id": "key-id-flux",
                "is_active": True,
            },
            {
                "id": "model-id-sd3",
                "model_name": "sd3",
                "provider": "stability",
                "display_name": "Stable Diffusion 3",
                "api_keys_id": "key-id-sd3",
                "is_active": True,
            },
        ]

        self.sample_app_assignments = {
            IMAGE_APP_ARTICLE_IMAGE: "model-id-flux",
            IMAGE_APP_INFOGRAPHICS: "model-id-sd3",
        }

        self.keys_store = {
            "key-id-flux": "secret-flux-key-12345",
            "key-id-sd3": "secret-sd3-key-67890",
        }

    def test_resolve_by_application_article_image(self):
        with patch("supabase_client.get_supabase_client", return_value=self.mock_client), \
             patch("supabase_client._fetch_image_provider_rows", return_value=self.sample_image_rows), \
             patch("supabase_client._fetch_image_application_assignments", return_value=self.sample_app_assignments), \
             patch("supabase_client._fetch_api_key_value_by_id", side_effect=lambda c, k: self.keys_store.get(k)):

            resolved = resolve_image_provider(application="article_image")

            self.assertEqual(resolved["model"], "flux-kontext-pro")
            self.assertEqual(resolved["provider"], "flux")
            self.assertEqual(resolved["api_key"], "secret-flux-key-12345")
            self.assertEqual(resolved["display_name"], "Flux Kontext Pro")
            self.assertEqual(resolved["llm_image_id"], "model-id-flux")
            self.assertEqual(resolved["application"], "article_image")
            self.assertEqual(resolved["source"], "used_for")

    def test_resolve_by_application_infographics(self):
        with patch("supabase_client.get_supabase_client", return_value=self.mock_client), \
             patch("supabase_client._fetch_image_provider_rows", return_value=self.sample_image_rows), \
             patch("supabase_client._fetch_image_application_assignments", return_value=self.sample_app_assignments), \
             patch("supabase_client._fetch_api_key_value_by_id", side_effect=lambda c, k: self.keys_store.get(k)):

            resolved = resolve_image_provider(application="infographics")

            self.assertEqual(resolved["model"], "sd3")
            self.assertEqual(resolved["provider"], "stability")
            self.assertEqual(resolved["api_key"], "secret-sd3-key-67890")
            self.assertEqual(resolved["display_name"], "Stable Diffusion 3")
            self.assertEqual(resolved["llm_image_id"], "model-id-sd3")
            self.assertEqual(resolved["application"], "infographics")
            self.assertEqual(resolved["source"], "used_for")

    def test_resolve_by_explicit_model_name(self):
        with patch("supabase_client.get_supabase_client", return_value=self.mock_client), \
             patch("supabase_client._fetch_image_provider_rows", return_value=self.sample_image_rows), \
             patch("supabase_client._fetch_image_application_assignments", return_value=self.sample_app_assignments), \
             patch("supabase_client._fetch_api_key_value_by_id", side_effect=lambda c, k: self.keys_store.get(k)):

            resolved = resolve_image_provider(model="sd3")

            self.assertEqual(resolved["model"], "sd3")
            self.assertEqual(resolved["provider"], "stability")
            self.assertEqual(resolved["api_key"], "secret-sd3-key-67890")
            self.assertEqual(resolved["source"], "explicit")

    def test_resolve_default_fallback_to_article_image(self):
        with patch("supabase_client.get_supabase_client", return_value=self.mock_client), \
             patch("supabase_client._fetch_image_provider_rows", return_value=self.sample_image_rows), \
             patch("supabase_client._fetch_image_application_assignments", return_value=self.sample_app_assignments), \
             patch("supabase_client._fetch_api_key_value_by_id", side_effect=lambda c, k: self.keys_store.get(k)):

            resolved = resolve_image_provider()

            self.assertEqual(resolved["model"], "flux-kontext-pro")
            self.assertEqual(resolved["provider"], "flux")
            self.assertEqual(resolved["api_key"], "secret-flux-key-12345")
            self.assertEqual(resolved["application"], "article_image")
            self.assertEqual(resolved["source"], "used_for")

    def test_get_image_provider_for_application_helper(self):
        with patch("supabase_client.get_supabase_client", return_value=self.mock_client), \
             patch("supabase_client._fetch_image_provider_rows", return_value=self.sample_image_rows), \
             patch("supabase_client._fetch_image_application_assignments", return_value=self.sample_app_assignments), \
             patch("supabase_client._fetch_api_key_value_by_id", side_effect=lambda c, k: self.keys_store.get(k)):

            provider, model, key = get_image_provider_for_application("article_image")
            self.assertEqual(provider, "flux")
            self.assertEqual(model, "flux-kontext-pro")
            self.assertEqual(key, "secret-flux-key-12345")

    def test_get_image_applications_config(self):
        with patch("supabase_client.get_supabase_client", return_value=self.mock_client), \
             patch("supabase_client._fetch_image_provider_rows", return_value=self.sample_image_rows), \
             patch("supabase_client._fetch_image_application_assignments", return_value=self.sample_app_assignments), \
             patch("supabase_client._fetch_api_key_value_by_id", side_effect=lambda c, k: self.keys_store.get(k)):

            config = get_image_applications_config()

            self.assertIn("article_image", config)
            self.assertIn("infographics", config)

            self.assertEqual(config["article_image"]["model_name"], "flux-kontext-pro")
            self.assertEqual(config["article_image"]["provider"], "flux")
            self.assertTrue(config["article_image"]["has_api_key"])
            self.assertNotIn("api_key", config["article_image"])

            self.assertEqual(config["infographics"]["model_name"], "sd3")
            self.assertEqual(config["infographics"]["provider"], "stability")
            self.assertTrue(config["infographics"]["has_api_key"])

    def test_resolve_by_alias_nano_banana_pro(self):
        rows_with_banana = self.sample_image_rows + [
            {
                "id": "model-id-banana",
                "model_name": "gemini-3-pro-image-preview",
                "provider": "google",
                "display_name": "Nano Banana Pro",
                "api_keys_id": "key-id-banana",
                "is_active": True,
            }
        ]
        keys = dict(self.keys_store)
        keys["key-id-banana"] = "secret-google-gemini-key"

        with patch("supabase_client.get_supabase_client", return_value=self.mock_client), \
             patch("supabase_client._fetch_image_provider_rows", return_value=rows_with_banana), \
             patch("supabase_client._fetch_image_application_assignments", return_value=self.sample_app_assignments), \
             patch("supabase_client._fetch_api_key_value_by_id", side_effect=lambda c, k: keys.get(k)):

            resolved = resolve_image_provider(model="nano banana pro")
            self.assertEqual(resolved["model"], "gemini-3-pro-image-preview")
            self.assertEqual(resolved["provider"], "google")
            self.assertEqual(resolved["api_key"], "secret-google-gemini-key")
            self.assertEqual(resolved["source"], "explicit")

    def test_resolve_by_alias_flux_2(self):
        rows_with_flux2 = self.sample_image_rows + [
            {
                "id": "model-id-flux2",
                "model_name": "flux-2/flex",
                "provider": "kie.ai",
                "display_name": "Flux 2",
                "api_keys_id": "key-id-flux2",
                "is_active": True,
            }
        ]
        keys = dict(self.keys_store)
        keys["key-id-flux2"] = "secret-kie-key"

        with patch("supabase_client.get_supabase_client", return_value=self.mock_client), \
             patch("supabase_client._fetch_image_provider_rows", return_value=rows_with_flux2), \
             patch("supabase_client._fetch_image_application_assignments", return_value=self.sample_app_assignments), \
             patch("supabase_client._fetch_api_key_value_by_id", side_effect=lambda c, k: keys.get(k)):

            resolved = resolve_image_provider(model="Flux 2")
            self.assertEqual(resolved["model"], "flux-2/flex")
            self.assertEqual(resolved["provider"], "kie.ai")
            self.assertEqual(resolved["api_key"], "secret-kie-key")


if __name__ == "__main__":
    unittest.main()
