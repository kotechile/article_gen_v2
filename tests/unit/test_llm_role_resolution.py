"""Tests for LLM role resolution: silent downgrades must become loud.

Production case this reproduces: `final_review` was assigned to
gemini/gemini-3.1-pro-preview, whose api_keys row had an empty key_value. The
resolver skipped it without a word, fell through to the `is_default` row
(gemini-3.8-flash - inactive, and mapped to research topic generation), and the
long-form polishing pass ran on a model nobody had chosen for it.
"""

import sys
import unittest
from unittest.mock import patch

# Other unit tests stub sys.modules["supabase_client"] with a SimpleNamespace to
# import tasks.py in isolation. Drop such a stub so this suite always binds the
# real module, whatever order pytest collects in.
_stub = sys.modules.get("supabase_client")
if _stub is not None and not hasattr(_stub, "_fetch_llm_provider_rows"):
    del sys.modules["supabase_client"]

import supabase_client as sc

PROVIDERS = [
    {
        "id": "prov-strong",
        "name": "gemini-3.1-pro-preview",
        "provider": "gemini",
        "model_name": "gemini-3.1-pro-preview",
        "api_keys_id": "key-missing",
        "base_url": None,
        "is_default": False,
        "is_active": False,
        "used_for": [],
    },
    {
        "id": "prov-research",
        "name": "gemini-3.8-flash",
        "provider": "gemini",
        "model_name": "gemini-3.8-flash",
        "api_keys_id": "key-present",
        "base_url": None,
        "is_default": True,
        "is_active": False,
        "used_for": [],
    },
]

ROLE_ASSIGNMENTS = {"final_review": "prov-strong"}
KEYS = {"key-missing": "", "key-present": "x" * 39}


class RoleResolutionTests(unittest.TestCase):
    def _resolve(self, role_assignments):
        with patch.object(sc, "get_supabase_client", return_value=object()), \
             patch.object(sc, "_fetch_llm_provider_rows", return_value=[dict(r) for r in PROVIDERS]), \
             patch.object(sc, "_fetch_llm_role_assignments", return_value=dict(role_assignments)), \
             patch.object(sc, "_fetch_api_key_value_by_id", side_effect=lambda _c, kid: KEYS.get(kid, "")):
            return sc.resolve_llm_provider(task_role="final_review")

    def test_downgrade_is_reported_not_silent(self):
        resolved = self._resolve(ROLE_ASSIGNMENTS)
        self.assertEqual(resolved["model"], "gemini-3.8-flash")
        self.assertEqual(resolved["source"], "default")
        warning = resolved.get("resolution_warning") or ""
        self.assertIn("gemini-3.1-pro-preview", warning)
        self.assertIn("no key_value", warning)
        self.assertIn("Falling back to", warning)

    def test_assigned_model_with_a_key_is_used(self):
        with patch.dict(KEYS, {"key-missing": "y" * 39}):
            resolved = self._resolve(ROLE_ASSIGNMENTS)
        self.assertEqual(resolved["model"], "gemini-3.1-pro-preview")
        self.assertEqual(resolved["source"], "task_role")
        self.assertIsNone(resolved.get("resolution_warning"))

    def test_no_warning_when_there_is_nothing_skipped(self):
        resolved = self._resolve({})
        self.assertIsNone(resolved.get("resolution_warning"))

    def test_internal_links_uses_used_for_article_generation(self):
        providers = [
            {
                "id": "prov-deepseek",
                "name": "deepseek-flash",
                "provider": "deepseek",
                "model_name": "deepseek-flash",
                "api_keys_id": "key-ds",
                "base_url": None,
                "is_default": False,
                "is_active": True,
                "used_for": [],
            },
            {
                "id": "prov-gemini",
                "name": "gemini-3.8-flash",
                "provider": "gemini",
                "model_name": "gemini-3.8-flash",
                "api_keys_id": "key-gemini",
                "base_url": None,
                "is_default": True,
                "is_active": False,
                "used_for": [],
            },
        ]
        role_map = {"article_generation": "prov-deepseek"}
        keys = {"key-ds": "sk-deepseek-key-12345", "key-gemini": "gemini-key"}

        with patch.object(sc, "get_supabase_client", return_value=object()), \
             patch.object(sc, "_fetch_llm_provider_rows", return_value=providers), \
             patch.object(sc, "_fetch_llm_role_assignments", return_value=role_map), \
             patch.object(sc, "_fetch_api_key_value_by_id", side_effect=lambda _c, kid: keys.get(kid, "")):
            # internal_links should alias to article_generation from used_for
            resolved = sc.resolve_llm_provider(task_role="internal_links")
            self.assertEqual(resolved["provider"], "deepseek")
            self.assertEqual(resolved["model"], "deepseek-flash")

            # task_role=None should also consult used_for
            resolved_default = sc.resolve_llm_provider()
            self.assertEqual(resolved_default["provider"], "deepseek")
            self.assertEqual(resolved_default["model"], "deepseek-flash")


if __name__ == "__main__":
    unittest.main()
