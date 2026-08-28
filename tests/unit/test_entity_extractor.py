"""
Unit tests for EntityExtractor.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from src.services.context_image.entity_extractor import EntityExtractor, EntityExtractionResult


class TestEntityExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = EntityExtractor(
            provider="gemini",
            model="gemini-2.5-flash",
            api_key="test-fake-key"
        )

    def test_extract_empty_text_returns_empty_result(self):
        result = self.extractor.extract("")
        self.assertFalse(result.has_physical_entity)
        self.assertEqual(result.main_object, "")
        self.assertEqual(result.search_query, "")
        self.assertEqual(result.generation_prompt, "")

    @patch.object(EntityExtractor, "_call_llm")
    def test_extract_valid_entity(self, mock_call_llm):
        mock_call_llm.return_value = json.dumps({
            "has_physical_entity": True,
            "main_object": "Apple Watch Ultra 2",
            "search_query": "Apple Watch Ultra 2 product photo white background studio",
            "generation_prompt": "A close-up shot of an Apple Watch Ultra 2 on the wrist of an open-water swimmer cutting through ocean waves, water droplets splashing, cinematic lighting, 35mm photography",
            "object_fidelity_weight": 0.8
        })

        article_text = "The Apple Watch Ultra 2 represents Apple's most rugged wearable yet, with high water resistance and a bright display."
        result = self.extractor.extract(article_text)

        self.assertTrue(result.has_physical_entity)
        self.assertEqual(result.main_object, "Apple Watch Ultra 2")
        self.assertIn("Apple Watch Ultra 2", result.search_query)
        self.assertIn("swimmer", result.generation_prompt)
        self.assertEqual(result.object_fidelity_weight, 0.8)

    @patch.object(EntityExtractor, "_call_llm")
    def test_extract_handles_markdown_code_fences(self, mock_call_llm):
        mock_call_llm.return_value = """```json
{
  "has_physical_entity": true,
  "main_object": "1984 Macintosh 128k",
  "search_query": "1984 Macintosh 128k computer studio photo",
  "generation_prompt": "A vintage 1984 Macintosh 128k on a minimalist dark walnut desk, warm lighting",
  "object_fidelity_weight": 0.75
}
```"""

        result = self.extractor.extract("The design of the original 1984 Macintosh 128k set the tone for personal computing.")
        self.assertTrue(result.has_physical_entity)
        self.assertEqual(result.main_object, "1984 Macintosh 128k")
        self.assertEqual(result.search_query, "1984 Macintosh 128k computer studio photo")

    @patch.object(EntityExtractor, "_call_llm")
    def test_extract_fallback_on_malformed_json(self, mock_call_llm):
        mock_call_llm.return_value = "Non-JSON response error from model"

        text = "Sony A7 IV camera with 24-70mm lens"
        result = self.extractor.extract(text)

        self.assertTrue(result.has_physical_entity)
        self.assertIn("Sony", result.main_object)
        self.assertIn("product photo", result.search_query)

    @patch.object(EntityExtractor, "_call_llm")
    def test_extract_metaphorical_object_for_abstract_text(self, mock_call_llm):
        mock_call_llm.return_value = json.dumps({
            "has_physical_entity": False,
            "entity_type": "metaphorical",
            "is_metaphorical": True,
            "main_object": "An antique brass balancing scale with coins and feathers",
            "search_query": "antique brass balancing scale with coins and feathers studio photo",
            "generation_prompt": "An antique brass scale finely balanced between gleaming gold coins and white feathers on a polished oak surface, soft dramatic chiaroscuro lighting, 35mm editorial photography",
            "object_fidelity_weight": 0.60
        })

        abstract_text = "Navigating market inflation and interest rate policy requires a delicate monetary balancing act."
        result = self.extractor.extract(abstract_text)

        self.assertFalse(result.has_physical_entity)
        self.assertTrue(result.is_metaphorical)
        self.assertEqual(result.entity_type, "metaphorical")
        self.assertEqual(result.main_object, "An antique brass balancing scale with coins and feathers")
        self.assertIn("scale", result.search_query)
        self.assertEqual(result.object_fidelity_weight, 0.60)


if __name__ == "__main__":
    unittest.main()
