"""
Unit tests for ContextImagePipeline.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.services.context_image.context_pipeline import ContextImagePipeline
from src.services.context_image.entity_extractor import EntityExtractionResult
from src.services.context_image.reference_search import ReferenceImageItem


class TestContextImagePipeline(unittest.TestCase):
    def setUp(self):
        self.mock_extractor = MagicMock()
        self.mock_search = MagicMock()
        self.mock_preprocessor = MagicMock()

        self.pipeline = ContextImagePipeline(
            entity_extractor=self.mock_extractor,
            search_client=self.mock_search,
            preprocessor=self.mock_preprocessor
        )

    def test_analyze_context_flow(self):
        self.mock_extractor.extract.return_value = EntityExtractionResult(
            has_physical_entity=True,
            main_object="Porsche 911 GT3",
            search_query="Porsche 911 GT3 clean studio photo",
            generation_prompt="A Porsche 911 GT3 speeding through a mountain pass at sunset",
            object_fidelity_weight=0.85
        )

        self.mock_search.search_reference_images.return_value = [
            ReferenceImageItem(
                url="https://images.example.com/porsche_gt3.jpg",
                thumbnail_url="https://images.example.com/porsche_gt3_thumb.jpg",
                title="Porsche 911 GT3 Front",
                source_domain="porsche.com",
                provider="tavily"
            )
        ]

        text = "The new Porsche 911 GT3 delivers race-bred performance with an atmospheric engine."
        res = self.pipeline.analyze_context(text)

        self.assertTrue(res["has_physical_entity"])
        self.assertEqual(res["main_object"], "Porsche 911 GT3")
        self.assertEqual(res["search_query"], "Porsche 911 GT3 clean studio photo")
        self.assertEqual(len(res["candidate_references"]), 1)
        self.assertEqual(res["candidate_references"][0]["url"], "https://images.example.com/porsche_gt3.jpg")

    def test_analyze_context_skips_search_for_metaphorical_concept(self):
        self.mock_extractor.extract.return_value = EntityExtractionResult(
            has_physical_entity=False,
            entity_type="metaphorical",
            is_metaphorical=True,
            main_object="Antique brass balancing scale",
            search_query="",
            generation_prompt="An antique brass scale weighing gold coins against feathers, 35mm editorial photography",
            object_fidelity_weight=0.0
        )

        text = "Monetary inflation requires a delicate balancing act by the Federal Reserve."
        res = self.pipeline.analyze_context(text)

        self.assertFalse(res["has_physical_entity"])
        self.assertTrue(res["is_metaphorical"])
        self.assertEqual(res["candidate_references"], [])
        self.mock_search.search_reference_images.assert_not_called()

    def test_prepare_reference_asset(self):
        self.mock_preprocessor.prepare_reference.return_value = (b"image-bytes", "b64string")

        ref_bytes, ref_url = self.pipeline.prepare_reference_asset(
            reference_url="https://images.example.com/sample.jpg",
            isolate_bg=False,
            user_id=None
        )

        self.assertEqual(ref_bytes, b"image-bytes")
        self.assertEqual(ref_url, "https://images.example.com/sample.jpg")


if __name__ == "__main__":
    unittest.main()
