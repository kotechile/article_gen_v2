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

    def test_analyze_context_with_art_direction(self):
        self.mock_extractor.extract.return_value = EntityExtractionResult(
            has_physical_entity=True,
            main_object="Electric Vehicle backing up home during blackout",
            hero_subject="Modern electric vehicle connected via umbilical cable",
            core_thesis="EVs are becoming decentralized residential power plants",
            core_conflict="Grid vulnerability vs localized vehicle storage capacity",
            composition="Low-angle asymmetric 16:9 framing, vehicle hero off-centre",
            style_id="cinematic_still",
            style_label="Cinematic still",
            alt_text="Modern electric vehicle connected to dark house during outage",
            caption="An electric vehicle powers a residential home during a blackout.",
            title="Vehicle to Home Energy Backup",
            negative_prompt="text, lettering, numbers, watermark, UI",
            search_query="electric vehicle home backup",
            generation_prompt="A cinematic 35mm photograph of an unbranded modern electric vehicle parked on a residential driveway at dusk",
            object_fidelity_weight=0.75
        )

        res = self.pipeline.analyze_context(
            text="Vehicle to home power solves grid outages.",
            article_title="How EVs Power the Grid",
            article_context={"vertical": "residential / energy", "one_big_thing": "EV backup transforms homes"}
        )

        self.assertEqual(res["hero_subject"], "Modern electric vehicle connected via umbilical cable")
        self.assertEqual(res["core_thesis"], "EVs are becoming decentralized residential power plants")
        self.assertEqual(res["core_conflict"], "Grid vulnerability vs localized vehicle storage capacity")
        self.assertEqual(res["composition"], "Low-angle asymmetric 16:9 framing, vehicle hero off-centre")
        self.assertEqual(res["alt_text"], "Modern electric vehicle connected to dark house during outage")
        self.assertEqual(res["style_id"], "cinematic_still")

    def test_alt_text_sanitation(self):
        from src.services.context_image.entity_extractor import EntityExtractor
        sanitized = EntityExtractor._sanitize_alt_text("A cinematic photograph of an electric vehicle charging station")
        self.assertEqual(sanitized, "an electric vehicle charging station")

        sanitized_render = EntityExtractor._sanitize_alt_text("3D render showing modular circuit components on desk")
        self.assertEqual(sanitized_render, "modular circuit components on desk")

    def test_entity_extractor_preserves_named_brands(self):
        from src.services.context_image.entity_extractor import SYSTEM_PROMPT, EDITORIAL_STYLES, IMAGE_GUARD

        # Ensure system prompt mandates brand preservation
        self.assertIn("PRESERVE NAMED BRANDS AND MODELS", SYSTEM_PROMPT)
        self.assertIn("Tesla Model Y", SYSTEM_PROMPT)
        self.assertIn("NEVER genericize them into 'an unbranded vehicle'", SYSTEM_PROMPT)
        self.assertIn("ALWAYS PRESERVE NAMED BRANDS & MODELS", SYSTEM_PROMPT)

        # Ensure editorial styles do not enforce generic unbranded objects
        self.assertNotIn("generic unbranded object", EDITORIAL_STYLES["studio_object"]["craft"])
        self.assertIn("authentic product design and silhouette", EDITORIAL_STYLES["studio_object"]["craft"])

        # Ensure wordmarks is guarded without forbidding factory badges or car models
        self.assertIn("wordmarks", IMAGE_GUARD)


if __name__ == "__main__":
    unittest.main()

