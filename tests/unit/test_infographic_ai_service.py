"""
Unit tests for InfographicAIService (7 archetypes + auto detection).
"""

import unittest
from src.services.infographic_ai_service import InfographicAIService, ARCHETYPE_DESCRIPTIONS


class TestInfographicAIService(unittest.TestCase):
    def test_all_7_archetypes_present(self):
        expected_archetypes = [
            "technical_scientific",
            "step_by_step",
            "flowchart_whiteboard",
            "modular_explainer",
            "timeline_historical",
            "data_visualization",
            "playful_viral"
        ]
        for arch in expected_archetypes:
            self.assertIn(arch, ARCHETYPE_DESCRIPTIONS)

    def test_auto_detect_timeline(self):
        text = "The timeline and history of modern computing started in the 20th century, reaching the 1980s."
        detected = InfographicAIService.auto_detect_archetype(text)
        self.assertEqual(detected, "timeline_historical")

    def test_auto_detect_step_by_step(self):
        text = "Here is a recipe and step 1 preparation guide for assembling the device."
        detected = InfographicAIService.auto_detect_archetype(text)
        self.assertEqual(detected, "step_by_step")

    def test_auto_detect_flowchart(self):
        text = "A logic flowchart and decision tree to determine if yes or if no."
        detected = InfographicAIService.auto_detect_archetype(text)
        self.assertEqual(detected, "flowchart_whiteboard")

    def test_auto_detect_data_viz(self):
        text = "Quarterly revenue grew by 45% with strong growth rate and positive roi."
        detected = InfographicAIService.auto_detect_archetype(text)
        self.assertEqual(detected, "data_visualization")

    def test_auto_detect_technical_scientific(self):
        text = "Deploying Kubernetes pods across distributed architecture nodes via database protocols."
        detected = InfographicAIService.auto_detect_archetype(text)
        self.assertEqual(detected, "technical_scientific")

    def test_auto_detect_playful_viral(self):
        text = "10 humorous life hacks and funny tips in a playful illustrated menu."
        detected = InfographicAIService.auto_detect_archetype(text)
        self.assertEqual(detected, "playful_viral")

    def test_prompt_synthesis_explicit_archetype(self):
        text = "Photosynthesis converting sunlight into energy inside chloroplasts."
        prompt, archetype = InfographicAIService.synthesize_prompt(
            text=text,
            archetype="technical_scientific",
            user_instructions="Use emerald green tones"
        )
        self.assertEqual(archetype, "technical_scientific")
        self.assertIn("scientific and technical infographic diagram", prompt)
        self.assertIn("Photosynthesis", prompt)
        self.assertIn("Creative Instructions: Use emerald green tones", prompt)

    def test_prompt_synthesis_auto_archetype(self):
        text = "Step 1: Whisk eggs. Step 2: Heat skillet. Step 3: Serve."
        prompt, archetype = InfographicAIService.synthesize_prompt(
            text=text,
            archetype="auto"
        )
        self.assertEqual(archetype, "step_by_step")
        self.assertIn("step-by-step visual instructional infographic", prompt)


if __name__ == "__main__":
    unittest.main()
