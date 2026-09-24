"""
Unit tests for InfographicAIService (7 archetypes + 21 curated styles + auto detection).
"""

import unittest
from src.services.infographic_ai_service import (
    InfographicAIService,
    ARCHETYPE_DESCRIPTIONS,
    STYLE_PRESETS,
    INFOGRAPHIC_CATEGORIES
)


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

    def test_all_21_style_presets_present(self):
        expected_styles = [
            # Process & Sequential Flow
            "step_by_step_isometric",
            "timeline_modern",
            "timeline_historical_vintage",
            "user_journey_flat",
            "lifecycle_loop_watercolor",
            # Comparison & Contrast
            "side_by_side_neon",
            "pros_cons_scandinavian",
            "venn_diagram_glassmorphism",
            "quadrant_matrix_bauhaus",
            # Data & Statistics
            "corporate_dashboard_ui",
            "typography_stat_sheet_swiss",
            "geographic_map_hologram",
            "funnel_chart_neumorphism",
            # Structure & Hierarchy
            "pyramid_hierarchy_lowpoly",
            "hub_and_spoke_material",
            "anatomy_exploded_blueprint",
            "mind_map_doodle",
            # Lists & Summaries
            "checklist_synthwave",
            "top_10_listicle_popart",
            "cheat_sheet_monochrome",
            "problem_solution_duotone"
        ]
        for style_key in expected_styles:
            self.assertIn(style_key, STYLE_PRESETS)
            preset = STYLE_PRESETS[style_key]
            self.assertTrue(preset.get("name"))
            self.assertTrue(preset.get("style_tag"))
            self.assertTrue(preset.get("category_id"))
            self.assertTrue(preset.get("prompt_template"))

    def test_categories_present(self):
        categories = InfographicAIService.get_categories()
        self.assertIn("process_sequential", categories)
        self.assertIn("comparison_contrast", categories)
        self.assertIn("data_statistics", categories)
        self.assertIn("structure_hierarchy", categories)
        self.assertIn("lists_summaries", categories)

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

    def test_prompt_synthesis_isometric_flowchart(self):
        text = "How to onboard new software engineers"
        prompt, effective = InfographicAIService.synthesize_prompt(
            text=text,
            archetype="step_by_step_isometric"
        )
        self.assertEqual(effective, "step_by_step_isometric")
        self.assertIn("3D isometric infographic", prompt)
        self.assertIn("Smart Brevity", prompt)
        self.assertIn("How to onboard new software engineers", prompt)

    def test_prompt_synthesis_side_by_side_duel_comparison(self):
        text = "React vs Vue.js for enterprise frontend development"
        prompt, effective = InfographicAIService.synthesize_prompt(
            text=text,
            archetype="side_by_side_neon"
        )
        self.assertEqual(effective, "side_by_side_neon")
        self.assertIn("cyberpunk aesthetic with glowing neon accents", prompt)
        self.assertIn("React", prompt)
        self.assertIn("Vue.js", prompt)

    def test_prompt_synthesis_modern_timeline_no_antique_look(self):
        text = "The Evolution of Contextual Fidelity in Multi-Agent AI"
        prompt, effective = InfographicAIService.synthesize_prompt(
            text=text,
            archetype="timeline_modern"
        )
        self.assertEqual(effective, "timeline_modern")
        self.assertIn("sleek, modern chronological timeline", prompt)
        self.assertIn("Contemporary high-tech aesthetic", prompt)
        self.assertIn("without any antique or parchment textures", prompt)

    def test_prompt_synthesis_historical_timeline_vintage(self):
        text = "The Industrial Revolution across 18th and 19th centuries"
        prompt, effective = InfographicAIService.synthesize_prompt(
            text=text,
            archetype="timeline_historical_vintage"
        )
        self.assertEqual(effective, "timeline_historical_vintage")
        self.assertIn("vertical timeline infographic", prompt)
        self.assertIn("Vintage/retro aesthetic with muted earthy tones", prompt)
        self.assertIn("textured paper background", prompt)

    def test_default_timeline_archetype_modernized(self):
        text = "AI agents roadmap 2024 to 2026"
        prompt, effective = InfographicAIService.synthesize_prompt(
            text=text,
            archetype="timeline_historical"
        )
        self.assertEqual(effective, "timeline_historical")
        self.assertIn("sleek modern chronological timeline", prompt)
        self.assertIn("no antique scrolls or parchment textures", prompt)
        self.assertNotIn("museum-grade", prompt)

    def test_style_param_overrides_archetype(self):
        text = "10 tips to boost developer productivity"
        prompt, effective = InfographicAIService.synthesize_prompt(
            text=text,
            archetype="technical_scientific",
            style="top_10_listicle_popart"
        )
        self.assertEqual(effective, "top_10_listicle_popart")
        self.assertIn("Vintage comic book/Pop Art", prompt)

    def test_venn_diagram_adapts_to_two_items(self):
        text = "Comparing React vs Vue for interactive dashboards"
        prompt, effective = InfographicAIService.synthesize_prompt(
            text=text,
            style="venn_diagram_glassmorphism"
        )
        self.assertEqual(effective, "venn_diagram_glassmorphism")
        self.assertIn("Two intersecting circles", prompt)
        self.assertIn("React", prompt)
        self.assertIn("Vue", prompt)
        self.assertNotIn("Three intersecting circles", prompt)

    def test_venn_diagram_adapts_to_three_items(self):
        text = "The intersection of Design, Engineering, and Business in product teams"
        prompt, effective = InfographicAIService.synthesize_prompt(
            text=text,
            style="venn_diagram_glassmorphism"
        )
        self.assertEqual(effective, "venn_diagram_glassmorphism")
        self.assertIn("Three intersecting circles", prompt)
        self.assertIn("Design", prompt)
        self.assertIn("Engineering", prompt)
        self.assertIn("Business", prompt)

    def test_venn_diagram_adapts_to_four_items(self):
        text = "Overlap of Frontend, Backend, DevOps, and QA in modern software delivery"
        prompt, effective = InfographicAIService.synthesize_prompt(
            text=text,
            style="venn_diagram_glassmorphism"
        )
        self.assertEqual(effective, "venn_diagram_glassmorphism")
        self.assertIn("Four intersecting circles", prompt)
        self.assertIn("Frontend", prompt)
        self.assertIn("Backend", prompt)

    def test_lifecycle_loop_adapts_to_stages(self):
        text = "Water Cycle:\n1. Evaporation\n2. Condensation\n3. Precipitation\n4. Collection"
        prompt, effective = InfographicAIService.synthesize_prompt(
            text=text,
            style="lifecycle_loop_watercolor"
        )
        self.assertEqual(effective, "lifecycle_loop_watercolor")
        self.assertIn("4 arrows forming a closed circle", prompt)
        self.assertIn("Evaporation", prompt)
        self.assertIn("Collection", prompt)

    def test_hub_and_spoke_adapts_to_node_count(self):
        text = "Microservices platform components:\n- Auth Service\n- Billing Service\n- Database Cluster\n- Notification Queue\n- Gateway"
        prompt, effective = InfographicAIService.synthesize_prompt(
            text=text,
            style="hub_and_spoke_material"
        )
        self.assertEqual(effective, "hub_and_spoke_material")
        self.assertIn("5 surrounding nodes", prompt)
        self.assertIn("Auth Service", prompt)

    def test_user_journey_adapts_to_phases(self):
        text = "User onboarding:\n1. Discovery\n2. Sign Up\n3. First Project\n4. Team Invite\n5. Subscription"
        prompt, effective = InfographicAIService.synthesize_prompt(
            text=text,
            style="user_journey_flat"
        )
        self.assertEqual(effective, "user_journey_flat")
        self.assertIn("divided into 5 distinct phases", prompt)
        self.assertIn("Discovery", prompt)
        self.assertIn("Subscription", prompt)

    def test_funnel_adapts_to_layers(self):
        text = "Conversion funnel:\n- Website Visitors\n- Lead Signups\n- Active Trials"
        prompt, effective = InfographicAIService.synthesize_prompt(
            text=text,
            style="funnel_chart_neumorphism"
        )
        self.assertEqual(effective, "funnel_chart_neumorphism")
        self.assertIn("divided into 3 horizontal layers", prompt)
        self.assertIn("Website Visitors", prompt)

    def test_top_listicle_adapts_to_count(self):
        text = "7 Essential Habits of High-Performing Engineers"
        prompt, effective = InfographicAIService.synthesize_prompt(
            text=text,
            style="top_10_listicle_popart"
        )
        self.assertEqual(effective, "top_10_listicle_popart")
        self.assertIn("top 7 facts about", prompt)
        self.assertIn("(1-7)", prompt)


if __name__ == "__main__":
    unittest.main()

