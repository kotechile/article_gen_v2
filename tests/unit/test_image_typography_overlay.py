"""
Unit tests for Editorial Cover Typography Overlay:
- src.services.image_typography_overlay
- POST /api/v1/images/generate-overlay-copy
- POST /api/v1/images/apply-overlay
"""

import base64
import io
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

from src.services.image_typography_overlay import (
    HOOK_MAX,
    KICKER_MAX,
    SWITCH_MARGIN,
    TITLE_MAX,
    OverlayCopy,
    OverlayResult,
    _binding_rgb_and_clutter,
    _create_feathered_scrim,
    _fit_single_line,
    _fit_wrapped_title,
    _solve_scrim_alpha,
    apply_editorial_overlay,
    clean,
    contrast_ratio,
    detect_best_corner,
    generate_overlay_copy,
    relative_luminance,
)


class TestImageTypographyOverlay(unittest.TestCase):
    def test_clean_text_formatting_and_word_truncation(self):
        # 1. Strips markdown and smart quotes
        raw = '### **Analysis**: “Generative Models” — The Next Frontier.'
        cleaned = clean(raw, 40, all_caps=True)
        self.assertNotIn("*", cleaned)
        self.assertNotIn("“", cleaned)
        self.assertNotIn("”", cleaned)
        self.assertTrue(cleaned.isupper())

        # 2. Word-boundary truncation
        long_text = "This is an extremely long subtitle that will exceed the limit and must truncate safely"
        truncated = clean(long_text, 34, all_caps=False)
        self.assertLessEqual(len(truncated), 34)
        # Should end at a whole word, not split mid-word
        self.assertFalse(truncated.endswith(" "))
        self.assertTrue(long_text.startswith(truncated))

    def test_detect_best_corner_with_top_left_hysteresis(self):
        # Uniform canvas -> top-left should stay default
        uniform_img = Image.new("RGB", (1200, 675), (200, 200, 200))
        corner = detect_best_corner(uniform_img, switch_margin=SWITCH_MARGIN)
        self.assertEqual(corner, "top-left")

        # Heavy edges in top-left, top-right, bottom-left; empty in bottom-right
        cluttered_img = Image.new("RGB", (1200, 675), (255, 255, 255))
        draw = ImageDraw.Draw(cluttered_img)
        # Top-left clutter
        for y in range(0, 300, 8):
            draw.line([(0, y), (500, y)], fill=(0, 0, 0), width=3)
        # Top-right clutter
        for y in range(0, 300, 8):
            draw.line([(700, y), (1200, y)], fill=(0, 0, 0), width=3)
        # Bottom-left clutter
        for y in range(350, 675, 8):
            draw.line([(0, y), (500, y)], fill=(0, 0, 0), width=3)

        # Bottom-right is completely clean -> should switch to bottom-right
        best = detect_best_corner(cluttered_img, switch_margin=SWITCH_MARGIN)
        self.assertEqual(best, "bottom-right")

    def test_adaptive_contrast_and_dual_palette(self):
        # Dark image -> INK_DARK palette
        dark_img = Image.new("RGB", (800, 450), (15, 20, 25))
        _, meta_dark = apply_editorial_overlay(
            dark_img,
            kicker="LLM HARDWARE // COMPUTE",
            title="THE MEMORY WALL",
            hook="Bandwidth limits are reshaping inference cluster topologies.",
            corner="top-left"
        )
        self.assertEqual(meta_dark.ink_palette, "dark")
        self.assertGreaterEqual(meta_dark.contrast_ratio, 4.5)

        # Light image -> INK_LIGHT palette
        light_img = Image.new("RGB", (800, 450), (250, 248, 245))
        _, meta_light = apply_editorial_overlay(
            light_img,
            kicker="GRID CAPEX // INFRASTRUCTURE",
            title="POWER CONSUMPTION",
            hook="Data center queues strain regional electrical substations.",
            corner="top-left"
        )
        self.assertEqual(meta_light.ink_palette, "light")
        self.assertGreaterEqual(meta_light.contrast_ratio, 4.5)

    def test_title_wrapped_at_most_two_lines(self):
        font, lines, tw, th = _fit_wrapped_title(
            text="THE CONCURRENCY CEILING IN PRODUCTION",
            bold=True,
            initial_pt=50,
            max_w=300,
            max_h=150
        )
        self.assertLessEqual(len(lines), 2)
        self.assertLessEqual(tw, 300)

    def test_dimension_capping_max_width(self):
        huge_img = Image.new("RGB", (2560, 1440), (40, 40, 40))
        res_img, meta = apply_editorial_overlay(
            huge_img,
            kicker="TEST KICKER",
            title="TEST TITLE",
            hook="Test hook line.",
            max_width=1920
        )
        self.assertEqual(meta.base_width, 2560)
        self.assertEqual(meta.final_width, 1920)
        self.assertEqual(res_img.size, (1920, 1080))

    def test_feathered_scrim_generation(self):
        scrim = _create_feathered_scrim(
            corner="top-left",
            box_w=200,
            box_h=150,
            scrim_alpha=120,
            scrim_color=(0, 0, 0)
        )
        self.assertEqual(scrim.size, (200, 150))
        self.assertEqual(scrim.mode, "RGBA")


if __name__ == "__main__":
    unittest.main()
