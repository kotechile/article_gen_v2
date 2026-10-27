"""Section count must follow the requested length.

The complaint this answers: a 2,500-word request produced ten sections, so every
section was a ~250-word summary and the article read as a list of abstracts. Depth
per section is a planning decision, not a writing one.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from article_structure_generator import (  # noqa: E402
    SECTION_WORDS_MAX,
    SECTION_WORDS_MIN,
    plan_section_budget,
)


class PlanSectionBudgetTests(unittest.TestCase):
    def test_2500_words_is_four_to_five_sections(self):
        plan = plan_section_budget(2500)
        self.assertIn(plan["section_count"], (4, 5))
        self.assertGreaterEqual(plan["words_per_section"], 450)
        self.assertLessEqual(plan["words_per_section"], 700)

    def test_section_words_stay_in_the_band_for_typical_lengths(self):
        # 800 words cannot reach 450/section with two sections; everything above can.
        for target in (900, 1200, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000):
            plan = plan_section_budget(target)
            self.assertGreaterEqual(
                plan["words_per_section"], SECTION_WORDS_MIN, f"{target} produced {plan}"
            )
            self.assertLessEqual(
                plan["words_per_section"], SECTION_WORDS_MAX, f"{target} produced {plan}"
            )

    def test_word_budget_adds_up_to_the_request(self):
        for target in (1200, 2500, 5000, 8000):
            plan = plan_section_budget(target)
            total = plan["section_count"] * plan["words_per_section"]
            self.assertLessEqual(abs(total - target), target * 0.15, f"{target}: {plan}")

    def test_short_piece_keeps_a_floor_of_sections(self):
        plan = plan_section_budget(300)
        self.assertGreaterEqual(plan["section_count"], 2)

    def test_long_piece_caps_the_section_count(self):
        plan = plan_section_budget(40000)
        self.assertLessEqual(plan["section_count"], 12)

    def test_count_is_monotonic(self):
        counts = [plan_section_budget(t)["section_count"] for t in (600, 1200, 2500, 4000, 6000)]
        self.assertEqual(counts, sorted(counts))

    def test_band_is_derived_from_the_target(self):
        plan = plan_section_budget(2500)
        self.assertLess(plan["min_words"], plan["words_per_section"])
        self.assertGreater(plan["max_words"], plan["words_per_section"])


if __name__ == "__main__":
    unittest.main()
