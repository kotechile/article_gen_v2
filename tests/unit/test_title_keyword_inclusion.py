"""Tests for focus-keyword inclusion in titles.

The title engine scored clickability only: a headline could be punchy and rank
for nothing. These cover the keyword dimension and the deterministic insertion
that runs after the model has had its turn.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.services.viral_title_engine import ViralTitleEngine

KEYWORD = "agentic rag"


class KeywordStatusTests(unittest.TestCase):
    def test_exact_phrase_detected(self):
        status = ViralTitleEngine.keyword_status("Agentic RAG: Why Pilots Stall", KEYWORD)
        self.assertEqual(status["status"], "exact")
        self.assertTrue(status["front_loaded"])

    def test_reordered_tokens_are_partial_not_exact(self):
        # Both keyword tokens present, but not as the phrase.
        status = ViralTitleEngine.keyword_status("Agentic Pipelines Break RAG Agents", KEYWORD)
        self.assertEqual(status["status"], "partial")
        self.assertFalse(status["exact"])

    def test_missing_keyword_detected(self):
        self.assertEqual(ViralTitleEngine.keyword_status("Why Pilots Stall", KEYWORD)["status"], "missing")

    def test_no_keyword_is_not_evaluated(self):
        self.assertEqual(ViralTitleEngine.keyword_status("Anything", "")["status"], "not_evaluated")


class ScoreTests(unittest.TestCase):
    def test_keyword_raises_the_score(self):
        title = "Agentic RAG Pilots Stall"
        without = ViralTitleEngine.evaluate_title_heuristics(title)
        with_front = ViralTitleEngine.evaluate_title_heuristics(title, focus_keyword=KEYWORD)
        self.assertEqual(without["keyword_status"], "not_evaluated")
        self.assertEqual(with_front["keyword_status"], "exact")
        self.assertGreater(with_front["score"], without["score"])
        self.assertTrue(with_front["keyword_front_loaded"])

    def test_keyword_far_from_the_front_earns_less(self):
        front = ViralTitleEngine.evaluate_title_heuristics("Agentic RAG Pilots Stall", focus_keyword=KEYWORD)
        back = ViralTitleEngine.evaluate_title_heuristics(
            "Why Most Enterprise Pilots Break Agentic RAG", focus_keyword=KEYWORD
        )
        self.assertTrue(back["keyword_exact"])
        self.assertGreater(front["keyword_points"], back["keyword_points"])

    def test_missing_keyword_is_reported_in_feedback(self):
        result = ViralTitleEngine.evaluate_title_heuristics("Why Pilots Stall", focus_keyword=KEYWORD)
        self.assertTrue(any("keyword" in line.lower() for line in result["feedback"]))


class EnsureKeywordTests(unittest.TestCase):
    def test_title_already_carrying_the_keyword_is_untouched(self):
        result = ViralTitleEngine.ensure_keyword_in_title("Agentic RAG: 3 Fatal Traps", KEYWORD)
        self.assertFalse(result["changed"])
        self.assertEqual(result["title"], "Agentic RAG: 3 Fatal Traps")

    def test_missing_keyword_is_prepended_with_the_hook_kept(self):
        result = ViralTitleEngine.ensure_keyword_in_title("3 Fatal Traps Costing Teams", KEYWORD)
        self.assertTrue(result["changed"])
        self.assertTrue(result["title"].lower().startswith("agentic rag"))
        self.assertIn("Fatal", result["title"])
        self.assertLessEqual(len(result["title"]), 60)

    def test_filler_is_dropped_before_the_hook(self):
        result = ViralTitleEngine.ensure_keyword_in_title("The Hidden Cost Of The Pilot Program", KEYWORD)
        self.assertTrue(result["title"].lower().startswith("agentic rag"))
        self.assertNotIn(" The ", result["title"])
        self.assertLessEqual(len(result["title"]), 60)

    def test_result_always_fits_the_length_budget(self):
        long_keyword = "retrieval augmented generation for enterprise agents"
        result = ViralTitleEngine.ensure_keyword_in_title("A Very Long Headline About Many Things Breaking", long_keyword)
        self.assertLessEqual(len(result["title"]), 60)

    def test_no_keyword_means_no_change(self):
        result = ViralTitleEngine.ensure_keyword_in_title("Some Title", "")
        self.assertFalse(result["changed"])
        self.assertEqual(result["keyword_status"], "not_evaluated")

    def test_keyword_survives_even_when_title_is_pure_hook(self):
        result = ViralTitleEngine.ensure_keyword_in_title("Why Pilots Stall", KEYWORD)
        self.assertIn("agentic rag", result["title"].lower())
        self.assertIn("pilot", result["title"].lower())

    def test_identical_phrase_in_another_case_is_left_alone(self):
        # "Agentic RAG" already is the phrase; only the casing differs.
        result = ViralTitleEngine.ensure_keyword_in_title("Agentic RAG: Why Pilots Stall", KEYWORD)
        self.assertFalse(result["changed"])
        self.assertEqual(result["keyword_status"], "exact")

    def test_adjacent_partial_match_is_upgraded_in_place(self):
        # Same words, wrong order: the phrase is rebuilt where it stands.
        result = ViralTitleEngine.ensure_keyword_in_title("Agentic RAG: Why Pilots Stall", "rag agentic")
        self.assertTrue(result["changed"])
        self.assertEqual(result["keyword_status"], "exact")
        self.assertTrue(result["title"].startswith("RAG agentic"))
        self.assertIn("Why Pilots Stall", result["title"])

    def test_scattered_partial_match_gets_the_phrase_prefixed(self):
        result = ViralTitleEngine.ensure_keyword_in_title("Agentic Pipelines Break RAG Agents", KEYWORD)
        self.assertTrue(result["title"].lower().startswith("agentic rag"))
        self.assertEqual(result["keyword_status"], "exact")

    def test_keyword_keeps_headline_casing(self):
        result = ViralTitleEngine.ensure_keyword_in_title("Why Pilots Stall", "rag agentic ai")
        self.assertTrue(result["title"].startswith("RAG agentic AI"))


class SeoTitleTests(unittest.TestCase):
    def test_short_title_without_keyword_gets_the_keyword(self):
        from tasks import _truncate_seo_title

        seo = _truncate_seo_title("Why Pilots Stall", 60, focus_keyword=KEYWORD)
        self.assertTrue(seo.lower().startswith("agentic rag"))
        self.assertLessEqual(len(seo), 60)

    def test_seo_title_never_exceeds_the_limit(self):
        from tasks import _truncate_seo_title

        seo = _truncate_seo_title(
            "An Extremely Long Headline That Would Definitely Be Truncated In Search Results",
            60,
            focus_keyword=KEYWORD,
        )
        self.assertLessEqual(len(seo), 60)
        self.assertIn("agentic rag", seo.lower())

    def test_title_already_containing_keyword_is_not_prefixed(self):
        from tasks import _truncate_seo_title

        seo = _truncate_seo_title("Agentic RAG Fails Quietly", 60, focus_keyword=KEYWORD)
        self.assertEqual(seo, "Agentic RAG Fails Quietly")


if __name__ == "__main__":
    unittest.main()
