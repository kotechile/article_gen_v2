"""The article title must carry the keyword the piece targets.

Production case: an article whose primary keyword was "rag agentic ai" shipped with
the title "RAG explained" - the bare topic seed. The SEO title got the keyword, the
article title never did, and `focus_keyword` was resolved by a length heuristic that
picked the secondary "progress agentic rag" instead of the selected primary.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks import (  # noqa: E402
    _keyword_led_title,
    _primary_keyword_for,
    _title_case_keyword,
)


class KeywordCasingTests(unittest.TestCase):
    def test_short_tokens_read_as_acronyms(self):
        self.assertEqual(_title_case_keyword("rag agentic ai"), "RAG Agentic AI")

    def test_stopwords_stay_lowercase(self):
        self.assertEqual(_title_case_keyword("cost of ownership"), "Cost of Ownership")

    def test_multiword_terms_are_capitalised(self):
        self.assertEqual(_title_case_keyword("data pipeline latency"), "Data Pipeline Latency")


class KeywordLedTitleTests(unittest.TestCase):
    def test_seed_title_gains_the_keyword(self):
        self.assertEqual(
            _keyword_led_title("RAG explained", "rag agentic ai"), "RAG Agentic AI Explained"
        )

    def test_unrelated_title_gets_a_keyword_prefix(self):
        self.assertEqual(
            _keyword_led_title("Why teams fail", "rag agentic ai"),
            "RAG Agentic AI: Why teams fail",
        )

    def test_title_that_already_has_the_keyword_is_untouched(self):
        title = "RAG Agentic AI: 3 Productivity Killers"
        self.assertEqual(_keyword_led_title(title, "rag agentic ai"), title)

    def test_partial_match_is_accepted(self):
        title = "Agentic AI for retrieval"
        self.assertEqual(_keyword_led_title(title, "agentic ai"), title)

    def test_no_keyword_means_no_change(self):
        self.assertEqual(_keyword_led_title("RAG explained", ""), "RAG explained")

    def test_length_is_capped_but_the_keyword_survives(self):
        out = _keyword_led_title(
            "A very long descriptive subtitle about enterprise retrieval architecture and its limits",
            "rag agentic ai",
            max_length=60,
        )
        self.assertLessEqual(len(out), 60)
        self.assertIn("RAG Agentic AI", out)

    def test_empty_title_becomes_the_keyword(self):
        self.assertEqual(_keyword_led_title("", "rag agentic ai"), "RAG Agentic AI")


class PrimaryKeywordTests(unittest.TestCase):
    def test_explicit_structure_keyword_wins(self):
        self.assertEqual(
            _primary_keyword_for({"focus_keyword": "kv cache"}, {"primary_keyword": "other"}, ""),
            "kv cache",
        )

    def test_research_primary_beats_the_length_heuristic(self):
        # "progress agentic rag" sorts first by length closeness; the selected
        # primary must still win.
        self.assertEqual(
            _primary_keyword_for(
                {"keywords": ["rag agentic ai", "progress agentic rag"]},
                {"primary_keyword": "rag agentic ai"},
                "rag agentic ai, progress agentic rag",
            ),
            "rag agentic ai",
        )

    def test_search_phrase_is_used_when_no_primary_is_set(self):
        self.assertEqual(_primary_keyword_for({}, {"search_phrase": "vector db cost"}, ""), "vector db cost")

    def test_primary_keywords_list_is_used(self):
        self.assertEqual(
            _primary_keyword_for({"primary_keywords": ["rag agentic ai"]}, {}, ""), "rag agentic ai"
        )

    def test_falls_back_to_the_keyword_string(self):
        self.assertEqual(_primary_keyword_for({}, {}, "alpha, beta"), "alpha")


if __name__ == "__main__":
    unittest.main()
