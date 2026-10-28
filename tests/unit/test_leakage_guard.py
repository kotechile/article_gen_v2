"""A model answering the prompt instead of the task must never reach the article.

The exact failure this covers, from a real run on a cheaper model: the polishing pass
returned its own deliberation and a template skeleton as the article body.

    The user wants me to act as an expert executive editor and produce a polished
    HTML article that follows a specific editorial architecture. ... Let me analyze
    what I need to do: ... Wait, but the original content doesn't have a FAQ ...

    Section 1 header
    ... content...
    [1 paragraph context] [3 bolded bullets] [Table: Classic vs Agentic RAG] ...

It passed every length check, because it was long.
"""

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.services.model_output_cleanup import (  # noqa: E402
    leakage_summary,
    strip_model_scaffolding,
)

LEAKED = """<p>rag agentic ai: 3 fatal mistakes</p>
<p><strong>Why it matters:</strong> Nearly 40% of enterprise RAG implementations fail.</p>
<p>The user wants me to act as an expert executive editor and produce a polished HTML article that follows a specific editorial architecture. They've provided original HTML content and very detailed instructions about structure.</p>
<p>Let me analyze what I need to do:</p>
<p>Wait, but the original content doesn't have a FAQ section. The instructions say to include a FAQ section with 3-5 realistic reader questions. Let me re-read.</p>
<p>I need to create a References section. And I need to create FAQ. This is tricky.</p>
<p>Let me think about what's best. That seems like the best approach.</p>
<h2>Section 1 header</h2>
<p>... content...</p>
<p>[1 paragraph context] [3 bolded bullets] [Table: Classic vs Agentic RAG]</p>
<h2>Section 2 header</h2>
<p>... content...</p>"""

CLEAN = """<p><strong>Why it matters:</strong> Nearly 40% of enterprises that ship agentic retrieval rebuild it within a year.</p>
<h2>What Actually Changes</h2>
<p>Retrieval stops being a lookup and becomes a control loop. The agent plans a query, fetches, grades what came back, and retries when the evidence is thin.</p>
<p>That loop is what makes multi-hop questions answerable, and it is also what makes the failure modes expensive.</p>"""


class LeakageDetectionTests(unittest.TestCase):
    def test_the_real_failure_is_detected(self):
        report = leakage_summary(LEAKED)
        self.assertTrue(report["leaked"], report)
        self.assertGreaterEqual(report["reasoning_paragraphs"], 2)

    def test_first_person_planning_is_named(self):
        report = leakage_summary(LEAKED)
        joined = " ".join(report["markers"])
        self.assertIn("the user wants", joined)
        self.assertIn("let me analyze", joined)

    def test_template_placeholders_are_named(self):
        report = leakage_summary(LEAKED)
        text = " ".join(report["placeholders"]).lower()
        self.assertTrue("section 1 header" in text or "... content" in text or "[1 paragraph" in text, report)

    def test_clean_article_is_not_flagged(self):
        self.assertFalse(leakage_summary(CLEAN)["leaked"])

    def test_ordinary_first_person_prose_is_not_flagged(self):
        article = (
            "<p>I have seen teams rebuild the same retrieval layer twice because the first "
            "version never recorded why a document was chosen.</p>"
            "<p>The fix is unglamorous: log the query, the candidates, and the score.</p>"
        )
        self.assertFalse(leakage_summary(article)["leaked"])

    def test_one_stray_sentence_is_removed_not_fatal(self):
        # The gate runs after stripping, so a single slip is deleted and the pass
        # survives; the raw draft is still flagged, which is what makes stripping
        # happen at all.
        article = CLEAN * 15 + "<p>Let me analyze what I need to do next.</p>"
        self.assertTrue(leakage_summary(article)["leaked"])
        cleaned, report = strip_model_scaffolding(article)
        self.assertEqual(report["removed_blocks"], 1)
        after = leakage_summary(cleaned)
        self.assertFalse(after["leaked"], after)
        self.assertEqual(after["reasoning_paragraphs"], 0)


class StrippingTests(unittest.TestCase):
    def test_reasoning_paragraphs_are_removed(self):
        cleaned, report = strip_model_scaffolding(LEAKED)
        self.assertGreater(report["removed_blocks"], 0)
        self.assertNotIn("The user wants me to", cleaned)
        self.assertNotIn("Let me analyze", cleaned)
        self.assertIn("Nearly 40%", cleaned)

    def test_placeholder_skeleton_is_removed(self):
        cleaned, _ = strip_model_scaffolding(LEAKED)
        self.assertNotIn("Section 1 header", cleaned)
        self.assertNotIn("[1 paragraph context]", cleaned)

    def test_stripping_then_checking_leaves_a_clean_draft(self):
        cleaned, _ = strip_model_scaffolding(LEAKED + CLEAN)
        report = leakage_summary(cleaned)
        self.assertFalse(report["leaked"], report)


class GuardTests(unittest.TestCase):
    def test_the_guard_discards_leaked_candidates(self):
        import tasks

        self.assertIsNone(tasks._reject_if_leaked("test", LEAKED))
        self.assertEqual(tasks._reject_if_leaked("test", CLEAN), CLEAN)

    def test_refinement_guard_keeps_the_previous_text(self):
        import tasks

        previous = "<p>" + "substantive sentence about retrieval quality and latency. " * 20 + "</p>"
        leaked = "<p>The user wants me to rewrite this section. Let me analyze what I need to do.</p>"
        out = tasks._apply_refinement_guard("Body", "cleanup", previous, leaked)
        self.assertEqual(out, previous)


if __name__ == "__main__":
    unittest.main()
