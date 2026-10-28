"""Tests for the LLM-editing guards: leaked reasoning and lost bodies.

Both fixtures are verbatim from a production run in which the polishing pass
returned 1,085 characters from a 20,445-character draft, opening with the model
deliberating about two of its own instructions.
"""

import unittest

from src.services.model_output_cleanup import (
    leakage_summary,
    looks_like_reasoning,
    polish_preserved_body,
    strip_model_scaffolding,
)

# Verbatim from the 2026-10-27 run: a 23 KB reply with no HTML at all, opening with the model's
# plan and then echoing the prompt's own architecture skeleton as the article.
PLAIN_TEXT_MONOLOGUE = (
    "rag agentic ai hidden costs\n\n"
    "The Big Picture: 1-2 sentences\n\n"
    "Go deeper: Review the detailed breakdown, comparative analysis, and step-by-step guidance below.\n\n"
    "Go Deeper\n\n"
    " The user wants me to rewrite this as an expert executive editor. I need to understand the task. "
    "The user says \"Return the complete article HTML.\" Let me analyze the requirements carefully:\n\n"
    "Must match the editorial architecture:\n\n"
    "The Big Picture at top (1-2 sentences)\n\nGo Deeper section with substantive body sections\n\n"
    "What I'd Watch\n\nThe Catch\n\nAt a Glance\n\nFAQ section\n\nReferences\n\n"
    "Fact discipline: keep all numbers, citations in place, no new numbers.\n\n"
    "Every substantive topic section must follow this exact formula: "
    "[Action-Oriented or Metric-Driven Header] with exactly 3 bolded key metric bullets:\n\n"
    "* [Key Metric or Term]: 1-2 sentences explaining why it matters. Cut all introductory "
    "throat-clearing.\n\n* [Key Metric or Term]: Focus on empirical mechanics, code constraints, "
    "or hard numbers.\n\n* [Key Metric or Term]: Highlight the trade-off, cost, or hidden failure "
    "mode.\n\nTables: If the section contains comparative data, options, or matrices, format as "
    "clean HTML tables.\n\nSo I need to restructure into the specified architecture. Let me think "
    "about how to do this practically. I will take the original content and restructure it, then "
    "return the complete article HTML. I need to preserve all the original text, tables, FAQs and "
    "citations while reorganising the headers and adding the required elements."
)

LEAKED_OUTPUT = (
    '. Wait, let\'s check the rule: "Do not restate the same statistic in more than one section." '
    '"CRITICAL ALIGNMENT - COMPLETE CONCEPT DEVELOPMENT UNDER \'GO DEEPER\'": Every single topic, '
    'theme, concept, statistic, data point, and comparison mentioned in the Executive Smart Brevity '
    'Lead must be developed inside the substantive body sections under\n'
    '<p><strong>The big picture:</strong> Autonomy is not the upgrade - accountability is.</p>\n'
    '<p>Notice: It says every single concept, theme, statistic mentioned in FAQ must be developed in '
    'the body. If a statistic is mentioned in the FAQ, but the rule says "Do not restate the same '
    'statistic", how to reconcile? Simply do not put duplicate numbers in the FAQ!</p>\n'
    '<p><strong>Why it matters:</strong> Loop budgets and routing keep the bill predictable.</p>\n'
    '<h2>What I\'d Watch</h2>\n'
    '<p>Track latency and concurrency spikes under peak traffic.</p>\n'
)

REAL_DRAFT = (
    "<p><strong>The big picture:</strong> " + ("Agentic retrieval costs tokens. " * 60) + "</p>\n"
    "<h2>Go Deeper</h2>\n"
    "<h2>Query Planning</h2><p>" + ("Decomposition raises recall on multi-hop questions. " * 20) + "</p>\n"
    "<h2>Grading Gates</h2><p>" + ("Programmatic grading filters irrelevant context. " * 20) + "</p>\n"
    "<h2>Cost Control</h2><p>" + ("Small routing models cut the per-query expense. " * 20) + "</p>\n"
)

TRUNCATED_POLISH = (
    "<p><strong>The big picture:</strong> Autonomy is not the upgrade - accountability is.</p>\n"
    "<h2>What I'd Watch</h2>\n<p>Track latency.</p>\n"
)


class StripScaffoldingTests(unittest.TestCase):
    def test_leading_deliberation_is_removed(self):
        cleaned, report = strip_model_scaffolding(LEAKED_OUTPUT)
        self.assertNotIn("Wait, let's check the rule", cleaned)
        self.assertIn("The big picture", cleaned)
        self.assertGreaterEqual(report["removed_blocks"], 1)
        self.assertTrue(report["samples"])

    def test_instruction_echo_paragraph_is_removed(self):
        cleaned, report = strip_model_scaffolding(LEAKED_OUTPUT)
        self.assertNotIn("how to reconcile", cleaned)
        self.assertNotIn("Simply do not put duplicate numbers", cleaned)
        self.assertIn("Why it matters", cleaned)  # genuine content survives

    def test_clean_article_is_untouched(self):
        cleaned, report = strip_model_scaffolding(REAL_DRAFT)
        self.assertEqual(cleaned.strip(), REAL_DRAFT.strip())
        self.assertEqual(report["removed_blocks"], 0)

    def test_plain_text_monologue_is_leaked(self):
        """No block markup + planning phrases: the 2026-10-27 failure, which shipped."""
        report = leakage_summary(PLAIN_TEXT_MONOLOGUE)
        self.assertTrue(report["leaked"])
        self.assertTrue(report["plain_text_only"])
        self.assertEqual(report["html_blocks"], 0)
        self.assertGreaterEqual(len(report["markers"]), 2)
        self.assertIn("the user wants", report["markers"])

    def test_prompt_skeleton_echoed_as_prose_is_a_placeholder(self):
        report = leakage_summary(PLAIN_TEXT_MONOLOGUE)
        joined = " | ".join(report["placeholders"])
        self.assertIn("1-2 sentences", joined)
        self.assertIn("Action-Oriented or Metric-Driven", joined)

    def test_a_real_article_with_no_markers_is_not_leaked(self):
        report = leakage_summary(REAL_DRAFT)
        self.assertFalse(report["leaked"], report)
        self.assertGreater(report["html_blocks"], 2)

    def test_two_planning_phrases_inside_markup_are_still_leaked(self):
        """Markers are evidence even when the body is otherwise well formed."""
        html = (
            "<p>First, let me analyze the retrieval numbers in the draft.</p>"
            "<p>The user wants a cost comparison, so here is a table.</p>"
            "<p>Retrieval costs compound across hops, and the ledger shows it.</p>"
        )
        self.assertTrue(leakage_summary(html)["leaked"])

    def test_reasoning_detection(self):
        self.assertTrue(looks_like_reasoning(
            "Notice: It says every single concept must be developed. If a statistic is mentioned, how to reconcile?"
        ))
        self.assertFalse(looks_like_reasoning(
            "Programmatic grading removes irrelevant context before synthesis, which keeps token spend flat."
        ))


class PolishGuardTests(unittest.TestCase):
    def test_catastrophic_shrinkage_is_rejected(self):
        ok, reason = polish_preserved_body(REAL_DRAFT, TRUNCATED_POLISH)
        self.assertFalse(ok)
        self.assertIn("chars", reason)

    def test_real_production_case_is_rejected(self):
        # 1,085 chars out of a 20,445-char draft.
        ok, reason = polish_preserved_body("x" * 20445, "y" * 1085)
        self.assertFalse(ok)
        self.assertIn("5%", reason)

    def test_healthy_edit_is_accepted(self):
        edited = REAL_DRAFT.replace("costs tokens", "burns tokens")
        ok, _ = polish_preserved_body(REAL_DRAFT, edited)
        self.assertTrue(ok)

    def test_empty_output_is_rejected(self):
        ok, reason = polish_preserved_body(REAL_DRAFT, "")
        self.assertFalse(ok)
        self.assertIn("empty", reason)

    def test_section_loss_is_rejected_even_when_length_is_close(self):
        # Same size, but the body sections were collapsed away.
        original = "<h2>Go Deeper</h2><h3>A</h3><p>x</p><h3>B</h3><p>y</p><h3>C</h3><p>z</p>"
        polished = "<h2>Go Deeper</h2><p>" + ("filler words here " * 20) + "</p>"
        ok, reason = polish_preserved_body(original, polished)
        self.assertFalse(ok)
        self.assertIn("sections", reason)


if __name__ == "__main__":
    unittest.main()
