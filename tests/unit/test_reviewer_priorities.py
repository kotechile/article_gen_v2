"""The reviewer optimises for quality, not for a word count.

The refinement prompt used to say "LENGTH TARGET: HARD CONSTRAINT - you MUST
compress or expand the content to meet this constraint". That makes the editor
trade substance for arithmetic, and it is why passes came back shorter and duller
than the draft. Quality comes first: humanization, readability, substance.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _guard(*args, **kwargs):
    # Lazy: importing tasks at module scope would pre-empt test files that stub
    # sys.modules before importing it themselves.
    import tasks

    return tasks._apply_refinement_guard(*args, **kwargs)


def _build_prompt(*args, **kwargs):
    import tasks

    return tasks._build_refinement_user_message(*args, **kwargs)

SECTION = "<p>The system is designed to process data and it is important to note that teams often use this architecture.</p>"


class RefinementPromptTests(unittest.TestCase):
    def _prompt(self, **kwargs):
        return _build_prompt(
            "professional", SECTION, target_word_count=600, current_word_count=480, **kwargs
        )

    def test_no_word_count_hard_constraint(self):
        prompt = self._prompt()
        self.assertNotIn("HARD CONSTRAINT", prompt)
        self.assertNotIn("MUST compress", prompt)
        self.assertNotIn("must be between", prompt.lower())

    def test_quality_priorities_are_stated_in_order(self):
        prompt = self._prompt()
        humanization = prompt.index("HUMANIZATION")
        readability = prompt.index("READABILITY")
        substance = prompt.index("SUBSTANCE")
        self.assertLess(humanization, readability)
        self.assertLess(readability, substance)

    def test_length_is_context_only(self):
        prompt = self._prompt()
        self.assertIn("LENGTH CONTEXT", prompt)
        self.assertIn("480", prompt)          # the current size
        self.assertNotIn("510", prompt)       # the old 0.85x floor
        self.assertIn("Do not pad", prompt)

    def test_keeps_evidence(self):
        prompt = self._prompt()
        self.assertIn("Never invent facts", prompt)
        self.assertIn("citation", prompt.lower())

    def test_at_a_glance_rules_survive(self):
        prompt = _build_prompt(
            "professional", SECTION, current_word_count=120, section_title="At a Glance"
        )
        self.assertIn("BULLET", prompt.upper())
        self.assertIn("60 words", prompt)


class GuardAllowsRealEditingTests(unittest.TestCase):
    """A quality pass may tighten prose by a third; it may not delete the section."""

    def _text(self, words):
        return "<p>" + " ".join(["word"] * words) + "</p>"

    def test_a_third_shorter_is_accepted(self):
        before = self._text(600)
        candidate = self._text(400)
        self.assertEqual(
            _guard("Body", "humanization", before, candidate), candidate
        )

    def test_deletion_is_still_refused(self):
        before = self._text(600)
        candidate = self._text(60)
        self.assertEqual(_guard("Body", "humanization", before, candidate), before)

    def test_takeaways_may_shrink_further(self):
        before = self._text(600)
        candidate = self._text(320)  # 53%
        self.assertEqual(
            _guard(
                "At a Glance", "cleanup", before, candidate, min_ratio=0.5
            ),
            candidate,
        )


if __name__ == "__main__":
    unittest.main()
