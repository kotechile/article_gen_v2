"""Tests for the per-section refinement guard.

The production case: nine sections generated 8,496 words; after three LLM
refinement passes per section the assembled article held ~3,000 words, and
nothing compared before/after. The publish gate refused it (correctly), but the
loss happened a stage earlier - here.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LONG_SECTION = "<h3>Third-Hop Failure</h3>" + "".join(
    f"<p>Paragraph {i} explains the retrieval failure with concrete evidence and a citation [1].</p>"
    for i in range(10)
)
SHORT_REWRITE = "<h3>Third-Hop Failure</h3><p>Retrieval fails on the third hop.</p>"


class RefinementGuardTests(unittest.TestCase):
    def setUp(self):
        from tasks import _apply_refinement_guard

        self.guard = _apply_refinement_guard

    def test_catastrophic_shrinkage_is_discarded(self):
        result = self.guard("Third-Hop Failure", "cleanup", LONG_SECTION, SHORT_REWRITE)
        self.assertEqual(result, LONG_SECTION)

    def test_edit_that_keeps_the_body_is_accepted(self):
        edited = LONG_SECTION.replace("explains the retrieval failure", "shows the retrieval failure")
        result = self.guard("Third-Hop Failure", "cleanup", LONG_SECTION, edited)
        self.assertEqual(result, edited)

    def test_empty_result_keeps_previous_text(self):
        self.assertEqual(self.guard("Section", "humanization", LONG_SECTION, ""), LONG_SECTION)

    def test_takeaways_may_shrink_further_when_allowed(self):
        # 50% is the allowance for 'At a glance'; half the text survives here.
        half = LONG_SECTION[: len(LONG_SECTION) // 2] + "</p>"
        self.assertEqual(self.guard("At a Glance", "cleanup", LONG_SECTION, half, min_ratio=0.5), half)

    def test_same_allowance_still_rejects_a_stub(self):
        self.assertEqual(self.guard("At a Glance", "cleanup", LONG_SECTION, SHORT_REWRITE, min_ratio=0.5), LONG_SECTION)


if __name__ == "__main__":
    unittest.main()
