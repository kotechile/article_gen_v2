"""Story shape: paragraphs that announce their claim instead of making it are measurable.

The two fixtures are the same article before and after a rewrite by the user -
the flat draft whose paragraphs read as separate notes, and the version that
reads as one argument. The module's job is to separate them, so both directions
are pinned here: the flat draft must fail, the rewrite must pass, and a
legitimate edit must not be mistaken for a regression.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.services.narrative_flow import (  # noqa: E402
    analyze_narrative_flow,
    compare_flow,
    flow_evidence,
)

FLAT = """
<p>Attribution tells you which factor created the cut; prevention has to act before the order is accepted. The first-attempt completeness standard [1] is what exposes the weakness in the alert-based model: a regional DC can commit a full week of inventory cover in a single Monday morning EDI release, and by the time a planner opens a Tuesday exception report those orders are already confirmed and cannot be changed.</p>
<p>Product Allocation closes that window. SAP IBP sends weekly constrained allocations to S/4HANA, which checks every incoming order against its ceiling before confirmation. When a retail customer submits an unforecasted order at 6 a.m., the system caps the confirmation at the allocated quantity and holds the balance back.</p>
<p>Alternative-Based Confirmation handles the local stockout differently. Instead of rejecting the line, it checks adjacent DCs and either changes the shipping plant or splits the order based on rules the business sets for full confirmation versus on-time confirmation. Backorder Processing then reallocates whatever is genuinely scarce across strategic tiers: Win, Gain, Redistribute, Fill, Lose.</p>
<p>The payoff is measurable. Teams that enable these controls typically report a 4 to 6 percentage point improvement in Case Fill Rate and OTIF, along with roughly a 24% reduction in expedited freight spend. The difference is rarely effort; it is structure.</p>
"""

STORY = """
<p>Root-cause attribution explains why an order was cut, but preventing the cut requires intervening before the order is confirmed. Standard first-come, first-served order processing exposes the fatal flaw of exception reports: a single retail customer can wipe out a full week of regional DC inventory in one 6:00 a.m. Monday EDI drop.</p>
<p>SAP S/4HANA closes this exposure by enforcing three automated controls at the exact moment an order arrives. Product Allocation (PAL) eliminates forecast cannibalization: weekly supply limits flow from the planning system into S/4HANA, which checks every incoming order against its customer ceiling before confirming.</p>
<p>Alternative-Based Confirmation (ABC) bypasses local stockouts. Instead of rejecting an order when the primary DC is empty, S/4HANA automatically checks alternate DCs and reroutes or splits the shipment based on pre-defined business rules.</p>
<p>Backorder Processing (BOP) protects strategic accounts when national supply runs short. Rather than leaving triage to manual planner intervention, the system automatically reallocates scarce stock across five priority tiers (Win, Gain, Redistribute, Fill, Lose).</p>
<p>The ROI is structural and sustained: organizations enabling these controls typically gain 4 to 6 percentage points in Case Fill Rate and OTIF while reducing expedited freight spend by roughly 24%.</p>
"""


class FlatDraftTests(unittest.TestCase):
    def test_frame_opener_is_caught(self):
        report = analyze_narrative_flow(FLAT)
        self.assertEqual(report["meta_frame_openers"], 1, report)
        self.assertTrue(any("frame instead of a claim" in r for r in report["reasons"]), report)

    def test_reasons_name_the_failing_paragraph(self):
        evidence = flow_evidence(analyze_narrative_flow(FLAT))
        self.assertTrue(any("The payoff is measurable." in line for line in evidence), evidence)

    def test_flat_draft_scores_lower_than_the_rewrite(self):
        flat = analyze_narrative_flow(FLAT)
        story = analyze_narrative_flow(STORY)
        self.assertLess(flat["story_score"], story["story_score"], (flat, story))
        self.assertGreater(flat["nominalisation_density"], story["nominalisation_density"])


class StoryDraftTests(unittest.TestCase):
    def test_rewrite_is_clean(self):
        report = analyze_narrative_flow(STORY)
        self.assertEqual(report["meta_frame_openers"], 0, report)
        self.assertEqual(report["anchorless_paragraphs"], 0, report)
        self.assertEqual(report["reasons"], [], report)
        self.assertEqual(report["story_score"], 1.0, report)

    def test_every_mechanism_paragraph_carries_an_anchor(self):
        report = analyze_narrative_flow(STORY)
        for paragraph in report["paragraphs"]:
            self.assertTrue(paragraph["anchor"], paragraph)


class ComparisonTests(unittest.TestCase):
    def test_flattening_an_article_is_a_regression(self):
        comparison = compare_flow(analyze_narrative_flow(STORY), analyze_narrative_flow(FLAT))
        self.assertTrue(comparison["regressed"], comparison)
        self.assertTrue(any("frame-opener" in r for r in comparison["regressions"]), comparison)

    def test_improving_an_article_is_not_a_regression(self):
        comparison = compare_flow(analyze_narrative_flow(FLAT), analyze_narrative_flow(STORY))
        self.assertFalse(comparison["regressed"], comparison)
        self.assertGreater(comparison["deltas"]["story_score"], 0)

    def test_tightening_without_flattening_is_allowed(self):
        trimmed = STORY.replace(
            " Standard first-come, first-served order processing exposes the fatal flaw of "
            "exception reports: a single retail customer can wipe out a full week of regional "
            "DC inventory in one 6:00 a.m. Monday EDI drop.",
            " At 6:00 a.m. on Monday, one retail customer can wipe out a full week of DC inventory.",
        )
        comparison = compare_flow(analyze_narrative_flow(STORY), analyze_narrative_flow(trimmed))
        self.assertFalse(comparison["regressed"], comparison)


class EdgeCaseTests(unittest.TestCase):
    def test_plain_text_is_supported(self):
        report = analyze_narrative_flow(
            "The payoff is measurable. Teams report a 4 to 6 point gain.\n\n"
            "SAP S/4HANA closes this. Product Allocation checks every order."
        )
        self.assertEqual(report["paragraph_count"], 2)
        self.assertEqual(report["meta_frame_openers"], 1)

    def test_list_items_are_not_paragraphs(self):
        html = "<ul><li><strong>Retrieval gap:</strong> one</li><li>two</li></ul><p>" + "word " * 6 + "</p>"
        report = analyze_narrative_flow(html)
        self.assertEqual(report["paragraph_count"], 1)

    def test_empty_input_is_safe(self):
        report = analyze_narrative_flow("")
        self.assertEqual(report["paragraph_count"], 0)
        self.assertEqual(report["story_score"], 1.0)
        self.assertEqual(report["reasons"], [])

    def test_citation_markers_do_not_count_as_anchors(self):
        html = "<p>The completeness standard in the alert model [1] [2] holds the balance back.</p>"
        report = analyze_narrative_flow(html)
        self.assertEqual(report["anchorless_paragraphs"], 1, report)


if __name__ == "__main__":
    unittest.main()
