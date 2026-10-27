"""Unit tests for the citation integrity gate.

The fixture is the reference list of a real generated article: four entries with
no URL, generic institutional authors and 2024 dates - the shape invented
sources take - sitting beside real papers, and a reference list numbered by
insertion order rather than first mention.
"""

import re
import unittest

from src.services.citation_integrity import enforce_citation_integrity

BODY = (
    "<p>Dynamic task decomposition improves retrieval accuracy by 15 to 30 percentage points [1].</p>"
    "<p>Relevance grading holds a 0.85 fidelity floor before synthesis [2].</p>"
    "<p>An 8B router keeps inference budgets flat [3].</p>"
    "<p>Routing cuts orchestration cost to $0.01 per transaction [4].</p>"
)

CITATIONS = [
    {"title": "Benchmark Studies on Multi-Hop Question Answering", "url": "", "author": "", "publication_date": "2024"},
    {"title": "TruLens & Ragas Documentation", "url": "", "author": "", "publication_date": "2024"},
    {"title": "Edge AI and Systems Optimization Lab: Inference Economics", "url": "", "author": "", "publication_date": "2024"},
    {"title": "Toolformer: Language Models Can Teach Themselves to Use Tools", "url": "https://arxiv.org/abs/2302.04761", "author": "Schick, T.", "publication_date": "2023"},
]


class CitationIntegrityTests(unittest.TestCase):
    def test_url_less_references_are_dropped(self):
        _, kept, report = enforce_citation_integrity(BODY, CITATIONS)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["url"], "https://arxiv.org/abs/2302.04761")
        self.assertEqual(len(report["dropped_without_url"]), 3)

    def test_markers_pointing_at_dropped_references_are_removed(self):
        content, kept, report = enforce_citation_integrity(BODY, CITATIONS)
        self.assertEqual(len(kept), 1)
        # Only the verifiable source keeps a marker, renumbered to [1]; the
        # markers that pointed at unverifiable entries are gone with them.
        markers = re.findall(r"\[(\d+)\]", content)
        self.assertEqual(markers, ["1"])
        self.assertIn("$0.01 per transaction", content)  # the sentence is kept, its bad marker is not
        self.assertGreater(report["markers_repaired"], 0)

    def test_reference_order_follows_first_appearance(self):
        body = "<p>First claim [2].</p><p>Second claim [1].</p>"
        citations = [
            {"title": "Second source", "url": "https://example.com/b"},
            {"title": "First source", "url": "https://example.com/a"},
        ]
        content, kept, _ = enforce_citation_integrity(body, citations)
        self.assertEqual([c["title"] for c in kept], ["First source", "Second source"])
        self.assertIn("First claim [1]", content)
        self.assertIn("Second claim [2]", content)

    def test_nothing_survives_when_no_reference_has_a_url(self):
        citations = [{"title": "Some lab report", "url": "", "publication_date": "2024"}]
        content, kept, report = enforce_citation_integrity("<p>A claim [1].</p>", citations)
        self.assertEqual(kept, [])
        self.assertEqual(content, "<p>A claim.</p>")
        self.assertEqual(report["kept"], 0)

    def test_require_urls_can_be_disabled_explicitly(self):
        _, kept, _ = enforce_citation_integrity(BODY, CITATIONS, require_urls=False)
        self.assertEqual(len(kept), 4)

    def test_cited_only_and_verifiable_reference_list_stays_intact(self):
        body = "<p>Claim one [1].</p><p>Claim two [2].</p>"
        citations = [
            {"title": "A", "url": "https://example.com/a"},
            {"title": "B", "url": "https://example.com/b"},
        ]
        content, kept, report = enforce_citation_integrity(body, citations)
        self.assertEqual(len(kept), 2)
        self.assertEqual(report["markers_repaired"], 0)
        self.assertIn("[1]", content)
        self.assertIn("[2]", content)

    def test_empty_inputs_are_safe(self):
        content, kept, report = enforce_citation_integrity("", [])
        self.assertEqual(content, "")
        self.assertEqual(kept, [])
        self.assertEqual(report["kept"], 0)


if __name__ == "__main__":
    unittest.main()
