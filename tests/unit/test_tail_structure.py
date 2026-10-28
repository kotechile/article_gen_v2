"""The end of the article: no invented modules, no restating the lead, no empty FAQ.

Taken from a real draft whose tail read:

    What I'd Watch: 3 Indicators Before Scaling RAG Agentic AI   <- the real section
      ...content...
    What I'd Watch:                                              <- boilerplate duplicate
      ...Monitor latency, concurrency spikes, and system resource saturation...
    The Catch
      Every tactical gain introduces architectural trade-offs...
    At a Glance
      At a glance, the argument comes back to the same silent failure: static
      retrieval syndrome stalls 7 in 10 mid-career promotions...   <- repeats the lead
    FAQ                                                          <- heading, no questions
    References
"""

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.services.article_structure_normalizer import normalize_document_structure  # noqa: E402

LEAD = (
    "<p><strong>The big picture:</strong> The defining failure mode of agentic AI is not "
    "hallucination - it is unchecked retrieval. RAG only creates durable advantage when it is "
    "treated as an auditable control plane.</p>"
    "<p><strong>Why it matters:</strong> The agentic RAG trap costing careers is static retrieval "
    "syndrome - treating retrieval as a one-time setup - and it silently stalls 7 in 10 "
    "mid-career promotions.</p>"
)

BODY = "".join(
    f"<h2>Section {i}: Decision {i}</h2>"
    f"<p>Decision {i} determines whether deployment creates value or risk, and the evidence "
    f"for that claim sits in the evaluation set rather than the demo [1].</p>"
    f"<p>The engineering effort lands in the retrieval layer, where identity, residency and "
    f"logging are enforced together rather than bolted on afterwards [2].</p>"
    for i in range(1, 4)
)

REAL_WATCH = (
    "<h2>What I'd Watch: 3 Indicators Before Scaling RAG Agentic AI</h2>"
    "<p>Before scaling an agentic RAG program, track three indicators. High-stakes agent "
    "accuracy should stay within 2% of the human baseline, approval latency should stay under "
    "a four-hour service level, and DLP false positives should remain below 10%.</p>"
)

BOILER_WATCH = (
    "<h3>What I'd Watch:</h3>\n<ul>\n"
    "<li><strong>Operational Bottlenecks:</strong> Monitor latency, concurrency spikes, and "
    "system resource saturation under peak traffic.</li>\n"
    "<li><strong>Integration Friction:</strong> Track compatibility drifts, breaking changes, "
    "and configuration overhead across dependencies.</li>\n</ul>"
)

BOILER_CATCH = (
    "<h3>The Catch</h3>\n<p>Every tactical gain introduces architectural trade-offs; early "
    "efficiency improvements in agentic RAG must be weighed against operational complexity and "
    "organizational constraints.</p>"
)

RESTATING_GLANCE = (
    "<h3>At a Glance</h3>\n<p>At a glance, the argument comes back to the same silent failure: "
    "static retrieval syndrome stalls 7 in 10 mid-career promotions because retrieval is treated "
    "as a one-time setup.</p>"
)

TAIL = (
    REAL_WATCH + BOILER_WATCH + BOILER_CATCH + RESTATING_GLANCE
    + "<h2>FAQ</h2><h2>References</h2><p>[1] source one</p><p>[2] source two</p>"
)

ARTICLE = "<h1>RAG Agentic AI Trap Costing Careers</h1>" + LEAD + "<h2>Go Deeper</h2>" + BODY + TAIL


class TailTests(unittest.TestCase):
    def setUp(self):
        self.out, self.report = normalize_document_structure(ARTICLE, title="RAG Agentic AI Trap")

    def test_one_what_id_watch_and_it_is_the_real_one(self):
        self.assertEqual(len(re.findall(r"What I'd Watch", self.out, re.IGNORECASE)), 1)
        self.assertIn("three indicators", self.out)
        self.assertIn("2% of the human baseline", self.out)

    def test_generic_boilerplate_never_ships(self):
        self.assertNotIn("concurrency spikes", self.out)
        self.assertNotIn("Every tactical gain introduces architectural trade-offs", self.out)

    def test_a_glance_that_restates_the_lead_is_dropped(self):
        self.assertNotIn("At a Glance", self.out)
        self.assertNotIn("At a glance, the argument comes back", self.out)

    def test_empty_faq_heading_is_dropped(self):
        self.assertNotIn("<h2>FAQ</h2>", self.out)

    def test_references_stay_last(self):
        self.assertTrue(self.out.rstrip().endswith("</p>"))
        self.assertLess(self.out.index("three indicators"), self.out.index("References"))

    def test_lead_survives_intact(self):
        self.assertIn("auditable control plane", self.out)
        self.assertIn("7 in 10 mid-career promotions", self.out)


class BoilerplateFreeTests(unittest.TestCase):
    def test_no_closing_modules_are_synthesised_by_finalize(self):
        """tasks.py must not invent What I'd Watch / The Catch / At a Glance."""
        source = (Path(REPO_ROOT) / "tasks.py").read_text()
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertNotIn("closing_blocks.append", code)
        self.assertNotIn("Every tactical gain introduces architectural trade-offs", code)
        self.assertNotIn("Monitor latency, concurrency spikes", code)
        self.assertNotIn("Execute disciplined, measurable implementations", code)


if __name__ == "__main__":
    unittest.main()
