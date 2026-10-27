"""Unit tests for numeric claim grounding and citation integrity.

The fixtures reproduce a real generated article whose "By the numbers" block
published fragments with no subject ("0.85: before proceeding [2]") and whose
cost tables carried figures that appear in no retained source. Both must be
caught deterministically - no LLM in the loop.
"""

import unittest

from src.services.fact_grounding import (
    audit_numbers,
    build_evidence_corpus,
    canonical_number,
    extract_number_claims,
    render_numbers_block,
    strip_numbers_block,
    verify_claim,
)

# Verbatim excerpts of the published article.
ARTICLE_HTML = """
<p>The Big Picture: Enterprise implementations of rag agentic ai fail when engineering teams treat dynamic, iterative control loops as simple drop-in upgrades for static, single-turn retrieval pipelines.</p>
<p><strong>By the numbers:</strong></p>
<ul>
<li><strong>15 to 30:</strong> percentage points over static semantic baselines on complex multi-hop questions [1].</li>
<li><strong>0.85:</strong> before proceeding [2].</li>
<li><strong>8:</strong> B-parameter routing models preserves inference budgets without sacrificing dynamic routing intelligence [3].</li>
<li><strong>$0.01:</strong> per transaction [4].</li>
<li><strong>1 to 3:</strong> seconds, multi-cycle agent reflection pushes tail p95 response times out to 8 to 15 seconds [3].</li>
</ul>
<h2>Resolve Multi-Hop Retrieval Gaps Through Dynamic Query Planning</h2>
<p>Empirical Hit-Rate Delta: Dynamic task decomposition improves retrieval accuracy by 15 to 30 percentage points over static semantic baselines on complex multi-hop questions [1].</p>
<h2>Contain Token Inflation and Tail Latency With SLM Routers</h2>
<p>Routing Cost Arbitrage: Routing classification tasks to an optimized SLM reduces per-query orchestration expenses from approximately $0.09 down to $0.01 per transaction [4].</p>
<p>p95 Latency Realities: While standard single-pass RAG completes in 1 to 3 seconds, multi-cycle agent reflection pushes tail p95 response times out to 8 to 15 seconds [3].</p>
<h2>Scaling Economics For Enterprise Programs</h2>
<p>Operational Dimension Cost per 1,000 Complex Queries: $90.00 to $150.00 for a frontier model agent [4].</p>
<p>Daily Infrastructure Run-Rate: $1,500 to $5,000 at scaled volume [3].</p>
"""

EVIDENCE = {
    "research_dossier": {
        "summary": "Agentic RAG evaluation and inference cost notes.",
        "source_map": [
            {
                "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
                "url": "https://arxiv.org/abs/2312.10997",
                "content": "Dynamic task decomposition improves retrieval accuracy by 15 to 30 percentage points over static semantic baselines on complex multi-hop questions.",
            },
            {
                "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
                "url": "https://arxiv.org/abs/2302.04761",
                "content": "An 8B-parameter routing model dispatches queries to vector indices, SQL databases or external APIs while preserving inference budgets.",
            },
            {
                "title": "Inference routing cost study",
                "url": "https://example.com/routing-cost-study",
                "content": "Routing classification to a small language model reduces per-query orchestration expense from about $0.09 to $0.01 per transaction.",
            },
            {
                "title": "Tail latency measurements for multi-cycle agent reflection",
                "url": "https://example.com/latency-agent-loops",
                "content": "Standard single-pass RAG completes in 1 to 3 seconds, while multi-cycle agent reflection pushes tail p95 response times out to 8 to 15 seconds.",
            },
        ],
    }
}

EVIDENCE_BY_TITLE_ONLY = {
    "research_dossier": {
        "source_map": [
            # A "reference" with no URL - the shape invented sources take.
            {
                "title": "Cloud Security Alliance: Secure Implementation Guidelines for Autonomous Large Language Model Agents",
                "content": "Agents should enforce least-privilege entitlements and identity-bound access for every retrieval call.",
            }
        ]
    }
}


class CanonicalNumberTests(unittest.TestCase):
    def test_strips_currency_and_thousands_separators(self):
        self.assertEqual(canonical_number("$1,500"), "1500")
        self.assertEqual(canonical_number("90.00"), "90")

    def test_keeps_decimals(self):
        self.assertEqual(canonical_number("0.85"), "0.85")


class ExtractClaimTests(unittest.TestCase):
    def test_keeps_subject_and_unit(self):
        claims = [c for c in extract_number_claims(ARTICLE_HTML) if "percentage points" in c.raw]
        self.assertTrue(claims)
        claim = claims[0]
        self.assertEqual(claim.values, ("15", "30"))
        self.assertIn("percentage points", claim.raw)
        self.assertIn("retrieval accuracy", claim.label.lower())

    def test_year_and_citation_markers_are_not_claims(self):
        claims = extract_number_claims("<p>In 2024 the platform shipped 4 features [3].</p>")
        values = {value for claim in claims for value in claim.values}
        self.assertNotIn("2024", values)
        self.assertNotIn("3", values)

    def test_broken_block_is_stripped_before_extraction(self):
        stripped = strip_numbers_block(ARTICLE_HTML)
        self.assertNotIn("By the numbers", stripped)
        self.assertNotIn("0.85:", stripped)


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.corpus = build_evidence_corpus(EVIDENCE)

    def test_corpus_keeps_urls(self):
        urls = {item.url for item in self.corpus}
        self.assertIn("https://arxiv.org/abs/2312.10997", urls)
        self.assertTrue(any(item.citable for item in self.corpus))

    def test_grounded_claim_is_accepted_with_its_source(self):
        claim = next(c for c in extract_number_claims(ARTICLE_HTML) if "percentage points" in c.raw)
        result = verify_claim(claim, self.corpus)
        self.assertTrue(result.grounded)
        self.assertEqual(result.evidence.url, "https://arxiv.org/abs/2312.10997")

    def test_invented_cost_claim_is_rejected(self):
        # "$90.00 to $150.00 per 1,000 queries" appears in no source.
        claim = next(c for c in extract_number_claims(ARTICLE_HTML) if "$90" in c.raw)
        result = verify_claim(claim, self.corpus)
        self.assertFalse(result.grounded)
        self.assertEqual(result.reason, "no_matching_evidence")

    def test_number_without_evidence_corpus_is_never_grounded(self):
        claim = next(c for c in extract_number_claims(ARTICLE_HTML) if "percentage points" in c.raw)
        result = verify_claim(claim, [])
        self.assertFalse(result.grounded)
        self.assertEqual(result.reason, "no_evidence")

    def test_same_number_in_wrong_unit_is_not_grounded(self):
        # 0.85 exists in evidence as a fidelity score, never as $0.85.
        corpus = build_evidence_corpus({
            "research_dossier": {"source_map": [{
                "title": "Grading docs",
                "url": "https://example.com/grading",
                "content": "Relevance grading measures context fidelity against a 0.85 floor before synthesis.",
            }]}
        })
        claim = next(c for c in extract_number_claims(
            "<p>Inference cost falls to $0.85 per thousand retrieved chunks in production.</p>"
        ))
        result = verify_claim(claim, corpus)
        self.assertFalse(result.grounded)


class NumbersBlockTests(unittest.TestCase):
    def setUp(self):
        self.corpus = build_evidence_corpus(EVIDENCE)

    def test_block_is_rebuilt_from_grounded_claims(self):
        html, report = render_numbers_block(ARTICLE_HTML, self.corpus)
        self.assertIn("By the numbers", html)
        self.assertNotIn("0.85: before proceeding", html)
        self.assertNotIn("$90.00", html)
        self.assertNotIn("$1,500", html)
        for link in ("https://arxiv.org/abs/2312.10997", "https://example.com/routing-cost-study"):
            self.assertIn(link, html)
        self.assertGreaterEqual(report["items_rendered"], 3)

    def test_each_bullet_has_subject_unit_and_link(self):
        html, _ = render_numbers_block(ARTICLE_HTML, self.corpus)
        bullets = [line for line in html.splitlines() if line.strip().startswith("<li>")]
        self.assertTrue(bullets)
        for bullet in bullets:
            self.assertRegex(bullet, r"<strong>[^<]+:</strong>")  # subject
            self.assertIn("<a href=\"http", bullet)  # verifiable source

    def test_block_is_omitted_when_nothing_can_be_grounded(self):
        html, report = render_numbers_block(ARTICLE_HTML, [])
        self.assertEqual(html, "")
        self.assertTrue(report["skipped"])
        self.assertIn("no evidence", report["reason"])

    def test_block_is_omitted_rather_than_padded(self):
        thin_corpus = build_evidence_corpus({
            "research_dossier": {"source_map": [{
                "title": "One stat only",
                "url": "https://example.com/one",
                "content": "Dynamic task decomposition improves retrieval accuracy by 15 to 30 percentage points.",
            }]}
        })
        html, report = render_numbers_block(ARTICLE_HTML, thin_corpus, min_items=3)
        self.assertEqual(html, "")
        self.assertTrue(report["skipped"])
        self.assertLess(report["items_rendered"], 3)


class AuditTests(unittest.TestCase):
    def test_audit_reports_ungrounded_claims(self):
        report = audit_numbers(ARTICLE_HTML, build_evidence_corpus(EVIDENCE))
        claims = [sample["claim"] for sample in report["ungrounded_samples"]]
        self.assertGreater(report["ungrounded"], 0)
        self.assertTrue(any("$90" in claim for claim in claims))
        self.assertGreater(report["grounded"], 0)

    def test_url_less_evidence_cannot_ground_a_published_number(self):
        corpus = build_evidence_corpus(EVIDENCE_BY_TITLE_ONLY)
        html, report = render_numbers_block(
            "<p>Enterprises must implement identity-following retrieval for least-privilege access today.</p>",
            corpus,
        )
        self.assertEqual(html, "")
        self.assertEqual(report["items_rendered"], 0)


if __name__ == "__main__":
    unittest.main()
