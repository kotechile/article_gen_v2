"""The excerpt must describe the article that was actually written.

The excerpt is written at outline time, from the brief. The finished piece follows
its own outline, so the stored excerpt can advertise sections, evidence or an
argument the article never contains - and it is the first sentence a reader (and
every social card) sees.

Two stages check it: the structure generator writes the excerpt from the planned
sections and rejects one they do not support, and finalization reconciles it
against the finished document. Both repair only from grounded material.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.services.excerpt_alignment import (  # noqa: E402
    derive_excerpt_from_article,
    derive_excerpt_from_outline,
    evaluate_excerpt_alignment,
)

LEDE = (
    "Naive retrieval pipelines drop 41% of multi-hop queries because every hop "
    "re-embeds the same context instead of carrying the answer forward."
)

ARTICLE = (
    "<h1>RAG Agentic AI Explained</h1>"
    f"<p>{LEDE}</p>"
    "<h2>Auditing Distributed Lock Contention</h2>"
    + "".join(
        f"<p>Reranking the retrieved context adds 30% token overhead per hop, and teams "
        f"that skip the audit pay $250k a year for the same failed queries.</p>"
        for _ in range(12)
    )
)

MISMATCHED = (
    "A practical guide to dashboard rituals, growth loops and team rituals that lift "
    "revenue fast without hiring a data team."
)

MATCHED = (
    "Naive retrieval drops 41% of multi-hop queries because each hop re-embeds context; "
    "reranking the retrieved passages adds 30% token overhead per hop."
)


class ExcerptAlignmentTests(unittest.TestCase):
    def test_an_excerpt_about_another_article_does_not_match(self):
        report = evaluate_excerpt_alignment(MISMATCHED, ARTICLE)
        self.assertFalse(report["matches"], report)
        self.assertIn("dashboard", report["unsupported_terms"])
        self.assertIn("terms appear in the article", report["reason"])

    def test_a_paraphrase_of_the_article_matches(self):
        report = evaluate_excerpt_alignment(MATCHED, ARTICLE)
        self.assertTrue(report["matches"], report)
        self.assertGreaterEqual(report["support"], 0.5)

    def test_a_number_the_article_dropped_fails_even_when_wording_matches(self):
        # Every term is supported; the 87% came from the brief and never made it in.
        report = evaluate_excerpt_alignment(
            "Naive retrieval drops 87% of multi-hop queries; reranking adds 30% token overhead.",
            ARTICLE,
        )
        self.assertFalse(report["matches"], report)
        self.assertEqual(report["missing_numbers"], ["87"])
        self.assertIn("numbers not in the article", report["reason"])

    def test_framing_words_alone_do_not_carry_an_excerpt(self):
        # "guide/practical/insight" are what every standfirst says; they must not
        # let a brief-driven excerpt clear the support bar.
        report = evaluate_excerpt_alignment(
            "A practical guide with real insight into better workflows.", ARTICLE
        )
        self.assertFalse(report["matches"], report)

    def test_an_empty_excerpt_never_matches(self):
        report = evaluate_excerpt_alignment("", ARTICLE)
        self.assertFalse(report["matches"])
        self.assertEqual(report["reason"], "empty excerpt")

    def test_derived_excerpt_comes_from_the_article_lede(self):
        derived = derive_excerpt_from_article(ARTICLE)
        self.assertTrue(derived.startswith("Naive retrieval pipelines drop 41%"), derived)
        self.assertTrue(derived.endswith("."), derived)
        self.assertTrue(evaluate_excerpt_alignment(derived, ARTICLE)["matches"])

    def test_derived_excerpt_drops_module_labels_and_citation_markers(self):
        body = f"<p><strong>The Big Picture:</strong> {LEDE} [2] {LEDE}</p>"
        derived = derive_excerpt_from_article(body)
        self.assertNotIn("The Big Picture", derived)
        self.assertNotIn("[2]", derived)
        self.assertIn("Naive retrieval", derived)

    def test_derived_excerpt_is_capped(self):
        derived = derive_excerpt_from_article(f"<p>{LEDE} {LEDE} {LEDE}</p>", max_words=12)
        self.assertLessEqual(len(derived.split()), 12)

    def test_outline_derived_excerpt_uses_the_planners_own_points(self):
        sections = [
            {"title": "Auditing Distributed Lock Contention", "key_points": ["Short point"]},
            {
                "title": "Slashing Database Latency by 40%",
                "key_points": [
                    "Lock contention, not query volume, is what pushes p99 latency past the "
                    "SLO in most write-heavy services."
                ],
            },
        ]
        derived = derive_excerpt_from_outline(sections)
        self.assertIn("Lock contention", derived)
        self.assertNotIn("Short point", derived)  # too thin to be an excerpt sentence

    def test_outline_without_usable_points_returns_nothing(self):
        # No synthesised standfirst: an empty result lets the caller keep what it has
        # and log the reason instead of publishing a sentence that says nothing.
        self.assertEqual(derive_excerpt_from_outline([{"title": "Only a title"}]), "")


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.prompts = []

    def generate(self, messages, model=None):
        self.calls += 1
        self.prompts.append(messages)
        return _FakeResponse(self.payload)


class FinalizeExcerptTests(unittest.TestCase):
    """`_reconcile_excerpt` is the final validator's excerpt gate."""

    def setUp(self):
        import tasks

        self.tasks = tasks
        self._orig = (tasks.create_llm_client, tasks.get_llm_provider_for_role)

    def tearDown(self):
        self.tasks.create_llm_client, self.tasks.get_llm_provider_for_role = self._orig

    def _run(self, excerpt, payload=MATCHED):
        client = _FakeClient(payload)
        self.tasks.create_llm_client = lambda **kwargs: client
        self.tasks.get_llm_provider_for_role = lambda role: ("gemini", "gemini-3.1-pro", "key")
        out, report = self.tasks._reconcile_excerpt(
            excerpt,
            ARTICLE,
            research_data={},
            structure={"title": "RAG Agentic AI Explained"},
        )
        return out, report, client

    def test_a_matching_excerpt_is_left_alone_without_a_model_call(self):
        out, report, client = self._run(MATCHED)
        self.assertEqual(out, MATCHED)
        self.assertFalse(report["applied"])
        self.assertEqual(client.calls, 0)

    def test_a_mismatched_excerpt_is_rewritten_against_the_article(self):
        out, report, client = self._run(MISMATCHED)
        self.assertEqual(client.calls, 1)
        self.assertTrue(report["applied"], report)
        self.assertEqual(report["source"], "model")
        self.assertEqual(out, MATCHED)
        # The rewrite is asked against the finished article, not the brief.
        prompt = client.prompts[0][1]["content"]
        self.assertIn("multi-hop", prompt)

    def test_a_rewrite_that_still_does_not_match_falls_back_to_the_lede(self):
        out, report, client = self._run(MISMATCHED, payload=MISMATCHED)
        self.assertEqual(client.calls, 1)
        self.assertTrue(report["applied"], report)
        self.assertEqual(report["source"], "derived")
        self.assertTrue(out.startswith("Naive retrieval pipelines drop 41%"), out)
        self.assertTrue(evaluate_excerpt_alignment(out, ARTICLE)["matches"])

    def test_a_rewrite_quoting_a_number_the_article_lacks_is_rejected(self):
        invented = (
            "Naive retrieval drops 87% of multi-hop queries, and reranking the retrieved "
            "passages adds 30% token overhead per hop."
        )
        out, report, client = self._run(MISMATCHED, payload=invented)
        self.assertEqual(report["source"], "derived")
        self.assertNotIn("87", out)

    def test_markup_from_the_model_is_rejected(self):
        out, report, _ = self._run(MISMATCHED, payload="<p>" + MATCHED + "</p>")
        self.assertEqual(report["source"], "derived")
        self.assertFalse(out.startswith("<p>"))

    def test_a_title_echoing_the_headline_is_rejected(self):
        out, report, _ = self._run(MISMATCHED, payload="RAG Agentic AI Explained")
        self.assertEqual(report["source"], "derived")

    def test_a_failed_model_call_still_repairs_the_excerpt(self):
        def boom(**kwargs):
            raise RuntimeError("provider down")

        self.tasks.create_llm_client = boom
        self.tasks.get_llm_provider_for_role = lambda role: ("gemini", "m", "key")
        out, report = self.tasks._reconcile_excerpt(
            MISMATCHED, ARTICLE, research_data={}, structure={}
        )
        self.assertTrue(report["applied"], report)
        self.assertEqual(report["source"], "derived")
        self.assertIn("provider down", report["reason"])


class OutlineStageTests(unittest.TestCase):
    """The structure generator writes the excerpt from the plan, then checks it."""

    def _generator(self, payloads):
        from article_structure_generator import ArticleStructureGenerator

        class _Client:
            def __init__(self):
                self.payloads = list(payloads)
                self.prompts = []
                self.calls = 0

            def generate(self, messages, model=None):
                self.calls += 1
                self.prompts.append(messages)
                payload = self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]
                return _FakeResponse(payload)

        client = _Client()
        generator = ArticleStructureGenerator(llm_client=client)
        return generator, client

    @staticmethod
    def _sections():
        from article_structure_generator import SectionOutline

        return [
            SectionOutline(
                title="Auditing Distributed Lock Contention",
                key_points=[
                    "Lock contention, not query volume, pushes p99 latency past the SLO in "
                    "write-heavy services."
                ],
            ),
            SectionOutline(
                title="Slashing Database Latency by 40%",
                key_points=[
                    "Trimming the transaction that holds the row lock recovers most of the "
                    "latency without a schema change."
                ],
            ),
        ]

    def test_the_excerpt_prompt_carries_the_planned_sections(self):
        generator, client = self._generator(
            ["Lock contention pushes p99 latency past the SLO in write-heavy services, "
             "and trimming the transaction that holds the row lock recovers it."]
        )
        excerpt = generator._generate_excerpt("brief", [], 2000, "professional", sections=self._sections())
        self.assertIn("Lock contention", excerpt)
        prompt = "\n".join(m["content"] for m in client.prompts[0])
        self.assertIn("Auditing Distributed Lock Contention", prompt)
        self.assertIn("Planned sections", prompt)

    def test_an_excerpt_the_outline_does_not_support_is_rewritten(self):
        off_plan = (
            "This guide explores dashboard rituals, growth loops and hiring plans that "
            "lift revenue fast for every data team."
        )
        on_plan = (
            "Lock contention pushes p99 latency past the SLO in write-heavy services, and "
            "trimming the transaction that holds the row lock recovers the latency."
        )
        generator, client = self._generator([off_plan, on_plan])
        excerpt = generator._generate_excerpt("brief", [], 2000, "professional", sections=self._sections())
        self.assertEqual(client.calls, 2)
        self.assertEqual(excerpt, on_plan)

    def test_two_off_plan_attempts_fall_back_to_the_outline_key_points(self):
        off_plan = (
            "This guide explores dashboard rituals, growth loops and hiring plans that "
            "lift revenue fast for every data team."
        )
        generator, _ = self._generator([off_plan])
        excerpt = generator._generate_excerpt("brief", [], 2000, "professional", sections=self._sections())
        self.assertIn("Lock contention", excerpt)
        self.assertNotIn("dashboard", excerpt)


if __name__ == "__main__":
    unittest.main()
