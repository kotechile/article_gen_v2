"""An acronym is spelled out on first use, then run short.

The rule applies to every document the generator produces: the first appearance of
an acronym or initialism reads "retrieval-augmented generation (RAG)", and every
later mention uses the short form. The prompts ask the writer and the editor for it;
finalization audits the assembled document and repairs the remaining first uses with
a definition the model supplies and a deterministic spelling check validates.
"""

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.services.acronym_expansion import (  # noqa: E402
    audit_acronyms,
    expand_first_use,
    first_use_context,
    spelling_matches,
)

ARTICLE = (
    "<h1>RAG Guide</h1>"
    "<p>Teams adopt RAG early because it answers questions over their own documents.</p>"
    "<h2>Auditing the pipeline</h2>"
    "<p>Most RAG stacks fail on multi-hop questions, and the LLM hides it.</p>"
)


class SpellingTests(unittest.TestCase):
    def test_an_expansion_whose_initials_spell_the_acronym_passes(self):
        self.assertTrue(spelling_matches("retrieval-augmented generation", "RAG"))
        self.assertTrue(spelling_matches("large language models", "LLM"))
        self.assertTrue(spelling_matches("Federal Bureau of Investigation", "FBI"))
        self.assertTrue(spelling_matches("United States Postal Service", "USPS"))

    def test_an_expansion_that_does_not_spell_it_is_rejected(self):
        self.assertFalse(spelling_matches("rapid retrieval", "RAG"))
        self.assertFalse(spelling_matches("generative retrieval", "RAG"))
        self.assertFalse(spelling_matches("", "RAG"))

    def test_model_and_version_identifiers_have_no_expansion(self):
        self.assertFalse(spelling_matches("binary measure", "BM25"))
        self.assertFalse(spelling_matches("generative pretrained transformer", "GPT-4"))

    def test_an_over_long_phrase_is_not_a_definition(self):
        self.assertFalse(
            spelling_matches("retrieval augmented generation runs the whole stack today", "RAG")
        )


class AuditTests(unittest.TestCase):
    def test_a_bare_first_use_is_reported(self):
        report = audit_acronyms(ARTICLE)
        self.assertIn("RAG", report["unexpanded"])
        self.assertIn("LLM", report["unexpanded"])
        self.assertEqual(report["total"], 2)  # RAG and LLM

    def test_a_spelled_out_first_use_is_accepted(self):
        text = (
            "<p>Retrieval-augmented generation (RAG) answers questions over your own "
            "documents, and RAG fails on multi-hop questions.</p>"
        )
        report = audit_acronyms(text)
        self.assertNotIn("RAG", report["unexpanded"])
        self.assertEqual(report["expanded_count"], report["total"])

    def test_a_heading_is_not_running_text(self):
        # "RAG" in an H1 alone is not a first use the reader meets mid-sentence.
        self.assertEqual(audit_acronyms("<h1>RAG Guide</h1>")["total"], 0)

    def test_model_names_are_not_audited(self):
        report = audit_acronyms("<p>BM25 and GPT-4 are both ranked here.</p>")
        self.assertNotIn("BM25", report["unexpanded"])
        self.assertNotIn("GPT-4", report["unexpanded"])

    def test_a_bare_use_before_an_expansion_still_counts_as_bare(self):
        text = (
            "<p>RAG is popular. Retrieval-augmented generation (RAG) answers questions "
            "over documents.</p>"
        )
        self.assertIn("RAG", audit_acronyms(text)["unexpanded"])

    def test_first_use_context_returns_the_sentence(self):
        context = first_use_context(ARTICLE, "RAG")
        self.assertTrue(any("Teams adopt RAG early" in sentence for sentence in context), context)


class ExpandTests(unittest.TestCase):
    def test_the_first_body_use_is_expanded_and_later_uses_are_not(self):
        out, applied, mode = expand_first_use(ARTICLE, "RAG", "retrieval-augmented generation")
        self.assertTrue(applied)
        self.assertEqual(mode, "inlined")
        self.assertIn("Teams adopt retrieval-augmented generation (RAG) early", out)
        self.assertIn("Most RAG stacks fail", out)
        self.assertIn("<h1>RAG Guide</h1>", out)  # the heading is untouched

    def test_a_mismatched_expansion_changes_nothing(self):
        out, applied, _ = expand_first_use(ARTICLE, "RAG", "rapid retrieval")
        self.assertFalse(applied)
        self.assertEqual(out, ARTICLE)

    def test_an_already_expanded_acronym_is_left_alone(self):
        text = "<p>Retrieval-augmented generation (RAG) is a stack, and RAG is popular.</p>"
        out, applied, _ = expand_first_use(text, "RAG", "retrieval-augmented generation")
        self.assertFalse(applied)
        self.assertEqual(out, text)

    def test_emphasis_around_the_acronym_keeps_its_closing_tag(self):
        text = "<p>Teams adopt <strong>RAG</strong> early.</p>"
        out, applied, _ = expand_first_use(text, "RAG", "retrieval-augmented generation")
        self.assertTrue(applied)
        self.assertIn("<strong>retrieval-augmented generation (RAG)</strong>", out)


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def generate(self, messages, model=None):
        self.calls += 1
        return _FakeResponse(self.payload)


class FinalizeAcronymTests(unittest.TestCase):
    """`_expand_acronyms` is the finalization gate for the style rule."""

    def setUp(self):
        import tasks

        self.tasks = tasks
        self._orig = (tasks.create_llm_client, tasks.get_llm_provider_for_role)

    def tearDown(self):
        self.tasks.create_llm_client, self.tasks.get_llm_provider_for_role = self._orig

    def _run(self, payload, article=ARTICLE):
        client = _FakeClient(payload)
        self.tasks.create_llm_client = lambda **kwargs: client
        self.tasks.get_llm_provider_for_role = lambda role: ("gemini", "gemini-3.1-pro", "key")
        out, report = self.tasks._expand_acronyms(article, research_data={}, structure={"title": "t"})
        return out, report, client

    def test_expansions_are_inserted_at_first_use(self):
        payload = json.dumps(
            {
                "expansions": [
                    {"acronym": "RAG", "expansion": "retrieval-augmented generation"},
                    {"acronym": "LLM", "expansion": "large language models"},
                ]
            }
        )
        out, report, client = self._run(payload)
        self.assertEqual(client.calls, 1)
        self.assertIn("Teams adopt retrieval-augmented generation (RAG) early", out)
        self.assertIn("the large language models (LLM) hides it", out)
        self.assertEqual(set(report["applied"]), {"RAG", "LLM"})
        self.assertEqual(report["after"]["unexpanded"], [])

    def test_an_expansion_that_does_not_spell_the_acronym_is_rejected(self):
        payload = json.dumps(
            {"expansions": [{"acronym": "RAG", "expansion": "retrieval augmented pipeline"}]}
        )
        out, report, _ = self._run(payload)
        self.assertEqual(out, ARTICLE)
        self.assertIn("RAG", report["rejected"])
        self.assertIn("RAG", report["after"]["unexpanded"])

    def test_a_null_proposal_leaves_the_text_alone(self):
        payload = json.dumps({"expansions": [{"acronym": "RAG", "expansion": None}]})
        out, report, _ = self._run(payload)
        self.assertEqual(out, ARTICLE)
        self.assertEqual(report["applied"], {})
        self.assertIn("RAG", report["after"]["unexpanded"])

    def test_nothing_happens_without_an_unexpanded_acronym(self):
        clean = (
            "<p>Retrieval-augmented generation (RAG) answers questions, and RAG is popular.</p>"
        )
        out, report, client = self._run("{}", article=clean)
        self.assertEqual(client.calls, 0)
        self.assertEqual(out, clean)

    def test_unparseable_output_leaves_the_text_alone(self):
        out, report, _ = self._run("I think RAG means retrieval augmented generation.")
        self.assertEqual(out, ARTICLE)
        self.assertEqual(report["applied"], {})


if __name__ == "__main__":
    unittest.main()
