"""The whole-document editor may improve flow, never lose evidence.

Per-section refinement cannot see the joins between sections, which is why an
article assembled from good sections still reads like isolated queries. The document
pass is the fix - but it is a rewrite of the entire article by a model, so the guards
matter more than the prompt.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tasks  # noqa: E402

LONG = (
    "Retrieval quality decides whether an agent can answer a question at all, and the "
    "measured gap between single-pass and iterative retrieval is large enough to change "
    "an architecture decision on its own."
)

ARTICLE = (
    "<h1>RAG Agentic AI Explained</h1>"
    f"<p><strong>The Big Picture:</strong> {LONG}</p>"
    "<h2>Diagnosing Naive RAG Failures</h2>"
    + "".join(f"<p>{LONG} Teams lose 41% of multi-hop queries and pay $250k a year [1].</p>" for _ in range(6))
    + "<h2>Engineering Guardrails</h2>"
    + "".join(f"<p>{LONG} Reflection adds 30% to 48% token overhead [2].</p>" for _ in range(6))
)


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


class DocumentEditorTests(unittest.TestCase):
    def setUp(self):
        self._orig = (tasks.create_llm_client, tasks.get_llm_provider_for_role)

    def tearDown(self):
        tasks.create_llm_client, tasks.get_llm_provider_for_role = self._orig

    def _run(self, payload):
        client = _FakeClient(payload)
        tasks.create_llm_client = lambda **kwargs: client
        tasks.get_llm_provider_for_role = lambda role: ("gemini", "gemini-3.1-pro-preview", "key")
        out, report = tasks._unify_document_voice(ARTICLE, {}, {"title": "t"})
        return out, report, client

    def test_a_clean_rewrite_is_applied(self):
        revised = ARTICLE.replace("Teams lose 41%", "In practice, teams lose 41%")
        out, report, client = self._run(revised)
        self.assertEqual(client.calls, 1)
        self.assertTrue(report["applied"], report)
        self.assertIn("In practice", out)

    def test_a_lost_number_reverts_the_whole_pass(self):
        # 48% silently disappears - exactly what an editor must never do.
        revised = ARTICLE.replace("30% to 48% token overhead", "heavy token overhead")
        out, report, _ = self._run(revised)
        self.assertFalse(report["applied"])
        self.assertIn("numbers lost", report["reason"])
        self.assertEqual(out, ARTICLE)

    def test_a_lost_citation_marker_reverts_the_pass(self):
        revised = ARTICLE.replace(" [2]", "")
        out, report, _ = self._run(revised)
        self.assertFalse(report["applied"])
        self.assertIn("citation markers lost", report["reason"])
        self.assertEqual(out, ARTICLE)

    def test_changed_headings_revert_the_pass(self):
        revised = ARTICLE.replace("<h2>Engineering Guardrails</h2>", "<h2>Guardrails</h2>")
        out, report, _ = self._run(revised)
        self.assertFalse(report["applied"])
        self.assertIn("heading structure changed", report["reason"])

    def test_a_gutted_document_reverts_the_pass(self):
        revised = "<h1>RAG Agentic AI Explained</h1><p>" + LONG + " 41% $250k [1] 30% 48% [2]</p>"
        revised = revised.replace("<h2>", "")
        out, report, _ = self._run(revised)
        self.assertFalse(report["applied"])
        self.assertTrue(out == ARTICLE)

    def test_short_article_is_skipped_without_a_call(self):
        client = _FakeClient(ARTICLE)
        tasks.create_llm_client = lambda **kwargs: client
        tasks.get_llm_provider_for_role = lambda role: ("gemini", "m", "key")
        out, report = tasks._unify_document_voice("<h1>t</h1><p>short</p>", {}, {})
        self.assertFalse(report["applied"])
        self.assertEqual(client.calls, 0)
        self.assertIn("too short", report["reason"])

    def test_fenced_output_is_unwrapped(self):
        out, report, _ = self._run("```html\n" + ARTICLE + "\n```")
        self.assertTrue(report["applied"], report)
        self.assertFalse(out.startswith("```"))

    def test_editor_failure_keeps_the_original(self):
        def boom(**kwargs):
            raise RuntimeError("provider down")

        tasks.create_llm_client = boom
        tasks.get_llm_provider_for_role = lambda role: ("gemini", "m", "key")
        out, report = tasks._unify_document_voice(ARTICLE, {}, {})
        self.assertFalse(report["applied"])
        self.assertIn("provider down", report["reason"])
        self.assertEqual(out, ARTICLE)


if __name__ == "__main__":
    unittest.main()
