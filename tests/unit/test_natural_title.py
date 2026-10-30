"""Title: written to carry the keyword, and to still read like English.

"RAG Agentic AI Trap" contains the keyword and is not a headline. The mechanical
rewrite ("RAG Agentic AI Explained") is correct and lifeless, so a model writes the
title - but it only keeps it if the keyword guarantee still holds.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tasks  # noqa: E402


class _Resp:
    def __init__(self, content):
        self.content = content


class _Client:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def generate(self, messages, model=None):
        self.calls += 1
        if isinstance(self.payload, Exception):
            raise self.payload
        return _Resp(self.payload)


class NeedsRewriteTests(unittest.TestCase):
    def test_seed_with_keyword_still_needs_a_rewrite(self):
        self.assertTrue(tasks._title_needs_rewrite("RAG Agentic AI Trap", "rag agentic ai"))

    def test_missing_keyword_needs_a_rewrite(self):
        self.assertTrue(tasks._title_needs_rewrite("Why Retrieval Fails At Scale", "rag agentic ai"))

    def test_finished_headline_is_left_alone(self):
        self.assertFalse(
            tasks._title_needs_rewrite(
                "RAG Agentic AI: The Three Decisions That Break Enterprise Deployments",
                "rag agentic ai",
            )
        )

    def test_empty_title_needs_a_rewrite(self):
        self.assertTrue(tasks._title_needs_rewrite("", "rag agentic ai"))


class NaturalTitleTests(unittest.TestCase):
    def setUp(self):
        self._orig = tasks.create_llm_client

    def tearDown(self):
        tasks.create_llm_client = self._orig

    def _run(self, payload, title="RAG Agentic AI Trap", keyword="rag agentic ai"):
        client = _Client(payload)
        tasks.create_llm_client = lambda **kwargs: client
        return tasks._natural_title(title, keyword, provider="gemini", model="m", api_key="k")

    def test_a_written_headline_is_kept(self):
        out, report = self._run("RAG Agentic AI: The Three Decisions That Make or Break Deployment")
        self.assertTrue(report["applied"], report)
        self.assertIn("RAG Agentic AI", out)
        self.assertNotEqual(out, "RAG Agentic AI Trap")

    def test_a_headline_without_the_keyword_is_refused(self):
        out, report = self._run("The Three Decisions That Make or Break Deployment")
        self.assertFalse(report["applied"])
        self.assertIn("missing keyword", report["reason"])
        self.assertIn("RAG Agentic AI", out)  # mechanical fallback still carries it

    def test_an_overlong_headline_is_refused(self):
        out, report = self._run("RAG Agentic AI " + "and the very long tail of consequences " * 3)
        self.assertFalse(report["applied"])
        self.assertLessEqual(len(out), 70)

    def test_a_one_word_answer_is_refused(self):
        out, report = self._run("RAG Agentic AI")
        self.assertFalse(report["applied"])
        self.assertIn("words", report["reason"])

    def test_leaked_reasoning_in_the_title_is_refused(self):
        out, report = self._run("Let me analyze what I need to do with rag agentic ai here")
        self.assertFalse(report["applied"])
        self.assertIn("reasoning", report["reason"])

    def test_markdown_and_quotes_are_stripped(self):
        out, report = self._run('  "RAG Agentic AI: Pick Your Oversight Model"  ')
        self.assertTrue(report["applied"], report)
        self.assertFalse(out.startswith('"'))

    def test_model_failure_falls_back_to_the_mechanical_title(self):
        out, report = self._run(RuntimeError("provider down"))
        self.assertFalse(report["applied"])
        self.assertIn("provider down", report["reason"])
        self.assertIn("RAG Agentic AI", out)

    def test_no_model_means_no_call(self):
        client = _Client("whatever")
        tasks.create_llm_client = lambda **kwargs: client
        out, report = tasks._natural_title("RAG Agentic AI Trap", "rag agentic ai")
        self.assertFalse(report["applied"])
        self.assertEqual(client.calls, 0)
        self.assertIn("RAG Agentic AI", out)


class LeadTests(unittest.TestCase):
    def test_go_deeper_announcement_is_no_longer_written(self):
        source = (Path(REPO_ROOT) / "tasks.py").read_text()
        code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
        self.assertNotIn("Review the detailed breakdown, comparative analysis", code)


if __name__ == "__main__":
    unittest.main()
