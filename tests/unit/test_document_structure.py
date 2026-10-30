"""The finished document must have one title, one lead and one set of closing modules.

Fixture mirrors the document that reached the editor: the "Go deeper" line-in at the
very top, a "Go Deeper" heading above it all, a section heading followed by a second
headline as h1, and every section repeating its own lead and closing modules.
"""

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.services.article_structure_normalizer import normalize_document_structure  # noqa: E402

LEAD = '<p><strong>The Big Picture:</strong> {text}</p>\n<hr>'
GLANCE = "<h3>At a Glance</h3>\n<ul>\n<li><strong>The Big Shift:</strong> {text}</li>\n</ul>"
CATCH = "<h3>The Catch</h3>\n<p>{text}</p>\n<hr>"
WATCH = "<h3>What I'd Watch:</h3>\n<ul>\n<li><strong>Latency:</strong> {text}</li>\n</ul>"


def section(title, headline, body, lead="lead text", catch="catch text"):
    return (
        f"<h2>{title}</h2>\n"
        f"<h1>{headline}</h1>\n"
        f"{LEAD.format(text=lead)}\n"
        f"{GLANCE.format(text='shift ' + title)}\n"
        f"<h3>{title} mechanics</h3>\n<p>{body}</p>\n"
        f"{WATCH.format(text='watch ' + title)}\n"
        f"{CATCH.format(text=catch)}\n"
    )


def messy_document():
    return (
        '<p><strong>Go deeper:</strong> Review the detailed breakdown, comparative analysis, '
        "and step-by-step guidance below.</p>\n\n"
        "<h2>Go Deeper</h2>\n\n"
        + section("First Reckoning", "Killers Of Retrieval", "first body " * 30)
        + section("Second Reckoning", "Failures At Scale", "second body " * 30)
        + '<h2>References</h2>\n<p>[1] some source</p>\n'
    )


class TitleTests(unittest.TestCase):
    def test_exactly_one_h1_and_it_is_first(self):
        out, report = normalize_document_structure(messy_document(), title="RAG explained")
        self.assertEqual(report["h1_before"], 2)  # one per section
        self.assertEqual(len(re.findall(r"<h1>", out)), 1)
        self.assertTrue(out.startswith("<h1>RAG explained</h1>"))

    def test_nothing_precedes_the_title(self):
        out, _ = normalize_document_structure(messy_document(), title="RAG explained")
        self.assertNotIn("<p>", out[: out.index("<h1>")])

    def test_section_headlines_are_demoted_or_merged(self):
        out, _ = normalize_document_structure(messy_document(), title="RAG explained")
        self.assertNotIn("<h1>First Reckoning", out)
        self.assertNotIn("<h1>Killers Of Retrieval", out)
        self.assertIn("<h2>", out)


class GoDeeperTests(unittest.TestCase):
    def test_no_announcement_line_and_one_heading(self):
        out, report = normalize_document_structure(messy_document(), title="RAG explained")
        self.assertEqual(report["go_deeper_headings_before"], 1)
        # the "Review the detailed breakdown..." line announces the article to itself
        self.assertEqual(len(re.findall(r"<strong>Go deeper:</strong>", out)), 0)
        self.assertEqual(len(re.findall(r"<h2>Go Deeper</h2>", out)), 1)

    def test_divider_sits_after_the_lead_and_before_the_body(self):
        out, _ = normalize_document_structure(messy_document(), title="RAG explained")
        title_at = out.index("<h1>")
        big_picture_at = out.index("The Big Picture")
        heading_at = out.index("<h2>Go Deeper</h2>")
        body_at = out.index("First Reckoning")
        self.assertLess(title_at, big_picture_at)
        self.assertLess(big_picture_at, heading_at)
        self.assertLess(heading_at, body_at)

    def test_duplicate_lead_paragraphs_collapse_to_one_per_marker(self):
        out, report = normalize_document_structure(messy_document(), title="RAG explained")
        opening = out[: out.index("<h2>Go Deeper</h2>")]
        self.assertEqual(len(re.findall(r"The Big Picture:", opening)), 1)
        self.assertEqual(report["lead_paragraphs"], 1)
        # the later copy is not deleted - it stays as that section's opening
        self.assertEqual(len(re.findall(r"The Big Picture:", out)), 2)

    def test_content_under_a_go_deeper_heading_is_not_lost(self):
        messy = messy_document().replace(
            "<h2>Go Deeper</h2>",
            "<h2>Go Deeper</h2><p>Opening paragraph that must survive.</p>",
        )
        out, _ = normalize_document_structure(messy, title="RAG explained")
        self.assertIn("Opening paragraph that must survive.", out)


class ClosingModuleTests(unittest.TestCase):
    def test_modules_appear_once(self):
        out, report = normalize_document_structure(messy_document(), title="RAG explained")
        self.assertEqual(len(re.findall(r"At a Glance", out)), 1)
        self.assertEqual(len(re.findall(r"The Catch", out)), 1)
        self.assertEqual(len(re.findall(r"What I'd Watch", out)), 1)
        self.assertGreater(report["duplicate_modules_removed"], 0)

    def test_modules_are_at_the_end(self):
        out, _ = normalize_document_structure(messy_document(), title="RAG explained")
        last_body = out.rindex("second body")
        self.assertGreater(out.index("At a Glance"), last_body)
        self.assertGreater(out.index("The Catch"), last_body)
        self.assertGreater(out.index("What I'd Watch"), last_body)


class PreservationTests(unittest.TestCase):
    def test_body_prose_survives(self):
        out, _ = normalize_document_structure(messy_document(), title="RAG explained")
        self.assertIn("first body", out)
        self.assertIn("second body", out)
        self.assertIn("<h3>First Reckoning mechanics</h3>", out)

    def test_clean_document_is_a_no_op(self):
        clean = (
            "<h1>RAG explained</h1>\n"
            "<p><strong>The Big Picture:</strong> x</p>\n"

            "<h2>Go Deeper</h2>\n"
            "<h2>A section</h2>\n<p>body</p>\n"
            "<h3>At a Glance</h3>\n<ul><li>x</li></ul>"
        )
        out, report = normalize_document_structure(clean, title="RAG explained")
        self.assertEqual(len(re.findall(r"<h1>", out)), 1)
        self.assertFalse(report["changed"], out)


if __name__ == "__main__":
    unittest.main()
