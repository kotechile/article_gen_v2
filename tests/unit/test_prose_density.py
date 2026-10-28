"""Prose-first: bullet scaffolding is measurable, and stale references are removed."""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.services.prose_density import measure_prose_density, strip_references_section  # noqa: E402

LONG = "This sentence carries enough words to count as a real paragraph of prose here."

PROSE = "".join(f"<h3>Part {i}</h3><p>{LONG}</p><p>{LONG}</p>" for i in range(1, 4))

SLIDES = "".join(
    f"""<h3>Part {i}</h3>
<p>{LONG}</p>
<ul>
  <li><strong>Retrieval gap:</strong> {LONG}</li>
  <li><strong>Multi-hop failure:</strong> {LONG}</li>
  <li><strong>Rework tax:</strong> {LONG}</li>
</ul>"""
    for i in range(1, 4)
)


class DensityTests(unittest.TestCase):
    def test_prose_scores_low(self):
        report = measure_prose_density(PROSE)
        self.assertEqual(report["list_items"], 0)
        self.assertFalse(report["reads_like_slides"], report)

    def test_labelled_bullets_are_flagged(self):
        report = measure_prose_density(SLIDES)
        self.assertEqual(report["list_items"], 9)
        self.assertEqual(report["label_bullets"], 9)
        self.assertTrue(report["reads_like_slides"])
        self.assertTrue(report["reasons"])

    def test_more_lists_than_paragraphs_is_called_out(self):
        report = measure_prose_density(SLIDES)
        self.assertGreater(report["bullet_ratio"], 1.0)
        self.assertTrue(any("list items than paragraphs" in r for r in report["reasons"]))

    def test_short_paragraph_fragments_do_not_count_as_prose(self):
        html = "<p>Too short.</p>" * 10
        self.assertEqual(measure_prose_density(html)["paragraphs"], 0)

    def test_empty_input_is_safe(self):
        report = measure_prose_density("")
        self.assertEqual(report["list_items"], 0)
        self.assertFalse(report["reads_like_slides"])


class ReferenceStripTests(unittest.TestCase):
    def test_generated_references_block_is_removed(self):
        html = (
            "<h1>T</h1><p>body</p>\n"
            "<h2>References</h2>\n<p>[1] [Agentic RAG](https://www.</p>\n<p>[2] ### **3.</p>"
        )
        out = strip_references_section(html)
        self.assertNotIn("References", out)
        self.assertNotIn("[Agentic RAG]", out)
        self.assertIn("<p>body</p>", out)

    def test_document_without_references_is_untouched(self):
        html = "<h1>T</h1><p>body</p>"
        self.assertEqual(strip_references_section(html), html)

    def test_sources_heading_is_also_removed(self):
        html = "<h1>T</h1><p>body</p><h2>Sources</h2><p>[1] x</p>"
        self.assertNotIn("Sources", strip_references_section(html))


class SectionPromptTests(unittest.TestCase):
    """The prompt that produced the bullet farm must now ask for prose."""

    def _prompt(self):
        # Another test file replaces `content_generator` in sys.modules with a stub
        # before importing tasks; evict it so this reads the real prompt builder.
        module = sys.modules.get("content_generator")
        if module is not None and not hasattr(module, "ContentGenerator"):
            del sys.modules["content_generator"]
        from content_generator import ContentGenerator

        generator = ContentGenerator.__new__(ContentGenerator)
        return generator._get_demographic_and_smart_brevity_instructions({"target_audience": "general"})

    def test_prose_first_is_stated(self):
        prompt = self._prompt()
        self.assertIn("PROSE FIRST", prompt)
        self.assertIn("connected paragraphs", prompt)

    def test_labelled_bullets_are_forbidden(self):
        prompt = self._prompt()
        self.assertIn("NEVER use the", prompt)
        self.assertIn("**Label:** explanation", prompt)

    def test_old_bullet_factory_is_gone(self):
        prompt = self._prompt()
        self.assertNotIn("followed by exactly 3 bolded metric", prompt)


if __name__ == "__main__":
    unittest.main()
