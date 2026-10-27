"""Tests for the requested-length trim.

Production case this fixes: a 2,500-word request generated 10,606 words; the old
"proportional trim" cut a word-prefix of every section and dropped the remaining
blocks, so sections ended mid-argument and ``content`` disagreed with
``content_blocks``. The publish gate then refused the result.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _tasks():
    # Imported lazily: other unit files stub sys.modules before importing tasks,
    # and being the first to import it would hand them the wrong module.
    import tasks

    return tasks


def _resolve_requested_word_count(data):
    return _tasks()._resolve_requested_word_count(data)


def _section_word_count(section):
    return _tasks()._section_word_count(section)


def _trim_sections_to_budget(sections, requested):
    return _tasks()._trim_sections_to_budget(sections, requested)


def make_section(title, paragraphs, words_per_paragraph=100):
    return {
        "title": title,
        "content_blocks": [
            {"content": " ".join([f"w{i}"] * words_per_paragraph), "content_type": "paragraph"}
            for i in range(paragraphs)
        ],
        "total_word_count": paragraphs * words_per_paragraph,
        "content": "\n\n".join("old", ),  # stale value, must be rebuilt
    }


class RequestedWordCountTests(unittest.TestCase):
    def test_prefers_target_word_count(self):
        self.assertEqual(_resolve_requested_word_count({"target_word_count": 2500, "articleLength": 900}), 2500)

    def test_falls_back_to_article_length(self):
        self.assertEqual(_resolve_requested_word_count({"articleLength": "1800"}), 1800)

    def test_zero_when_unset(self):
        self.assertEqual(_resolve_requested_word_count({}), 0)
        self.assertEqual(_resolve_requested_word_count({"target_word_count": ""}), 0)


class TrimTests(unittest.TestCase):
    def _big_article(self):
        # Mirrors the production run: 10 sections, ~10,000 words, 2,500 requested.
        return [make_section(f"Section {i}", n) for i, n in enumerate([10, 7, 13, 8, 9, 8, 8, 12, 13, 8], start=1)]

    def test_article_is_brought_near_the_request(self):
        sections = self._big_article()
        report = _trim_sections_to_budget(sections, 2500)
        final = sum(_section_word_count(s) for s in sections)
        self.assertTrue(report["changed"])
        self.assertLessEqual(final, report["budget"])
        self.assertGreater(final, 1500)

    def test_only_whole_blocks_are_kept(self):
        sections = self._big_article()
        _trim_sections_to_budget(sections, 2500)
        for section in sections:
            for block in section["content_blocks"]:
                # never a partial paragraph
                self.assertEqual(len(block["content"].split()), 100)

    def test_every_section_keeps_at_least_one_block(self):
        sections = self._big_article()
        _trim_sections_to_budget(sections, 600)
        for section in sections:
            self.assertGreaterEqual(len(section["content_blocks"]), 1)

    def test_content_is_rebuilt_from_surviving_blocks(self):
        sections = self._big_article()
        _trim_sections_to_budget(sections, 2500)
        for section in sections:
            expected = "\n\n".join(block["content"] for block in section["content_blocks"])
            self.assertEqual(section["content"], expected)

    def test_section_word_count_reflects_the_trim(self):
        sections = self._big_article()
        _trim_sections_to_budget(sections, 2500)
        for section in sections:
            expected = sum(len(b["content"].split()) for b in section["content_blocks"])
            self.assertEqual(_section_word_count(section), expected)
            self.assertEqual(section["total_word_count"], expected)

    def test_article_within_budget_is_untouched(self):
        sections = [make_section("Small", 2)]  # 200 words
        before = sum(_section_word_count(s) for s in sections)
        report = _trim_sections_to_budget(sections, 2500)
        self.assertFalse(report["changed"])
        self.assertEqual(sum(_section_word_count(s) for s in sections), before)

    def test_no_requested_length_means_no_trim(self):
        sections = self._big_article()
        report = _trim_sections_to_budget(sections, 0)
        self.assertFalse(report["changed"])


if __name__ == "__main__":
    unittest.main()
