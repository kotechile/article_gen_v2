"""The story contract is wired into every stage that can produce prose.

Three stages decide whether paragraphs flow: the section writer (drafts them),
the per-section refinement pass (rewrites them), and the whole-document editor
(joins them). Each carries the same contract, and the editor's flow half is
measured rather than trusted - a pass that merges paragraphs or turns sentences
into labels is retried once with its own failing paragraphs quoted, and the
pre-edit draft is kept if the retry does not improve on it.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from content_generator import ContentBlock, ContentGenerator, SectionContent  # noqa: E402


class _Response:
    def __init__(self, content):
        self.content = content


class _StubClient:
    """Returns the queued replies in order and records the prompts it was given."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, messages):
        self.prompts.append(messages[-1]["content"])
        return _Response(self.replies.pop(0))


def _generator():
    return ContentGenerator(llm_client=_StubClient([]))


def _article(paragraphs):
    """A body long enough for the document pass, with unique headings per block.

    The editor's hard guards compare headings element by element, so a fixture
    that only reuses six headings across twelve paragraphs looks like a heading
    change the moment two paragraphs merge - and would test the wrong guard.
    """
    blocks = []
    for index, text in enumerate(paragraphs, 1):
        blocks.append("<h2>Mechanism %d</h2><p>%s</p>" % (index, text))
    return "<h1>Order Cuts</h1>" + "".join(blocks)


FLAT_PARAGRAPH = (
    "The payoff is measurable. Teams that enable these controls typically report a 4 to 6 "
    "percentage point improvement in case fill rate and a 24% reduction in expedited freight "
    "spend, which the operations team tracks every quarter across every region it serves."
)
STORY_PARAGRAPH = (
    "Product Allocation (PAL) eliminates forecast cannibalization: weekly supply limits flow "
    "into S/4HANA, which checks every incoming order against its customer ceiling before "
    "confirming, so a retailer ordering 3x forecast at 6:00 a.m. on Monday gets only their "
    "allocated volume and the rest of the 24% saving stays in the DC for other customers."
)


class StoryContractTests(unittest.TestCase):
    def test_section_writer_carries_the_contract(self):
        block = ContentGenerator._get_story_thread_instructions(None)
        self.assertIn("ONE CLAIM PER PARAGRAPH", block)
        self.assertIn("CONNECT BACKWARD", block)
        self.assertIn("frame that announces the claim instead of making it", block)
        self.assertIn("ONE MEMBER OF A SET PER PARAGRAPH", block)

    def test_refinement_pass_states_story_above_substance(self):
        import tasks

        prompt = tasks._build_refinement_user_message(
            "professional", "<p>Body text here.</p>", current_word_count=480
        )
        self.assertIn("READABILITY AND STORY", prompt)
        self.assertIn("announces the claim instead of making it", prompt)
        self.assertLess(prompt.index("READABILITY AND STORY"), prompt.index("SUBSTANCE"))

    def test_document_editor_repairs_paragraph_joints_first(self):
        import tasks

        prompt = tasks.DOCUMENT_EDITOR_PROMPT
        self.assertIn("PARAGRAPH JOINTS", prompt)
        self.assertLess(prompt.index("PARAGRAPH JOINTS"), prompt.index("BULLET SCAFFOLDING"))
        self.assertIn("Before returning, check these mechanically", prompt)
        self.assertIn("The paragraph count is the same or higher", prompt)


class PreviousSectionTailTests(unittest.TestCase):
    def _section(self, blocks):
        return SectionContent(
            title="Capping Monday Orders",
            subtitle=None,
            content_blocks=blocks,
            total_word_count=200,
            key_points_covered=[],
            citations=[],
            section_order=1,
        )

    def test_tail_is_the_closing_prose(self):
        section = self._section(
            [
                ContentBlock(content="<p>Opening sentence of the section.</p>", content_type="paragraph", word_count=6),
                ContentBlock(content="<p>The orders are already confirmed and the inventory is gone.</p>",
                             content_type="paragraph", word_count=11),
            ]
        )
        tail = ContentGenerator._previous_section_tail(section)
        self.assertTrue(tail.endswith("the inventory is gone."), tail)
        self.assertNotIn("<p>", tail)

    def test_tail_skips_tables_and_is_bounded(self):
        section = self._section(
            [
                ContentBlock(content="<p>" + " ".join(["alpha"] * 300) + "</p>", content_type="paragraph", word_count=300),
                ContentBlock(content="<table><tr><td>never</td></tr></table>", content_type="table", word_count=1),
            ]
        )
        tail = ContentGenerator._previous_section_tail(section, max_words=40)
        self.assertEqual(len(tail.split()), 40)
        self.assertNotIn("never", tail)

    def test_empty_section_is_safe(self):
        self.assertEqual(ContentGenerator._previous_section_tail(None), "")
        self.assertEqual(
            ContentGenerator._previous_section_tail(self._section([])),
            "",
        )

    def test_user_message_asks_the_writer_to_pick_up_the_thread(self):
        generator = _generator()
        message = generator._build_user_message(
            {
                "title": "Capping Monday Orders",
                "subtitle": None,
                "tone": "professional",
                "key_points": ["Hit the cut before confirmation"],
                "research_brief": "brief",
                "draft_title": "",
                "relevant_claims": [],
                "relevant_evidence": [],
                "previous_sections": "Section 1: The EDI drop - 600 words",
                "previous_tail": "The orders are already confirmed and the inventory is gone.",
                "keywords": "",
                "is_first_section": False,
                "include_in_text_citations": True,
            }
        )
        self.assertIn("WHERE THE PREVIOUS SECTION LEFT OFF", message)
        self.assertIn("the inventory is gone", message)


class DocumentEditorFlowGuardTests(unittest.TestCase):
    """The flow pass is judged on measured shape, not on the prompt it was given."""

    def _run(self, replies, article):
        import tasks

        client = _StubClient(replies)
        with patch.object(tasks, "create_llm_client", return_value=client), patch.object(
            tasks, "get_llm_provider_for_role", return_value=("openai", "gpt-test", "key")
        ):
            content, report = tasks._unify_document_voice(
                article, {"provider": "openai", "model": "gpt-test"}, {"title": "Order Cuts"}
            )
        return content, report, client

    def test_openers_that_announce_instead_of_claim_are_retried_with_evidence(self):
        draft = _article([STORY_PARAGRAPH] * 12)
        flattened = _article(["The payoff is measurable. " + STORY_PARAGRAPH] * 12)
        improved = _article([STORY_PARAGRAPH] * 12)

        content, report, client = self._run([flattened, improved], draft)

        self.assertEqual(len(client.prompts), 2, "the guard must retry once")
        self.assertIn("MADE THE FLOW WORSE", client.prompts[1])
        self.assertTrue(report["flow_retried"])
        self.assertEqual(report["reason"], "")
        self.assertEqual(report["flow_after"]["meta_frame_openers"], 0)

    def test_a_retry_that_also_flattens_keeps_the_draft(self):
        draft = _article([STORY_PARAGRAPH] * 12)
        flattened = _article(["The payoff is measurable. " + STORY_PARAGRAPH] * 12)

        content, report, _ = self._run([flattened, flattened], draft)

        self.assertEqual(content, draft)
        self.assertFalse(report["applied"])
        self.assertIn("kept the pre-edit draft", report["reason"])

    def test_an_improving_pass_is_accepted_without_a_retry(self):
        draft = _article([FLAT_PARAGRAPH] * 12)
        improved = _article([STORY_PARAGRAPH] * 12)

        content, report, client = self._run([improved], draft)

        self.assertEqual(len(client.prompts), 1)
        self.assertFalse(report["flow_retried"])
        self.assertTrue(report["applied"])
        self.assertEqual(report["flow_before"]["meta_frame_openers"], 12)
        self.assertEqual(report["flow_after"]["meta_frame_openers"], 0)


if __name__ == "__main__":
    unittest.main()
