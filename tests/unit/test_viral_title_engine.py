import unittest
from unittest.mock import MagicMock

from src.services.viral_title_engine import ViralTitleEngine
from src.services.article_quality_evaluator import build_article_quality_report, _title_virality_signals


class TestViralTitleEngine(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.engine = ViralTitleEngine(llm_client=self.mock_llm)

    def test_word_count_and_six_word_rule(self):
        # 6 words or fewer
        self.assertEqual(ViralTitleEngine.count_words("Why Remote Work Fails In 2026"), 6)
        res = ViralTitleEngine.evaluate_title_heuristics("Why Remote Work Fails In 2026")
        self.assertTrue(res["under_6_words"])
        self.assertEqual(res["word_count"], 6)

        # Over 6 words
        res_long = ViralTitleEngine.evaluate_title_heuristics("The Ultimate Complete Guide To Everything You Need To Know")
        self.assertFalse(res_long["under_6_words"])
        self.assertGreater(res_long["word_count"], 6)

    def test_front_loading_and_fluff_omission(self):
        # Front-loaded with strong noun / trigger, no leading fluff
        front_loaded = ViralTitleEngine.evaluate_title_heuristics("Remote Teams: 3 Fatal Traps")
        self.assertTrue(front_loaded["front_loaded"])

        # Title starting with fluff article
        fluff_front = ViralTitleEngine.evaluate_title_heuristics("The Best Ways For Remote Teams")
        self.assertFalse(fluff_front["front_loaded"])
        self.assertGreater(fluff_front["fluff_count"], 0)

    def test_psychological_triggers(self):
        # Curiosity gap
        res_curiosity = ViralTitleEngine.evaluate_title_heuristics("Why Smart Leaders Fail Fast")
        self.assertIn("curiosity_gap", res_curiosity["triggers"])

        # Loss aversion / negative framing
        res_loss = ViralTitleEngine.evaluate_title_heuristics("1 Fatal Mistake In Hiring")
        self.assertIn("loss_aversion", res_loss["triggers"])

        # Counter-intuitive authority
        res_counter = ViralTitleEngine.evaluate_title_heuristics("Skip SEO: Do This Instead")
        self.assertIn("counter_intuitive", res_counter["triggers"])

    def test_specificity_and_metrics(self):
        res_metrics = ViralTitleEngine.evaluate_title_heuristics("3 Costly Pricing Traps")
        self.assertTrue(res_metrics["has_metrics"])

        res_no_metrics = ViralTitleEngine.evaluate_title_heuristics("Pricing Traps To Avoid")
        self.assertFalse(res_no_metrics["has_metrics"])

    def test_title_review_during_generation(self):
        # Mock LLM response for candidate generation and review
        self.mock_llm.generate.side_effect = [
            # Candidate generation response
            MagicMock(content="""{
              "candidates": [
                {"title": "Why Fast Growth Kills Startups", "angle": "curiosity_gap", "promise": "Explain why premature scaling kills founders"},
                {"title": "1 Deadly Scaling Trap Revealed", "angle": "loss_aversion", "promise": "Identify the hidden cash burn trap"},
                {"title": "Stop Hiring: Do This Instead", "angle": "counter_intuitive", "promise": "Why smaller teams win"},
                {"title": "3 Startup Traps Costing Millions", "angle": "metric_specificity", "promise": "Calculate lost capital"}
              ]
            }"""),
            # Review and selection response
            MagicMock(content="""{
              "selected_title": "Why Fast Growth Kills Startups",
              "trigger_type": "curiosity_gap",
              "core_promise": "Unpack the cash-flow crunch behind rapid scaling",
              "review_critique": "Front-loads the counter-intuitive consequence with zero fluff under 6 words"
            }""")
        ]

        result = self.engine.generate_and_review_title(
            brief="Article on startup scaling pitfalls and cash management",
            keywords="startup growth, cash flow",
            target_audience="Founders",
        )

        self.assertEqual(result["title"], "Why Fast Growth Kills Startups")
        self.assertEqual(result["trigger_type"], "curiosity_gap")
        self.assertEqual(result["core_promise"], "Unpack the cash-flow crunch behind rapid scaling")
        self.assertTrue(result["evaluation"]["under_6_words"])
        self.assertGreaterEqual(result["evaluation"]["score"], 60)

    def test_first_paragraph_alignment_zero_bait_and_switch(self):
        title = "Why Fast Growth Kills Startups"
        promise = "Unpack cash-flow crunch behind rapid scaling"

        # Well-aligned first paragraph directly addressing growth, startup, kill, cash flow
        good_para = "<p>Fast growth kills startups faster than slow stagnation. When order volume quadruples overnight, cash flow implodes before invoices clear.</p>"
        align_good = self.engine.verify_first_paragraph_alignment(title, promise, good_para)
        self.assertTrue(align_good["aligned"])
        self.assertIsNone(align_good["warning"])

        # Unaligned throat-clearing opening (bait and switch)
        bad_para = "<p>In today's fast-paced digital world, running a business involves many multifaceted elements and strategic considerations for modern stakeholders.</p>"
        align_bad = self.engine.verify_first_paragraph_alignment(title, promise, bad_para)
        self.assertFalse(align_bad["aligned"])
        self.assertIsNotNone(align_bad["warning"])

    def test_article_quality_evaluator_integration(self):
        title = "Why Fast Growth Kills Startups"
        html_content = "<p>Fast growth kills startups when founders confuse vanity revenue with actual bank balance.</p>"
        plain_text = "Fast growth kills startups when founders confuse vanity revenue with actual bank balance."

        report = build_article_quality_report(
            title=title,
            html_content=html_content,
            plain_text=plain_text,
            citations=[{"url": "https://example.com", "title": "Study"}],
            sections=[{"title": "Intro"}],
            evidence_count=3,
        )

        self.assertIn("title_viral_score", report)
        self.assertIn("title_virality", report["diagnostics"])
        title_diag = report["diagnostics"]["title_virality"]
        self.assertTrue(title_diag["under_6_words"])
        self.assertTrue(title_diag["first_paragraph_aligned"])
        self.assertIn("curiosity_gap", title_diag["triggers"])


if __name__ == "__main__":
    unittest.main()
