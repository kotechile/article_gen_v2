from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.services.research_strategy_service import ResearchStrategyService


def test_score_trend_handles_rising_stable_declining():
    service = ResearchStrategyService()

    rising_score, rising_meta = service._score_trend(
        "heat pump roi",
        {
            "result_summary_json": {
                "top_items": [
                    {"values": [10, 12, 18, 26, 32, 38]},
                ],
            },
        },
    )
    declining_score, declining_meta = service._score_trend(
        "solar panels value",
        {
            "result_summary_json": {
                "top_items": [
                    {"values": [40, 36, 28, 20, 16, 12]},
                ],
            },
        },
    )
    stable_score, stable_meta = service._score_trend(
        "smart thermostat",
        {
            "result_summary_json": {
                "top_items": [
                    {"values": [20, 21, 19, 22, 20, 21]},
                ],
            },
        },
    )

    assert rising_score == 0.8
    assert rising_meta["direction"] == "rising"
    assert declining_score == 0.2
    assert declining_meta["direction"] == "declining"
    assert stable_score == 0.5
    assert stable_meta["direction"] == "stable"


def test_serp_articleability_gate_rejects_tool_dominant_serp():
    service = ResearchStrategyService()
    rows = [
        {"title": "Mortgage Calculator Tool", "url": "https://example.com/calculator", "domain": "example.com"},
        {"title": "Heat Pump ROI Calculator", "url": "https://tools.example.com/roi-calculator", "domain": "tools.example.com"},
        {"title": "Window Value Tool", "url": "https://example.com/tool/window-value", "domain": "example.com"},
        {"title": "Estimate Your Upgrade Value", "url": "https://example.com/tool/estimate", "domain": "example.com"},
        {"title": "Forum discussion", "url": "https://reddit.com/r/home", "domain": "reddit.com"},
    ]

    result = service._classify_serp(
        query_text="heat pump roi",
        rows=rows,
        article_format="decision_guide",
        route_hint="article",
    )

    assert result["articleability_passed"] is False
    assert result["classification"] == "tool_dominant"
    assert "tool_dominant_serp" in result["reason_codes"]
    assert result["route_hint"] == "software"


def test_extract_article_competitor_urls_filters_non_article_pages():
    service = ResearchStrategyService()
    rows = [
        {"title": "Do New Windows Increase Home Value?", "url": "https://site.com/blog/new-windows-home-value", "domain": "site.com", "rank_group": 1},
        {"title": "Window replacement service near me", "url": "https://site.com/services/window-replacement", "domain": "site.com", "rank_group": 2},
        {"title": "Home Value Calculator", "url": "https://site.com/calculator/home-value", "domain": "site.com", "rank_group": 3},
    ]

    filtered = service._extract_article_competitor_urls(rows)

    assert len(filtered) == 1
    assert filtered[0]["url"] == "https://site.com/blog/new-windows-home-value"


def test_cluster_keywords_uses_median_kd_and_competitor_overlap():
    service = ResearchStrategyService()
    rows = [
        {
            "keyword": "do new windows increase home value",
            "source_url": "https://a.com/post-1",
            "keyword_difficulty": 20,
            "cpc": 4.0,
            "competition_index": 0.7,
            "rank_group": 4,
        },
        {
            "keyword": "new windows increase resale value",
            "source_url": "https://b.com/post-2",
            "keyword_difficulty": 50,
            "cpc": 3.0,
            "competition_index": 0.5,
            "rank_group": 7,
        },
        {
            "keyword": "window replacement roi",
            "source_url": "https://b.com/post-2",
            "keyword_difficulty": 80,
            "cpc": 5.0,
            "competition_index": 0.8,
            "rank_group": 9,
        },
    ]

    clusters = service._cluster_keywords(rows)

    assert clusters
    first = clusters[0]
    assert first["supporting_urls"]
    assert 0.0 < first["competitor_support_score"] <= 1.0
    assert first["kd_median_score"] < 1.0


def test_qualify_competitor_keywords_prefers_relevant_competitive_terms():
    service = ResearchStrategyService()
    rows = [
        {
            "keyword": "ai second brain app",
            "search_volume": 500,
            "rank_group": 8,
            "keyword_difficulty": 32,
            "intent": "informational",
        },
        {
            "keyword": "best pizza dough recipe",
            "search_volume": 3000,
            "rank_group": 3,
            "keyword_difficulty": 20,
            "intent": "informational",
        },
        {
            "keyword": "second brain workflow ai",
            "search_volume": 120,
            "rank_group": 12,
            "keyword_difficulty": 41,
            "intent": "informational",
        },
    ]

    qualified = service._qualify_competitor_keywords(
        topic_text="Intelligent Knowledge Management Second Brain AI",
        bet_text="AI second brain workflows",
        rows=rows,
    )

    keywords = [row["keyword"] for row in qualified]
    assert "ai second brain app" in keywords
    assert "second brain workflow ai" in keywords
    assert "best pizza dough recipe" not in keywords


def test_qualify_competitor_keywords_can_keep_high_quality_terms_without_seed_overlap():
    service = ResearchStrategyService()
    rows = [
        {
            "keyword": "best ai meeting assistants",
            "search_volume": 1900,
            "rank_group": 6,
            "keyword_difficulty": 38,
            "intent": "commercial",
            "source_title": "9 Best AI Meeting Assistants in 2026",
            "source_url": "https://www.read.ai/resources/best-ai-meeting-assistants-2026",
        },
        {
            "keyword": "microsoft copilot pricing",
            "search_volume": 90,
            "rank_group": 30,
            "keyword_difficulty": 77,
            "intent": "commercial",
            "source_title": "Microsoft Copilot pricing",
            "source_url": "https://example.com/copilot-pricing",
        },
    ]

    qualified = service._qualify_competitor_keywords(
        topic_text="Fireflies and Granola vs ms copilot to Execute Your Meeting Action Items",
        bet_text="Fireflies and Granola vs ms copilot",
        rows=rows,
    )

    keywords = [row["keyword"] for row in qualified]
    assert "best ai meeting assistants" in keywords
    assert "microsoft copilot pricing" not in keywords


def test_mixed_serp_can_still_survive_as_article_candidate():
    service = ResearchStrategyService()
    rows = [
        {"title": "Best AI second brain apps", "url": "https://site.com/blog/best-ai-second-brain-apps", "domain": "site.com"},
        {"title": "How to build a second brain with AI", "url": "https://site.com/guides/ai-second-brain", "domain": "site.com"},
        {"title": "Second brain comparison", "url": "https://medium.com/second-brain-comparison", "domain": "medium.com"},
        {"title": "Community discussion", "url": "https://reddit.com/r/productivity", "domain": "reddit.com"},
        {"title": "Product page", "url": "https://app.example.com/product", "domain": "app.example.com"},
    ]

    result = service._classify_serp(
        query_text="ai second brain app",
        rows=rows,
        article_format="tool_evaluation",
        route_hint="article",
    )

    assert result["classification"] in {"mixed", "article_friendly"}
    assert result["articleability_score"] >= 0.4


def test_comparison_serp_survives_when_article_results_are_present():
    service = ResearchStrategyService()
    rows = [
        {
            "title": "Granola vs Microsoft Copilot for Meetings",
            "url": "https://zackproser.com/blog/granola-vs-microsoft-copilot-meetings",
            "domain": "zackproser.com",
        },
        {
            "title": "9 Best AI Meeting Assistants in 2026",
            "url": "https://www.read.ai/resources/best-ai-meeting-assistants-2026",
            "domain": "read.ai",
        },
        {
            "title": "Top 5 AI note takers for teams: Granola, Fireflies, and more",
            "url": "https://www.linkedin.com/posts/example-top-5-ai-note-takers",
            "domain": "linkedin.com",
        },
        {
            "title": "7 Best AI Meeting Notetakers 2026: Fathom vs Fireflies vs Granola",
            "url": "https://get-alfred.ai/blog/best-ai-meeting-notetakers-2026",
            "domain": "get-alfred.ai",
        },
    ]

    result = service._classify_serp(
        query_text="Fireflies and Granola vs ms copilot to Execute Your Meeting Action Items",
        rows=rows,
        article_format="tool_evaluation",
        route_hint="article",
    )

    assert result["articleability_passed"] is True
    assert result["classification"] == "article_friendly"
    assert "article_friendly_serp" in result["reason_codes"]


def test_select_attractive_competitor_targets_prefers_repeat_domains_and_excludes_large_sites():
    service = ResearchStrategyService()
    rows = [
        {"url": "https://smallsite.com/guide/crewai-vs-autogpt", "title": "CrewAI vs AutoGPT", "domain": "smallsite.com", "rank_group": 3, "probe_query_id": "a"},
        {"url": "https://smallsite.com/blog/autogpt-business-automation", "title": "AutoGPT for business automation", "domain": "smallsite.com", "rank_group": 6, "probe_query_id": "b"},
        {"url": "https://tomshardware.com/review/crewai", "title": "CrewAI review", "domain": "tomshardware.com", "rank_group": 2, "probe_query_id": "a"},
    ]

    selected = service._select_attractive_competitor_targets(rows)

    assert selected
    assert selected[0]["domain"] == "smallsite.com"
    assert selected[0]["analysis_target"] == "smallsite.com"
    assert selected[0]["domain_hits"] == 2
    assert all(item["domain"] != "tomshardware.com" for item in selected)


def test_materialize_keyword_opportunities_allows_single_keyword_article_path():
    service = ResearchStrategyService()
    opportunities = service._materialize_keyword_opportunities(
        rows=[
            {
                "keyword": "crewai for business automation",
                "search_volume": 90,
                "keyword_difficulty": 31,
                "cpc": 3.2,
                "competition_index": 0.4,
                "rank_group": 8,
                "relevance_score": 0.71,
                "qualification_score": 0.69,
                "seed_overlap": 2,
                "source_domain": "smallsite.com",
                "source_url": "https://smallsite.com/guide/crewai-vs-autogpt",
                "source_title": "CrewAI vs AutoGPT",
                "intent": "commercial",
            }
        ],
        bet={
            "serp_weakness_score": 0.58,
            "trend_score": 0.5,
            "article_fit_score": 0.64,
            "serp_articleability_score": 0.74,
        },
    )

    assert len(opportunities) == 1
    first = opportunities[0]
    assert first["cluster_type"] == "keyword_opportunity"
    assert first["primary_keyword_candidate"] == "crewai for business automation"
    assert first["supporting_urls"] == ["https://smallsite.com/guide/crewai-vs-autogpt"]
    assert first["cluster_metadata"]["source_domain"] == "smallsite.com"
