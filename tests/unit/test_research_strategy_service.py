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
