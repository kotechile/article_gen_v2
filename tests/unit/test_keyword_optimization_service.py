"""
Unit tests for KeywordOptimizationService.
"""

from __future__ import annotations

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from src.services.keyword_optimization_service import (
    KeywordOptimizationService,
    calculate_opportunity_score,
    count_keyword_occurrences,
)


@pytest.fixture
def service():
    return KeywordOptimizationService()


def test_calculate_opportunity_score():
    # Low KD, High Volume -> High Score
    score_high = calculate_opportunity_score(search_volume=2000, keyword_difficulty=15.0)
    assert score_high >= 70

    # High KD, Low Volume -> Low Score
    score_low = calculate_opportunity_score(search_volume=50, keyword_difficulty=85.0)
    assert score_low <= 40

    # None handling
    score_none = calculate_opportunity_score(search_volume=None, keyword_difficulty=None)
    assert 0 <= score_none <= 100


def test_count_keyword_occurrences():
    html = "<h1>Heat Pump Rebates 2026</h1><p>Learn about heat pump rebates and electric upgrades.</p>"
    assert count_keyword_occurrences(html, "heat pump rebates") == 2
    assert count_keyword_occurrences(html, "electric upgrades") == 1
    assert count_keyword_occurrences(html, "gas furnace") == 0


def test_discover_keywords_for_article(service):
    mock_related = [
        {
            "keyword": "heat pump rebate 2026",
            "search_volume": 1200,
            "keyword_difficulty": 20.0,
            "cpc": 1.50,
            "intent": "informational",
            "competition": "LOW",
        },
        {
            "keyword": "electric home incentives",
            "search_volume": 800,
            "keyword_difficulty": 35.0,
            "cpc": 2.10,
            "intent": "commercial",
            "competition": "MEDIUM",
        }
    ]

    with patch("src.services.keyword_optimization_service.dataforseo_api.get_related_keywords_labs_live", new_callable=AsyncMock) as mock_get_related:
        mock_get_related.return_value = mock_related

        raw_res = asyncio.run(service.discover_keywords_for_article(
            title="Electric Heat Pump Guide",
            content="<p>Understanding new federal rebates for electric heat pumps.</p>",
            custom_seed="heat pump rebate",
        ))

        results = raw_res["keywords"] if isinstance(raw_res, dict) else raw_res
        assert len(results) >= 2
        top = results[0]
        assert top["keyword"] == "heat pump rebate 2026"
        assert top["search_volume"] == 1200
        assert top["keyword_difficulty"] == 20.0
        assert top["opportunity_score"] > 50


def test_weave_keywords_into_content(service):
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = """
    {
      "modified_title": "Heat Pump Rebates 2026: The Complete Electric Upgrade Guide",
      "modified_html": "<h1>Electric Guide with Heat Pump Rebates 2026</h1><p>Check your electric home incentives today.</p>",
      "changes": [
        "Updated title to include primary keyword 'Heat Pump Rebates 2026'",
        "Added 'Heat Pump Rebates 2026' to H1",
        "Added 'electric home incentives' to intro paragraph"
      ]
    }
    """
    mock_llm.generate.return_value = mock_response

    with patch.object(service, "_get_llm_client", return_value=mock_llm):
        res = asyncio.run(service.weave_keywords_into_content(
            title="Electric Guide",
            html_content="<h1>Electric Guide</h1><p>Check your home today.</p>",
            primary_keyword="heat pump rebates 2026",
            secondary_keywords=["electric home incentives"],
        ))

        assert res["success"] is True
        assert res["title"] == "Heat Pump Rebates 2026: The Complete Electric Upgrade Guide"
        assert "Heat Pump Rebates 2026" in res["html"]
        assert len(res["changes"]) == 3
        assert len(res["placements"]) == 2
