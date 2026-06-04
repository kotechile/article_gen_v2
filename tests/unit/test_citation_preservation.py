import sys
from pathlib import Path

import pytest

# Ensure repo root is importable when tests run without an installed package.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("celery")
pytest.importorskip("kombu")

import tasks  # noqa: E402


def test_finalize_article_preserves_prior_citations_when_generation_returns_none():
    body_text = " ".join(["Mortgage prepayment decisions require grounded evidence."] * 120)
    prior_citations = [
        {
            "title": "Federal Reserve Mortgage Outlook",
            "url": "https://example.gov/fed-mortgage-outlook",
            "author": "Federal Reserve",
            "content": "Mortgage rates and opportunity cost should be compared carefully.",
            "source_type": "web",
        },
        {
            "title": "Consumer Finance Study",
            "url": "https://example.org/consumer-finance-study",
            "author": "Research Institute",
            "content": "Liquidity and emergency reserves affect prepayment choices.",
            "source_type": "web",
        },
    ]

    result = {
        "structure": {
            "title": "Should You Prepay Your Mortgage in 2026?",
            "meta_description": "A grounded look at mortgage prepayment trade-offs.",
            "hook": "Paying down your mortgage early can be smart, but only with evidence.",
            "excerpt": "Evidence matters when deciding between principal reduction and investing.",
            "call_to_action": "Review your liquidity and return assumptions before prepaying.",
            "keywords": ["mortgage prepayment", "opportunity cost", "interest rates"],
            "tone": "professional",
            "target_audience": "professionals",
        },
        "content": {
            "sections": [
                {
                    "title": "What the Evidence Says",
                    "content": body_text,
                }
            ],
            "word_count": len(body_text.split()),
        },
        "citations": [],
        "claim_bundles": [],
        "research_data": {
            "prior_citations": prior_citations,
            "articleLength": 1200,
            "include_in_text_citations": True,
        },
    }

    finalized = tasks._finalize_article(result)
    final_article = finalized["final_article"]

    assert len(final_article["citations"]) == 2
    assert final_article["citations"][0]["title"] == "Federal Reserve Mortgage Outlook"
    assert "References" in final_article["html_content"]


def test_polish_and_format_article_uses_llm_and_preserves_content(mocker):
    # Mock create_llm_client
    mock_client = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.content = "<p>Polished text with citation [1] and [2].</p><table><tr><td>Table</td></tr></table>"
    mock_client.generate.return_value = mock_response
    mocker.patch("tasks.create_llm_client", return_value=mock_client)

    html_in = "<p>Raw text with citation [1] and [2].</p>"
    research_data = {
        "tone": "professional",
        "writer_notes": "Use firsthand style",
        "primary_keyword": "test keyword",
    }
    structure = {"title": "Test Title"}

    out = tasks._polish_and_format_article(html_in, research_data, structure)

    # Verify generate was called with expected prompt elements
    mock_client.generate.assert_called_once()
    assert "[1]" in out
    assert "[2]" in out
    assert "<table>" in out

