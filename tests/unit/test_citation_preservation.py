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
