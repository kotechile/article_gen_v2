import sys
from pathlib import Path
from types import SimpleNamespace
import types

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_task_import_stubs():
    class DummyCeleryApp:
        def task(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

    sys.modules.setdefault("celery", SimpleNamespace(current_task=None))
    sys.modules.setdefault("celery_config", SimpleNamespace(celery=DummyCeleryApp()))
    sys.modules.setdefault(
        "supabase_client",
        SimpleNamespace(
            LLM_ROLE_ARTICLE_GENERATION="article_generation",
            LLM_ROLE_FINAL_REVIEW="final_review",
            get_supabase_client=lambda: None,
            get_llm_api_key=lambda *args, **kwargs: "",
            get_linkup_api_key=lambda: "",
            get_default_llm_provider=lambda: ("openai", "gpt-4", ""),
            get_llm_provider_for_role=lambda role: ("openai", "gpt-4", ""),
        ),
    )
    sys.modules.setdefault("llm_client", SimpleNamespace(create_llm_client=lambda **kwargs: None))
    sys.modules.setdefault("rag_client", SimpleNamespace(create_rag_client=lambda **kwargs: None, RAGQuery=dict))
    sys.modules.setdefault("linkup_client", SimpleNamespace(create_linkup_client=lambda **kwargs: None, SearchQuery=dict))
    sys.modules.setdefault("article_structure_generator", SimpleNamespace(create_article_structure_generator=lambda *args, **kwargs: None))
    sys.modules.setdefault(
        "content_generator",
        SimpleNamespace(
            create_content_generator=lambda *args, **kwargs: None,
            get_tone_specific_instructions=lambda tone: str(tone or ""),
        ),
    )
    sys.modules.setdefault(
        "citation_generator",
        SimpleNamespace(create_citation_generator=lambda *args, **kwargs: None, CitationStyle=types.SimpleNamespace()),
    )
    sys.modules.setdefault("src.utils.config", SimpleNamespace(get_config=lambda: {}))


_install_task_import_stubs()

import pytest
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


def test_polish_and_format_article_uses_llm_and_preserves_content(monkeypatch):
    from unittest.mock import MagicMock
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "<p>Polished text with citation [1] and [2].</p><table><tr><td>Table</td></tr></table>"
    mock_client.generate.return_value = mock_response
    monkeypatch.setattr(tasks, "create_llm_client", lambda **kwargs: mock_client)

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

