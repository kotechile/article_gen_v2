"""
Unit tests for target audience calibration and smart brevity directives.
"""

import pytest
from src.core.models.research import ResearchRequest
from src.api.schemas.research import ResearchRequestSchema
from article_structure_generator import ArticleStructureGenerator
from content_generator import ContentGenerator


def test_research_request_target_audience_model():
    """Verify target_audience field is accepted in ResearchRequest."""
    req = ResearchRequest(
        brief="Test brief for mid-career professionals looking to move abroad",
        keywords="expat taxes, international schooling",
        target_audience="Mid-career professionals (aged 35–45)"
    )
    assert req.target_audience == "Mid-career professionals (aged 35–45)"


def test_research_request_schema_target_audience():
    """Verify target_audience is accepted in ResearchRequestSchema."""
    schema = ResearchRequestSchema(
        brief="Valid brief for testing schema target audience",
        keywords="remote work, visas",
        target_audience="Remote software engineers"
    )
    assert schema.target_audience == "Remote software engineers"


def test_article_structure_generator_target_audience_precedence():
    """Verify ArticleStructureGenerator uses explicitly supplied target_audience."""
    generator = ArticleStructureGenerator(llm_client=None)
    research_data = {
        "target_audience": "Mid-career professionals (aged 35–45)",
        "brief": "General discussion about starting an online business",
        "tone": "journalistic",
    }
    audience = generator._determine_target_audience(
        brief=research_data["brief"],
        tone=research_data["tone"],
        research_data=research_data
    )
    assert audience == "Mid-career professionals (aged 35–45)"


def test_article_structure_generator_target_audience_fallback():
    """Verify ArticleStructureGenerator falls back intelligently when no audience is provided."""
    generator = ArticleStructureGenerator(llm_client=None)
    audience = generator._determine_target_audience(
        brief="Advanced developer guide to Kubernetes deployments",
        tone="technical",
        research_data={}
    )
    assert audience == "Experts and technical professionals"


def test_content_generator_demographic_instructions_35_45_bracket():
    """Verify ContentGenerator injects mid-career 35-45 wealth & logistics calibration."""
    generator = ContentGenerator(llm_client=None)
    context = {
        "target_audience": "Mid-career professionals (aged 35–45)",
        "tone": "journalistic"
    }
    instructions = generator._get_demographic_and_smart_brevity_instructions(context)
    assert "Wealth over Salary" in instructions
    assert "Family Logistics" in instructions
    assert "Financial Variables" in instructions
    assert "Lifestyle ROI" in instructions
    assert "SMART BREVITY FORMATTING RULES" in instructions
    assert "**The big picture:**" in instructions


def test_content_generator_demographic_instructions_custom_audience():
    """Verify ContentGenerator adapts to any arbitrary audience."""
    generator = ContentGenerator(llm_client=None)
    context = {
        "target_audience": "First-time real estate investors",
        "tone": "journalistic"
    }
    instructions = generator._get_demographic_and_smart_brevity_instructions(context)
    assert "First-time real estate investors" in instructions
    assert "SMART BREVITY FORMATTING RULES" in instructions


def test_finalize_article_smart_brevity_structure():
    """Verify that finalization invokes polishing agent and does not prepend crude GEO boilerplate."""
    import sys
    from unittest.mock import MagicMock, patch

    if "celery" not in sys.modules:
        mock_celery_mod = MagicMock()
        mock_celery_app = MagicMock()
        mock_celery_app.task = lambda *args, **kwargs: (lambda f: f)
        mock_celery_mod.Celery = MagicMock(return_value=mock_celery_app)
        mock_celery_mod.current_task = MagicMock()
        sys.modules["celery"] = mock_celery_mod

    if "celery_config" not in sys.modules:
        mock_cfg = MagicMock()
        mock_cfg.celery = MagicMock()
        mock_cfg.celery.task = lambda *args, **kwargs: (lambda f: f)
        sys.modules["celery_config"] = mock_cfg

    if "litellm" not in sys.modules:
        mock_litellm = MagicMock()
        mock_litellm.exceptions = MagicMock()
        sys.modules["litellm"] = mock_litellm
        sys.modules["litellm.exceptions"] = mock_litellm.exceptions

    import tasks

    smart_brevity_html = (
        "<p>A $200K U.S. tech salary often shrinks below €140K in Europe, yet the savviest professionals still make the jump.</p>\n"
        "<p><strong>The big picture:</strong> Deciding to relocate for a job requires looking past the gross salary offer and running the dual-country math to find your true net cash flow.</p>\n"
        "<p><strong>Why it matters:</strong> Hidden variables like partner visa restrictions and international school tuitions can turn a lucrative offer into a financial liability.</p>\n"
        "<p><strong>By the numbers:</strong></p>\n"
        "<table><tr><th>Expense</th><th>US</th><th>EU</th></tr><tr><td>Net Cash Flow</td><td>$110k</td><td>€85k</td></tr></table>\n"
        "<p><strong>The reality check:</strong> Quality of life gains often outweigh purely monetary discrepancies.</p>\n"
        "<p><strong>Go deeper:</strong></p>\n"
        "<ul><li><strong>Tax optimization:</strong> Review bilateral treaties beforehand.</li></ul>"
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = smart_brevity_html
    mock_client.generate.return_value = mock_response

    raw_body = " ".join(["Relocation analysis details and data points for international career growth."] * 100)
    result = {
        "structure": {
            "title": "Relocating Overseas: The True Value of a Job-Driven Move",
            "meta_description": "Compare net cash flow, visa rules, and family logistics for expat moves.",
            "hook": "A $200K U.S. tech salary shrinks below €140K in Europe.",
            "excerpt": "Relocating requires dual-country math.",
            "target_audience": "Mid-career professionals (aged 35–45)",
            "tone": "analytical",
            "keywords": ["relocating overseas", "expat tax", "international school tuition"],
        },
        "content": {
            "sections": [
                {
                    "title": "Introduction",
                    "content": raw_body,
                }
            ],
            "word_count": len(raw_body.split()),
        },
        "citations": [
            {
                "title": "Expat Financial Review",
                "url": "https://example.com/expat-finance",
                "author": "Global Mobility Inst",
                "publication_date": "2026",
            }
        ],
        "research_data": {
            "target_audience": "Mid-career professionals (aged 35–45)",
            "articleLength": 800,
            "include_in_text_citations": True,
        }
    }

    with patch("tasks.create_llm_client", return_value=mock_client):
        finalized = tasks._finalize_article(result)
        html = finalized["final_article"]["html_content"]

    # Assert no crude wrapper headers leaked into output
    assert "<h2>Short Answer</h2>" not in html
    assert "<h2>At a glance</h2>" not in html
    assert "<h2>Introduction</h2>" not in html

    # Assert Smart Brevity axiom components are front and center
    assert "<strong>The big picture:</strong>" in html
    assert "<strong>Why it matters:</strong>" in html
    assert "<strong>By the numbers:</strong>" in html
    assert "<table>" in html
    assert "<strong>The reality check:</strong>" in html
    assert "<strong>Go deeper:</strong>" in html
    assert "<h2>References</h2>" in html

