"""
Unit tests for EditorialFactoryService and content transformation pipeline.
"""

import re
import pytest
from unittest.mock import MagicMock, patch
from src.services.editorial_factory_service import EditorialFactoryService


@pytest.fixture
def service():
    return EditorialFactoryService()


def test_markdown_to_html_headings_and_paragraphs(service):
    markdown = """# Main Headline
## Subheading Level 2
### Section Header

This is a paragraph with **bold text** and *italic words* and `inline code`.
"""
    html = service.markdown_to_html(markdown)
    assert "<h1>Main Headline</h1>" in html
    assert "<h2>Subheading Level 2</h2>" in html
    assert "<h3>Section Header</h3>" in html
    assert "<strong>bold text</strong>" in html
    assert "<em>italic words</em>" in html
    assert "<code>inline code</code>" in html
    assert "<p>" in html


def test_markdown_to_html_lists_and_tables(service):
    markdown = """
- Bullet item 1
- Bullet item 2

1. Numbered item 1
2. Numbered item 2

| Feature | Description | Status |
| --- | --- | --- |
| GEO | Generative Engine Optimization | Active |
| SEO | Search Engine Optimization | Ready |
"""
    html = service.markdown_to_html(markdown)
    assert "<ul" in html
    assert "<li>Bullet item 1</li>" in html
    assert "<ol" in html
    assert "<li>Numbered item 1</li>" in html
    assert "<table" in html
    assert "Feature</th>" in html
    assert "GEO</td>" in html


def test_extract_citations_from_text(service):
    text = """
The study showed significant results [1] https://example.com/study-1.
Another finding was noted in [^2]: [Research Paper](https://university.edu/paper).

## References
[1] https://example.com/study-1
[2] Research Paper: https://university.edu/paper
"""
    citations = service.extract_citations_from_text(text)
    assert len(citations) >= 2
    urls = [c["url"] for c in citations]
    assert "https://example.com/study-1" in urls
    assert "https://university.edu/paper" in urls


def test_extract_citations_various_formats(service):
    # Test Markdown list format
    markdown_list = """
Main article body with stats.

## References:
1. [Department of Energy](https://energy.gov/rebates)
2. [Rewiring America](https://rewiringamerica.org/calculator)
- [IRS Home Credit](https://irs.gov/credit)
"""
    citations = service.extract_citations({"content": markdown_list})
    assert len(citations) == 3
    assert any(c["url"] == "https://energy.gov/rebates" for c in citations)
    assert any(c["url"] == "https://rewiringamerica.org/calculator" for c in citations)
    assert any(c["url"] == "https://irs.gov/credit" for c in citations)

    # Test Database JSON field
    article_with_json_citations = {
        "content": "Article without bottom references",
        "raw_data": {
            "citations": [
                {"title": "DOE Report", "url": "https://energy.gov/report", "author": "DOE", "publication_date": "2026"},
                {"title": "EPA Clean Energy", "url": "https://epa.gov/clean", "author": "EPA"}
            ]
        }
    }
    citations_json = service.extract_citations(article_with_json_citations)
    assert len(citations_json) == 2
    assert citations_json[0]["title"] == "DOE Report"
    assert citations_json[0]["url"] == "https://energy.gov/report"


def test_synthesize_metadata(service):
    article = {
        "title": "Future of AI in Content [1]",
        "content": """
[1] . Artificial intelligence is radically transforming modern digital journalism [1][3].
Publishers who adopt generative search optimization early will capture emerging AI answer traffic [2].
Key principles include:
- Maintain high citation density across all claims [1]
- Structure direct answers for LLM ingestion [2]
- Build interactive utilities for repeated workflows
""",
        "tags": ["AI", "Content Strategy", "GEO"]
    }
    meta = service.synthesize_metadata(article)
    assert "[1]" not in meta["hook"]
    assert "[3]" not in meta["hook"]
    assert "Artificial intelligence is radically transforming" in meta["hook"]
    assert "[2]" not in meta["thesis"]
    assert "Publishers who adopt generative search optimization" in meta["thesis"]
    assert len(meta["takeaways"]) >= 2
    for t in meta["takeaways"]:
        assert "[" not in t
    assert meta["primary_keyword"] == "AI"
    assert "GEO" in meta["secondary_keywords"]


def test_inject_key_takeaways_html(service):
    html_body = "<h1>Article Title</h1>\n<p>First paragraph intro.</p>"
    takeaways = [
        "• **Built after 2010:** Modern codes require higher seismic standards.",
        "Takeaway 2: Citations build trust"
    ]

    enriched = service.inject_key_takeaways_html(html_body, takeaways)
    assert "geo-key-takeaways" in enriched
    assert "<h2>At a glance</h2>" in enriched
    assert "<li><strong>Built after 2010:</strong> Modern codes require higher seismic standards.</li>" in enriched
    assert "<li>Takeaway 2: Citations build trust</li>" in enriched
    assert "**" not in enriched
    assert "•" not in enriched


def test_import_article_to_titles(service):
    mock_article = {
        "id": "ef-123",
        "title": "Imported Editorial Strategy",
        "content": "## Core Thesis\nEditorial excellence demands structured citations and direct answers.",
        "summary": "A deep dive into modern editorial standards.",
        "tags": ["Strategy", "Publishing"],
        "author": "Chief Editor",
        "created_at": "2026-09-01T12:00:00Z",
    }

    mock_local_supabase = MagicMock()
    mock_insert_builder = MagicMock()
    mock_insert_builder.execute.return_value = MagicMock(data=[{"id": "new-title-uuid", "Title": "Imported Editorial Strategy"}])
    mock_local_supabase.table.return_value.insert.return_value = mock_insert_builder

    with patch.object(service, "get_article", return_value=mock_article), \
         patch("src.services.editorial_factory_service.get_supabase_client", return_value=mock_local_supabase):

        success, new_id, res = service.import_article_to_titles(
            article_id="ef-123",
            user_id="user-456",
            target_domain="buildomain.com"
        )

        assert success is True
        assert new_id == "new-title-uuid"
        mock_local_supabase.table.assert_called_with("Titles")


def test_list_articles_with_imported_flags(service):
    mock_ef_articles = [
        {
            "id": "ef-101",
            "title": "Article One: Emerging Tech",
            "content": "Body text for article one",
            "summary": "Summary one",
            "tags": ["Tech"],
            "created_at": "2026-09-10T10:00:00Z",
            "author": "Author A"
        },
        {
            "id": "ef-102",
            "title": "Article Two: Clean Energy",
            "content": "Body text for article two",
            "summary": "Summary two",
            "tags": ["Energy"],
            "created_at": "2026-09-11T10:00:00Z",
            "author": "Author B"
        },
        {
            "id": "ef-103",
            "title": "Article Three: Title Match Only",
            "content": "Body text for article three",
            "summary": "Summary three",
            "tags": ["Finance"],
            "created_at": "2026-09-12T10:00:00Z",
            "author": "Author C"
        }
    ]

    mock_client = MagicMock()
    mock_query = MagicMock()
    mock_query.order.return_value.range.return_value.execute.return_value = MagicMock(data=mock_ef_articles)
    mock_client.table.return_value.select.return_value = mock_query

    # Mock local Titles returning ef-101 (by editorial_factory_id) and ef-103 (by Title)
    mock_local_supabase = MagicMock()
    mock_titles_query = MagicMock()
    mock_titles_query.execute.return_value = MagicMock(data=[
        {
            "id": "title-uuid-101",
            "Title": "Article One: Emerging Tech",
            "idea_metadata": {
                "source": "editorial-factory",
                "editorial_factory_id": "ef-101",
                "imported_at": "2026-09-12T15:00:00Z"
            },
            "dateCreatedOn": "2026-09-12T15:00:00Z",
            "domain": "giniloh.com",
            "user_id": "user-123"
        },
        {
            "id": "title-uuid-103",
            "Title": "Article Three: Title Match Only",
            "idea_metadata": None,
            "dateCreatedOn": "2026-09-13T10:00:00Z",
            "domain": "giniloh.com",
            "user_id": "user-123"
        }
    ])
    # Handle chain for eq
    mock_titles_query.eq.return_value = mock_titles_query
    mock_local_supabase.table.return_value.select.return_value = mock_titles_query

    with patch.object(service, "get_client", return_value=mock_client), \
         patch("src.services.editorial_factory_service.get_supabase_client", return_value=mock_local_supabase):

        results = service.list_articles(search="", limit=10, user_id="user-123", domain="giniloh.com")

        assert len(results) == 3

        # ef-101 should be marked imported via editorial_factory_id
        assert results[0]["id"] == "ef-101"
        assert results[0]["is_imported"] is True
        assert results[0]["imported_title_id"] == "title-uuid-101"
        assert results[0]["imported_at"] == "2026-09-12T15:00:00Z"

        # ef-102 should NOT be imported
        assert results[1]["id"] == "ef-102"
        assert results[1]["is_imported"] is False
        assert results[1]["imported_title_id"] is None

        # ef-103 should be marked imported via Title match
        assert results[2]["id"] == "ef-103"
        assert results[2]["is_imported"] is True
        assert results[2]["imported_title_id"] == "title-uuid-103"


def test_import_article_strips_duplicate_takeaways_at_end(service):
    raw_markdown = """# The four patterns that actually won Google's AI Agents Challenge

On September 2, Google shared a review of its Artificial Intelligence (AI) Agents Challenge. The contest drew thousands of builders.

The deeper limit is that these patterns stack, but they do not scale on their own. The system choice happens long before you write the code.

Most "multi-agent" systems are just one model chaining prompts with agent names attached; the Google Artificial Intelligence Agents Challenge winners stood out with four predictable patterns, not bigger models.

The four moves: expose your agent's own tools as a Model Context Protocol server, replace call chains with an event bus, force every fallback through one checking function, and route cheap checks before the model.

Exposing reasoning to outside callers demands real access control, and the 40 percent routing figure is one team's own count rather than a universal standard.
"""

    mock_article = {
        "id": "ef-multi-agent",
        "title": "The four patterns that actually won Google's AI Agents Challenge",
        "content": raw_markdown,
        "summary": "Google AI Agents Challenge breakdown",
        "tags": ["AI", "Multi-Agent"],
        "author": "Editorial Factory",
        "created_at": "2026-09-12T12:00:00Z",
    }

    mock_local_supabase = MagicMock()
    mock_insert_builder = MagicMock()
    mock_insert_builder.execute.return_value = MagicMock(data=[{"id": "title-multi-agent-uuid"}])
    mock_local_supabase.table.return_value.insert.return_value = mock_insert_builder

    with patch.object(service, "get_article", return_value=mock_article), \
         patch("src.services.editorial_factory_service.get_supabase_client", return_value=mock_local_supabase):

        success, new_id, _ = service.import_article_to_titles(
            article_id="ef-multi-agent",
            user_id="user-123",
            target_domain="giniloh.com"
        )

        assert success is True
        # Inspect the payload sent to Supabase
        call_args = mock_local_supabase.table.return_value.insert.call_args[0][0]
        html_article = call_args["htmlArticle"]

        # Verify "At a glance" section is injected at the top
        assert '<section class="geo-key-takeaways" data-geo-injected="key-takeaways">' in html_article
        assert '<h2>At a glance</h2>' in html_article

        # Verify the end of the article ends with the real last paragraph, NOT the duplicated takeaway paragraphs
        assert html_article.endswith('<p>The deeper limit is that these patterns stack, but they do not scale on their own. The system choice happens long before you write the code.</p>')
        # The body should not have the takeaways repeated outside the geo-key-takeaways section
        body_without_takeaways_section = re.sub(r'<section class="geo-key-takeaways"[\s\S]*?</section>', '', html_article)
        assert 'Most &quot;multi-agent&quot; systems are just one model chaining prompts' not in body_without_takeaways_section
        assert 'The four moves: expose your agent&#39;s own tools' not in body_without_takeaways_section


def test_import_article_generates_and_stores_excerpt(service):
    article_with_explicit_excerpt = {
        "id": "ef-excerpt-1",
        "title": "Enterprise AI ROI",
        "content": "Adoption of AI agents reached 40 percent in 2026. However EBIT impact remains flat for most organizations.",
        "excerpt": "A deep dive into why enterprise AI agent adoption does not automatically convert to EBIT impact.",
        "summary": "AI agents challenge and ROI breakdown",
    }

    meta = service.synthesize_metadata(article_with_explicit_excerpt)
    assert meta["excerpt"] == "A deep dive into why enterprise AI agent adoption does not automatically convert to EBIT impact."

    # Test synthesized fallback when excerpt is missing
    article_without_excerpt = {
        "id": "ef-excerpt-2",
        "title": "Build vs Buy Software in 2026",
        "content": "Thirty-two percent of enterprises now build internal software tools using automated coding assistants. The traditional procurement cycle has shifted toward fast in-house prototyping.",
    }

    meta_synth = service.synthesize_metadata(article_without_excerpt)
    assert meta_synth["excerpt"] != ""
    assert "Thirty-two percent of enterprises" in meta_synth["excerpt"]

    # Test database payload on import
    mock_local_supabase = MagicMock()
    mock_insert_builder = MagicMock()
    mock_insert_builder.execute.return_value = MagicMock(data=[{"id": "title-excerpt-uuid"}])
    mock_local_supabase.table.return_value.insert.return_value = mock_insert_builder

    with patch.object(service, "get_article", return_value=article_without_excerpt), \
         patch("src.services.editorial_factory_service.get_supabase_client", return_value=mock_local_supabase):

        success, new_id, _ = service.import_article_to_titles(
            article_id="ef-excerpt-2",
            user_id="user-456",
            target_domain="giniloh.com"
        )

        assert success is True
        call_args = mock_local_supabase.table.return_value.insert.call_args[0][0]
        assert call_args["excerpt"] == meta_synth["excerpt"]
        assert call_args["wp_excerpt_auto_generated"] == meta_synth["excerpt"]
        assert call_args["idea_metadata"]["excerpt"] == meta_synth["excerpt"]
