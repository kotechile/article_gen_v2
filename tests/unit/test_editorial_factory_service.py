"""
Unit tests for EditorialFactoryService and content transformation pipeline.
"""

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
    takeaways = ["Takeaway 1: Scannability matters", "Takeaway 2: Citations build trust"]

    enriched = service.inject_key_takeaways_html(html_body, takeaways)
    assert "geo-key-takeaways" in enriched
    assert "<h2>TL;DR</h2>" in enriched
    assert "<li>Takeaway 1: Scannability matters</li>" in enriched
    assert "<li>Takeaway 2: Citations build trust</li>" in enriched


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
