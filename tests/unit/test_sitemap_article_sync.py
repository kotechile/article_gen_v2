import os
import sys
import types
from unittest.mock import MagicMock, patch

try:
    import flask
except ImportError:
    mock_flask = types.ModuleType("flask")
    mock_flask.Blueprint = lambda *args, **kwargs: types.SimpleNamespace(
        route=lambda *a, **kw: (lambda f: f),
        before_request=lambda f: f
    )
    mock_flask.jsonify = lambda d: d
    mock_flask.request = types.SimpleNamespace(json={}, args={})
    sys.modules["flask"] = mock_flask

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.services.sitemap_article_sync_service import (
    _slug_to_title,
    _clean_page_title,
    _extract_primary_and_secondary_keywords,
    SitemapArticleSyncService,
    DEFAULT_SITEMAP_DOMAINS,
    ensure_sitemap_articles_synced,
)


def test_slug_to_title():
    assert _slug_to_title("benefits-of-living-in-the-city-2026-pros-and-cons") == "Benefits Of Living In The City 2026 Pros And Cons"
    assert _slug_to_title("2026-03-01-my-first-post") == "My First Post"
    assert _slug_to_title("") == "Untitled Article"


def test_clean_page_title():
    assert _clean_page_title("Benefits of Living in the City: 2026 Pros and Cons | WellRoost") == "Benefits of Living in the City: 2026 Pros and Cons"
    assert _clean_page_title("Your Disks: 35% Dead by Year Six | Giniloh") == "Your Disks: 35% Dead by Year Six"
    assert _clean_page_title("100-Case Eval - Aichieve") == "100-Case Eval"
    assert _clean_page_title("Standalone Title") == "Standalone Title"


def test_extract_keywords():
    primary, secondaries = _extract_primary_and_secondary_keywords(
        "benefits-of-living-in-the-city-2026-pros-and-cons",
        "Benefits of Living in the City"
    )
    assert primary != ""
    assert isinstance(secondaries, list)


def test_parse_sitemap_xml_urlset():
    xml_data = b"""<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url>
            <loc>https://wellroost.com/</loc>
            <lastmod>2026-10-17</lastmod>
        </url>
        <url>
            <loc>https://wellroost.com/benefits-of-living-in-the-city-2026-pros-and-cons/</loc>
            <lastmod>2026-10-17</lastmod>
        </url>
        <url>
            <loc>https://wellroost.com/about/</loc>
            <lastmod>2026-10-16</lastmod>
        </url>
    </urlset>"""

    service = SitemapArticleSyncService(supabase=MagicMock())
    entries = service._parse_sitemap_xml(xml_data, "wellroost.com")
    assert len(entries) == 3

    articles = service.filter_and_extract_articles(entries, "wellroost.com")
    assert len(articles) == 1
    assert articles[0]["slug"] == "benefits-of-living-in-the-city-2026-pros-and-cons"
    assert articles[0]["domain"] == "wellroost.com"


def test_build_article_records():
    candidates = [
        {
            "loc": "https://aichieve.net/agentic-ai-adoption-soars-but-profits-stall-in-2026/",
            "slug": "agentic-ai-adoption-soars-but-profits-stall-in-2026",
            "path": "/agentic-ai-adoption-soars-but-profits-stall-in-2026/",
            "lastmod": "2026-10-17",
            "domain": "aichieve.net",
        }
    ]

    service = SitemapArticleSyncService(supabase=MagicMock())
    records = service.build_article_records(candidates, user_id="user_test_99", domain="aichieve.net", max_head_fetch=0)
    assert len(records) == 1
    rec = records[0]
    assert rec["user_id"] == "user_test_99"
    assert rec["source_site"] == "aichieve.net"
    assert rec["status"] == "publish"
    assert rec["link"] == "https://aichieve.net/agentic-ai-adoption-soars-but-profits-stall-in-2026/"
    assert "agentic" in rec["slug"].lower()
    assert rec["post_id"] is not None


def test_ensure_sitemap_articles_synced_skips_when_present():
    mock_sb = MagicMock()
    mock_resp = MagicMock()
    mock_resp.count = 25
    mock_sb.table().select().eq().limit().execute.return_value = mock_resp

    res = ensure_sitemap_articles_synced(user_id="user_123", force=False, supabase=mock_sb)
    assert res is not None
    assert res.get("already_synced") is True
    assert res.get("count") == 25


def test_default_sitemap_domains():
    assert "giniloh.com" in DEFAULT_SITEMAP_DOMAINS
    assert "wellroost.com" in DEFAULT_SITEMAP_DOMAINS
    assert "aichieve.net" in DEFAULT_SITEMAP_DOMAINS
