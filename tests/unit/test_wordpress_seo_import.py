import os
import sys
import types

# Provide lightweight mock for flask/supabase if not in current environment
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

from src.api.wordpress import (
    extract_wordpress_seo_metadata,
    _build_titles_payload_from_imported_post,
    _clean_html_to_plain_text,
    _insert_with_schema_fallback,
)


def test_clean_html_text():
    assert _clean_html_to_plain_text("<p>Hello <strong>world</strong>!</p>") == "Hello world!"
    assert _clean_html_to_plain_text("&amp; &#8217; &#8220; &#8221;") == "& ’ “ ”"
    assert _clean_html_to_plain_text(None) == ""


def test_extract_wordpress_seo_metadata_yoast():
    raw_post = {
        "id": 101,
        "slug": "best-running-shoes-2026",
        "title": {"rendered": "Best Running Shoes for Marathon Runners &#8211; 2026 Guide"},
        "excerpt": {"rendered": "<p>A quick summary of running shoes.</p>"},
        "content": {"rendered": "<p>Full content of the running shoes review...</p>"},
        "date": "2026-03-01T10:00:00",
        "modified": "2026-03-05T14:30:00",
        "link": "https://example.com/best-running-shoes-2026/",
        "categories": [12, 15],
        "tags": [45, 88],
        "_embedded": {
            "wp:term": [
                [
                    {"id": 12, "name": "Footwear", "taxonomy": "category"},
                    {"id": 15, "name": "Running Gear", "taxonomy": "category"},
                ],
                [
                    {"id": 45, "name": "Marathon", "taxonomy": "post_tag"},
                    {"id": 88, "name": "Shoe Reviews", "taxonomy": "post_tag"},
                ],
            ],
            "wp:featuredmedia": [
                {
                    "id": 999,
                    "source_url": "https://example.com/wp-content/uploads/2026/03/shoes.jpg",
                    "alt_text": "Top marathon running shoes on a track",
                    "title": {"rendered": "Marathon Shoes Overview"},
                    "caption": {"rendered": "<p>Photo of marathon shoes</p>"},
                }
            ],
        },
        "yoast_head_json": {
            "title": "Best Running Shoes 2026 | Top Marathon Gear",
            "description": "Comprehensive review of the best marathon running shoes for 2026.",
            "canonical": "https://example.com/best-running-shoes-2026/",
            "og_title": "Top Running Shoes 2026",
            "og_description": "Our editors picked the top running shoes.",
            "og_image": [{"url": "https://example.com/wp-content/uploads/2026/03/shoes-og.jpg"}],
            "twitter_title": "Best Running Shoes 2026",
            "twitter_description": "Find your next pair of shoes.",
            "twitter_image": "https://example.com/wp-content/uploads/2026/03/shoes-og.jpg",
            "twitter_card": "summary_large_image",
            "robots": {"index": "index", "follow": "follow"},
            "schema": {"@context": "https://schema.org", "@graph": []},
        },
        "meta": {
            "_yoast_wpseo_focuskw": "best running shoes",
            "_yoast_wpseo_focuskeywords": '[{"keyword":"marathon running shoes"},{"keyword":"best marathon shoes 2026"}]',
            "_yoast_wpseo_linkdex": "85",
            "_yoast_wpseo_content_score": "78",
        },
    }

    extracted = extract_wordpress_seo_metadata(raw_post, "example.com")

    assert extracted["title"] == "Best Running Shoes for Marathon Runners – 2026 Guide"
    assert extracted["slug"] == "best-running-shoes-2026"
    assert extracted["seo_title"] == "Best Running Shoes 2026 | Top Marathon Gear"
    assert extracted["seo_description"] == "Comprehensive review of the best marathon running shoes for 2026."
    assert extracted["focus_keyword"] == "best running shoes"
    assert extracted["primary_keyword"] == "best running shoes"
    assert "marathon running shoes" in extracted["secondary_keywords"]
    assert "best marathon shoes 2026" in extracted["secondary_keywords"]
    assert extracted["category_names"] == ["Footwear", "Running Gear"]
    assert extracted["tag_names"] == ["Marathon", "Shoe Reviews"]
    assert extracted["featured_image_url"] == "https://example.com/wp-content/uploads/2026/03/shoes.jpg"
    assert extracted["featured_image_alt"] == "Top marathon running shoes on a track"
    assert extracted["seo_optimization_score"] >= 85
    assert extracted["readability_score"] >= 70
    assert extracted["canonical_url"] == "https://example.com/best-running-shoes-2026/"
    assert extracted["seo_metadata"]["metaTitle"] == "Best Running Shoes 2026 | Top Marathon Gear"


def test_extract_wordpress_seo_metadata_rankmath():
    raw_post = {
        "id": 202,
        "slug": "how-to-train-for-10k",
        "title": {"rendered": "How to Train for a 10K Race"},
        "excerpt": {"rendered": "<p>10K training plan summary.</p>"},
        "content": {"rendered": "<p>Complete 8-week training plan...</p>"},
        "date": "2026-02-15T08:00:00",
        "link": "https://runnersworld.example/how-to-train-for-10k",
        "rank_math_seo": {
            "title": "How to Train for a 10K in 8 Weeks | Expert Guide",
            "description": "Follow our beginner to intermediate 10k training plan.",
            "focus_keyword": "train for a 10k, 10k training plan, running 10k",
            "canonical_url": "https://runnersworld.example/how-to-train-for-10k",
            "og_title": "Train for a 10K",
            "og_description": "8-week roadmap.",
            "og_image": "https://runnersworld.example/images/10k.png",
            "score": 92,
        },
    }

    extracted = extract_wordpress_seo_metadata(raw_post, "runnersworld.example")

    assert extracted["title"] == "How to Train for a 10K Race"
    assert extracted["seo_title"] == "How to Train for a 10K in 8 Weeks | Expert Guide"
    assert extracted["seo_description"] == "Follow our beginner to intermediate 10k training plan."
    assert extracted["focus_keyword"] == "train for a 10k"
    assert "10k training plan" in extracted["secondary_keywords"]
    assert "running 10k" in extracted["secondary_keywords"]
    assert extracted["seo_optimization_score"] >= 85
    assert extracted["seo_metadata"]["canonicalUrl"] == "https://runnersworld.example/how-to-train-for-10k"


def test_build_titles_payload_from_imported_post():
    extracted = {
        "wp_post_id": 303,
        "site_id": 5,
        "domain": "myblog.com",
        "title": "Complete SEO Checklist 2026",
        "slug": "seo-checklist-2026",
        "excerpt": "Checklist for modern search ranking.",
        "content_html": "<h2>Step 1: On-page SEO</h2><p>Optimize title tags...</p>",
        "published_at": "2026-01-20T12:00:00",
        "modified_at": "2026-01-22T15:00:00",
        "link": "https://myblog.com/seo-checklist-2026/",
        "category_ids": [10, 20],
        "category_names": ["Search Marketing", "Optimization"],
        "tag_ids": [30, 40],
        "tag_names": ["Checklist", "Rankings"],
        "featured_image_url": "https://myblog.com/img/seo-checklist.jpg",
        "featured_image_alt": "SEO Checklist graphic",
        "featured_image_title": "SEO Graphic",
        "featured_image_caption": "Caption here",
        "seo_title": "Ultimate SEO Checklist (2026) | Rank #1 on Google",
        "seo_description": "Follow this comprehensive 45-point SEO checklist.",
        "focus_keyword": "seo checklist 2026",
        "primary_keyword": "seo checklist 2026",
        "secondary_keywords": ["seo audit", "on page seo tips", "technical seo checklist"],
        "canonical_url": "https://myblog.com/seo-checklist-2026/",
        "seo_plugin": "yoast",
        "seo_score": 88,
        "readability_score": 80,
        "open_graph": {"title": "SEO Checklist 2026"},
        "twitter": {"card": "summary_large_image"},
        "robots": {"index": "index", "follow": "follow"},
        "schema_json": {},
        "raw_post_json": {"id": 303},
    }

    payload = _build_titles_payload_from_imported_post(
        user_id="user_test_123",
        site_id=5,
        domain="myblog.com",
        extracted=extracted,
    )

    assert payload["user_id"] == "user_test_123"
    assert payload["Title"] == "Complete SEO Checklist 2026"
    assert payload["primary_keyword"] == "seo checklist 2026"
    assert payload["primary_keywords"] == ["seo checklist 2026"]
    assert payload["secondary_keywords"] == ["seo audit", "on page seo tips", "technical seo checklist"]
    assert "seo audit" in payload["Keywords"]
    assert payload["search_phrase"] == "seo checklist 2026"
    assert payload["deck"] == "Follow this comprehensive 45-point SEO checklist."
    assert payload["userDescription"] == "Follow this comprehensive 45-point SEO checklist."
    assert payload["htmlArticle"] == "<h2>Step 1: On-page SEO</h2><p>Optimize title tags...</p>"
    assert payload["status"] == "WP Published"
    assert payload["published"] is True
    assert payload["last_wp_site_id"] == "5"
    assert payload["last_wp_post_status"] == "publish"
    assert payload["featuredImageUrl"] == "https://myblog.com/img/seo-checklist.jpg"
    assert payload["featuredImageURL"] == "https://myblog.com/img/seo-checklist.jpg"
    assert payload["mediaAltText"] == "SEO Checklist graphic"
    assert payload["seo_optimization_score"] == 88
    assert payload["readability_score"] == 80
    assert payload["wordpress_category_id"] == 10
    assert payload["category"] == "Search Marketing"

    # Check idea_metadata
    idea_meta = payload["idea_metadata"]
    assert idea_meta["seo_metadata"]["meta_title"] == "Ultimate SEO Checklist (2026) | Rank #1 on Google"
    assert idea_meta["seo_metadata"]["meta_description"] == "Follow this comprehensive 45-point SEO checklist."
    assert idea_meta["seo_metadata"]["focus_keyword"] == "seo checklist 2026"
    assert idea_meta["slug"] == "seo-checklist-2026"
    assert idea_meta["canonical_url"] == "https://myblog.com/seo-checklist-2026/"
    assert idea_meta["category_context"]["primary_category_name"] == "Search Marketing"
    assert idea_meta["category_context"]["secondary_category_name"] == "Optimization"

    # Check selected_keyword_metrics_json
    kw_metrics = payload["selected_keyword_metrics_json"]
    assert kw_metrics["primary"]["keyword"] == "seo checklist 2026"
    assert len(kw_metrics["secondary"]) == 3


def test_insert_with_schema_fallback_adaptive_retries():
    # Simulate a Supabase table that lacks multiple columns (e.g., 'seo_title', 'featured_image_url', 'slug')
    allowed_cols = {"user_id", "title", "link", "excerpt"}
    received_records = []

    class MockTableQuery:
        def __init__(self, records):
            self.records = records

        def execute(self):
            # Check if any record has columns outside allowed_cols
            for rec in self.records:
                for col in rec.keys():
                    if col not in allowed_cols:
                        # Emulate PostgREST error format
                        raise Exception(f"Could not find the '{col}' column of 'wordpress_imported_posts' in the schema cache")
            received_records.extend(self.records)
    class MockTable:
        def insert(self, records):
            return MockTableQuery(records)

    class MockSupabase:
        def table(self, name):
            return MockTable()

    records_to_insert = [
        {
            "user_id": "u123",
            "title": "Test Title",
            "link": "https://example.com/test",
            "excerpt": "Test excerpt",
            "seo_title": "SEO Title",
            "featured_image_url": "https://example.com/img.jpg",
            "slug": "test-slug",
        }
    ]

    mock_sb = MockSupabase()
    success = _insert_with_schema_fallback(mock_sb, "wordpress_imported_posts", records_to_insert, log_label="Test")

    assert success is True
    assert len(received_records) == 1
    assert "seo_title" not in received_records[0]
    assert "featured_image_url" not in received_records[0]
    assert "slug" not in received_records[0]
    assert received_records[0]["title"] == "Test Title"
    assert received_records[0]["user_id"] == "u123"


if __name__ == "__main__":
    test_clean_html_text()
    test_extract_wordpress_seo_metadata_yoast()
    test_extract_wordpress_seo_metadata_rankmath()
    test_build_titles_payload_from_imported_post()
    test_insert_with_schema_fallback_adaptive_retries()
    print("All WordPress SEO import tests passed successfully!")

