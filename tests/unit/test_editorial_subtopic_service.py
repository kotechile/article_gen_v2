import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.services.editorial_subtopic_service import EditorialSubtopicService


def test_prompt_emphasizes_distinct_non_overlapping_subtopics():
    service = EditorialSubtopicService()

    prompt = service._build_prompt(
        {
            "topic_title": "Email marketing software",
            "topic_description": "Tools and decisions for choosing an email platform",
        },
        max_subtopics=6,
    )

    assert "Every subtopic must represent a meaningfully different concept" in prompt
    assert "remove any near-duplicate or synonym-based variation" in prompt
    assert "same article outline" in prompt


def test_parse_dedupes_near_duplicate_subtopics():
    service = EditorialSubtopicService()

    response_text = """
[SUBTOPIC]
TITLE: Email Platform Pricing Comparison
SUMMARY: Compare pricing models across email tools for different list sizes.
DECISION_TYPE: comparison
USER_PROBLEM: Need to compare tool pricing before choosing a provider.
TARGET_AUDIENCE: SaaS founders
SEED_PHRASES: email pricing comparison, email tool cost, email platform pricing, compare email software
GEO_ENTITY_HINTS: Mailchimp, Klaviyo, ConvertKit
COMMERCIAL_PATHS: software, consulting
[END]
[SUBTOPIC]
TITLE: Choosing an Affordable Email Marketing Platform
SUMMARY: Evaluate low-cost platform options without overpaying for features.
DECISION_TYPE: comparison
USER_PROBLEM: Need to compare software costs and pricing tradeoffs before selecting a platform.
TARGET_AUDIENCE: SaaS founders
SEED_PHRASES: affordable email platform, email software pricing, low cost email tools, email platform cost
GEO_ENTITY_HINTS: Mailchimp, Klaviyo, ConvertKit
COMMERCIAL_PATHS: software, consulting
[END]
[SUBTOPIC]
TITLE: Email Automation Migration Checklist
SUMMARY: Plan the steps, risks, and dependencies when moving to a new platform.
DECISION_TYPE: checklist
USER_PROBLEM: Need a safe migration path from one email system to another.
TARGET_AUDIENCE: SaaS founders
SEED_PHRASES: email migration checklist, move email automation, email platform migration, switch email tools
GEO_ENTITY_HINTS: HubSpot, Mailchimp, Customer.io
COMMERCIAL_PATHS: software, services
[END]
"""

    parsed = service._dedupe_distinct_subtopics(service._parse(response_text), max_subtopics=6)

    titles = [item["title"] for item in parsed]
    assert len(parsed) == 2
    assert "Email Platform Pricing Comparison" in titles
    assert "Email Automation Migration Checklist" in titles
