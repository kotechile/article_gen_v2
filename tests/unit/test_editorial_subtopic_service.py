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
    assert "<<SUBTOPIC>>" in prompt
    assert "<<TITLE>>" in prompt
    assert "Do not use JSON." in prompt


def test_parse_accepts_tagged_subtopic_output():
    service = EditorialSubtopicService()

    response_text = """
<<SUBTOPIC>>
<<TITLE>>
Email Platform Pricing Comparison
<</TITLE>>
<<SUMMARY>>
Compare pricing models across email tools for different list sizes.
<</SUMMARY>>
<<DECISION_TYPE>>
comparison
<</DECISION_TYPE>>
<<USER_PROBLEM>>
Need to compare tool pricing before choosing a provider.
<</USER_PROBLEM>>
<<TARGET_AUDIENCE>>
SaaS founders
<</TARGET_AUDIENCE>>
<<SEED_PHRASES>>
email pricing comparison | email tool cost | email platform pricing | compare email software
<</SEED_PHRASES>>
<<GEO_ENTITY_HINTS>>
Mailchimp | Klaviyo | ConvertKit
<</GEO_ENTITY_HINTS>>
<<COMMERCIAL_PATHS>>
software | consulting
<</COMMERCIAL_PATHS>>
<</SUBTOPIC>>
<<SUBTOPIC>>
<<TITLE>>
Choosing an Affordable Email Marketing Platform
<</TITLE>>
<<SUMMARY>>
Evaluate low-cost platform options without overpaying for features.
<</SUMMARY>>
<<DECISION_TYPE>>
comparison
<</DECISION_TYPE>>
<<USER_PROBLEM>>
Need to compare software costs and pricing tradeoffs before selecting a platform.
<</USER_PROBLEM>>
<<TARGET_AUDIENCE>>
SaaS founders
<</TARGET_AUDIENCE>>
<<SEED_PHRASES>>
affordable email platform | email software pricing | low cost email tools | email platform cost
<</SEED_PHRASES>>
<<GEO_ENTITY_HINTS>>
Mailchimp | Klaviyo | ConvertKit
<</GEO_ENTITY_HINTS>>
<<COMMERCIAL_PATHS>>
software | consulting
<</COMMERCIAL_PATHS>>
<</SUBTOPIC>>
<<SUBTOPIC>>
<<TITLE>>
Email Automation Migration Checklist
<</TITLE>>
<<SUMMARY>>
Plan the steps, risks, and dependencies when moving to a new platform.
<</SUMMARY>>
<<DECISION_TYPE>>
checklist
<</DECISION_TYPE>>
<<USER_PROBLEM>>
Need a safe migration path from one email system to another.
<</USER_PROBLEM>>
<<TARGET_AUDIENCE>>
SaaS founders
<</TARGET_AUDIENCE>>
<<SEED_PHRASES>>
email migration checklist | move email automation | email platform migration | switch email tools
<</SEED_PHRASES>>
<<GEO_ENTITY_HINTS>>
HubSpot | Mailchimp | Customer.io
<</GEO_ENTITY_HINTS>>
<<COMMERCIAL_PATHS>>
software | services
<</COMMERCIAL_PATHS>>
<</SUBTOPIC>>
"""

    parsed = service._parse(response_text)

    titles = [item["title"] for item in parsed]
    assert len(parsed) == 3
    assert "Email Platform Pricing Comparison" in titles
    assert "Choosing an Affordable Email Marketing Platform" in titles
    assert "Email Automation Migration Checklist" in titles


def test_parse_accepts_numbered_gemini_style_structured_output():
    service = EditorialSubtopicService()

    response_text = """
1. The "Next Dollar" Hierarchy

TITLE: The "Next Dollar" Hierarchy: Beyond Static Asset Allocation
SUMMARY: A framework for high-velocity cash flow management that replaces fixed portfolio percentages with a real-time marginal utility ladder.
DECISION_TYPE: Sequential Logic / Prioritization Framework.
USER_PROBLEM: Decision paralysis when faced with surplus cash and outdated percentage-based models.
TARGET_AUDIENCE: Fractional CFOs, agile entrepreneurs, and HNWIs with monthly surplus income.
SEED_PHRASES: marginal utility of capital, incremental ROI framework, cash flow deployment strategy, next dollar principle
GEO_ENTITY_HINTS: Global financial markets, New York, London, California
COMMERCIAL_PATHS: cash-flow modeling software, financial advisory services, treasury management tools

2. Yield vs. Resilience

TITLE: Yield vs. Resilience: Calculating the Risk-Adjusted Marginal Utility of Optionality
SUMMARY: This article explores why the highest theoretical ROI is not always the correct move and teaches how to value optionality and liquidity.
DECISION_TYPE: Risk-Adjusted Optimization / Trade-off Analysis.
USER_PROBLEM: Over-leveraging into illiquid assets and missing distressed buying opportunities due to lack of dry powder.
TARGET_AUDIENCE: Real estate investors, venture capitalists, and contrarian stock traders.
SEED_PHRASES: opportunity cost of liquidity, dry powder strategy, risk-adjusted marginal utility, optionality in investing
GEO_ENTITY_HINTS: Emerging markets, US real estate markets
COMMERCIAL_PATHS: high-yield savings accounts, money market funds, private equity calls
"""

    parsed = service._parse(response_text)

    assert len(parsed) == 2
    assert parsed[0]["title"] == 'The "Next Dollar" Hierarchy: Beyond Static Asset Allocation'
    assert parsed[0]["decision_type"] == "sequential logic / prioritization framework."
    assert parsed[0]["seed_phrases"] == [
        "marginal utility of capital",
        "incremental ROI framework",
        "cash flow deployment strategy",
        "next dollar principle",
    ]
    assert parsed[1]["title"] == "Yield vs. Resilience: Calculating the Risk-Adjusted Marginal Utility of Optionality"


def test_parse_still_accepts_legacy_subtopic_blocks():
    service = EditorialSubtopicService()

    response_text = """
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

    parsed = service._parse(response_text)

    assert len(parsed) == 1
    assert parsed[0]["title"] == "Email Automation Migration Checklist"
