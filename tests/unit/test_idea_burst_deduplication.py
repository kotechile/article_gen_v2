import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.api.endpoints.research_topics import parse_idea_response


def test_parse_idea_response_dedupes_near_duplicate_blog_ideas():
    response_text = """
BLOG_IDEA: 1
TITLE: Incremental ROI Calculator for Marketing Spend
DESCRIPTION: Calculate whether the next tranche of spend should go into the same channel.
SEARCH_PHRASE: roi calculator
INPUT_KEYWORDS: roi calculator, marketing roi, spend calculator, budget roi
INTENT: informational
FORMAT: calculator-guide
USER_DECISION_HELPED: Decide whether the next dollar should go into the same marketing channel.
INTERNAL_LINK_HOOK: Link from channel budget planning pages
MONETIZATION: Affiliate links to analytics tools
VIABILITY: 78
END_IDEA

BLOG_IDEA: 2
TITLE: Marketing Budget ROI Tool for the Next Dollar
DESCRIPTION: Estimate whether an extra budget increase will produce enough return.
SEARCH_PHRASE: budget roi tool
INPUT_KEYWORDS: budget roi, marketing roi tool, spend roi, next dollar roi
INTENT: informational
FORMAT: calculator-guide
USER_DECISION_HELPED: Decide whether another dollar should go into the same marketing channel.
INTERNAL_LINK_HOOK: Link from paid media budget pages
MONETIZATION: Affiliate links to analytics tools
VIABILITY: 76
END_IDEA

BLOG_IDEA: 3
TITLE: Channel Saturation Warning Signs
DESCRIPTION: Show how to spot when incremental spend is starting to flatten out.
SEARCH_PHRASE: channel saturation
INPUT_KEYWORDS: channel saturation, ad fatigue signs, diminishing returns, budget plateau
INTENT: informational
FORMAT: checklist
USER_DECISION_HELPED: Decide when to stop increasing spend in a channel and look for alternatives.
INTERNAL_LINK_HOOK: Link from campaign optimization articles
MONETIZATION: Affiliate links to measurement tools
VIABILITY: 80
END_IDEA
"""

    ideas = parse_idea_response(
        text=response_text,
        content_type="blog",
        topic_id="topic-1",
        user_id="user-1",
        subtopic_name="Incremental ROI Decision Logic",
        primary_user_outcome="Choose where the next marketing dollar should go",
    )

    titles = [idea["title"] for idea in ideas]
    assert len(ideas) == 2
    assert "Channel Saturation Warning Signs" in titles
    assert (
        "Incremental ROI Calculator for Marketing Spend" in titles
        or "Marketing Budget ROI Tool for the Next Dollar" in titles
    )
