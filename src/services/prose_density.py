"""Measure how much of an article is bullet scaffolding, and strip stale references.

Two deterministic checks that back the prose-first prompts:

* ``measure_prose_density`` counts the shapes that make text read as machine
  output - list items, "**Label:** explanation" bullets, tables per section - so a
  section that is really a slide deck is visible as a number instead of a feeling.
* ``strip_references_section`` removes a References block that the generator wrote
  into the body, so the curated reference list is not appended on top of it.
"""

from __future__ import annotations

import re
from typing import Any, Dict

LIST_ITEM_RE = re.compile(r"<li\b[^>]*>(.*?)</li>", re.IGNORECASE | re.DOTALL)
PARAGRAPH_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)
TABLE_RE = re.compile(r"<table\b", re.IGNORECASE)
HEADING_RE = re.compile(r"<h([1-6])\b[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
REFERENCES_HEADING_RE = re.compile(
    r"<h[1-6]\b[^>]*>\s*(?:references|sources|bibliography)\s*</h[1-6]>",
    re.IGNORECASE,
)
LABEL_BULLET_RE = re.compile(
    r"<li\b[^>]*>\s*<(?:strong|b)>\s*[^<]{1,80}?\s*:?\s*</(?:strong|b)>",
    re.IGNORECASE,
)


def _plain(fragment: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", fragment or "")).strip()


def measure_prose_density(html: str) -> Dict[str, Any]:
    """Count prose against bullet scaffolding.

    ``bullet_ratio`` is list items per paragraph. Above ~1.0 the article reads as a
    deck; a well-integrated article sits well under it.
    """
    html = html or ""
    items = LIST_ITEM_RE.findall(html)
    paragraphs = [p for p in PARAGRAPH_RE.findall(html) if len(_plain(p).split()) >= 15]
    tables = len(TABLE_RE.findall(html))
    label_bullets = len(LABEL_BULLET_RE.findall(html))
    headings = len(HEADING_RE.findall(html))

    item_words = sum(len(_plain(item).split()) for item in items)
    paragraph_words = sum(len(_plain(p).split()) for p in paragraphs)
    body_words = max(1, item_words + paragraph_words)

    bullet_ratio = round(len(items) / max(1, len(paragraphs)), 2)
    list_word_share = round(item_words / body_words, 3)

    reasons = []
    if bullet_ratio > 1.0:
        reasons.append(f"more list items than paragraphs ({len(items)} vs {len(paragraphs)})")
    if list_word_share > 0.35:
        reasons.append(f"{int(list_word_share * 100)}% of body words sit in lists")
    if label_bullets >= 3 and label_bullets > len(paragraphs) * 0.5:
        reasons.append(f"{label_bullets} 'Label: explanation' bullets")
    if tables and tables > max(2, headings // 2):
        reasons.append(f"{tables} tables for {headings} headings")

    return {
        "list_items": len(items),
        "paragraphs": len(paragraphs),
        "tables": tables,
        "label_bullets": label_bullets,
        "headings": headings,
        "bullet_ratio": bullet_ratio,
        "list_word_share": list_word_share,
        "reads_like_slides": bool(reasons),
        "reasons": reasons,
    }


def strip_references_section(html: str) -> str:
    """Drop a References/Sources block that already sits in the body.

    The generator sometimes writes its own reference list with truncated titles;
    the curated one is appended afterwards, so the body copy has to go first or the
    article publishes both.
    """
    html = html or ""
    matches = list(REFERENCES_HEADING_RE.finditer(html))
    if not matches:
        return html
    # Everything from the first References heading to the end is the stale block:
    # references are always last, and the curated list replaces them wholesale.
    return html[: matches[0].start()].rstrip()
