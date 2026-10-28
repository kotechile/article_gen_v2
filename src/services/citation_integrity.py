"""Citation integrity gate.

A reference a reader cannot open is not a citation - it is an assertion wearing
a citation's clothes. That is exactly the shape an invented source takes:

    [3] Edge AI and Systems Optimization Lab: Inference Economics, Tail Latency,
        and Memory Footprint in Multi-Turn Agent Workflows, 2024.

No URL, no author, a plausible-sounding lab, and it carries the most specific
numbers in the article. Editors cannot check it and neither can readers.

This gate runs after citation reconciliation and before rendering:

* a reference survives only with a resolvable ``http(s)`` URL;
* references are renumbered to first appearance in the body, so ``[1]`` is the
  first thing the reader is asked to trust;
* markers pointing at dropped references are removed with them - a dangling
  ``[4]`` is worse than no citation at all;
* everything dropped is reported, so the loss is visible in logs instead of
  silently changing the article.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_MARKER_RE = re.compile(r"\[\^?(\d+)\]")

# A reference URL must name a host with a real TLD. `startswith("https://")` accepted
# "https://www." - a scraped URL truncated mid-domain - and the reader got a link to nothing.
_VALID_URL = re.compile(r"^https?://[^\s/?#]+\.[A-Za-z]{2,}(?::\d+)?(?:[/?#]\S*)?$")

# A reference title is a title, not a page fragment: "### **3." and "### Key Takeaways -
# **Progress Software Launch." are scrape noise, and neither can be checked or cited.
_FRAGMENT_TITLE = re.compile(r"^\s*(?:#{1,6}|[-*•])\s|^\W*\d+\.?\W*$")


def _usable_url(url: str) -> bool:
    return bool(_VALID_URL.match((url or "").strip()))


def _usable_title(title: str) -> bool:
    """False for scrape noise: a heading marker that leaked in, or a bare number.

    Length is not the test - a real title can be short ("S-1 filing") - the shape is:
    "### **3." and "### Key Takeaways - **Progress Software Launch." both begin with the
    markup of the page they were scraped from, and neither can be cited by a reader.
    """
    raw = (title or "").strip()
    if not raw or _FRAGMENT_TITLE.match(raw):
        return False
    text = re.sub(r"[*_`#>\[\]()]", " ", raw).strip()
    return bool(re.search(r"[A-Za-z]", text))


def enforce_citation_integrity(
    content: str,
    citations: List[Dict[str, Any]],
    require_urls: bool = True,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Drop unverifiable references and keep in-text markers in sync.

    Returns ``(content, citations, report)``.
    """
    report: Dict[str, Any] = {
        "kept": 0,
        "dropped_without_url": [],
        "dropped_malformed": [],
        "dropped_fragment": [],
        "dropped_uncited": [],
        "markers_repaired": 0,
        "require_urls": require_urls,
    }
    if not content or not citations:
        report["kept"] = len(citations or [])
        return content, citations or [], report

    # 1. Which references are verifiable at all?
    survives: List[Optional[Dict[str, Any]]] = []
    for citation in citations:
        url = str((citation or {}).get("url") or "").strip()
        title = str((citation or {}).get("title") or "")
        if require_urls and not url.startswith(("http://", "https://")):
            report["dropped_without_url"].append(title[:120] or "untitled")
            survives.append(None)
        elif require_urls and not _usable_url(url):
            report.setdefault("dropped_malformed", []).append(f"{url[:80]} ({title[:60]})")
            survives.append(None)
        elif not _usable_title(title):
            report.setdefault("dropped_fragment", []).append(f"{title[:80]} ({url[:60]})")
            survives.append(None)
        else:
            survives.append(citation)

    kept: List[Dict[str, Any]] = []
    old_to_new: Dict[int, Optional[int]] = {}

    # 2. Walk the body in order: keep only cited, verifiable references, and
    #    renumber to first appearance so the list matches the reader's journey.
    for raw_num in _MARKER_RE.findall(content):
        try:
            original = int(raw_num)
        except ValueError:
            continue
        if original in old_to_new:
            continue
        candidate = survives[original - 1] if 0 < original <= len(survives) else None
        if candidate is None:
            old_to_new[original] = None
            continue
        old_to_new[original] = len(kept) + 1
        kept.append(candidate)

    if not kept:
        for citation in survives:
            if citation is not None:
                report["dropped_uncited"].append(str(citation.get("title") or "untitled")[:120])

    def _replace(match: re.Match) -> str:
        try:
            original = int(match.group(1))
        except ValueError:
            return match.group(0)
        new_num = old_to_new.get(original)
        if new_num is None:
            report["markers_repaired"] += 1
            return ""
        if new_num != original:
            report["markers_repaired"] += 1
            return f"[{new_num}]"
        return match.group(0)

    updated = _MARKER_RE.sub(_replace, content)
    updated = re.sub(r"\s+([.,;:!?])", r"\1", updated)
    updated = re.sub(r"[ \t]{2,}", " ", updated)

    report["kept"] = len(kept)
    return updated, kept, report
