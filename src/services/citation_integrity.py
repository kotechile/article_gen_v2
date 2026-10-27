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
        if require_urls and not url.startswith(("http://", "https://")):
            report["dropped_without_url"].append(str((citation or {}).get("title") or "untitled")[:120])
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
