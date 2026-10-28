"""Enforce one canonical document skeleton on the assembled article.

The pipeline assembles a document from several writers - a smart-brevity lead, one
call per planned section, a polish pass, closing modules - and each of them has its
own idea of the top of the page. The result that reached the editor looked like
this:

    <p><strong>Go deeper:</strong> Review the detailed breakdown ...</p>
    <h2>Go Deeper</h2>
    <h2>RAG Agentic AI: The 3-Shift Reckoning for Mid-Career Architects</h2>
    <h1>3 Productivity Killers: RAG Agentic AI</h1>          <- plus four more h1s
    <p><strong>The Big Picture:</strong> ...</p>
    ...
    <h3>At a Glance</h3> ...                                  <- once per section
    <h3>The Catch</h3> ...

So the lead-in that belongs *after* the opening appeared at the very top, twice, the
article had five h1s and no single title, and every section repeated the closing
modules.

This module fixes the document as a whole, after everyone has written:

    <h1>title</h1>
    lead paragraphs (big picture / why it matters / by the numbers / bottom line)
    <p><strong>Go deeper:</strong> ...</p>
    <h2>Go Deeper</h2>
    body sections (one heading each, no h1s)
    closing modules, once, at the end

It moves and merges blocks; it never rewrites prose.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

HEADING_RE = re.compile(r"<h([1-6])\b([^>]*)>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
GO_DEEPER_PARA_RE = re.compile(
    r"<p\b[^>]*>\s*(?:<strong>\s*go\s+deeper\s*:?\s*</strong>|<b>\s*go\s+deeper\s*:?\s*</b>)"
    r"(.*?)</p>",
    re.IGNORECASE | re.DOTALL,
)
LEAD_MARKERS = (
    "the big picture",
    "why it matters",
    "by the numbers",
    "the bottom line",
    "the big shift",
)
CLOSING_MODULES = {
    "what i'd watch": "what_i_watch",
    "what i’d watch": "what_i_watch",
    "what id watch": "what_i_watch",
    "what to watch": "what_i_watch",
    "the catch": "the_catch",
    "at a glance": "at_a_glance",
    "key takeaways": "takeaways",
    "key takeaway": "takeaways",
    "tl;dr": "takeaways",
    "faq": "faq",
    "references": "references",
    "sources": "references",
}
MODULE_ORDER = ("what_i_watch", "the_catch", "at_a_glance", "takeaways", "faq", "references")
GO_DEEPER_DEFAULT = (
    "Review the detailed breakdown, comparative analysis, and step-by-step guidance below."
)


def _plain(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", " ", fragment or "")
    text = re.sub(r"&nbsp;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", _plain(text).lower()).strip()


class _Block:
    """A heading and the body that follows it (or the preamble, which has none)."""

    __slots__ = ("level", "heading_raw", "heading_text", "body")

    def __init__(self, level: Optional[int], heading_raw: str, heading_text: str, body: str):
        self.level = level
        self.heading_raw = heading_raw
        self.heading_text = heading_text
        self.body = body

    def render(self) -> str:
        return f"{self.heading_raw}{self.body}" if self.heading_raw else self.body

    @property
    def words(self) -> int:
        return len(_plain(self.body).split())


def _segment(html: str) -> List[_Block]:
    parts = re.split(r"(?=<h[1-6]\b)", html or "", flags=re.IGNORECASE)
    blocks: List[_Block] = []
    for part in parts:
        if not part.strip():
            continue
        match = HEADING_RE.match(part.lstrip())
        if match:
            lead_ws = part[: len(part) - len(part.lstrip())]
            level = int(match.group(1))
            blocks.append(
                _Block(level, f"{lead_ws}{match.group(0)}", _plain(match.group(3)), part[match.end():])
            )
        else:
            blocks.append(_Block(None, "", "", part))
    return blocks


def _module_key(block: _Block) -> Optional[str]:
    """Recognise a closing module by its heading, allowing a descriptive suffix.

    "What I'd Watch: 3 Indicators Before Scaling RAG Agentic AI" is the same module
    as "What I'd Watch". Matching the whole heading only missed the version that had
    real content, so the boilerplate copy survived instead.
    """
    if not block.heading_text:
        return None
    full = _norm(block.heading_text)
    if full in CLOSING_MODULES:
        return CLOSING_MODULES[full]
    for name, key in CLOSING_MODULES.items():
        if full.startswith(name):
            return key
    return None


def _shingles(text: str, size: int = 6) -> set:
    words = _plain(text).lower().split()
    return {" ".join(words[i:i + size]) for i in range(max(0, len(words) - size + 1))}


# Sentences the pipeline used to synthesise when a module was missing. They are
# recognisable filler, so a document that still carries them can be cleaned.
GENERIC_FILLER_RE = re.compile(
    r"(?:"
    r"every tactical gain introduces"
    r"|acknowledge real-?world edge cases"
    r"|monitor latency,? concurrency spikes"
    r"|execute disciplined,? measurable implementations"
    r"|track compatibility drifts"
    r"|identify edge cases where conventional architectures"
    r"|mastering the fundamentals of"
    r"|understanding the fundamental tension of"
    r"|the market is accelerating the transition toward"
    r")",
    re.IGNORECASE,
)

_CONTENT_STOPWORDS = {
    "that", "this", "with", "from", "have", "been", "they", "their", "which", "when",
    "what", "will", "would", "there", "these", "those", "into", "over", "than", "then",
    "also", "such", "more", "most", "many", "some", "your", "each", "about", "because",
    "does", "doesn", "where", "while", "after", "before", "under", "between",
}


def _content_words(text: str) -> set:
    return {
        word for word in re.findall(r"[a-z][a-z-]{3,}", _plain(text).lower())
        if word not in _CONTENT_STOPWORDS
    }


def _duplicates_earlier(block: _Block, earlier_text: str) -> bool:
    """True when a module just restates prose the reader already met above it.

    Uses content words rather than n-gram shingles: "At a Glance" restates the lead
    by paraphrasing it ("stalls 7 in 10 mid-career promotions" / "static retrieval
    syndrome"), which no exact phrase match will catch.
    """
    body = _content_words(block.body)
    if len(body) < 4:
        return False
    overlap = len(body & _content_words(earlier_text)) / len(body)
    return overlap >= 0.5


def _is_lead_paragraph(fragment: str) -> bool:
    marker = _plain(fragment).lower()
    return any(marker.startswith(m) or f"{m}:" in marker[:40] for m in LEAD_MARKERS)


def _lead_marker(fragment: str) -> str:
    """Which lead slot a paragraph fills - the first of each marker survives."""
    marker = _plain(fragment).lower()
    for name in LEAD_MARKERS:
        if marker.startswith(name) or f"{name}:" in marker[:40]:
            return name
    return marker[:40]


def _extract_lead_paragraphs(body: str, taken: set) -> Tuple[List[str], str]:
    """Pull *unclaimed* lead paragraphs out of a section body.

    One paragraph per marker is hoisted to the top of the article. A later section's
    version of "The Big Picture" stays where it is: it is that section's opening,
    and moving or deleting it would take its evidence with it.
    """
    found: List[str] = []
    paragraphs = re.findall(r"<p\b[^>]*>.*?</p>", body or "", re.IGNORECASE | re.DOTALL)
    for paragraph in paragraphs:
        if not _is_lead_paragraph(paragraph):
            continue
        marker = _lead_marker(paragraph)
        if marker in taken:
            continue
        taken.add(marker)
        found.append(paragraph.strip())
    if not found:
        return [], body
    remaining = body
    for paragraph in found:
        remaining = remaining.replace(paragraph, "")
    return found, remaining


def _strip_hrs(html: str) -> str:
    return re.sub(r"(?:<hr\s*/?>\s*)+$", "", html.strip())


def normalize_document_structure(
    html: str, title: str = "", *, require_lead: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """Return ``(normalised_html, report)``.

    ``title`` wins when the document carries several competing headlines.
    """
    report: Dict[str, Any] = {
        "changed": False,
        "title": "",
        "title_source": "",
        "h1_before": 0,
        "go_deeper_paragraphs_before": 0,
        "go_deeper_headings_before": 0,
        "duplicate_modules_removed": 0,
        "lead_paragraphs": 0,
        "words_before": 0,
        "words_after": 0,
    }
    source = html or ""
    original_source = source
    report["words_before"] = len(_plain(source).split())

    h1_matches = [m for m in HEADING_RE.finditer(source) if m.group(1) == "1"]
    report["h1_before"] = len(h1_matches)
    report["go_deeper_paragraphs_before"] = len(GO_DEEPER_PARA_RE.findall(source))
    report["go_deeper_headings_before"] = sum(
        1
        for m in HEADING_RE.finditer(source)
        if _norm(m.group(3)) == "go deeper"
    )

    # 1. The lead-in belongs after the lead: take it out of the flow entirely.
    go_deeper_line = ""
    for match in GO_DEEPER_PARA_RE.finditer(source):
        if not go_deeper_line:
            tail = " ".join(match.group(1).split()) or GO_DEEPER_DEFAULT
            go_deeper_line = f"<p><strong>Go deeper:</strong> {tail}</p>"
    source = GO_DEEPER_PARA_RE.sub("", source)

    blocks = _segment(source)

    # 2. Drop every "Go Deeper" heading; exactly one is re-inserted below. Content
    #    that followed the heading belongs to the body and must survive it.
    rebuilt: List[_Block] = []
    for block in blocks:
        if block.heading_text and _norm(block.heading_text) == "go deeper":
            if _plain(block.body):
                rebuilt.append(_Block(None, "", "", block.body))
            continue
        rebuilt.append(block)
    blocks = rebuilt

    # 3. Choose the title: an explicit title beats a document with several h1s.
    title_blocks = [b for b in blocks if b.level == 1]
    chosen_title_block: Optional[_Block] = None
    clean_title = _plain(title)
    if clean_title and len(title_blocks) != 1:
        report["title_source"] = "argument"
    elif title_blocks:
        chosen_title_block = title_blocks[0]
        report["title_source"] = "document"
    elif clean_title:
        report["title_source"] = "argument"
    if chosen_title_block is not None:
        title_text = chosen_title_block.heading_text
        blocks.remove(chosen_title_block)
        # Whatever sat under the title belongs to the article, not to the title:
        # re-queue it so the lead extraction and the body flow still see it.
        if _plain(chosen_title_block.body):
            blocks.insert(0, _Block(None, "", "", chosen_title_block.body))
    else:
        title_text = clean_title or (title_blocks[0].heading_text if title_blocks else "Untitled article")
    report["title"] = title_text

    # 4. Collapse "section heading, then a headline with no body between them":
    #    the headline is a second title for the same section, not a new section.
    merged: List[_Block] = []
    for block in blocks:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and block.level == 1
            and previous.level is not None
            and not _plain(previous.body)
        ):
            previous.body = block.body
            continue
        merged.append(block)
    blocks = merged

    # 5. Any remaining h1 is a section headline.
    for block in blocks:
        if block.level == 1:
            block.level = 2
            block.heading_raw = re.sub(r"^(\s*)<h1", r"\1<h2", block.heading_raw, flags=re.IGNORECASE)
            block.heading_raw = re.sub(r"</h1>\s*$", "</h2>", block.heading_raw, flags=re.IGNORECASE)

    # 6. Lead paragraphs move to the top, in document order, one per marker. Each
    #    section used to write its own lead, so five "The Big Picture" paragraphs
    #    would otherwise stack up as the opening of the article.
    lead: List[str] = []
    seen_markers: set[str] = set()
    for block in blocks:
        found, block.body = _extract_lead_paragraphs(block.body, seen_markers)
        lead.extend(found)
    report["lead_paragraphs"] = len(lead)

    # 7. Closing modules: keep one copy of each - the richest, since sections wrote
    #    their own version with their own caveats - and move it to the end.
    lead_text = " ".join(lead)
    modules: Dict[str, _Block] = {}
    body_blocks: List[_Block] = []
    for block in blocks:
        key = _module_key(block)
        if key and not _plain(block.body):
            report["duplicate_modules_removed"] += 1
            continue
        if key and GENERIC_FILLER_RE.search(_plain(block.body)):
            report["boilerplate_modules_removed"] = report.get("boilerplate_modules_removed", 0) + 1
            continue
        if key and _duplicates_earlier(block, lead_text):
            # An "At a Glance" that repeats the lead is not a summary, it is a
            # second copy of the opening.
            report["restated_modules_removed"] = report.get("restated_modules_removed", 0) + 1
            continue
        if key:
            existing = modules.get(key)
            if existing is None:
                modules[key] = block
            else:
                report["duplicate_modules_removed"] += 1
                if block.words >= existing.words:
                    modules[key] = block
            continue
        body_blocks.append(block)

    # 8. Assemble.
    pieces: List[str] = [f"<h1>{title_text}</h1>" if not title_text.startswith("<") else f"<h1>{_plain(title_text)}</h1>"]
    pieces.extend(lead)
    if not go_deeper_line:
        go_deeper_line = f"<p><strong>Go deeper:</strong> {GO_DEEPER_DEFAULT}</p>"
    pieces.append(go_deeper_line)
    pieces.append("<h2>Go Deeper</h2>")
    for index, block in enumerate(body_blocks):
        rendered = block.render().strip()
        if not rendered:
            continue
        if block.level is not None and not _plain(block.body):
            # A heading with nothing of its own is either a section opener (deeper
            # headings follow it) or a leftover; keep the former, drop the latter.
            following = body_blocks[index + 1] if index + 1 < len(body_blocks) else None
            opens_a_section = (
                following is not None
                and following.level is not None
                and following.level > block.level
                and _plain(following.body)
            )
            if not opens_a_section:
                continue
        pieces.append(_strip_hrs(rendered))
    for key in MODULE_ORDER:
        block = modules.pop(key, None)
        if block is None:
            continue
        rendered = _strip_hrs(block.render().strip())
        if rendered:
            pieces.append(rendered)

    result = "\n\n".join(piece for piece in pieces if piece.strip())
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    result = re.sub(r"^(\s*<hr\s*/?>\s*)+", "", result)

    report["words_after"] = len(_plain(result).split())
    # Whitespace and re-wrapping are not changes; moved or dropped blocks are.
    report["changed"] = re.sub(r"\s+", "", result) != re.sub(r"\s+", "", original_source)
    return result, report
