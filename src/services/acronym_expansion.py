"""Spell an acronym out on first use, then use only the short form.

The rule: the first time an acronym or initialism appears in the text it is written
out in full, with the short form in parentheses — "retrieval-augmented generation
(RAG)" — and every later mention uses the short form alone. A reader who meets
"RAG" cold has to go and look it up; a reader who has met it once carries it through
the piece.

The prompts ask for it. This module is the check and the deterministic half of the
repair:

* `audit_acronyms()` reports which acronyms the text spells out and which it does
  not. It reads running text only - a heading or a module label is not prose.
* `spelling_matches()` validates a proposed expansion against the acronym's letters.
  A model supplies the expansion (it knows the context); this rejects anything whose
  initials do not spell the acronym, so "AI" can never be defined as "accuracy of
  identity" because those words happen to appear nearby. That failure is not
  hypothetical: discovering expansions by matching initials against an article's own
  phrases produced exactly that on a real draft, which is why this is validation.
* `expand_first_use()` writes the expansion at the first bare use and nowhere else.
  It adds a definition, never a claim, and leaves every other sentence alone.
"""

from __future__ import annotations

import html as html_module
import re
from typing import Any, Dict, List, Optional, Tuple

# 2-6 characters, not glued to another word or to a version suffix ("GPT-4").
_ACRONYM_RE = re.compile(r"(?<![A-Za-z0-9\-])[A-Z][A-Z0-9]{1,5}(?![A-Za-z0-9]|-\d)")
_WORD_RE = re.compile(r"[A-Za-z]+")
_TAG_RE = re.compile(r"<[^>]*>")
_HEADING_RE = re.compile(r"<h[1-6]\b[^>]*>.*?</h[1-6]>", re.I | re.S)

# Abbreviations conventionally written bare, and module labels an article always
# carries. Flagging these trains the reader to ignore the audit.
_IGNORE = frozenset({"OK", "AM", "PM", "FAQ", "TLDR", "US", "UK", "EU", "USA", "UAE"})

# Words that may sit inside a spelled-out form without owning an initial:
# "Federal Bureau of Investigation" -> FBI, "Retrieval Augmented Generation" -> RAG.
_FILLER = frozenset(
    {"of", "and", "for", "the", "in", "on", "to", "with", "a", "an", "by", "at",
     "from", "or", "via", "using", "into", "as", "per"}
)


def _plain(value: str) -> str:
    text = _TAG_RE.sub(" ", value or "")
    return re.sub(r"\s+", " ", html_module.unescape(text)).strip()


def _spans(value: str, pattern: re.Pattern) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in pattern.finditer(value or "")]


def _inside(position: int, spans: List[Tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)


def _body_like_plain(value: str) -> str:
    """Plain text with headings removed: a module label is not prose to expand."""
    without_headings = _HEADING_RE.sub(" ", value or "")
    return _plain(without_headings)


def _expanded_before(text: str, acronym: str, limit: int) -> bool:
    """Is there an "<expansion> (ACR)" before ``limit`` with no bare use ahead of it?"""
    pattern = re.compile(r"\(\s*" + re.escape(acronym) + r"\s*\)")
    for match in pattern.finditer(text[:limit]):
        prefix = text[max(0, match.start() - 80): match.start()]
        # The token immediately before the parenthesis must be a spelled-out word,
        # not punctuation or another acronym's tail.
        if not re.search(r"[A-Za-z][A-Za-z\-']*\s+$", prefix + " "):
            continue
        bare = [m.group(0) for m in _ACRONYM_RE.finditer(text[: match.start()])]
        if acronym in bare:
            continue
        return True
    return False


def spelling_matches(expansion: str, acronym: str) -> bool:
    """Do the expansion's initials spell the acronym?

    The shape check on a model-proposed expansion: "RAG" as retrieval-augmented
    generation passes, "RAG" as "rapid answer generation" does not spell the letters
    the same way and is rejected. It cannot tell two spellable readings apart - "AI"
    as artificial intelligence and as "accuracy of identity" both pass - which is why
    the model is given the sentence and a wrong reading is dropped rather than
    silently published.
    """
    if not acronym or any(ch.isdigit() for ch in acronym):
        return False
    letters = [c for c in acronym.upper() if c.isalpha()]
    if len(letters) < 2:
        return False
    words = _WORD_RE.findall(expansion or "")
    if not words or len(words) > len(letters) + 4:
        return False
    content = [w for w in words if w.lower() not in _FILLER]
    if len(content) < 2 and len(letters) > 2:
        return False
    initials = "".join(w[0].upper() for w in content)
    return initials == "".join(letters)


def audit_acronyms(value: str) -> Dict[str, Any]:
    """Report which acronyms the text spells out on first use.

    ``unexpanded`` is the work list: acronyms whose first appearance is bare.
    Headings are excluded from the scan - a module label is not running text.
    """
    plain = _body_like_plain(value)
    report: Dict[str, Any] = {
        "acronyms": [],
        "unexpanded": [],
        "total": 0,
        "expanded_count": 0,
    }

    first_use: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    for match in _ACRONYM_RE.finditer(plain):
        token = match.group(0)
        # "BM25", "R2", "GPT-4": model and version identifiers, not expandable terms.
        if token in _IGNORE or any(ch.isdigit() for ch in token):
            continue
        first_use.setdefault(token, match.start())
        counts[token] = counts.get(token, 0) + 1

    for token, position in first_use.items():
        expanded = _expanded_before(plain, token, len(plain))
        report["acronyms"].append(
            {
                "acronym": token,
                "count": counts[token],
                "expanded": expanded,
                "first_use": position,
            }
        )
        if not expanded:
            report["unexpanded"].append(token)

    report["total"] = len(report["acronyms"])
    report["expanded_count"] = report["total"] - len(report["unexpanded"])
    return report


def first_use_context(value: str, acronym: str, *, window: int = 2) -> List[str]:
    """The sentence holding the first bare use of ``acronym``, plus its neighbours.

    What an expansion has to be right for: "SB" in an article about solar rebates and
    "SB" in one about Senate procedure are different words.
    """
    plain = _body_like_plain(value)
    position = _first_bare_use(plain, acronym)
    if position is None:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", plain) if s.strip()]
    index = 0
    offset = 0
    for i, sentence in enumerate(sentences):
        if offset <= position < offset + len(sentence) + 1:
            index = i
            break
        offset += len(sentence) + 1
    return sentences[max(0, index - 1): index + window + 1]


def _first_bare_use(value: str, acronym: str) -> Optional[int]:
    """First use of the acronym outside tags and outside headings."""
    tag_spans = _spans(value, _TAG_RE)
    heading_spans = _spans(value, _HEADING_RE)
    for match in re.finditer(r"(?<![A-Za-z0-9\-])" + re.escape(acronym) + r"(?![A-Za-z0-9])", value or ""):
        start = match.start()
        if _inside(start, tag_spans) or _inside(start, heading_spans):
            continue
        # "(RAG)" is a parenthesised short form, i.e. part of an expansion already.
        if value[max(0, start - 2):start].strip().endswith("("):
            continue
        return start
    return None


def expand_first_use(value: str, acronym: str, expansion: str) -> Tuple[str, bool, str]:
    """Write the first bare use of ``acronym`` as "<expansion> (ACR)".

    Only the first use is touched, and only with an expansion whose initials spell
    the acronym (:func:`spelling_matches`). Returns ``(html, applied, mode)`` where
    mode is ``"inlined"`` or ``"unchanged"``.
    """
    if not acronym or not expansion:
        return value, False, "unchanged"
    expansion = " ".join((expansion or "").split()).strip(" .;:")
    if not spelling_matches(expansion, acronym):
        return value, False, "unchanged"

    plain = _body_like_plain(value)
    if _expanded_before(plain, acronym, len(plain)):
        return value, False, "unchanged"

    first_use = _first_bare_use(value, acronym)
    if first_use is None:
        return value, False, "unchanged"

    end = first_use + len(acronym)
    # Whatever wrapper surrounds the acronym (<strong>, <em>) stays: the replacement
    # goes inside it, and the closing tag is left exactly where it was. An earlier
    # version swallowed "</strong>" and published an unbalanced paragraph.
    return value[:first_use] + f"{expansion} ({acronym})" + value[end:], True, "inlined"
