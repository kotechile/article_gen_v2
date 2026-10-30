"""Keep the excerpt describing the article that was actually written.

The excerpt is written at outline time, from the brief, before a single section
exists. The finished piece then follows its own outline - different sections,
different evidence, sometimes a different argument - and the stored excerpt keeps
advertising the intention. WordPress prints it under the title, social cards reuse
it, and the reader who clicks through finds another article.

Two pieces live here:

* :func:`evaluate_excerpt_alignment` - a deterministic comparison of the excerpt
  against the text it will be published beside. Numbers are checked first (an
  excerpt quote the article dropped is the mismatch a reader notices), then the
  share of the excerpt's content words the article actually supports.
* :func:`derive_excerpt_from_article` / :func:`derive_excerpt_from_outline` -
  repairs built from real material (the article's own lede, or the planner's own
  key points), so a mismatch is never fixed by inventing a claim.

Callers decide what to do with a mismatch: the structure generator rewrites the
excerpt from the section plan, and finalization reconciles it against the finished
document. Neither path may answer the check by loosening the threshold.
"""

from __future__ import annotations

import html
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

DEFAULT_MIN_SUPPORT = 0.5

# Function words carry no theme, so counting them would make every excerpt look
# supported by every article.
_STOPWORDS = frozenset(
    """
    the and for that with this from your you are was were will have has had not but they
    them their there here what when which while into onto over under about after before
    because than then also just only very more most much many some such other each these
    those been being does did doing can could should would may might must its our ours who
    whom whose how why all any both every few own same too now out off down again further
    once thus therefore however per via upon also whose whom within without across during
    between among around toward towards toward still even ever never always often
    """.split()
)

# Framing words every standfirst uses, whatever the article says. Counting them as
# theme would let a brief-driven excerpt scrape past the support threshold on
# "guide", "practical" and "learn" alone.
_FRAMING_WORDS = frozenset(
    """
    guide guides practical actionable insight insights discover learn learning everything
    essential complete comprehensive article articles piece pieces explore explores
    exploring journey landscape real really actually today world future better best good
    great effective efficiently simply clear clearly deep dive look need needs reader
    readers understand understanding help helps start starting
    """.split()
)

_SUFFIXES = ("ingly", "ings", "ing", "ies", "ied", "ers", "est", "ed", "es", "ly", "s")

_TAG_RE = re.compile(r"<[^>]+>")
_PARAGRAPH_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.I | re.S)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CITATION_RE = re.compile(r"\[\^?\d{1,3}\]")
_LABEL_RE = re.compile(
    r"^(?:the\s+)?(?:big\s+picture|big\s+shift|bottom\s+line|why\s+it\s+matters|"
    r"the\s+catch|reality\s+check|by\s+the\s+numbers|go\s+deeper|at\s+a\s+glance)\s*:\s*",
    re.I,
)


def plain_text(value: str) -> str:
    """Tags out, entities decoded, whitespace collapsed."""
    text = _TAG_RE.sub(" ", value or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def content_terms(text: str) -> set:
    """Theme-bearing terms, crudely stemmed so 'retrieval' matches 'retrievals'."""
    words = re.findall(r"[A-Za-z][A-Za-z'\-]{3,}", plain_text(text))
    terms = set()
    for word in words:
        lowered = word.lower().strip("'-")
        if lowered.endswith("'s"):
            lowered = lowered[:-2]
        if len(lowered) < 4 or lowered in _STOPWORDS or lowered in _FRAMING_WORDS:
            continue
        terms.add(_stem(lowered))
    return terms


def excerpt_numbers(text: str) -> set:
    """Digits only, so '41%', '41 percent' and '41' compare equal."""
    numbers = set()
    for match in _NUMBER_RE.findall(plain_text(text)):
        numbers.add(match.replace(",", "").rstrip("."))
    return numbers


def evaluate_excerpt_alignment(
    excerpt: str,
    body: str,
    *,
    min_support: float = DEFAULT_MIN_SUPPORT,
) -> Dict[str, Any]:
    """Compare an excerpt with the text it will be published beside.

    ``matches`` requires both: every number the excerpt quotes exists in the body,
    and at least ``min_support`` of its content words are supported. A paraphrase
    passes; a claim about a section the article does not contain does not.
    """
    excerpt_text = (excerpt or "").strip()
    body_plain = plain_text(body or "")
    report: Dict[str, Any] = {
        "matches": False,
        "reason": "",
        "support": 0.0,
        "word_count": len(excerpt_text.split()),
        "terms": 0,
        "supported_terms": 0,
        "unsupported_terms": [],
        "missing_numbers": [],
    }
    if not excerpt_text:
        report["reason"] = "empty excerpt"
        return report
    if len(body_plain.split()) < 20:
        report["reason"] = "no body to compare against"
        return report

    excerpt_terms = content_terms(excerpt_text)
    body_terms = content_terms(body_plain)
    supported = excerpt_terms & body_terms
    unsupported = sorted(excerpt_terms - body_terms)
    support = (len(supported) / len(excerpt_terms)) if excerpt_terms else 1.0

    missing_numbers = sorted(excerpt_numbers(excerpt_text) - excerpt_numbers(body_plain))

    report.update(
        support=round(support, 3),
        terms=len(excerpt_terms),
        supported_terms=len(supported),
        unsupported_terms=unsupported[:12],
        missing_numbers=missing_numbers[:12],
    )

    if missing_numbers:
        report["reason"] = f"numbers not in the article: {missing_numbers[:6]}"
    elif support < min_support:
        report["reason"] = (
            f"only {round(support * 100)}% of the excerpt's terms appear in the article "
            f"(min {round(min_support * 100)}%): {unsupported[:8]}"
        )
    else:
        report["matches"] = True
        report["reason"] = "excerpt is supported by the article"
    return report


def _clean_sentence(sentence: str) -> str:
    text = _CITATION_RE.sub("", sentence or "")
    text = _LABEL_RE.sub("", text.strip())
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lstrip("—-–•*· ").strip()
    return text


def _trim(text: str, max_words: int, max_chars: int) -> str:
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(",;:")
        if not text.endswith((".", "!", "?")):
            text += "."
    if len(text) > max_chars:
        cut = text[:max_chars].rsplit(" ", 1)[0].rstrip(",;:")
        text = cut + "…" if not cut.endswith((".", "!", "?")) else cut
    return text.strip()


def _first_paragraph(body: str) -> str:
    if "<p" not in (body or "").lower():
        return plain_text(body or "")
    for match in _PARAGRAPH_RE.finditer(body or ""):
        candidate = match.group(1)
        text = plain_text(candidate)
        if len(text.split()) >= 8 and "reference" not in text.lower()[:40]:
            return text
    return plain_text(body or "")


def derive_excerpt_from_article(
    body: str,
    *,
    max_words: int = 70,
    max_chars: int = 300,
) -> str:
    """An excerpt cut from the article's own lede.

    Every word here is published material, which is the point: the repair for a
    mismatched excerpt is not a better guess, it is text the article actually
    contains.
    """
    paragraph = _first_paragraph(body or "")
    if not paragraph:
        return ""
    sentences = [_clean_sentence(s) for s in _SENTENCE_RE.split(paragraph)]
    sentences = [s for s in sentences if s and len(s.split()) >= 4]
    if not sentences:
        return ""
    picked = sentences[0]
    for sentence in sentences[1:]:
        if len(picked.split()) >= 25:
            break
        picked = f"{picked} {sentence}"
    return _trim(picked, max_words, max_chars)


def _field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def derive_excerpt_from_outline(
    sections: Optional[Sequence[Any]],
    *,
    max_chars: int = 300,
    min_chars: int = 150,
) -> str:
    """An excerpt built from the outline's own key points.

    Used when the excerpt written from the brief does not match the plan and the
    model's second attempt does not either. Key points are the planner's claims
    about the sections it just laid out, so the result promises what the article
    was actually planned to deliver. Returns "" when the outline carries no usable
    material rather than composing a sentence that says nothing.
    """
    picked: List[str] = []
    for section in sections or []:
        for point in _field(section, "key_points") or []:
            text = _clean_sentence(str(point))
            if len(text.split()) < 6:
                continue
            if not text.endswith((".", "!", "?")):
                text += "."
            picked.append(text)
            if len(" ".join(picked)) >= min_chars:
                break
        if len(" ".join(picked)) >= min_chars:
            break
    if not picked:
        return ""
    return _trim(" ".join(picked), max_words=80, max_chars=max_chars)
