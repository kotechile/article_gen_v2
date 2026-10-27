"""Numeric claim grounding.

The article generator must not publish a number it cannot trace back to a
retrieved source. Prose is checked by editors; numbers are the part readers and
reviewers check first, and they are the easiest thing for a language model to
invent with complete confidence.

This module is deliberately deterministic - no LLM in the loop:

1. ``build_evidence_corpus`` collects every piece of retained evidence (research
   dossier statistics, RAG passages, prior citations, claims) into normalised
   items that know their own URL.
2. ``extract_number_claims`` finds numeric claims in the article HTML, together
   with the sentence that carries them (the *subject*) and the unit class the
   number is expressed in.
3. ``verify_claim`` accepts a claim only when the same value appears in an
   evidence item **in a compatible unit** *and* the surrounding wording overlaps
   the evidence. Matching on the value alone is not enough: "8" is everywhere.
4. ``render_numbers_block`` produces a Smart Brevity "By the numbers" list from
   grounded claims only, each with a citation marker pointing at the evidence
   that supports it.

The contract is fail-closed: no evidence, no number. A thin block (or no block
at all) is the correct output when the research did not support the claims -
inventing a plausible bullet is never an option.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# numeric vocabulary
# --------------------------------------------------------------------------- #
_UNIT_PATTERNS: Dict[str, re.Pattern] = {
    "currency": re.compile(r"(\$|\busd\b|\bdollars?\b|\beur\b|\bgbp\b)", re.IGNORECASE),
    "percent": re.compile(r"(%)|(\bper\s?cent\b)|(\bpercentage\s+points?\b)|(\bpp\b)", re.IGNORECASE),
    "magnitude": re.compile(r"(\bbillion\b|\bmillion\b|\btrillion\b|\bbn\b|(?<=\d)\s*[bmk]\b)", re.IGNORECASE),
    "time": re.compile(r"(\bms\b|\bmilliseconds?\b|\bseconds?\b|\bsecs?\b|\bminutes?\b|\bhours?\b)", re.IGNORECASE),
    "multiple": re.compile(r"(?<=\d)\s*x\b", re.IGNORECASE),
    "count": re.compile(r"(\bqueries\b|\btransactions?\b|\brequests?\b|\bcalls?\b|\busers?\b|\btokens?\b)", re.IGNORECASE),
}

_NUMBER_RE = re.compile(
    r"(?<![\w.])"
    r"(?P<prefix>\$)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s*(?:to|through|-|–|—)\s*(?P<prefix2>\$)?(?P<num2>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?))?"
    r"(?P<suffix>%|x|X|pp|k|K|m|M|b|B)?"
    r"(?![\w])"
)

_CITATION_MARKER_RE = re.compile(r"\[\^?(\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of to in on for with without by as at from
    is are was were be been being it its their there here we you they he she i not no more most less
    much many very can could should would may might will do does did have has had into over under
    about across also between such when where which who whom while because so up out per each
    """.split()
)

# Units that must agree with the evidence for a claim to count as grounded.
_UNIT_CLASSES = ("currency", "percent", "magnitude", "time", "multiple")


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
@dataclass
class EvidenceItem:
    """A single retained evidence item and the text it supports."""

    id: int  # 1-based; matches the citation marker ``[id]``
    text: str
    title: str = ""
    url: str = ""
    source_type: str = ""
    numbers: Dict[str, set] = field(default_factory=dict)  # canonical value -> unit classes

    @property
    def citable(self) -> bool:
        return bool(self.url and self.url.startswith(("http://", "https://")))


@dataclass
class NumberClaim:
    """A numeric claim plus the wording that gives it meaning."""

    raw: str
    values: Tuple[str, ...]
    unit: str
    sentence: str
    label: str
    markers: Tuple[int, ...] = ()

    def key(self) -> Tuple[str, str]:
        return ("|".join(self.values), self.label.lower()[:60])


@dataclass
class GroundingResult:
    claim: NumberClaim
    grounded: bool
    evidence: Optional[EvidenceItem] = None
    reason: str = ""
    matched_snippet: str = ""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def strip_html(text: str) -> str:
    """Collapse HTML to plain text (tags removed, whitespace normalised)."""
    if not text:
        return ""
    no_tags = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    no_tags = re.sub(r"<[^>]+>", " ", no_tags)
    return re.sub(r"\s+", " ", no_tags).strip()


def strip_numbers_block(html: str) -> str:
    """Remove an existing "By the numbers" block.

    The block is rebuilt from grounded body prose, so whatever the writer model
    put there (typically numbers scraped out of sentences and stripped of their
    subject) is discarded rather than trusted.
    """
    if not html:
        return html
    pattern = re.compile(
        r"<p>\s*(?:<strong>|<b>)?\s*By\s+the\s+numbers:?\s*(?:</strong>|</b>)?\s*</p>\s*"
        r"(?:<ul>.*?</ul>|<ol>.*?</ol>)?",
        re.IGNORECASE | re.DOTALL,
    )
    return pattern.sub("", html)


def canonical_number(token: str) -> str:
    """Canonicalise a number token: ``$1,500`` -> ``1500``, ``0.85`` -> ``0.85``."""
    cleaned = token.replace(",", "").replace("$", "").strip()
    if not cleaned:
        return ""
    try:
        if "." in cleaned:
            return str(float(cleaned)).rstrip("0").rstrip(".") or "0"
        return str(int(cleaned))
    except ValueError:
        return cleaned


def _unit_window(sentence: str, match: "re.Match") -> str:
    """Text that legitimately belongs to one number.

    Scanning a fixed number of characters around a match crosses into the *next*
    number's unit, so "$15,000 in expenses and a 49% increase" would label the
    dollar amount as a percentage. Stop at the neighbouring number - including
    its currency sign, which sits just before its digits.
    """
    number_token = re.compile(r"[$€£]?\d[\d,.]*")

    before = sentence[max(0, match.start() - 20): match.start()]
    previous = None
    for previous in number_token.finditer(before):
        pass
    if previous:
        before = before[previous.end():]

    tail = sentence[match.end(): match.end() + 40]
    following = number_token.search(tail)
    if following:
        tail = tail[: following.start()]

    return f"{before}{sentence[match.start(): match.end()]}{tail}"


def _unit_class(window: str) -> str:
    """Best-effort unit classification for the text around a number."""
    for unit, pattern in _UNIT_PATTERNS.items():
        if pattern.search(window):
            return unit
    return ""


def _content_words(text: str) -> set:
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", text.lower())
    return {w for w in words if len(w) > 3 and w not in _STOPWORDS}


def _numbers_in(text: str) -> Dict[str, set]:
    """Map canonical value -> set of unit classes found with it."""
    found: Dict[str, set] = {}
    for match in _NUMBER_RE.finditer(text):
        window_start = max(0, match.start() - 30)
        window_end = min(len(text), match.end() + 30)
        window = text[window_start:window_end]
        unit = _unit_class(window)
        values = [canonical_number(match.group("num"))]
        if match.group("num2"):
            values.append(canonical_number(match.group("num2")))
        for value in values:
            if value:
                found.setdefault(value, set()).add(unit)
    return found


# --------------------------------------------------------------------------- #
# evidence corpus
# --------------------------------------------------------------------------- #
_TEXT_KEYS = ("content", "content_excerpt", "snippet", "excerpt", "summary", "claim", "text",
              "statement", "passage", "chunk", "body", "description")
_URL_KEYS = ("url", "source", "link", "source_url", "canonical_url")
_TITLE_KEYS = ("title", "source_title", "name", "heading")


def _evidence_from_dict(item: Dict[str, Any], index: int) -> Optional[EvidenceItem]:
    raw_metadata = item.get("metadata")
    metadata: Dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    text = ""
    for key in _TEXT_KEYS:
        value = item.get(key)
        if isinstance(value, str) and len(value.strip()) >= 20:
            text = value.strip()
            break
    if not text:
        for key in _TEXT_KEYS:
            value = metadata.get(key)
            if isinstance(value, str) and len(value.strip()) >= 20:
                text = value.strip()
                break
    if not text:
        return None

    url = ""
    for key in _URL_KEYS:
        candidate = item.get(key) or metadata.get(key)
        if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
            url = candidate
            break

    title = ""
    for key in _TITLE_KEYS:
        candidate = item.get(key) or metadata.get(key)
        if isinstance(candidate, str) and candidate.strip():
            title = candidate.strip()
            break

    return EvidenceItem(
        id=index,
        text=text,
        title=title,
        url=url,
        source_type=str(item.get("source_type") or metadata.get("source_type") or ""),
        numbers=_numbers_in(text),
    )


def build_evidence_corpus(research_data: Optional[Dict[str, Any]]) -> List[EvidenceItem]:
    """Collect every citable/grounding piece of evidence available to the writer.

    Walks the research payload rather than one rigid schema: dossier statistics,
    RAG passages, live-web results and claims all end up in the same corpus, and
    duplicates (same URL or same leading text) are dropped so citation markers
    stay meaningful.
    """
    corpus: List[EvidenceItem] = []
    if not isinstance(research_data, dict):
        return corpus

    candidates: List[Dict[str, Any]] = []

    def _collect(value: Any, depth: int = 0) -> None:
        if depth > 6 or value is None:
            return
        if isinstance(value, dict):
            if any(isinstance(value.get(k), str) for k in _TEXT_KEYS):
                candidates.append(value)
            for nested in value.values():
                if isinstance(nested, (dict, list)):
                    _collect(nested, depth + 1)
        elif isinstance(value, (list, tuple)):
            for entry in value:
                _collect(entry, depth + 1)

    dossier_keys = (
        "research_dossier", "evidence", "sources", "citations", "prior_citations", "claims",
        "rag_chunks", "rag_results", "live_web_results", "web_results", "deep_research",
        "research_sources", "key_statistics", "important_statistics",
    )
    for key in dossier_keys:
        if key in research_data:
            _collect(research_data[key])

    seen_urls: set = set()
    seen_text: set = set()
    for candidate in candidates:
        item = _evidence_from_dict(candidate, len(corpus) + 1)
        if not item:
            continue
        url_key = item.url.rstrip("/").lower()
        text_key = re.sub(r"[^a-z0-9]", "", item.text.lower()[:120])
        if url_key and url_key in seen_urls:
            continue
        if text_key and text_key in seen_text:
            continue
        if url_key:
            seen_urls.add(url_key)
        if text_key:
            seen_text.add(text_key)
        corpus.append(item)

    return corpus


# --------------------------------------------------------------------------- #
# claim extraction
# --------------------------------------------------------------------------- #
_LABEL_LEADING = frozenset(
    "while whereas and but or so then also yet in on at for with by from that this it its we you they "
    "a an the meanwhile however still even".split()
)
_LABEL_TRAILING = frozenset(
    "by to of at in from than approximately about around down up out over under and or the a an is are "
    "was were be completes reduces improves increases pushes adds costs consumes raises cuts drops grows "
    "reaches delivers brings versus vs".split()
)


def _trim_label(words: List[str]) -> List[str]:
    words = words[-8:]
    while words and re.sub(r"[^\w%-]", "", words[0]).lower() in _LABEL_LEADING:
        words.pop(0)
    while words and re.sub(r"[^\w%-]", "", words[-1]).lower() in _LABEL_TRAILING:
        words.pop()
    return words


def _label_for(sentence: str, match: re.Match) -> str:
    """Derive the *subject* a number refers to.

    A number without a subject is not a fact, it is noise - "0.85: before
    proceeding" is exactly what happens when the subject is dropped. Sources, in
    order:

    1. an explicit ``Label:`` prefix written by the author ("Empirical Hit-Rate
       Delta: Dynamic task decomposition improves…"), which is already the best
       possible label;
    2. the nearest noun-ish phrase before the number, trimmed of connectives and
       trailing verbs/prepositions ("…reduces per-query orchestration expenses
       from ~$0.09" -> "Per-query orchestration expenses");
    3. when the number opens a clause and there is nothing usable before it, the
       words that follow it ("Around 49% of relocators face…" -> "Relocators face
       unexpected living costs").

    A label that contains another number is not a subject, it is leftover
    context, so it is discarded rather than published.
    """
    stripped = sentence.strip()
    prefix = re.match(r"^([A-Za-z0-9][^:]{2,48}):\s", stripped)
    if prefix and prefix.end() <= match.start() and 1 <= len(prefix.group(1).split()) <= 8:
        return prefix.group(1).strip()

    before_words = sentence[: match.start()].strip(" ,;:-–—").split()
    # If the preceding clause contains another number, the subject starts after
    # it: "$14,289" preceded by "49% of workers struggle with rent, while" has
    # the subject "workers struggle with rent", not the whole clause.
    last_number_word = None
    for index, word in enumerate(before_words):
        if re.search(r"\d", word):
            last_number_word = index
    if last_number_word is not None:
        before_words = before_words[last_number_word + 1:]

    words = _trim_label(before_words)
    if len(words) >= 2 and not any(re.search(r"\d", word) for word in words):
        return " ".join(words).strip()

    after_words = sentence[match.end():].strip(" ,;:-–—").split()
    after_words = _trim_label(after_words)
    if any(re.search(r"\d", word) for word in after_words):
        # Skip past the stray number and try the clause that follows it.
        last_number_word = None
        for index, word in enumerate(after_words):
            if re.search(r"\d", word):
                last_number_word = index
        after_words = _trim_label(after_words[(last_number_word or 0) + 1:])
    if len(after_words) >= 2 and not any(re.search(r"\d", word) for word in after_words):
        return " ".join(after_words).strip()

    return " ".join(words).strip() if len(words) >= 2 else ""


def _expression_for(sentence: str, match: re.Match) -> str:
    """Return the number plus the unit words that belong to it.

    ``15 to 30`` alone is not publishable; ``15 to 30 percentage points`` is.
    Units are looked for in a window wide enough to include multi-word forms
    ("percentage points", "per transaction", "tail p95").
    """
    raw = re.sub(r"\s+", " ", sentence[match.start(): match.end()]).strip()

    # A move between two values is one fact, not two: "from $0.09 to $0.01 per
    # transaction" reads as a saving, "$0.09" alone reads as a cost.
    range_match = re.search(
        r"from\s+(?:approximately\s+|about\s+|around\s+|nearly\s+)?(\$?\d[\d,.]*)\s*(?:down\s+|up\s+)?to\s+(\$?\d[\d,.]*)\s*([%xX]|\bpercentage points?\b)?",
        sentence,
        re.IGNORECASE,
    )
    if range_match and range_match.start() <= match.start() and range_match.end() >= match.end():
        raw = f"from {range_match.group(1)} to {range_match.group(2)}"
        if range_match.group(3):
            raw = f"{raw} {range_match.group(3)}"
        trailing = re.match(r"\s*(per\s+\w+|a\s+\w+|each\s+\w+)", sentence[range_match.end(): range_match.end() + 30])
        if trailing:
            raw = f"{raw} {trailing.group(1)}"
        return raw.strip()

    window = _unit_window(sentence, match)
    lowered = window.lower()

    unit = ""
    if re.search(r"percentage\s+points?", lowered):
        unit = "percentage points"
    elif "%" in window or re.search(r"\bper\s?cent\b", lowered):
        unit = "%"
    elif re.search(r"\b(billion|million|trillion)\b", lowered):
        found = re.search(r"(billion|million|trillion)", lowered)
        unit = found.group(1) if found else ""
    elif re.search(r"\b(ms|milliseconds?|seconds?|secs?|minutes?|hours?)\b", lowered):
        found = re.search(r"(ms|milliseconds?|seconds?|secs?|minutes?|hours?)", lowered)
        unit = found.group(1) if found else ""
    elif re.search(r"\d\s*x\b", lowered):
        unit = "x"

    if unit and unit.lower() not in raw.lower():
        raw = f"{raw} {unit}"

    trailing = re.match(r"\s*(per\s+\w+|a\s+\w+|each\s+\w+)", sentence[match.end(): match.end() + 30])
    if trailing and trailing.group(1).lower() not in raw.lower():
        raw = f"{raw} {trailing.group(1)}"
    return raw.strip()


def extract_number_claims(html_or_text: str, *, min_value_words: int = 0) -> List[NumberClaim]:
    """Extract numeric claims from article HTML/text, in document order.

    Any existing "By the numbers" block is removed first: its bullets are
    already formatted (and usually already decontextualised), so re-reading them
    would double-count claims and inherit the defect we are fixing.
    """
    text = strip_html(strip_numbers_block(html_or_text))
    if not text:
        return []

    claims: List[NumberClaim] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if len(sentence) < 25:
            continue
        markers = tuple(int(m) for m in _CITATION_MARKER_RE.findall(sentence))
        clean_sentence = _CITATION_MARKER_RE.sub("", sentence)
        for match in _NUMBER_RE.finditer(clean_sentence):
            # Citation markers, years and bare ordinals are not claims.
            token = match.group("num").replace(",", "")
            if not match.group("prefix") and token.isdigit() and 1900 <= int(token) <= 2100 and not match.group("suffix"):
                continue
            values = [canonical_number(match.group("num"))]
            if match.group("num2"):
                values.append(canonical_number(match.group("num2")))
            values = [v for v in values if v]
            if not values:
                continue
            label = _label_for(clean_sentence, match)
            if len(label.split()) < min_value_words:
                continue
            claims.append(
                NumberClaim(
                    raw=_expression_for(clean_sentence, match),
                    values=tuple(dict.fromkeys(values)),
                    unit=_unit_class(_unit_window(clean_sentence, match)),
                    sentence=sentence,
                    label=label,
                    markers=markers,
                )
            )
    return claims


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def _snippet(item: EvidenceItem, values: Sequence[str], width: int = 160) -> str:
    for value in values:
        match = re.search(rf"(?<![\w.]){re.escape(value)}(?:[\d.,]*)", item.text.replace(",", ""))
        if match:
            start = max(0, match.start() - width // 2)
            return item.text[start:start + width].strip()
    return item.text[:width].strip()


def verify_claim(claim: NumberClaim, corpus: Sequence[EvidenceItem], *, min_overlap: int = 2) -> GroundingResult:
    """Check one claim against the corpus. Fail-closed on any doubt."""
    if not corpus:
        return GroundingResult(claim, False, reason="no_evidence")

    if not claim.label or len(claim.label.split()) < 2:
        return GroundingResult(claim, False, reason="no_subject")

    claim_words = _content_words(claim.sentence)
    best: Optional[Tuple[int, EvidenceItem]] = None

    for item in corpus:
        if not all(value in item.numbers for value in claim.values):
            continue
        # Unit compatibility: a "$0.01" claim is not grounded by "0.01%" evidence.
        if claim.unit in _UNIT_CLASSES:
            units_for_values = [item.numbers.get(v, set()) for v in claim.values]
            if not all((claim.unit in units) or ("" in units) for units in units_for_values):
                continue
        overlap = len(claim_words & _content_words(item.text))
        if overlap < min_overlap:
            continue
        if best is None or overlap > best[0]:
            best = (overlap, item)

    if best is None:
        return GroundingResult(claim, False, reason="no_matching_evidence")

    overlap, item = best
    return GroundingResult(
        claim,
        True,
        evidence=item,
        reason=f"overlap={overlap}",
        matched_snippet=_snippet(item, claim.values),
    )


def ground_claims(claims: Iterable[NumberClaim], corpus: Sequence[EvidenceItem]) -> List[GroundingResult]:
    """Verify a batch of claims, preserving order."""
    return [verify_claim(claim, corpus) for claim in claims]


def audit_numbers(html_or_text: str, corpus: Sequence[EvidenceItem], *, limit: int = 50) -> Dict[str, Any]:
    """Report which numeric claims in a finished article are unbacked."""
    claims = extract_number_claims(html_or_text)
    results = ground_claims(claims, corpus)
    ungrounded = [r for r in results if not r.grounded]
    return {
        "claims_checked": len(results),
        "grounded": len(results) - len(ungrounded),
        "ungrounded": len(ungrounded),
        "evidence_items": len(corpus),
        "ungrounded_samples": [
            {"claim": r.claim.raw, "subject": r.claim.label, "reason": r.reason}
            for r in ungrounded[:limit]
        ],
    }


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _clean_label(label: str) -> str:
    label = re.sub(r"\s+", " ", label).strip(" .,;:-–—")
    label = re.sub(r"^(the|a|an)\s+", "", label, flags=re.IGNORECASE)
    if not label:
        return ""
    # Prefer the head of a long clause: "Dynamic task decomposition improves retrieval accuracy by"
    label = re.sub(r"\s+(by|to|of|at|from|in)$", "", label, flags=re.IGNORECASE)
    return label[0].upper() + label[1:]


def _source_label(item: EvidenceItem) -> str:
    """Short, readable label for the source behind a number."""
    if item.title:
        label = item.title.strip()
        if len(label) > 48:
            label = label[:45].rstrip() + "…"
        return label
    host = re.sub(r"^https?://(www\.)?", "", item.url).split("/")[0]
    return host or "source"


def render_numbers_block(
    html_or_text: str,
    corpus: Sequence[EvidenceItem],
    *,
    max_items: int = 5,
    min_items: int = 3,
) -> Tuple[str, Dict[str, Any]]:
    """Build a "By the numbers" list from grounded claims only.

    Each bullet carries a real link to the evidence that supports it - not a
    ``[n]`` marker, which depends on reference-list ordering that is recomputed
    later in the pipeline. A number you can click is a number you can check.

    Returns ``(html, report)``. ``html`` is '' when fewer than ``min_items``
    claims could be grounded - the correct output when the research did not
    support three verifiable numbers.
    """
    claims = extract_number_claims(strip_numbers_block(html_or_text))
    results = ground_claims(claims, corpus)

    items: List[str] = []
    seen: set = set()
    seen_pairs: set = set()
    for result in results:
        if not result.grounded or result.evidence is None:
            continue
        if not result.evidence.citable:
            continue  # no URL -> not a reference we can publish
        label = _clean_label(result.claim.label)
        if len(label.split()) < 2:
            continue
        # One bullet per *subject* per sentence: "from $0.09 down to $0.01 per
        # transaction" is a single fact, but "$15,000 in moving expenses and a
        # 49% cost increase" is genuinely two.
        sentence_key = re.sub(r"[^a-z0-9]", "", result.claim.sentence.lower())[:120]
        subject_key = (sentence_key, " ".join(label.lower().split()[:3]))
        if subject_key in seen_pairs:
            continue
        key = (label.lower(), "|".join(result.claim.values))
        if key in seen:
            continue
        seen.add(key)
        seen_pairs.add(subject_key)
        source = (
            f'<a href="{result.evidence.url}" target="_blank" rel="noopener noreferrer">'
            f"{_source_label(result.evidence)}</a>"
        )
        items.append(f"<strong>{label}:</strong> {result.claim.raw} ({source})")
        if len(items) >= max_items:
            break

    report = {
        "claims_checked": len(results),
        "grounded": sum(1 for r in results if r.grounded),
        "items_rendered": len(items),
        "evidence_items": len(corpus),
        "rejected": [
            {"claim": r.claim.raw, "subject": r.claim.label, "reason": r.reason}
            for r in results
            if not r.grounded
        ][:20],
    }

    if len(items) < min_items:
        report["skipped"] = True
        report["reason"] = (
            "fewer than %d verifiable numbers in the evidence corpus" % min_items
            if corpus
            else "no evidence corpus available"
        )
        return "", report

    body = "\n".join(f"  <li>{item}</li>" for item in items)
    html = f"<p><strong>By the numbers:</strong></p>\n<ul>\n{body}\n</ul>"
    return html, report
