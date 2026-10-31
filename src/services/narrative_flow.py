"""Measure whether an article's paragraphs carry a story, or just sit next to each other.

Per-section drafting writes one call per section, so no stage ever sees two
paragraphs together until the document editor runs. The failure that produces is
a body of paragraphs that each restate a mechanism without ever taking the
reader forward:

    The payoff is measurable. Teams that enable these controls typically report
    a 4 to 6 percentage point improvement ...

That opening sentence announces a *category* of claim ("the payoff", "the
result", "the lesson") and states nothing; the reader has to reach the next
sentence to learn what the paragraph is about. The same document usually opens
other paragraphs on a deictic that points at something the previous paragraph
never supplied ("Product Allocation closes that window."), names abstractions
where it could name actions ("first-attempt completeness standard",
"cannibalization stops being a behaviour problem"), and drops every concrete
anchor after the first paragraph.

Three shapes are mechanically checkable, and they separate a flat draft from a
rewritten one on the pair that motivated this module:

* ``meta_frame_openers`` - a paragraph whose first sentence is a frame around a
  claim rather than the claim (``The payoff is measurable.``).
* ``anchorless_paragraphs`` - a paragraph with no number, time, named system,
  acronym or quoted title anywhere in it. Nothing pins it to the story.
* ``nominalisation_density`` - nominalisations per 100 words. A paragraph that
  names *things* (``attribution``, ``completeness``, ``confirmation``,
  ``intervention``) where it could name *actions* reads abstract even when the
  sentences are grammatical. The flat draft measured 5.7/100 against 3.0/100
  for the rewrite.

These are diagnostics, not a style score: a paragraph can be concrete and still
be dull. They exist so the whole-document editor's flow pass can be judged by a
number on either side of the edit instead of by eye, and so a pass that turns
prose into labelled notes is rejected rather than published.

CLI (reads HTML or plain text from a file or stdin)::

    python3 -m src.services.narrative_flow article.html
"""

from __future__ import annotations

import re
import sys
from typing import Any, Dict, List

_CITATION_RE = re.compile(r"\[\d{1,3}\]")
_TAG_RE = re.compile(r"<[^>]+>")
_BLOCK_RE = re.compile(r"<(p|li)\b[^>]*>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")

# A paragraph opener that names the *slot* its claim will fill instead of making
# the claim. The list is deliberately small and only matches at the start of a
# paragraph: "The difference is rarely effort; it is structure" is a fine closing
# line and a poor opening one, which is exactly the distinction being measured.
_META_FRAME_RE = re.compile(
    r"^\s*(?:and\s+|but\s+|so\s+)?(?:the\s+|its\s+|their\s+|this\s+|that\s+)?"
    r"(?:payoff|result|results|answer|key|lesson|lessons|takeaway|takeaways|"
    r"bottom\s+line|difference|point|points|reason|reasons|solution|problem|"
    r"upshot|benefit|benefits|catch|verdict|reality|caveat|thing|things|"
    r"story|case|idea|rule)\s+"
    r"(?:is|are|was|were|lies|lie|remains|remain|comes|come|starts|begins|"
    r"sounds|looks|seems)\b",
    re.IGNORECASE,
)

_NOMINALISATION_RE = re.compile(
    r"\b[a-z]{4,}(?:tion|tion's|sion|ment|ments|ance|ence|ity|ities|ness|"
    r"isation|ization|ship)\b"
)

# Anything that pins a paragraph to the world the article describes: a quantity,
# a clock time, a weekday, an acronym or ticker-shaped token, a model/version
# string, or a Title Case pair (S/4HANA, Case Fill Rate).
_ANCHOR_RE = re.compile(
    r"(?:\b\d[\d,.]*\s*(?:%|percent|percentage|words?|days?|weeks?|months?|years?|"
    r"hours?|minutes?|seconds?|million|billion|thousand|k\b|x\b))"
    r"|(?:\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b)"
    r"|(?:\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b)"
    r"|(?:\b[A-Z]{2,6}\b)"
    r"|(?:\b[A-Z][a-z]+(?:[/-][A-Z0-9][A-Za-z0-9]*)+)"
    r"|(?:\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b)"
    r"|(?:\b[A-Z][a-z]*\d[A-Za-z0-9]*\b)",
)

_LABEL_OPENER_RE = re.compile(r"^\s*<(?:strong|b)>\s*[^<]{1,80}?(?::|</(?:strong|b)>)", re.IGNORECASE)


def _plain(fragment: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", fragment or "")).strip()


def _blocks(html: str) -> List[Dict[str, str]]:
    """Paragraph-level blocks in document order.

    ``<li>`` items are included because a body that is really a list of labelled
    notes is the shape this module exists to catch; they are tagged so a caller
    can decide whether list items count. Headings are skipped - a heading is a
    label by definition.
    """
    blocks: List[Dict[str, str]] = []
    if "<p" not in (html or "").lower() and "<li" not in (html or "").lower():
        # Plain text: split on blank lines so the check works on a pasted draft.
        for chunk in re.split(r"\n\s*\n", html or ""):
            text = _plain(chunk)
            if text:
                blocks.append({"kind": "p", "html": chunk.strip(), "text": text})
        return blocks
    for match in _BLOCK_RE.finditer(html or ""):
        text = _plain(match.group(2))
        if text:
            blocks.append({"kind": match.group(1).lower(), "html": match.group(2), "text": text})
    return blocks


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split((text or "").strip()) if s.strip()]


def _words(text: str) -> List[str]:
    return _WORD_RE.findall(_CITATION_RE.sub(" ", text or ""))


def _count_nominalisations(text: str) -> int:
    return len(_NOMINALISATION_RE.findall((text or "").lower()))


def _meta_frame_opener(first_sentence: str) -> bool:
    return bool(_META_FRAME_RE.match(first_sentence or ""))


def analyze_narrative_flow(html: str) -> Dict[str, Any]:
    """Report per-paragraph story shape and the three aggregate measures.

    Returns a dict with ``paragraphs`` (one entry per body block), the aggregate
    counts, and ``reasons`` naming each shape that fired. ``story_score`` is the
    share of paragraphs that both open on a claim and carry a concrete anchor -
    the single number to read on either side of an edit.
    """
    blocks = [b for b in _blocks(html) if b["kind"] == "p"]
    paragraphs: List[Dict[str, Any]] = []
    meta_frames: List[Dict[str, Any]] = []
    anchorless: List[Dict[str, Any]] = []
    total_words = 0
    total_nominals = 0

    for index, block in enumerate(blocks):
        text = block["text"]
        sentences = _sentences(text)
        if not sentences:
            continue
        first = sentences[0]
        words = _words(text)
        word_count = len(words)
        nominals = _count_nominalisations(text)
        total_words += word_count
        total_nominals += nominals

        # Only a bolded/strong label counts. A colon inside a written sentence
        # ("The ROI is structural and sustained: organizations gain 4 to 6 points")
        # is prose doing its job, and flagging it would fail the rewrite this
        # module exists to approve.
        label_opener = bool(_LABEL_OPENER_RE.match(block["html"]))
        meta_frame = _meta_frame_opener(first) and not label_opener
        anchor = bool(_ANCHOR_RE.search(text))
        opens_on_claim = not meta_frame and not label_opener

        entry = {
            "index": index,
            "words": word_count,
            "first_sentence": first,
            "first_sentence_words": len(first.split()),
            "nominals": nominals,
            "anchor": anchor,
            "meta_frame": meta_frame,
            "label_opener": label_opener,
            "opens_on_claim": opens_on_claim,
        }
        paragraphs.append(entry)
        if meta_frame:
            meta_frames.append(entry)
        if not anchor:
            anchorless.append(entry)

    paragraph_count = len(paragraphs)
    density = (total_nominals / total_words * 100) if total_words else 0.0
    scored = [p for p in paragraphs if p["opens_on_claim"] and p["anchor"]]
    story_score = (len(scored) / paragraph_count) if paragraph_count else 1.0

    reasons: List[str] = []
    if meta_frames:
        reasons.append(
            "%d paragraph(s) open on a frame instead of a claim: %s"
            % (
                len(meta_frames),
                "; ".join('"%s"' % p["first_sentence"][:70] for p in meta_frames[:3]),
            )
        )
    if paragraph_count and len(anchorless) / paragraph_count > 0.34:
        reasons.append(
            "%d of %d paragraph(s) carry no concrete anchor (number, time, system, name)"
            % (len(anchorless), paragraph_count)
        )
    # Calibrated on the flat/story pair: 5.7/100 against 4.8/100 on a piece whose
    # subject matter *is* nominalisations ("Product Allocation", "Backorder
    # Processing", "confirmation"). Only extreme abstraction fires on its own;
    # the useful signal is the delta across an edit (see compare_flow).
    if density > 6.0:
        reasons.append(
            "nominalisation density %.1f per 100 words - the prose names things where it "
            "could name actions" % density
        )

    return {
        "paragraphs": paragraphs,
        "paragraph_count": paragraph_count,
        "meta_frame_openers": len(meta_frames),
        "anchorless_paragraphs": len(anchorless),
        "nominalisation_density": round(density, 2),
        "words": total_words,
        "story_score": round(story_score, 3),
        "reasons": reasons,
    }


def flow_evidence(report: Dict[str, Any], limit: int = 4) -> List[str]:
    """The failing paragraphs, quoted, for a retry prompt.

    A model told "three paragraphs open on a frame" restores nothing; told which
    paragraphs and what their first sentence says, it knows what to rewrite.
    """
    evidence: List[str] = []
    for paragraph in report.get("paragraphs", []):
        problem = None
        if paragraph.get("meta_frame"):
            problem = "opens by announcing the claim instead of making it"
        elif paragraph.get("label_opener"):
            problem = "opens on a bolded label rather than a sentence"
        elif not paragraph.get("anchor"):
            problem = "carries no concrete detail anywhere"
        if problem:
            evidence.append(
                "paragraph %d — %s: \"%s\"" % (paragraph["index"] + 1, problem, paragraph["first_sentence"])
            )
        if len(evidence) >= limit:
            break
    return evidence


def compare_flow(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Did an editing pass improve the flow, leave it alone, or flatten it?

    A flow pass is allowed to change wording freely, so the guard has to compare
    measurable shape: more frame-openers, fewer paragraphs (merging two ideas
    into one hides the seam rather than fixing it), more abstraction, or fewer
    anchored paragraphs all mean the edit read the instruction and missed it.
    """
    deltas = {
        "paragraphs": after.get("paragraph_count", 0) - before.get("paragraph_count", 0),
        "meta_frame_openers": after.get("meta_frame_openers", 0) - before.get("meta_frame_openers", 0),
        "anchorless_paragraphs": after.get("anchorless_paragraphs", 0)
        - before.get("anchorless_paragraphs", 0),
        "nominalisation_density": round(
            after.get("nominalisation_density", 0.0) - before.get("nominalisation_density", 0.0), 2
        ),
        "story_score": round(after.get("story_score", 0.0) - before.get("story_score", 0.0), 3),
    }
    regressions: List[str] = []
    if deltas["meta_frame_openers"] > 0:
        regressions.append("%+d frame-opener(s)" % deltas["meta_frame_openers"])
    if deltas["anchorless_paragraphs"] > 0:
        regressions.append("%+d anchorless paragraph(s)" % deltas["anchorless_paragraphs"])
    if deltas["nominalisation_density"] > 1.5:
        regressions.append("nominalisation density %+.2f per 100 words" % deltas["nominalisation_density"])
    if deltas["paragraphs"] < 0:
        regressions.append("%+d paragraph(s) - ideas merged instead of joined" % deltas["paragraphs"])
    return {"deltas": deltas, "regressed": bool(regressions), "regressions": regressions}


def _main(argv: List[str]) -> int:
    if len(argv) > 1:
        with open(argv[1], "r", encoding="utf-8") as handle:
            payload = handle.read()
    else:
        payload = sys.stdin.read()
    report = analyze_narrative_flow(payload)
    print(
        "paragraphs=%s meta_frame_openers=%s anchorless=%s nominalisation_density=%s story_score=%s"
        % (
            report["paragraph_count"],
            report["meta_frame_openers"],
            report["anchorless_paragraphs"],
            report["nominalisation_density"],
            report["story_score"],
        )
    )
    for reason in report["reasons"]:
        print("  - %s" % reason)
    for line in flow_evidence(report, limit=8):
        print("  * %s" % line)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(_main(sys.argv))
