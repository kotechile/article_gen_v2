"""Guards for LLM editing passes: leaked reasoning and catastrophic shrinkage.

Two failures showed up in production on a long-form article:

1. The polishing pass returned 1,085 characters from a 20,445-character draft -
   95% of the article deleted - and the pipeline published the remainder,
   because nothing compared input to output.
2. The output began with the model's own deliberation:

       . Wait, let's check the rule: "Do not restate the same statistic in more
       than one section." ... Notice: It says every single concept ... must be
       developed in the body. If a statistic is mentioned in the FAQ, but the
       rule says ...

   A reasoning model that is unhappy about conflicting instructions resolves the
   conflict in prose, and that prose lands in the HTML stream.

Neither is a writing-quality problem; both are missing checks. This module holds
the two mechanical guards: strip instruction echo / chain-of-thought out of the
markup, and refuse an edit that lost the body.

Both are reported, never silent: a stripped block is logged with its text so the
author can see what the model said before it was removed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# Phrases that only appear when a model narrates its instructions or reasoning.
_SCAFFOLD_MARKERS = re.compile(
    r"(?:"
    # First-person planning. A published article never says "the user wants me to".
    r"\bthe user (?:wants|asks|provided|said)\b"
    r"|\blet me (?:analyze|think|re-?read|look|check|create|construct|start)\b"
    r"|\bi (?:need|have|should|must|will|'ll|'m going) to\b"
    r"|\bi (?:can|could) (?:create|build|write|include)\b"
    r"|\bmy (?:job|task|goal) is\b"
    r"|\bthey'?ve provided\b"
    r"|\bthe instructions? (?:say|says|ask|asks|require|state)\b"
    r"|\bthis is (?:tricky|challenging|a challenge)\b"
    r"|\bthat seems like the best approach\b"
    r"|\bbest approach (?:here|is to)\b"
    r"|\bwait,?\s+(?:let'?s|let me|no\b|but\b|actually\b)"
    r"|\blet'?s check\b"
    r"|\bthe rule says\b"
    r"|\bnotice:\s*it says\b"
    r"|\bthe prompt (?:says|asks|requires)\b"
    r"|\bper the (?:instructions|prompt|rules)\b"
    r"|\boutput instructions\b"
    r"|\bcritical alignment\b"
    r"|\bdo not restate\b"
    r"|\bdo not mention prompts\b"
    r"|\btarget audience demographic\b"
    r"|\bprimary keyword:\s"
    r"|\bsystem prompt\b"
    r"|\bhow to reconcile\b"
    r"|\bsimply do not put\b"
    r")",
    re.IGNORECASE,
)

_PLACEHOLDER_RE = re.compile(
    r"(?:"
    r"\[\s*\d+\s*-\s*\d+\s+sentences?"
    r"|\[\s*\d+\s+bolded"
    r"|\[\s*(?:table|paragraph|h3|faq|author|url|insert|add|content)"
    r"|\bsection \d+ header\b"
    r"|\bcontent\.\.\.|\b\.\.\.\s*content"
    r"|\.\.\.\s*etc\.\s*\]"
    r")",
    re.IGNORECASE,
)

_BLOCK_RE = re.compile(r"<(p|li|h[1-6])\b[^>]*>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_LEADING_CHATTER_RE = re.compile(r"^[\s.,;:!?\"'`—–-]*(?:wait\b|okay\b|ok\b|hmm\b|let me\b|let'?s\b)", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _plain(fragment: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", fragment or "")).strip()


def looks_like_reasoning(text: str) -> bool:
    """True when a fragment reads as deliberation about the instructions."""
    if not text:
        return False
    plain = _plain(text)
    if len(plain) < 20:
        return False
    if _SCAFFOLD_MARKERS.search(plain):
        return True
    # A paragraph that quotes the instructions while talking about them.
    quotes = plain.count('"') + plain.count("“") + plain.count("’")
    return quotes >= 2 and any(word in plain.lower() for word in ("rule", "instruction", "prompt", "requirement"))


def strip_model_scaffolding(html: str) -> Tuple[str, Dict[str, Any]]:
    """Remove leaked reasoning and instruction echo from model-produced HTML.

    Returns ``(clean_html, report)`` where the report lists everything removed.
    """
    report: Dict[str, Any] = {"removed_blocks": 0, "removed_chars": 0, "samples": []}
    if not html:
        return html, report

    cleaned = html

    # 1. Chatter before the first block element ("`. Wait, let's check the rule:`
    #    ... <p>") - if the prefix has no markup and reads like speech, it is not
    #    part of the article.
    first_block = _BLOCK_RE.search(cleaned)
    if first_block and first_block.start() > 0:
        prefix = cleaned[: first_block.start()]
        if not _TAG_RE.search(prefix) and _LEADING_CHATTER_RE.match(prefix.strip()):
            report["removed_blocks"] += 1
            report["removed_chars"] += len(prefix)
            report["samples"].append(_plain(prefix)[:200])
            cleaned = cleaned[first_block.start():]

    # 2. Whole blocks that narrate the instructions.
    def _drop(match: "re.Match") -> str:
        inner = match.group(2)
        if looks_like_reasoning(inner):
            report["removed_blocks"] += 1
            report["removed_chars"] += len(match.group(0))
            if len(report["samples"]) < 8:
                report["samples"].append(_plain(inner)[:200])
            return ""
        return match.group(0)

    cleaned = _BLOCK_RE.sub(_drop, cleaned)

    # 2b. Contiguous runs of placeholder/skeleton text ("Section 1 header",
    #     "... content...", "[3 bolded bullets]"). Individually these read like
    #     ordinary short lines, together they are a template the model returned
    #     instead of the article.
    def _drop_placeholders(match: "re.Match") -> str:
        if _PLACEHOLDER_RE.search(_plain(match.group(2))):
            report["removed_blocks"] += 1
            report["removed_chars"] += len(match.group(0))
            if len(report["samples"]) < 8:
                report["samples"].append(_plain(match.group(2))[:200])
            return ""
        return match.group(0)

    cleaned = _BLOCK_RE.sub(_drop_placeholders, cleaned)

    # 3. Stray leading punctuation/speech outside tags.
    cleaned = re.sub(r"^[\s.,;:]+(?=<)", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    report["removed_blocks"] = int(report["removed_blocks"])
    return cleaned, report


def _section_count(html: str) -> int:
    return len(re.findall(r"<h[23]\b", html or "", re.IGNORECASE))


def polish_preserved_body(
    original: str,
    polished: str,
    *,
    min_length_ratio: float = 0.6,
    min_section_ratio: float = 0.7,
) -> Tuple[bool, str]:
    """Decide whether an editing pass kept the article.

    An editor may rewrite every sentence; it may not delete the body. Anything
    that lost more than ``1 - min_length_ratio`` of the text, or most of the
    sections, is a failed edit and must not replace the draft.

    Returns ``(ok, reason)``.
    """
    if not polished or not polished.strip():
        return False, "empty output"

    original_len = len(original or "")
    polished_len = len(polished)
    if original_len:
        ratio = polished_len / original_len
        if ratio < min_length_ratio:
            return False, (
                f"kept {polished_len} of {original_len} chars "
                f"({ratio:.0%}, minimum {min_length_ratio:.0%})"
            )

    original_sections = _section_count(original)
    polished_sections = _section_count(polished)
    if original_sections and polished_sections < original_sections * min_section_ratio:
        return False, f"kept {polished_sections} of {original_sections} sections"

    return True, "ok"


def leakage_summary(html: str) -> Dict[str, Any]:
    """Measure how much of a document is the model talking to itself.

    ``strip_model_scaffolding`` removes what it can recognise; this decides whether
    the result can be published at all. A draft whose leaked deliberation is more
    than a stray sentence cannot be repaired by deleting paragraphs - the pass that
    produced it failed, and the caller must keep the previous draft.
    """
    html = html or ""
    plain = _plain(html)
    blocks = [(_plain(m.group(2))) for m in _BLOCK_RE.finditer(html)]
    reasoning = [block for block in blocks if looks_like_reasoning(block)]
    markers = sorted({m.group(0).lower() for m in _SCAFFOLD_MARKERS.finditer(plain)})
    placeholders = _PLACEHOLDER_RE.findall(plain)

    leak_words = sum(len(block.split()) for block in reasoning)
    total_words = max(1, len(plain.split()))
    share = round(leak_words / total_words, 3)

    leaked = bool(len(reasoning) >= 2 or share > 0.05 or len(placeholders) >= 2)
    return {
        "leaked": leaked,
        "reasoning_paragraphs": len(reasoning),
        "reasoning_word_share": share,
        "markers": markers[:8],
        "placeholders": [p.strip() for p in placeholders[:8]],
        "samples": [block[:200] for block in reasoning[:2]],
    }
