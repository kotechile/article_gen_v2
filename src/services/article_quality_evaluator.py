"""
Article quality evaluator utilities.

Phase 0 instrumentation:
- Produce a machine-readable quality report
- Surface basic humanization, grounding, and GEO diagnostics
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List


_FILLER_PATTERNS = [
    r"\bin (today'?s|the) (fast[- ]paced )?world\b",
    r"\bit (is|[' ]s) important to note\b",
    r"\bdelve (into)?\b",
    r"\bnavigate\b",
    r"\bmoreover\b",
    r"\bfurthermore\b",
    r"\bin conclusion\b",
    r"\boverall\b",
]

_LOW_SUBSTANCE_PATTERNS = [
    r"\bthis section covers\b",
    r"\bthe content provides detailed information and insights\b",
    r"\boffering practical guidance and actionable advice\b",
    r"\bthis article demonstrates that\b",
    r"\ba quick guide to understanding the key concepts\b",
]


def _strip_html(text: str) -> str:
    if not text:
        return ""
    no_tags = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", no_tags).strip()


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _std_dev(values: List[float]) -> float:
    if not values:
        return 0.0
    avg = sum(values) / len(values)
    var = sum((v - avg) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def _repetition_diagnostics(text: str, min_phrase_len: int = 3, max_phrase_len: int = 5) -> Dict[str, Any]:
    tokens = _words(text)
    phrase_counts: Dict[str, int] = {}
    if len(tokens) < min_phrase_len:
        return {"top_repeated_phrases": [], "high_repetition_count": 0}

    for n in range(min_phrase_len, max_phrase_len + 1):
        for i in range(0, len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

    repeated = [(p, c) for p, c in phrase_counts.items() if c >= 3]
    repeated.sort(key=lambda item: item[1], reverse=True)
    top_repeated = [{"phrase": p, "count": c} for p, c in repeated[:8]]
    high_repetition_count = sum(1 for _, c in repeated if c >= 4)
    return {
        "top_repeated_phrases": top_repeated,
        "high_repetition_count": high_repetition_count,
    }


def _filler_density(text: str) -> Dict[str, Any]:
    lower_text = text.lower()
    hits = 0
    matched_patterns: List[str] = []
    for pattern in _FILLER_PATTERNS:
        count = len(re.findall(pattern, lower_text))
        if count > 0:
            hits += count
            matched_patterns.append(pattern)
    word_count = max(1, len(_words(text)))
    density = (hits / word_count) * 1000.0
    return {
        "filler_hits": hits,
        "filler_density_per_1000_words": round(density, 2),
        "matched_pattern_count": len(matched_patterns),
    }


def _low_substance_signals(text: str) -> Dict[str, Any]:
    lower_text = (text or "").lower()
    hits = 0
    matched: List[str] = []
    for pattern in _LOW_SUBSTANCE_PATTERNS:
        count = len(re.findall(pattern, lower_text))
        if count > 0:
            hits += count
            matched.append(pattern)
    return {
        "placeholder_phrase_hits": hits,
        "matched_pattern_count": len(matched),
    }


def _geo_signals(html_content: str, plain_text: str) -> Dict[str, Any]:
    lower_html = (html_content or "").lower()
    lower_text = (plain_text or "").lower()

    has_faq = ("<h2>faq" in lower_html) or ("frequently asked questions" in lower_text)
    has_table = "<table" in lower_html
    has_takeaways = "key takeaways" in lower_text
    has_answer_style_intro = any(
        phrase in lower_text
        for phrase in [
            "short answer",
            "quick answer",
            "in short",
            "here's the answer",
        ]
    )
    has_definition_pattern = bool(
        re.search(r"\b[A-Z][A-Za-z0-9 ]{2,30}\s+is\s+a[n]?\s+", plain_text or "")
    )

    signal_count = sum(
        [
            has_faq,
            has_table,
            has_takeaways,
            has_answer_style_intro,
            has_definition_pattern,
        ]
    )

    return {
        "has_faq": has_faq,
        "has_table": has_table,
        "has_key_takeaways": has_takeaways,
        "has_answer_style_intro": has_answer_style_intro,
        "has_definition_pattern": has_definition_pattern,
        "geo_signal_count": signal_count,
    }


def _passage_quality_signals(html_content: str, plain_text: str) -> Dict[str, Any]:
    text = plain_text or _strip_html(html_content or "")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs and text:
        paragraphs = [text]

    para_word_counts = [len(_words(p)) for p in paragraphs]
    quotable_count = 0
    answer_like_count = 0
    for p in paragraphs:
        wc = len(_words(p))
        # "Quotable" passages are concise, information-dense paragraphs.
        if 25 <= wc <= 90 and re.search(r"\b(is|are|means|because|therefore|best|should)\b", p.lower()):
            quotable_count += 1
        if any(m in p.lower() for m in ("in short", "short answer", "the answer is", "key takeaway")):
            answer_like_count += 1

    avg_para_words = round(sum(para_word_counts) / max(len(para_word_counts), 1), 2) if para_word_counts else 0.0
    paragraph_variance = round(_std_dev([float(v) for v in para_word_counts]), 2) if para_word_counts else 0.0
    extractability_score = 0.0
    extractability_score += min(1.0, quotable_count / 3.0) * 45.0
    extractability_score += min(1.0, answer_like_count / 2.0) * 35.0
    if avg_para_words <= 120:
        extractability_score += 20.0
    return {
        "paragraph_count": len(paragraphs),
        "average_paragraph_words": avg_para_words,
        "paragraph_word_std_dev": paragraph_variance,
        "quotable_passage_count": quotable_count,
        "answer_like_passage_count": answer_like_count,
        "extractability_score": round(max(0.0, min(100.0, extractability_score)), 1),
    }


def _entity_clarity_signals(title: str, plain_text: str) -> Dict[str, Any]:
    text = plain_text or ""
    # Heuristic named entities: Title Case tokens/phrases and acronym-like entities.
    title_entities = re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", title or "")
    body_entities = re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", text)
    acronym_entities = re.findall(r"\b[A-Z]{2,8}\b", text)
    entities = list(dict.fromkeys(title_entities + body_entities[:120] + acronym_entities[:80]))
    defined_count = 0
    ambiguous_entities: List[str] = []
    for ent in entities[:40]:
        # Entity considered "defined" if an explanatory pattern appears near first mention.
        pattern = rf"\b{re.escape(ent)}\b[^.:\n]{{0,80}}\b(is|are|means|refers to)\b"
        if re.search(pattern, text):
            defined_count += 1
        else:
            # Skip very common words accidentally captured as entities.
            if ent.lower() not in {"the", "and", "for", "with", "this", "that"}:
                ambiguous_entities.append(ent)
    entity_count = max(1, len(entities[:40]))
    clarity_score = round((defined_count / entity_count) * 100.0, 1)
    return {
        "entity_count": len(entities[:40]),
        "defined_entity_count": defined_count,
        "ambiguous_entity_count": max(0, len(ambiguous_entities[:20])),
        "ambiguous_entities_sample": ambiguous_entities[:10],
        "entity_clarity_score": clarity_score,
    }


def build_article_quality_report(
    title: str,
    html_content: str,
    plain_text: str,
    citations: List[Dict[str, Any]] | None,
    sections: List[Dict[str, Any]] | None,
    evidence_count: int = 0,
) -> Dict[str, Any]:
    """
    Build a machine-readable quality report for generated articles.
    """
    citations = citations or []
    sections = sections or []

    text = (plain_text or "").strip() or _strip_html(html_content or "")
    sentence_list = _split_sentences(text)
    sentence_lengths = [len(_words(s)) for s in sentence_list]
    word_count = len(_words(text))
    paragraph_count = max(1, len([p for p in re.split(r"\n\s*\n", text) if p.strip()]))

    sentence_count = len(sentence_list)
    avg_sentence_len = round(sum(sentence_lengths) / sentence_count, 2) if sentence_count else 0.0
    sentence_len_std = round(_std_dev([float(v) for v in sentence_lengths]), 2)

    repetition = _repetition_diagnostics(text)
    filler = _filler_density(text)
    low_substance = _low_substance_signals(text)
    geo = _geo_signals(html_content, text)
    passage_quality = _passage_quality_signals(html_content, text)
    entity_clarity = _entity_clarity_signals(title, text)

    citations_count = len(citations)
    section_count = len(sections)
    evidence_support_ratio = round(citations_count / max(1, section_count), 2)

    # Humanization score
    humanization_score = 100.0
    if repetition["high_repetition_count"] > 2:
        humanization_score -= 12.0
    if filler["filler_density_per_1000_words"] > 2.5:
        humanization_score -= 10.0
    if sentence_len_std < 6.0 and sentence_count > 8:
        humanization_score -= 8.0
    if avg_sentence_len > 28.0:
        humanization_score -= 6.0
    humanization_score = max(0.0, round(humanization_score, 1))

    # Grounding score
    grounding_score = 100.0
    if citations_count == 0:
        grounding_score -= 45.0
    if evidence_count < 3:
        grounding_score -= 25.0
    if evidence_support_ratio < 0.5:
        grounding_score -= 10.0
    grounding_score = max(0.0, round(grounding_score, 1))

    # GEO score
    geo_score = 40.0 + (geo["geo_signal_count"] * 12.0)
    geo_score += (passage_quality["extractability_score"] - 50.0) * 0.20
    geo_score += (entity_clarity["entity_clarity_score"] - 50.0) * 0.15
    if len(title or "") > 60:
        geo_score -= 8.0
    geo_score = max(0.0, min(100.0, round(geo_score, 1)))

    overall_score = round((humanization_score * 0.35) + (grounding_score * 0.4) + (geo_score * 0.25), 1)

    warnings: List[str] = []
    if citations_count == 0:
        warnings.append("No citations found; factual grounding is weak.")
    if evidence_count < 3:
        warnings.append("Low evidence count; consider deeper research coverage.")
    if repetition["high_repetition_count"] > 2:
        warnings.append("High phrase repetition detected; article may sound robotic.")
    if not geo["has_answer_style_intro"]:
        warnings.append("No answer-first intro signal detected for GEO.")
    if not geo["has_key_takeaways"]:
        warnings.append("No explicit key takeaways block detected.")
    if passage_quality["extractability_score"] < 45:
        warnings.append("Passage extractability is weak for generative engines.")
    if entity_clarity["entity_clarity_score"] < 45:
        warnings.append("Entity clarity is low; add explicit definitions/disambiguation.")
    if low_substance["placeholder_phrase_hits"] >= 2:
        warnings.append("Template-like placeholder phrasing detected; revise for specificity.")

    return {
        "version": "phase0_v1",
        "overall_score": overall_score,
        "humanization_score": humanization_score,
        "grounding_score": grounding_score,
        "geo_score": geo_score,
        "diagnostics": {
            "word_count": word_count,
            "paragraph_count": paragraph_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_len,
            "sentence_length_std_dev": sentence_len_std,
            "citations_count": citations_count,
            "section_count": section_count,
            "evidence_count": evidence_count,
            "evidence_support_ratio": evidence_support_ratio,
            "repetition": repetition,
            "filler": filler,
            "low_substance": low_substance,
            "geo_signals": geo,
            "passage_quality": passage_quality,
            "entity_clarity": entity_clarity,
        },
        "warnings": warnings,
    }
