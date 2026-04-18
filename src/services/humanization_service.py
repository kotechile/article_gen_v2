"""
Humanization diagnostics and targeting helpers.

Phase 5:
- Detect robotic patterns
- Identify weak sections that need rewrite
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


_BANNED_PHRASES = [
    "in today's digital age",
    "game changer",
    "unlock the secrets",
    "take a deep dive",
    "it is important to note",
    "moreover",
    "furthermore",
]

_FILLER_WORDS = [
    "delve",
    "navigate",
    "leverage",
    "robust",
    "seamless",
    "transformative",
]

_HEDGING_WORDS = [
    "might", "may", "could", "possibly", "perhaps", "arguably", "generally", "typically"
]

_ABSTRACT_WORDS = [
    "framework", "ecosystem", "strategy", "optimization", "synergy", "methodology", "approach"
]


def _strip_html(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


def _sentence_lengths(text: str) -> List[int]:
    parts = re.split(r"(?<=[.!?])\s+", text or "")
    lengths = []
    for p in parts:
        wc = len(_words(p))
        if wc:
            lengths.append(wc)
    return lengths


def _std_dev(values: List[float]) -> float:
    if not values:
        return 0.0
    avg = sum(values) / len(values)
    var = sum((v - avg) ** 2 for v in values) / len(values)
    return var ** 0.5


def analyze_humanization(text: str) -> Dict[str, Any]:
    raw = _strip_html(text or "")
    words = _words(raw)
    wc = max(1, len(words))
    lower = raw.lower()
    sentence_lengths = _sentence_lengths(raw)

    banned_hits = [p for p in _BANNED_PHRASES if p in lower]
    filler_hits = sum(lower.count(w) for w in _FILLER_WORDS)
    hedging_hits = sum(len(re.findall(rf"\b{re.escape(w)}\b", lower)) for w in _HEDGING_WORDS)
    abstract_hits = sum(len(re.findall(rf"\b{re.escape(w)}\b", lower)) for w in _ABSTRACT_WORDS)
    sentence_std = _std_dev([float(v) for v in sentence_lengths])

    metrics = {
        "word_count": wc,
        "banned_phrase_hits": len(banned_hits),
        "banned_phrases": banned_hits[:8],
        "filler_density_per_1000": round((filler_hits / wc) * 1000.0, 2),
        "hedging_density_per_1000": round((hedging_hits / wc) * 1000.0, 2),
        "abstraction_density_per_1000": round((abstract_hits / wc) * 1000.0, 2),
        "sentence_length_std_dev": round(sentence_std, 2),
        "sentence_count": len(sentence_lengths),
    }

    weak_reasons: List[str] = []
    if metrics["banned_phrase_hits"] > 0:
        weak_reasons.append("banned_phrase")
    if metrics["filler_density_per_1000"] > 3.0:
        weak_reasons.append("filler_density")
    if metrics["hedging_density_per_1000"] > 8.0:
        weak_reasons.append("hedging_overload")
    if metrics["abstraction_density_per_1000"] > 10.0:
        weak_reasons.append("abstraction_density")
    if metrics["sentence_count"] >= 6 and metrics["sentence_length_std_dev"] < 5.0:
        weak_reasons.append("sentence_monotony")

    return {
        "metrics": metrics,
        "needs_humanization_rewrite": len(weak_reasons) > 0,
        "weak_reasons": weak_reasons,
    }

