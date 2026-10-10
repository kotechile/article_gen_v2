"""
Viral Title Generation and Review Engine.

Enforces core viral headline architecture:
1. Length & Scannability Rules (6-Word Rule, Front-Load Value, Omit Fluff)
2. Psychological Triggers & Angles (Curiosity Gap, Negative Framing/Loss Aversion, Counter-Intuitive Authority)
3. Specificity & Concrete Mechanics (Quantifiable Metrics, Explicit Target Identification, Zero Clickbait Bait-and-Switch)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Filler and fluff words to strip or penalize in viral headlines
FLUFF_TOKENS = {
    "a", "an", "the", "that", "this", "these", "those",
    "very", "really", "quite", "basically", "literally",
    "simply", "just", "how", "to", "guide", "introduction",
    "overview", "ultimate", "comprehensive", "essential",
    "simple", "easy", "steps", "things", "ways",
}

# Psychological trigger regex indicators
CURIOSITY_PATTERNS = [
    r"\bwhy\b",
    r"\bsecret\b",
    r"\bhidden\b",
    r"\bunexpected\b",
    r"\btruth\b",
    r"\brevealed\b",
    r"\bbehind\b",
    r"\bnobody\b",
    r"\bactually\b",
]

LOSS_AVERSION_PATTERNS = [
    r"\bmistake\b",
    r"\bmistakes\b",
    r"\bkills?\b",
    r"\btrap\b",
    r"\btraps\b",
    r"\bstop\b",
    r"\bnever\b",
    r"\bfail\b",
    r"\bfails\b",
    r"\bfailure\b",
    r"\bdanger\b",
    r"\bdangerous\b",
    r"\bcostly\b",
    r"\bwaste\b",
    r"\brisk\b",
    r"\bruin\b",
    r"\blosing\b",
]

COUNTER_INTUITIVE_PATTERNS = [
    r"\bwrong\b",
    r"\bmyth\b",
    r"\bdon'?t\b",
    r"\bignore\b",
    r"\bskip\b",
    r"\bquit\b",
    r"\binstead\b",
    r"\blie\b",
    r"\blies\b",
    r"\boverrated\b",
    r"\bworst\b",
]


class ViralTitleEngine:
    """Generates and reviews viral article titles and enforces first-paragraph payoff."""

    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def count_words(title: str) -> int:
        """Count clean words in title."""
        if not title:
            return 0
        clean = re.sub(r"[^\w\s\d%-]", " ", title)
        tokens = [t for t in clean.split() if t.strip()]
        return len(tokens)

    @staticmethod
    def extract_metrics(title: str) -> List[str]:
        """Extract numbers, percentages, or odd integers from title."""
        return re.findall(r"\b\d+(?:[\.,]\d+)?%?|\b(?:one|two|three|five|seven)\b", title, re.IGNORECASE)

    @classmethod
    def evaluate_title_heuristics(cls, title: str, target_audience: str = "") -> Dict[str, Any]:
        """
        Evaluate viral headline heuristics.
        
        Returns:
            Dictionary containing score (0-100), word count, trigger types, and detailed feedback.
        """
        raw_title = (title or "").strip().strip('"').strip("'")
        if not raw_title:
            return {
                "score": 0,
                "word_count": 0,
                "under_6_words": False,
                "front_loaded": False,
                "fluff_count": 0,
                "triggers": [],
                "has_metrics": False,
                "has_audience": False,
                "feedback": ["Title is empty."],
            }

        words = [w.strip() for w in re.sub(r"[^\w\s\d%-]", " ", raw_title).split() if w.strip()]
        word_count = len(words)
        under_6_words = word_count <= 6

        # Check front-loading: first 3 words should not be filler/articles
        first_three = words[:3]
        fluff_in_front = sum(1 for w in first_three if w.lower() in {"a", "an", "the", "in", "to", "for", "how"})
        front_loaded = (fluff_in_front == 0) and (len(first_three) > 0)

        # Fluff count
        fluff_tokens = [w for w in words if w.lower() in FLUFF_TOKENS]
        fluff_count = len(fluff_tokens)

        # Triggers
        lower_title = raw_title.lower()
        triggers = []
        if any(re.search(pat, lower_title) for pat in CURIOSITY_PATTERNS):
            triggers.append("curiosity_gap")
        if any(re.search(pat, lower_title) for pat in LOSS_AVERSION_PATTERNS):
            triggers.append("loss_aversion")
        if any(re.search(pat, lower_title) for pat in COUNTER_INTUITIVE_PATTERNS):
            triggers.append("counter_intuitive")

        # Metrics
        metrics = cls.extract_metrics(raw_title)
        has_metrics = len(metrics) > 0

        # Audience identification
        has_audience = False
        if target_audience:
            aud_tokens = [t.lower() for t in re.sub(r"[^\w\s]", " ", target_audience).split() if len(t) > 3]
            if any(tok in lower_title for tok in aud_tokens):
                has_audience = True

        # Calculate composite score
        score = 0
        # 1. Length & Scannability (35 pts max)
        if word_count <= 6:
            score += 25
        elif word_count <= 8:
            score += 18
        elif word_count <= 10:
            score += 10
        else:
            score += 5

        if front_loaded:
            score += 10
        elif fluff_in_front == 1:
            score += 5

        # Deduct for excess fluff
        score = max(0, score - (fluff_count * 2))

        # 2. Psychological Triggers (35 pts max)
        if len(triggers) >= 2:
            score += 35
        elif len(triggers) == 1:
            score += 25
        else:
            score += 10  # Baseline

        # 3. Specificity & Concrete Mechanics (30 pts max)
        if has_metrics:
            score += 15
        if has_audience:
            score += 15
        elif len(raw_title) <= 55 and len(raw_title) >= 15:
            score += 5

        score = min(100, max(0, score))

        feedback = []
        if not under_6_words:
            feedback.append(f"Title has {word_count} words; target is <= 6 words for zero truncation.")
        if not front_loaded:
            feedback.append("First 3 words contain filler; front-load the primary noun or core benefit.")
        if not triggers:
            feedback.append("Missing psychological hook (Curiosity Gap, Loss Aversion, or Counter-Intuitive Authority).")
        if not has_metrics:
            feedback.append("Consider adding a specific number, odd integer, or verifiable metric.")

        return {
            "title": raw_title,
            "score": score,
            "word_count": word_count,
            "under_6_words": under_6_words,
            "front_loaded": front_loaded,
            "fluff_count": fluff_count,
            "triggers": triggers,
            "has_metrics": has_metrics,
            "has_audience": has_audience,
            "char_count": len(raw_title),
            "feedback": feedback,
        }

    def generate_and_review_title(
        self,
        brief: str,
        keywords: str = "",
        article_type: str = "article",
        tone: str = "authoritative",
        draft_title: str = "",
        target_audience: str = "",
    ) -> Dict[str, Any]:
        """
        Full Title Generation and In-Generation Review Pipeline:
        1. Generates 4 diverse viral title candidates across psychological angles.
        2. Reviews each candidate against strict 6-word, scannability, fluff, and psychological triggers.
        3. Selects and optimizes the best title, returning the title + core promise for first-paragraph fulfillment.
        """
        if not self.llm_client:
            # Fallback heuristic mode
            candidate = draft_title or brief.split(".")[0][:50]
            candidate_cleaned = self._clean_title_text(candidate)
            eval_result = self.evaluate_title_heuristics(candidate_cleaned, target_audience)
            return {
                "title": candidate_cleaned,
                "core_promise": f"Actionable breakdown of {candidate_cleaned}",
                "evaluation": eval_result,
                "trigger_type": eval_result.get("triggers", ["authority"])[0] if eval_result.get("triggers") else "curiosity_gap",
            }

        try:
            # Step 1: Generate candidates across angles
            candidates = self._generate_viral_candidates(
                brief=brief,
                keywords=keywords,
                article_type=article_type,
                tone=tone,
                draft_title=draft_title,
                target_audience=target_audience,
            )

            # Step 2: In-Generation Review & Selection
            reviewed_result = self._review_and_select_candidate(
                candidates=candidates,
                brief=brief,
                keywords=keywords,
                target_audience=target_audience,
                tone=tone,
            )

            final_title = reviewed_result.get("selected_title") or (candidates[0].get("title") if candidates else draft_title)
            final_title = self._clean_title_text(final_title)
            core_promise = reviewed_result.get("core_promise") or f"Direct answer and solution for {final_title}"

            # Step 3: Heuristic verification & refinement loop
            eval_res = self.evaluate_title_heuristics(final_title, target_audience)
            
            # If still over 6 words or over 60 chars, run tight compression pass
            if eval_res["word_count"] > 6 or len(final_title) > 60:
                compressed = self._compress_to_six_words(final_title, keywords)
                if compressed and len(compressed) >= 10:
                    final_title = self._clean_title_text(compressed)
                    eval_res = self.evaluate_title_heuristics(final_title, target_audience)

            self.logger.info(
                f"Reviewed viral title: '{final_title}' (Words: {eval_res['word_count']}, "
                f"Score: {eval_res['score']}, Triggers: {eval_res['triggers']})"
            )

            return {
                "title": final_title,
                "core_promise": core_promise,
                "trigger_type": reviewed_result.get("trigger_type", "curiosity_gap"),
                "evaluation": eval_res,
                "candidates_reviewed": candidates,
            }

        except Exception as e:
            self.logger.error(f"Error in viral title generation and review: {e}")
            fallback = self._clean_title_text(draft_title or brief.split()[:5])
            if isinstance(fallback, list):
                fallback = " ".join(fallback)
            eval_res = self.evaluate_title_heuristics(fallback, target_audience)
            return {
                "title": fallback,
                "core_promise": f"Key insights into {fallback}",
                "evaluation": eval_res,
                "trigger_type": "loss_aversion",
            }

    def _generate_viral_candidates(
        self,
        brief: str,
        keywords: str,
        article_type: str,
        tone: str,
        draft_title: str,
        target_audience: str,
    ) -> List[Dict[str, str]]:
        """Generate candidate titles tailored to specific viral psychological angles."""
        prompt = f"""You are a master viral headline strategist.
Generate 4 DISTINCT, high-converting article title candidates for this piece.

CORE VIRAL HEADLINE RULES:
1. THE 6-WORD RULE: Aim for 4 to 6 words. MUST be under 60 characters total. Zero truncation on mobile.
2. FRONT-LOAD VALUE: Put the strongest noun or compelling benefit in the FIRST 3 WORDS.
3. OMIT FLUFF: Eliminate articles ("The", "A"), filler transitions, and passive words. Every word carries emotional or structural weight.
4. PSYCHOLOGICAL TRIGGERS: Each candidate MUST embody one specific psychological angle:
   - Candidate 1 (Curiosity Gap): Wide delta between what reader knows and what is promised (e.g., "Why Most X Fails", "Hidden Cost Behind X").
   - Candidate 2 (Negative Framing / Loss Aversion): Highlight pain avoidance, costly mistakes, or traps (e.g., "1 Fatal Mistake In X", "Stop Doing X Wrong").
   - Candidate 3 (Counter-Intuitive Authority): Challenge conventional wisdom or invert established norms (e.g., "Skip X: Do This Instead", "Why Experts Avoid X").
   - Candidate 4 (Metric & Target Specificity): Concrete odd integer, percentage, or explicit target audience callout (e.g., "3 Remote Traps Costing Thousands", "5 Freelancer Traps To Avoid").
5. ZERO CLICKBAIT BAIT-AND-SWITCH: Each headline must make a bold, verifiable promise that the article will satisfy in paragraph 1.

Article Context:
- Brief: {brief[:500]}
- Primary Keywords: {keywords}
- Tone: {tone}
- Target Audience: {target_audience or 'General audience'}
{f'- Draft Title: {draft_title}' if draft_title else ''}

Return STRICT JSON format:
{{
  "candidates": [
    {{
      "title": "4-6 word title",
      "angle": "curiosity_gap",
      "promise": "What specific promise this headline makes to the reader"
    }},
    {{
      "title": "4-6 word title",
      "angle": "loss_aversion",
      "promise": "What specific promise this headline makes to the reader"
    }},
    {{
      "title": "4-6 word title",
      "angle": "counter_intuitive",
      "promise": "What specific promise this headline makes to the reader"
    }},
    {{
      "title": "4-6 word title",
      "angle": "metric_specificity",
      "promise": "What specific promise this headline makes to the reader"
    }}
  ]
}}
"""
        messages = [
            {"role": "system", "content": "You generate high-converting viral headlines. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.generate(messages)
        content = (response.content or "").strip()
        
        # Parse JSON
        parsed = self._extract_json(content)
        if parsed and isinstance(parsed.get("candidates"), list) and len(parsed["candidates"]) > 0:
            return parsed["candidates"]

        # Fallback candidate list if JSON extraction failed
        lines = [line.strip().strip('"').strip("'") for line in content.split("\n") if line.strip() and not line.startswith("{")]
        candidates = []
        for line in lines[:4]:
            clean_line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if clean_line:
                candidates.append({"title": clean_line, "angle": "curiosity_gap", "promise": "Fulfill core premise"})
        return candidates or [{"title": draft_title or "Essential Guide", "angle": "curiosity_gap", "promise": "Deliver value"}]

    def _review_and_select_candidate(
        self,
        candidates: List[Dict[str, str]],
        brief: str,
        keywords: str,
        target_audience: str,
        tone: str,
    ) -> Dict[str, Any]:
        """
        Review step during generation:
        Critique each candidate against the 6-word rule, front-loading, fluff, psychological triggers, and bait-and-switch feasibility.
        Selects and polishes the winning headline.
        """
        candidates_json = json.dumps(candidates, indent=2)

        prompt = f"""You are the Chief Editor and Viral Headline Reviewer.
Your task is to REVIEW the candidate titles during generation and select the single most compelling, viral, and scannable title.

Review Criteria:
1. 6-Word Rule: Highest preference for titles with 4 to 6 words. Maximum 60 characters. Zero mobile truncation.
2. Front-Loaded Value: Strongest noun or benefit in the first 3 words.
3. Fluff Elimination: Strip articles ("The", "A"), filler modifiers, and passive words.
4. Psychological Power: High curiosity gap, loss aversion, or counter-intuitive authority.
5. Specificity: Odd numbers or clear target callouts where fitting.
6. Zero Clickbait Bait-and-Switch: Must make a concrete promise that the article's opening paragraph can deliver immediately.

Article Brief: {brief[:400]}
Keywords: {keywords}
Target Audience: {target_audience or 'General'}

Candidates to Review:
{candidates_json}

Select the best candidate or refine it into an even punchier version (strictly <= 6 words).
Return STRICT JSON:
{{
  "selected_title": "The reviewed, punchy <=6 word title",
  "trigger_type": "curiosity_gap|loss_aversion|counter_intuitive|metric_specificity",
  "core_promise": "The exact bold promise the first paragraph of the article must satisfy",
  "review_critique": "Brief explanation of why this title won and how fluff was eliminated"
}}
"""
        messages = [
            {"role": "system", "content": "You review and select viral titles. Return strict JSON only."},
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.generate(messages)
        content = (response.content or "").strip()
        parsed = self._extract_json(content)

        if parsed and parsed.get("selected_title"):
            return parsed

        # Fallback to candidate with best heuristic score
        best_cand = candidates[0]
        best_score = -1
        for cand in candidates:
            score_data = self.evaluate_title_heuristics(cand.get("title", ""), target_audience)
            if score_data["score"] > best_score:
                best_score = score_data["score"]
                best_cand = cand

        return {
            "selected_title": best_cand.get("title", ""),
            "trigger_type": best_cand.get("angle", "curiosity_gap"),
            "core_promise": best_cand.get("promise", "Immediate value delivery"),
            "review_critique": "Selected by heuristic ranking fallback",
        }

    def _compress_to_six_words(self, title: str, keywords: str) -> str:
        """Compress a title to 6 words or fewer while retaining its hook."""
        prompt = f"""Rewrite this headline to be STRICTLY 4 TO 6 WORDS while keeping its viral hook and core punch.
Remove articles (a, an, the) and all filler. Front-load the strongest noun.

Headline: "{title}"
Keywords to keep in mind: {keywords}

Output only the shortened title, no quotes."""

        messages = [
            {"role": "system", "content": "You compress headlines to 4-6 words. Output title only."},
            {"role": "user", "content": prompt},
        ]
        response = self.llm_client.generate(messages)
        compressed = (response.content or "").strip().strip('"').strip("'")
        return compressed

    def verify_first_paragraph_alignment(
        self,
        title: str,
        core_promise: str,
        first_paragraph_text: str,
    ) -> Dict[str, Any]:
        """
        Verify that the first paragraph satisfies the title promise (Zero Clickbait Bait-and-Switch).
        """
        clean_para = re.sub(r"<[^>]+>", " ", first_paragraph_text or "")
        clean_para = re.sub(r"\s+", " ", clean_para).strip()
        
        if not clean_para:
            return {
                "aligned": False,
                "score": 0,
                "warning": "First paragraph is missing; unable to verify promise satisfaction.",
            }

        # Check key terms from title in first paragraph
        title_words = [w.lower() for w in re.sub(r"[^\w\s]", " ", title).split() if len(w) > 3 and w.lower() not in FLUFF_TOKENS]
        para_lower = clean_para.lower()

        matched_terms = [w for w in title_words if w in para_lower]
        match_ratio = (len(matched_terms) / max(1, len(title_words)))

        # Promise check
        aligned = match_ratio >= 0.4 or any(w in para_lower for w in title_words)
        score = round(min(100.0, match_ratio * 100.0), 1)

        warning = None
        if not aligned:
            warning = f"Bait-and-switch risk: First paragraph does not clearly address title terms ({', '.join(title_words)})."

        return {
            "aligned": aligned,
            "score": score,
            "matched_terms": matched_terms,
            "warning": warning,
        }

    @staticmethod
    def _clean_title_text(text: str) -> str:
        """Strip markdown formatting, quotes, and colon labels from titles."""
        if not text:
            return ""
        cleaned = str(text).strip()
        cleaned = re.sub(r"^[\"']|[\"']$", "", cleaned)
        cleaned = re.sub(r"^(?:Title|Selected Title|Option \d+):\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[\*\_#]", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON object from LLM response."""
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try to find json code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        # Try raw braces
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        return None
