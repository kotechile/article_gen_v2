"""
Topic generation service.

Generates research topics with downstream routing metadata so the product can
decide whether a topic should follow the keyword-first, editorial-first, or
hybrid path.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List

from supabase_client import LLM_ROLE_RESEARCH_TOPIC_GENERATION
from .llm.llm_service import llm_service

logger = logging.getLogger(__name__)


class EditorialTopicGenerationService:
    """Generate category-aware research topics with mode and viability metadata."""

    _CONSULTANT_TERMS = {
        "framework": "guide",
        "audit": "checklist",
        "arbitrage": "cost gap",
        "scenario": "plan",
        "optimization": "improvements",
        "playbook": "step-by-step plan",
        "operating model": "way of working",
        "value chain": "cost chain",
        "capital allocation": "money decisions",
        "solvency": "financial stability",
    }

    _VALID_TOPIC_MODES = {"keyword_first", "editorial_first", "hybrid"}
    _VALID_VIABILITY_LABELS = {"high", "medium", "low"}
    _KEYWORD_SIGNAL_TERMS = {
        "best", "compare", "comparison", "pricing", "price", "cost", "software", "tool",
        "tools", "platform", "app", "calculator", "template", "review", "reviews",
        "alternatives", "alternative", "buying", "guide", "checklist",
    }
    _EDITORIAL_SIGNAL_TERMS = {
        "psychology", "behavior", "cultural", "future", "ethics", "narrative",
        "patterns", "mindset", "signals", "spending power", "decision patterns",
    }

    def _build_prompt(self, brief: Dict[str, Any], generation_mode: str) -> str:
        count = int(brief.get("count") or 10)
        trend_titles = brief.get("trend_titles") or []
        trends_line = ", ".join(trend_titles[:8]) if trend_titles else "None"
        category_path = brief.get("category_path") or "N/A"
        generation_mode = generation_mode if generation_mode in {"keyword_first", "editorial_first", "mixed"} else "mixed"

        mode_rules = {
            "keyword_first": """
- Focus only on topics with strong search potential.
- Prefer concrete user problems, comparisons, costs, alternatives, tools, platform selection, pricing, buying decisions, and repeated search behaviors.
- Titles should still be broad enough for multiple articles, but they must sound like a real search lane people actually explore.
- Avoid abstract strategy themes unless they clearly map to measurable search demand.
""",
            "editorial_first": """
- Focus on topics with strong informational or interpretive value.
- Prefer editorial themes, decision contexts, durable questions, user tradeoffs, and strategic framing.
- Do not force every topic into commercial search language.
- These topics should still be practical and human, but measurable keyword demand is optional.
""",
            "mixed": """
- Produce a balanced mix of keyword-friendly topics and editorial-first topics.
- At least one third of the output should be keyword-first or hybrid.
- At least one third of the output should be editorial-first or hybrid.
- Use the topic_mode field to make the downstream path explicit.
""",
        }[generation_mode]

        return f"""
You are a senior editorial strategist and SEO workflow planner.

Your job is to propose research TOPICS that can later become article ideas.
Every topic must also say which downstream path it belongs to:
- keyword_first
- editorial_first
- hybrid

TOPIC BRIEF
- Project: {brief.get("project_name")}
- Project Description: {brief.get("project_description")}
- Target Audience: {brief.get("target_audience")}
- Primary Category: {brief.get("primary_category_name")}
- Primary Category Description: {brief.get("primary_category_description")}
- Sub-Category: {brief.get("secondary_category_name")}
- Sub-Category Description: {brief.get("secondary_category_description")}
- Category Path: {category_path}
- Category Strategy Hint: {brief.get("category_strategy_hint")}
- Recent Trend Themes: {trends_line}

MODE
- Requested generation mode: {generation_mode}

RULES
- Generate exactly {count} topic candidates.
- Every topic must be broad enough to support multiple articles later, but narrow enough to stay tightly aligned with the category strategy.
- Titles must be plain, human language. Avoid consultant-speak and brochure wording.
- The topic title should sound like a real content lane, not a one-off headline and not a raw keyword string.
- Assign one topic_mode per topic:
  - keyword_first: strong chance of measurable keyword demand
  - editorial_first: valuable editorial topic even if keyword demand is weak
  - hybrid: could work in either path
- Assign one keyword_viability_score from 0-100 based on how likely this topic is to produce viable keyword candidates.
- Assign one keyword_viability_label:
  - high for 70-100
  - medium for 40-69
  - low for 0-39
- topic_generation_reasoning must explain in 1-2 short sentences why this topic belongs in that mode and why its keyword potential is high, medium, or low.
- Prefer practical user decisions, recurring user problems, comparisons, costs, evaluation paths, maintenance, lifecycle, support, tradeoffs, or tool potential when the topic naturally supports them.
- Ignore category wording if it conflicts with the actual topic opportunity.
{mode_rules}

OUTPUT FORMAT
Return only repeated blocks in this format:

[TOPIC]
TITLE: <broad topic title>
RATIONALE: <1-2 sentence rationale>
TOPIC_MODE: <keyword_first|editorial_first|hybrid>
KEYWORD_VIABILITY_SCORE: <0-100>
KEYWORD_VIABILITY_LABEL: <high|medium|low>
TOPIC_GENERATION_REASONING: <short explanation of mode + keyword potential>
INTENT_BUCKET: <informational_decision|commercial_evaluation|decision_financial|solution_enablement>
DECISION_FOCUS: <one sentence describing the user decision this topic helps with>
ANGLE_QUESTION: <one concrete question this topic should answer>
VALUE_LAYER_TAGS: <tag 1>, <tag 2>
RELATED_TERMS: <term 1>, <term 2>, <term 3>, <term 4>
SOURCE_SIGNALS: AI, Category Strategy
[END]
"""

    def _normalize_title_plain_language(self, title: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(title or "").strip())
        if not cleaned:
            return ""

        normalized = cleaned
        for jargon, simple in self._CONSULTANT_TERMS.items():
            normalized = re.sub(rf"\b{re.escape(jargon)}\b", simple, normalized, flags=re.IGNORECASE)

        normalized = re.sub(r"\s{2,}", " ", normalized).strip(" -:")
        return normalized

    def _heuristic_keyword_viability_score(self, title: str, rationale: str, related_terms: List[str], intent_bucket: str) -> int:
        text = " ".join([title, rationale, " ".join(related_terms or []), intent_bucket or ""]).lower()
        score = 50
        word_count = len([token for token in re.findall(r"[a-zA-Z0-9]+", title.lower()) if token])

        if word_count in (3, 4, 5):
            score += 8
        elif word_count <= 2:
            score -= 10
        elif word_count >= 7:
            score -= 8

        for term in self._KEYWORD_SIGNAL_TERMS:
            if term in text:
                score += 7

        for term in self._EDITORIAL_SIGNAL_TERMS:
            if term in text:
                score -= 8

        if intent_bucket in {"commercial_evaluation", "solution_enablement"}:
            score += 8
        elif intent_bucket == "informational_decision":
            score -= 4

        if re.search(r"\bhow\b|\bwhat\b|\bwhen\b", title.lower()):
            score += 4

        return max(0, min(100, score))

    def _finalize_topic_mode(self, llm_mode: str, viability_score: int, title: str, rationale: str) -> str:
        text = " ".join([title, rationale]).lower()
        if viability_score >= 75:
            return "keyword_first"
        if viability_score <= 34:
            return "editorial_first"
        if any(term in text for term in ["software", "pricing", "cost", "compare", "best", "platform", "tool"]):
            return "keyword_first"
        if any(term in text for term in ["psychology", "behavior", "future", "culture", "patterns"]):
            return "editorial_first"
        return llm_mode if llm_mode in self._VALID_TOPIC_MODES else "hybrid"

    def _coerce_topic_mode(self, value: Any, title: str, rationale: str, related_terms: List[str]) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in self._VALID_TOPIC_MODES:
            return normalized

        text = " ".join([title, rationale, " ".join(related_terms or [])]).lower()
        if any(term in text for term in ["software", "tool", "platform", "pricing", "cost", "best", "compare", "comparison", "calculator"]):
            return "keyword_first"
        if any(term in text for term in ["why", "future", "psychology", "behavior", "trade-offs", "tradeoffs", "spending power", "decision context"]):
            return "editorial_first"
        return "hybrid"

    def _coerce_viability_score(self, value: Any, topic_mode: str, title: str, related_terms: List[str]) -> int:
        try:
            score = int(float(value))
        except Exception:
            score = 0

        if score <= 0:
            text = " ".join([title, " ".join(related_terms or [])]).lower()
            if topic_mode == "keyword_first":
                score = 76 if any(term in text for term in ["cost", "pricing", "software", "tool", "compare", "best"]) else 68
            elif topic_mode == "editorial_first":
                score = 28 if any(term in text for term in ["psychology", "behavior", "future", "culture", "ethics"]) else 36
            else:
                score = 55
        return max(0, min(100, score))

    def _coerce_viability_label(self, value: Any, score: int) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in self._VALID_VIABILITY_LABELS:
            return normalized
        if score >= 70:
            return "high"
        if score >= 40:
            return "medium"
        return "low"

    def _build_generation_metadata(
        self,
        topic_mode: str,
        viability_score: int,
        viability_label: str,
        generation_mode: str,
        llm_topic_mode: str,
        llm_viability_score: int,
        heuristic_viability_score: int,
    ) -> Dict[str, Any]:
        return {
            "generator_version": "topic_mode_split_v1",
            "prompt_mode": generation_mode,
            "topic_mode": topic_mode,
            "keyword_viability_score": viability_score,
            "keyword_viability_label": viability_label,
            "llm_topic_mode": llm_topic_mode,
            "llm_keyword_viability_score": llm_viability_score,
            "heuristic_keyword_viability_score": heuristic_viability_score,
        }

    def _parse(self, text: str, generation_mode: str) -> List[Dict[str, Any]]:
        blocks = re.findall(r"\[TOPIC\](.*?)\[END\]", text, flags=re.DOTALL | re.IGNORECASE)
        parsed: List[Dict[str, Any]] = []
        for block in blocks:
            fields: Dict[str, str] = {}
            for raw_line in block.splitlines():
                line = raw_line.strip()
                if not line or ":" not in line:
                    continue
                key, val = line.split(":", 1)
                fields[key.strip().upper()] = val.strip()

            title = self._normalize_title_plain_language(fields.get("TITLE", ""))
            if not title:
                continue

            value_tags = [p.strip() for p in fields.get("VALUE_LAYER_TAGS", "").split(",") if p.strip()]
            related_terms = [p.strip() for p in fields.get("RELATED_TERMS", "").split(",") if p.strip()]
            source_signals = [p.strip() for p in fields.get("SOURCE_SIGNALS", "").split(",") if p.strip()]
            rationale = fields.get("RATIONALE", "")
            llm_topic_mode = self._coerce_topic_mode(fields.get("TOPIC_MODE", ""), title, rationale, related_terms)
            llm_viability_score = self._coerce_viability_score(fields.get("KEYWORD_VIABILITY_SCORE"), llm_topic_mode, title, related_terms)
            heuristic_viability_score = self._heuristic_keyword_viability_score(
                title=title,
                rationale=rationale,
                related_terms=related_terms,
                intent_bucket=fields.get("INTENT_BUCKET", "") or "informational_decision",
            )
            viability_score = int(round((llm_viability_score * 0.65) + (heuristic_viability_score * 0.35)))
            topic_mode = self._finalize_topic_mode(
                llm_mode=llm_topic_mode,
                viability_score=viability_score,
                title=title,
                rationale=rationale,
            )
            viability_label = self._coerce_viability_label(fields.get("KEYWORD_VIABILITY_LABEL", ""), viability_score)

            parsed.append(
                {
                    "title": title,
                    "rationale": rationale,
                    "topic_mode": topic_mode,
                    "keyword_viability_score": viability_score,
                    "keyword_viability_label": viability_label,
                    "topic_generation_reasoning": fields.get("TOPIC_GENERATION_REASONING", ""),
                    "topic_generation_metadata": self._build_generation_metadata(
                        topic_mode=topic_mode,
                        viability_score=viability_score,
                        viability_label=viability_label,
                        generation_mode=generation_mode,
                        llm_topic_mode=llm_topic_mode,
                        llm_viability_score=llm_viability_score,
                        heuristic_viability_score=heuristic_viability_score,
                    ),
                    "intent_bucket": fields.get("INTENT_BUCKET", "") or "informational_decision",
                    "decision_focus": fields.get("DECISION_FOCUS", ""),
                    "angle_question": fields.get("ANGLE_QUESTION", ""),
                    "value_layer_tags": value_tags[:4] or ["decision-support"],
                    "related_terms": related_terms[:8],
                    "source_signals": source_signals or ["AI", "Category Strategy"],
                }
            )
        return parsed

    def _fallback_topic(
        self,
        title: str,
        rationale: str,
        topic_mode: str,
        viability_score: int,
        intent_bucket: str,
        decision_focus: str,
        angle_question: str,
        value_layer_tags: List[str],
        related_terms: List[str],
        source_signals: List[str],
        generation_mode: str,
    ) -> Dict[str, Any]:
        viability_label = self._coerce_viability_label(None, viability_score)
        return {
            "title": self._normalize_title_plain_language(title),
            "rationale": rationale,
            "topic_mode": topic_mode,
            "keyword_viability_score": viability_score,
            "keyword_viability_label": viability_label,
            "topic_generation_reasoning": f"This topic is classified as {topic_mode} with {viability_label} keyword potential based on how concrete and search-shaped it is.",
            "topic_generation_metadata": self._build_generation_metadata(
                topic_mode=topic_mode,
                viability_score=viability_score,
                viability_label=viability_label,
                generation_mode=generation_mode,
                llm_topic_mode=topic_mode,
                llm_viability_score=viability_score,
                heuristic_viability_score=viability_score,
            ),
            "intent_bucket": intent_bucket,
            "decision_focus": decision_focus,
            "angle_question": angle_question,
            "value_layer_tags": value_layer_tags[:4],
            "related_terms": [term for term in related_terms if term][:8],
            "source_signals": source_signals or ["AI", "Category Strategy"],
        }

    def _fallback(self, brief: Dict[str, Any], generation_mode: str) -> List[Dict[str, Any]]:
        primary = (brief.get("primary_category_name") or "").strip()
        secondary = (brief.get("secondary_category_name") or "").strip()
        project_name = (brief.get("project_name") or "selected website").strip()
        trends = brief.get("trend_titles") or []
        trend_hint = trends[0].strip() if trends else ""
        lane = " / ".join([part for part in [primary, secondary] if part]) or "this category"
        core_name = secondary or primary or "Category"

        base_topics = [
            self._fallback_topic(
                title=f"{core_name} Costs and Tradeoffs",
                rationale=f"A search-shaped topic cluster for {project_name} that can expand into cost, comparison, and decision articles inside {lane}.",
                topic_mode="keyword_first",
                viability_score=74,
                intent_bucket="commercial_evaluation",
                decision_focus=f"Help readers compare the real cost and value tradeoffs inside {lane}.",
                angle_question=f"What tradeoffs matter most when someone evaluates {core_name.lower()} options?",
                value_layer_tags=["cost-vs-value", "decision-support"],
                related_terms=[primary, secondary, "cost", "comparison", "value"],
                source_signals=["AI", "Category Strategy"],
                generation_mode=generation_mode,
            ),
            self._fallback_topic(
                title=f"{core_name} Buying and Selection Guide",
                rationale=f"A keyword-friendly topic lane for recurring selection and comparison decisions inside {lane}.",
                topic_mode="keyword_first",
                viability_score=78,
                intent_bucket="commercial_evaluation",
                decision_focus=f"Help readers choose the best option inside {lane}.",
                angle_question=f"How should someone compare and select the right {core_name.lower()} option?",
                value_layer_tags=["decision-support", "roi-focused"],
                related_terms=[primary, secondary, "best", "compare", "buying guide"],
                source_signals=["AI", "Category Strategy"],
                generation_mode=generation_mode,
            ),
            self._fallback_topic(
                title=f"{core_name} Decision Patterns",
                rationale=f"A broader editorial topic for recurring user tradeoffs and decision behavior inside {lane}.",
                topic_mode="editorial_first",
                viability_score=34,
                intent_bucket="informational_decision",
                decision_focus=f"Help readers understand the recurring decisions and tradeoffs inside {lane}.",
                angle_question=f"What hidden decision patterns shape outcomes in {lane}?",
                value_layer_tags=["decision-support"],
                related_terms=[primary, secondary, "decision making", "tradeoffs"],
                source_signals=["AI", "Category Strategy"],
                generation_mode=generation_mode,
            ),
            self._fallback_topic(
                title=f"{core_name} Timing and Planning",
                rationale=f"A hybrid topic that can support both keyword-driven and editorial articles around timing, sequencing, and practical planning.",
                topic_mode="hybrid",
                viability_score=58,
                intent_bucket="decision_financial",
                decision_focus=f"Help readers decide when and how to act inside {lane}.",
                angle_question=f"When does acting now outperform waiting in {lane}?",
                value_layer_tags=["timing-decision", "decision-support"],
                related_terms=[primary, secondary, "timing", "plan", "strategy"],
                source_signals=["AI", "Category Strategy"],
                generation_mode=generation_mode,
            ),
        ]

        if trend_hint:
            base_topics.append(
                self._fallback_topic(
                    title=f"{core_name} Shifts Driven by {trend_hint}",
                    rationale=f"A hybrid trend-aware topic that can support both editorial interpretation and keyword-shaped follow-ups.",
                    topic_mode="hybrid",
                    viability_score=52,
                    intent_bucket="solution_enablement",
                    decision_focus=f"Help readers interpret how {trend_hint} changes decisions in {lane}.",
                    angle_question=f"How should readers adjust when {trend_hint} changes the landscape?",
                    value_layer_tags=["timing-decision", "decision-support"],
                    related_terms=[trend_hint, primary, secondary, "strategy shift"],
                    source_signals=["AI", "Category Strategy", "Trend Report"],
                    generation_mode=generation_mode,
                )
            )

        count = int(brief.get("count") or 10)
        normalized: List[Dict[str, Any]] = []
        seen = set()
        for topic in base_topics:
            title_key = (topic.get("title") or "").strip().lower()
            if not title_key or title_key in seen:
                continue
            seen.add(title_key)
            normalized.append(topic)
        return normalized[:count]

    async def generate(self, brief: Dict[str, Any], generation_mode: str = "mixed") -> List[Dict[str, Any]]:
        prompt = self._build_prompt(brief, generation_mode=generation_mode)
        count = int(brief.get("count") or 10)
        try:
            response = await asyncio.wait_for(
                llm_service.generate_text(
                    prompt=prompt,
                    task_role=LLM_ROLE_RESEARCH_TOPIC_GENERATION,
                    max_tokens=2600,
                ),
                timeout=45.0,
            )
            parsed = self._parse(response.content or "", generation_mode=generation_mode)
            if parsed:
                logger.info(
                    "Topic generation succeeded count=%s generation_mode=%s",
                    len(parsed),
                    generation_mode,
                )
                return parsed[:count]
        except Exception as e:
            logger.warning("Topic generation failed mode=%s err=%s", generation_mode, e)

        fallback = self._fallback(brief, generation_mode=generation_mode)
        logger.info("Topic generation used fallback count=%s mode=%s", len(fallback), generation_mode)
        return fallback[:count]


editorial_topic_generation_service = EditorialTopicGenerationService()
