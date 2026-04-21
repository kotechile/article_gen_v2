"""
Editorial topic generation service.
Generates research topics from project and category strategy, without SEO keyword validation.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List

from .llm.llm_service import llm_service

logger = logging.getLogger(__name__)


class EditorialTopicGenerationService:
    """Generate category-aware editorial research topics."""

    def _build_prompt(self, brief: Dict[str, Any]) -> str:
        count = int(brief.get("count") or 10)
        trend_titles = brief.get("trend_titles") or []
        trends_line = ", ".join(trend_titles[:8]) if trend_titles else "None"

        return f"""
You are a senior editorial strategist building research topic clusters for SEO and GEO workflows.

Your job is to propose editorial TOPICS, not keyword phrases and not article headlines.
Each topic must be broad enough to support multiple subtopics and articles later, but narrow enough to stay tightly aligned with the category strategy.

TOPIC BRIEF
- Project: {brief.get("project_name")}
- Project Description: {brief.get("project_description")}
- Primary Category: {brief.get("primary_category_name")}
- Primary Category Description: {brief.get("primary_category_description")}
- Sub-Category: {brief.get("secondary_category_name")}
- Sub-Category Description: {brief.get("secondary_category_description")}
- Category Path: {brief.get("category_path")}
- Category Strategy Hint: {brief.get("category_strategy_hint")}
- Recent Trend Themes: {trends_line}

RULES
- Generate exactly {count} editorial topic candidates.
- These topics must feel native to the category and sub-category strategy.
- Prefer decision spaces, frameworks, acquisition timing, comparisons, audits, operating models, and recurring high-value user problems.
- Avoid generic keyword buckets, vague lifestyle themes, and one-off article headlines.
- Avoid overly broad topics that could belong to almost any website.
- Use recent trend themes only as supporting freshness signals, never as the main driver.

OUTPUT FORMAT
Return only repeated blocks in this format:

[TOPIC]
TITLE: <broad editorial topic title>
RATIONALE: <1-2 sentence rationale for why this topic fits the site and can expand into multiple articles>
INTENT_BUCKET: <informational_decision|commercial_evaluation|decision_financial|solution_enablement>
DECISION_FOCUS: <one sentence describing the user decision this topic helps with>
ANGLE_QUESTION: <one concrete question this topic should answer>
VALUE_LAYER_TAGS: <tag 1>, <tag 2>
RELATED_TERMS: <term 1>, <term 2>, <term 3>, <term 4>
SOURCE_SIGNALS: AI, Category Strategy
[END]
"""

    def _parse(self, text: str) -> List[Dict[str, Any]]:
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

            title = fields.get("TITLE", "").strip()
            if not title:
                continue

            value_tags = [p.strip() for p in fields.get("VALUE_LAYER_TAGS", "").split(",") if p.strip()]
            related_terms = [p.strip() for p in fields.get("RELATED_TERMS", "").split(",") if p.strip()]
            source_signals = [p.strip() for p in fields.get("SOURCE_SIGNALS", "").split(",") if p.strip()]

            parsed.append(
                {
                    "title": title,
                    "rationale": fields.get("RATIONALE", ""),
                    "intent_bucket": fields.get("INTENT_BUCKET", "") or "informational_decision",
                    "decision_focus": fields.get("DECISION_FOCUS", ""),
                    "angle_question": fields.get("ANGLE_QUESTION", ""),
                    "value_layer_tags": value_tags[:3] or ["decision-support"],
                    "related_terms": related_terms[:6],
                    "source_signals": source_signals or ["AI", "Category Strategy"],
                }
            )
        return parsed

    def _fallback(self, brief: Dict[str, Any]) -> List[Dict[str, Any]]:
        primary = (brief.get("primary_category_name") or "").strip()
        secondary = (brief.get("secondary_category_name") or "").strip()
        project_name = (brief.get("project_name") or "selected website").strip()
        trends = brief.get("trend_titles") or []
        trend_hint = trends[0].strip() if trends else ""

        lane = " / ".join([part for part in [primary, secondary] if part]) or "this category"
        base_topics = [
            {
                "title": f"{secondary or primary or 'Category'} Decision Frameworks",
                "rationale": f"A foundational research cluster for {project_name} that organizes how readers make high-value decisions within {lane}.",
                "intent_bucket": "informational_decision",
                "decision_focus": f"Help readers make better decisions within {lane}.",
                "angle_question": f"What framework should readers use to evaluate opportunities in {lane}?",
                "value_layer_tags": ["decision-support", "roi-focused"],
                "related_terms": [primary, secondary, "decision framework", "evaluation model"],
                "source_signals": ["AI", "Category Strategy"],
            },
            {
                "title": f"{secondary or primary or 'Category'} Cost vs Value Audits",
                "rationale": f"A strong editorial lane for surfacing hidden costs, downside risk, and true value drivers inside {lane}.",
                "intent_bucket": "commercial_evaluation",
                "decision_focus": f"Help readers compare cost, risk, and upside before committing inside {lane}.",
                "angle_question": f"What hidden costs change the real value equation in {lane}?",
                "value_layer_tags": ["cost-vs-value", "hidden-cost-audit"],
                "related_terms": [primary, secondary, "cost of ownership", "hidden costs"],
                "source_signals": ["AI", "Category Strategy"],
            },
            {
                "title": f"{secondary or primary or 'Category'} Timing and Entry Strategy",
                "rationale": f"A durable topic cluster for users deciding when to act, wait, buy, enter, or re-evaluate within {lane}.",
                "intent_bucket": "decision_financial",
                "decision_focus": f"Help readers time actions and commitments more effectively inside {lane}.",
                "angle_question": f"When does acting now outperform waiting in {lane}?",
                "value_layer_tags": ["timing-decision", "roi-focused"],
                "related_terms": [primary, secondary, "timing strategy", "entry point"],
                "source_signals": ["AI", "Category Strategy"],
            },
            {
                "title": f"{secondary or primary or 'Category'} Comparison Models",
                "rationale": f"A comparison-oriented cluster that can branch into multiple subtopics and article formats while staying aligned to {lane}.",
                "intent_bucket": "commercial_evaluation",
                "decision_focus": f"Help readers compare competing options and paths inside {lane}.",
                "angle_question": f"What comparison model best reveals the right choice in {lane}?",
                "value_layer_tags": ["decision-support", "cost-vs-value"],
                "related_terms": [primary, secondary, "comparison", "tradeoffs"],
                "source_signals": ["AI", "Category Strategy"],
            },
        ]

        if trend_hint:
            base_topics.append(
                {
                    "title": f"{secondary or primary or 'Category'} Plays Shaped by {trend_hint}",
                    "rationale": f"A freshness-oriented cluster that uses recent trend signals while still staying grounded in {lane}.",
                    "intent_bucket": "solution_enablement",
                    "decision_focus": f"Help readers interpret how recent shifts affect strategy inside {lane}.",
                    "angle_question": f"How should readers adapt their approach in {lane} when {trend_hint} changes the environment?",
                    "value_layer_tags": ["timing-decision", "decision-support"],
                    "related_terms": [trend_hint, primary, secondary, "strategy shift"],
                    "source_signals": ["AI", "Category Strategy", "Trend Report"],
                }
            )

        count = int(brief.get("count") or 10)
        normalized = []
        seen = set()
        for topic in base_topics:
            title_key = (topic.get("title") or "").strip().lower()
            if not title_key or title_key in seen:
                continue
            seen.add(title_key)
            topic["related_terms"] = [term for term in topic.get("related_terms", []) if term]
            normalized.append(topic)
        return normalized[:count]

    async def generate(self, brief: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = self._build_prompt(brief)
        count = int(brief.get("count") or 10)
        try:
            response = await asyncio.wait_for(
                llm_service.generate_text(prompt=prompt, max_tokens=2200),
                timeout=40.0,
            )
            parsed = self._parse(response.content or "")
            if parsed:
                logger.info("Editorial topic generation succeeded count=%s", len(parsed))
                return parsed[:count]
        except Exception as e:
            logger.warning("Editorial topic generation failed: %s", e)

        fallback = self._fallback(brief)
        logger.info("Editorial topic generation used fallback count=%s", len(fallback))
        return fallback[:count]


editorial_topic_generation_service = EditorialTopicGenerationService()
